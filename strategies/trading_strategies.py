import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression, LinearRegression

class TradingStrategy():
    """Trading Strategy for Supervised Learning based models, implementing different trading strategies using Kelly criterion for optimal bet sizing."""
    def __init__(self, wallet_a, wallet_b, frac_kelly, trade_threshold):
        """Initialize the TradingStrategy class with the initial wallet balances and Kelly fraction option."""
        self.frac_kelly = frac_kelly
                                   
        self.trade_threshold = trade_threshold
        # Initialize wallets for different trading strategies
        strategy_keys = ['mean_reversion', 'trend', 'pure_forecasting', 'hybrid_mean_reversion', 'hybrid_trend', 'ensemble']
        self.wallet_a = {key: wallet_a for key in strategy_keys}
        self.wallet_b = {key: wallet_b for key in strategy_keys}
        
        # Profit, trades, wins/losses tracking for each strategy
        self.total_profit_or_loss = {key: 0 for key in strategy_keys}
        self.num_trades = {key: 0 for key in strategy_keys}
        self.num_wins = {key: 0 for key in strategy_keys}
        self.num_losses = {key: 0 for key in strategy_keys}
        self.total_gains = {key: 0 for key in strategy_keys}
        self.total_losses = {key: 0 for key in strategy_keys}
        
        # Position tracking (adding extra state for mean reversion waiting for reversion)
        self.open_positions = {
            key: {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0, 'waiting_for_reversion': False}
            for key in strategy_keys
        }

        self.min_trades_for_full_kelly = 50  # Minimum trades before using full Kelly
        self.fixed_position_size = 1000  # Fixed position size for training
        
        # Initialize linear regression model
        self.linear_model = LinearRegression()
        self.trained = False

        self.trend_coeff = 0
        self.forecasting_coeff = 0

        self.spread = 0
        self.trade_threshold = 0.001
        
    def calculate_profit_for_signals(self, curr_ratio, next_ratio):
        """Calculate potential profit for given signals using fixed position size."""
        max_profit = float('-inf')
        best_direction = 'no_trade'
        
        # Try both possible trade directions
        for direction in ['buy_currency_a', 'sell_currency_a']:
            # Calculate profit for this direction
            if direction == 'buy_currency_a':
                profit = self.fixed_position_size * (next_ratio - curr_ratio - self.spread / 2) / next_ratio
            else:  # sell_currency_a
                profit = self.fixed_position_size * (curr_ratio - next_ratio - self.spread / 2) / next_ratio
            
            # Update best direction if this profit is higher
            if profit > max_profit:
                max_profit = profit
                best_direction = direction
        
        return max_profit, best_direction
    
    def win_loss_ratio(self, strategy_name):
        """Calculate the win/loss ratio for a strategy with basic smoothing."""
        total_trades = self.num_wins[strategy_name] + self.num_losses[strategy_name]
        
        if total_trades == 0:
            return 1.5  # Conservative default
            
        # Use consistent scaling factor
        confidence = min(1.0, total_trades / self.min_trades_for_full_kelly)
        smoothing = max(0.1, 1.0 - confidence)
        
        # Calculate averages with basic error handling
        avg_gain = (self.total_gains[strategy_name] / self.num_wins[strategy_name]) if self.num_wins[strategy_name] else 1.0
        avg_loss = (self.total_losses[strategy_name] / self.num_losses[strategy_name]) if self.num_losses[strategy_name] else 1.0
        
        # Apply smoothing and return with floor
        return max(0.1, (avg_gain + smoothing) / (avg_loss + smoothing))

    def win_probability(self, strategy_name):
        """Calculate win probability with basic statistical adjustment."""
        total_trades = self.num_wins[strategy_name] + self.num_losses[strategy_name]
        
        if total_trades == 0:
            return 0.5  # Neutral default
            
        # Basic win rate
        win_rate = self.num_wins[strategy_name] / total_trades
        
        # Use consistent scaling
        confidence = min(1.0, total_trades / self.min_trades_for_full_kelly)
        adjusted_rate = (win_rate * confidence) + (0.5 * (1 - confidence))
        
        # Keep within reasonable bounds
        return max(0.1, min(0.9, adjusted_rate))

    def kelly_criterion(self, strategy_name):
        """Calculate Kelly fraction with basic risk controls."""
        # Get core metrics
        win_prob = self.win_probability(strategy_name)
        win_ratio = self.win_loss_ratio(strategy_name)
        total_trades = self.num_wins[strategy_name] + self.num_losses[strategy_name]

        # Basic Kelly calculation
        kelly = win_prob - ((1 - win_prob) / win_ratio)

        # Single confidence adjustment based on trade count
        confidence = min(1.0, total_trades / self.min_trades_for_full_kelly)
        kelly *= confidence

        # Return bounded result
        return max(0.01, min(0.25, kelly))

    def calculate_profit(self, strategy_name, trade_direction, curr_ratio, f_i, use_kelly):
        bid_price = curr_ratio + self.spread / 2
        ask_price = curr_ratio - self.spread / 2
        """Calculate profit/loss and handle position management"""        
        profit_in_base_currency = 0
        
        # Check if there's an open position
        if self.open_positions[strategy_name]['type'] is not None:
            # If new trade direction is different from current position type, close the position
            current_position_type = 'long' if self.open_positions[strategy_name]['type'] == 'long' else 'short'
            new_position_type = 'long' if trade_direction == 'buy_currency_a' else 'short' if trade_direction == 'sell_currency_a' else None
            
            if new_position_type is not None and new_position_type != current_position_type:
                profit_in_base_currency += self.close_position(strategy_name, curr_ratio)
            else:
                # If same type or no trade, don't make a new trade
                return profit_in_base_currency
        
        # Then open new position if there's a trade signal and no matching position type
        if trade_direction != 'no_trade':
            # Calculate total portfolio value in currency A
            total_value_in_a = self.wallet_a[strategy_name] + (self.wallet_b[strategy_name] / ask_price)

            if(use_kelly):
                base_bet_size_a = f_i * total_value_in_a
            else:
                base_bet_size_a = self.fixed_position_size
            
            if trade_direction == 'buy_currency_a':
                bet_size_a = min(base_bet_size_a, self.wallet_a[strategy_name])
                bet_size_b = bet_size_a * ask_price
                
                # Check if we have enough B
                if bet_size_b <= self.wallet_b[strategy_name]:
                    self.wallet_b[strategy_name] -= bet_size_b
                    self.wallet_a[strategy_name] += bet_size_a
                    
                    self.open_positions[strategy_name] = {
                        'type': 'long',
                        'size_a': bet_size_a,
                        'size_b': bet_size_b,
                        'entry_ratio': ask_price
                    }
                
            elif trade_direction == 'sell_currency_a':
                bet_size_a = min(base_bet_size_a, self.wallet_a[strategy_name])
                bet_size_b = bet_size_a * bid_price
                
                if bet_size_a <= self.wallet_a[strategy_name]:
                    self.wallet_a[strategy_name] -= bet_size_a
                    self.wallet_b[strategy_name] += bet_size_b
                    
                    self.open_positions[strategy_name] = {
                        'type': 'short',
                        'size_a': bet_size_a,
                        'size_b': bet_size_b,
                        'entry_ratio': bid_price
                    }
        
        return profit_in_base_currency
    
    def close_position(self, strategy_name, curr_ratio):
        """Close an open position and calculate profit/loss"""
        position = self.open_positions[strategy_name]
        profit = 0
        
        if position['type'] == 'long':
            # Close long position (sell currency A)
            exit_price = curr_ratio - self.spread / 2
            self.wallet_a[strategy_name] -= position['size_a']
            self.wallet_b[strategy_name] += position['size_a'] * exit_price
            # Calculate profit
            profit = position['size_a'] * (exit_price - position['entry_ratio']) / exit_price
            
        elif position['type'] == 'short':
            # Close short position (buy currency A)
            exit_price = curr_ratio + self.spread / 2
            self.wallet_b[strategy_name] -= position['size_b']
            self.wallet_a[strategy_name] += position['size_b'] / exit_price
            # Calculate profit
            profit = position['size_a'] * (position['entry_ratio'] - exit_price) / exit_price
        
        # Reset position tracking
        self.open_positions[strategy_name] = {
            'type': None,
            'size_a': 0,
            'size_b': 0,
            'entry_ratio': 0,
            'waiting_for_reversion': False
        }
        
        return profit
    
    def analyze_mean_reversion_window(self, recent_rates, min_crossings=10, deviation_multiplier=3):
        """
        Analyze the recent window of exchange rates.
        
        Parameters:
          recent_rates: list or np.array of exchange rates for the past N minutes.
          min_crossings: minimum number of median crossings required.
          deviation_multiplier: how many times the spread the deviation must exceed.
          
        Returns:
          (trade_direction, median_rate): trade_direction is 'buy_currency_a', 'sell_currency_a', or 'no_trade'
        """
        if len(recent_rates) < 2:
            return 'no_trade', None

        median_rate = np.median(recent_rates)
        crossings = 0
        for i in range(1, len(recent_rates)):
            # A crossing occurs when the sign of (rate - median) changes.
            if (recent_rates[i-1] - median_rate) * (recent_rates[i] - median_rate) < 0:
                crossings += 1

        if crossings < min_crossings:
            return 'no_trade', median_rate

        current_rate = recent_rates[-1]
        deviation = abs(current_rate - median_rate)
        threshold = deviation_multiplier * self.spread  # using spread as the transaction cost

        if deviation < threshold:
            return 'no_trade', median_rate

        # If current rate is above the median, expect a downward reversion, so sell.
        if current_rate > median_rate:
            return 'sell_currency_a', median_rate
        else:
            return 'buy_currency_a', median_rate

    def determine_trade_direction(self, strategy_name, base_ratio_change, predicted_ratio_change):
        """Determine the trade direction based on strategy and ratio changes."""
        trade_direction = 'no_trade'

        if(strategy_name == 'mean_reversion'):
            # Mean reversion strategy: trade against significant ratio changes
            if base_ratio_change > self.trade_threshold + self.spread / 2:
                trade_direction = 'buy_currency_a'
            elif base_ratio_change < -self.trade_threshold - self.spread / 2:
                trade_direction = 'sell_currency_a'

                
        elif(strategy_name == 'trend'):
            # Trend strategy: trade towards significant ratio changes
            if base_ratio_change > self.trade_threshold + self.spread / 2:
                trade_direction = 'buy_currency_a'
            elif base_ratio_change < -self.trade_threshold - self.spread / 2:
                trade_direction = 'sell_currency_a'
                
        elif(strategy_name == 'pure_forecasting'):
            # Pure forecasting strategy: trade based on predicted future ratio changes
            if predicted_ratio_change > self.trade_threshold + self.spread / 2:
                trade_direction = 'buy_currency_a'
            elif predicted_ratio_change < -self.trade_threshold - self.spread / 2:
                trade_direction = 'sell_currency_a'
                
        elif(strategy_name == 'hybrid_mean_reversion'):
            # Hybrid strategy: combine mean reversion and pure forecasting signals
            if base_ratio_change < -self.trade_threshold and predicted_ratio_change > self.trade_threshold:
                trade_direction = 'buy_currency_a'
            elif base_ratio_change > self.trade_threshold and predicted_ratio_change < -self.trade_threshold:
                trade_direction = 'sell_currency_a'
                
        elif(strategy_name == 'hybrid_trend'):
            # Hybrid strategy: combine trend and pure forecasting signals
            if base_ratio_change > self.trade_threshold and predicted_ratio_change > self.trade_threshold:
                trade_direction = 'buy_currency_a'
            elif base_ratio_change < -self.trade_threshold and predicted_ratio_change < -self.trade_threshold:
                trade_direction = 'sell_currency_a'
            
        return trade_direction
    
    def get_strategy_signals(self, base_ratio_change, predicted_ratio_change, curr_ratio, next_ratio):
        """Get the signals (+1, 0, -1) for each strategy."""
        signals = {}
        for strategy_name in ['trend', 'pure_forecasting']:
            trade_direction = self.determine_trade_direction(strategy_name, base_ratio_change, predicted_ratio_change)
            if trade_direction == 'buy_currency_a':
                signals[strategy_name] = self.fixed_position_size * (next_ratio - curr_ratio) / next_ratio
            elif trade_direction == 'sell_currency_a':
                signals[strategy_name] = self.fixed_position_size * (curr_ratio - next_ratio) / next_ratio
            else:
                signals[strategy_name] = 0
        return signals
    
    def train_linear_regression(self, historical_data):
        """Train the linear regression model on historical data."""
        X = []
        y = []
        for features, label in historical_data:
            # Only include data where label is +1 or -1
            X.append(features)
            y.append(label)
        if X:
            self.linear_model.fit(X, y)
            self.trained = True
            print("Linear regression model trained.")

            # Output the weights
            strategy_names = ['trend', 'pure_forecasting']
            # weights = self.logistic_model.coef_[0]
            # intercept = self.logistic_model.intercept_[0]
            weights = self.linear_model.coef_
            intercept = self.linear_model.intercept_
            print("Linear Regression Coefficients:")
            for name, weight in zip(strategy_names, weights):
                print(f"  {name}: {weight:.4f}")
            print(f"Intercept: {intercept:.4f}\n")
        else:
            print("Not enough data to train linear regression model.")
        
    def display_total_profit(self):
        """Display the total profit or loss for each strategy."""
        print(f"Total Profits - {self.total_profit_or_loss}")

    def display_profit_per_trade(self):
        """Display the average profit or loss per trade for each strategy."""
        print("Average Profit Per Trade:")
        for strategy_name in self.total_profit_or_loss.keys():
            total_trades = self.num_trades[strategy_name]
            if total_trades > 0:
                avg_profit = self.total_profit_or_loss[strategy_name] / total_trades
            else:
                avg_profit = 0
            print(f"Profit Per Trade - {strategy_name}: {avg_profit:.2f}")

    def display_final_wallet_amount(self):
        """Display the final amounts in both wallets for each strategy."""
        print(f"Final amount in Wallet A - {self.wallet_a}")
        print(f"Final amount in Wallet B - {self.wallet_b}")

    def simulate_trading_with_strategies(self, actual_rates, predicted_rates, use_kelly=True):
        """
        Simulate trading over a series of exchange rates.
        The mean-reversion strategy uses a window-based median analysis.
        """
        window_size = 150  # number of minutes to look back
        n = len(actual_rates)
        
        # First half: training phase for linear regression
        historical_data = []
        split_idx = n // 2
        for i in range(2, split_idx):
            curr_ratio = actual_rates[i - 1]
            prev_ratio = actual_rates[i - 2]
            predicted_next_ratio = predicted_rates[i]
            actual_next_ratio = actual_rates[i]
            if prev_ratio != 0 and curr_ratio != 0:
                base_pct = ((curr_ratio - prev_ratio) / prev_ratio) * 100
                forecast_pct = ((predicted_next_ratio - curr_ratio) / curr_ratio) * 100
            else:
                continue
            # Calculate potential profits for trend and forecasting strategies
            trend_profit = self.fixed_position_size * (actual_next_ratio - curr_ratio - self.spread/2) / actual_next_ratio if actual_next_ratio != 0 else 0
            forecast_profit = self.fixed_position_size * (actual_next_ratio - curr_ratio - self.spread/2) / actual_next_ratio if actual_next_ratio != 0 else 0
            historical_data.append(([trend_profit, forecast_profit], trend_profit + forecast_profit))
        
        if historical_data:
            X, y = zip(*historical_data)
            self.linear_model.fit(X, y) 
            self.trained = True
            self.trend_coeff = self.linear_model.coef_[0]
            self.forecasting_coeff = self.linear_model.coef_[1]
        
        # Second half: Trading phase
        for i in range(split_idx, n):
            curr_ratio = actual_rates[i - 1]
            predicted_next_ratio = predicted_rates[i]
            
            # Compute common signals
            prev_ratio = actual_rates[i - 2] if i >= split_idx + 2 else curr_ratio
            base_pct = ((curr_ratio - prev_ratio) / prev_ratio) * 100 if prev_ratio != 0 else 0
            forecast_pct = ((predicted_next_ratio - curr_ratio) / curr_ratio) * 100 if curr_ratio != 0 else 0
            
            # --- Mean Reversion Strategy ---
            if i >= window_size:
                recent_window = actual_rates[i - window_size:i]
                trade_direction, median_rate = self.analyze_mean_reversion_window(recent_window)
                if trade_direction != 'no_trade' and self.open_positions['mean_reversion']['type'] is None:
                    self.num_trades['mean_reversion'] += 1
                    f_i = self.kelly_criterion('mean_reversion')
                    total_value_in_a = self.wallet_a['mean_reversion'] + (self.wallet_b['mean_reversion'] / curr_ratio)
                    base_bet_size_a = f_i * total_value_in_a if use_kelly else self.fixed_position_size
                    
                    if trade_direction == 'buy_currency_a':
                        bet_size_a = min(base_bet_size_a, self.wallet_a['mean_reversion'])
                        bet_size_b = bet_size_a * curr_ratio
                        if bet_size_b <= self.wallet_b['mean_reversion']:
                            self.wallet_b['mean_reversion'] -= bet_size_b
                            self.wallet_a['mean_reversion'] += bet_size_a
                            self.open_positions['mean_reversion'] = {
                                'type': 'long',
                                'size_a': bet_size_a,
                                'size_b': bet_size_b,
                                'entry_ratio': median_rate,
                                'waiting_for_reversion': True
                            }
                    elif trade_direction == 'sell_currency_a':
                        bet_size_a = min(base_bet_size_a, self.wallet_a['mean_reversion'])
                        bet_size_b = bet_size_a * curr_ratio
                        if bet_size_a <= self.wallet_a['mean_reversion']:
                            self.wallet_a['mean_reversion'] -= bet_size_a
                            self.wallet_b['mean_reversion'] += bet_size_b
                            self.open_positions['mean_reversion'] = {
                                'type': 'short',
                                'size_a': bet_size_a,
                                'size_b': bet_size_b,
                                'entry_ratio': median_rate,
                                'waiting_for_reversion': True
                            }
            
            # Check and close mean-reversion position if reverted or end of simulation
            pos = self.open_positions['mean_reversion']
            if pos['waiting_for_reversion'] and pos['type'] is not None:
                tolerance = 3 * self.spread
                if abs(curr_ratio - pos['entry_ratio']) < tolerance or i == n - 1:
                    profit = self.close_position('mean_reversion', curr_ratio)
                    self.total_profit_or_loss['mean_reversion'] += profit
                    pos['waiting_for_reversion'] = False
            
            # --- Trend Strategy ---
            trade_direction_trend = self.determine_trade_direction('trend', base_pct, 0)
            if trade_direction_trend != 'no_trade':
                self.num_trades['trend'] += 1
                f_i = self.kelly_criterion('trend')
                profit = self.calculate_profit('trend', trade_direction_trend, curr_ratio, f_i, use_kelly)
                self.total_profit_or_loss['trend'] += profit
                if profit > 0:
                    self.num_wins['trend'] += 1
                    self.total_gains['trend'] += abs(profit)
                elif profit < 0:
                    self.num_losses['trend'] += 1
                    self.total_losses['trend'] += abs(profit)
            
            # --- Pure Forecasting Strategy ---
            trade_direction_pf = self.determine_trade_direction('pure_forecasting', 0, forecast_pct)
            if trade_direction_pf != 'no_trade':
                self.num_trades['pure_forecasting'] += 1
                f_i = self.kelly_criterion('pure_forecasting')
                profit = self.calculate_profit('pure_forecasting', trade_direction_pf, curr_ratio, f_i, use_kelly)
                self.total_profit_or_loss['pure_forecasting'] += profit
                if profit > 0:
                    self.num_wins['pure_forecasting'] += 1
                    self.total_gains['pure_forecasting'] += abs(profit)
                elif profit < 0:
                    self.num_losses['pure_forecasting'] += 1
                    self.total_losses['pure_forecasting'] += abs(profit)
            
            # --- Hybrid Mean Reversion Strategy ---
            trade_direction_hmr = self.determine_trade_direction('hybrid_mean_reversion', base_pct, forecast_pct)
            if trade_direction_hmr != 'no_trade':
                self.num_trades['hybrid_mean_reversion'] += 1
                f_i = self.kelly_criterion('hybrid_mean_reversion')
                profit = self.calculate_profit('hybrid_mean_reversion', trade_direction_hmr, curr_ratio, f_i, use_kelly)
                self.total_profit_or_loss['hybrid_mean_reversion'] += profit
                if profit > 0:
                    self.num_wins['hybrid_mean_reversion'] += 1
                    self.total_gains['hybrid_mean_reversion'] += abs(profit)
                elif profit < 0:
                    self.num_losses['hybrid_mean_reversion'] += 1
                    self.total_losses['hybrid_mean_reversion'] += abs(profit)
            
            # --- Hybrid Trend Strategy ---
            trade_direction_ht = self.determine_trade_direction('hybrid_trend', base_pct, forecast_pct)
            if trade_direction_ht != 'no_trade':
                self.num_trades['hybrid_trend'] += 1
                f_i = self.kelly_criterion('hybrid_trend')
                profit = self.calculate_profit('hybrid_trend', trade_direction_ht, curr_ratio, f_i, use_kelly)
                self.total_profit_or_loss['hybrid_trend'] += profit
                if profit > 0:
                    self.num_wins['hybrid_trend'] += 1
                    self.total_gains['hybrid_trend'] += abs(profit)
                elif profit < 0:
                    self.num_losses['hybrid_trend'] += 1
                    self.total_losses['hybrid_trend'] += abs(profit)
            
            # --- Ensemble Strategy ---
            if self.trained:
                signals = self.get_strategy_signals(base_pct, forecast_pct, curr_ratio, predicted_next_ratio)
                trend_signal = signals.get('trend', 0)
                forecast_signal = signals.get('pure_forecasting', 0)
                combined_signal = self.linear_model.predict([[trend_signal, forecast_signal]])[0]
                if combined_signal > self.trade_threshold:
                    ensemble_direction = 'buy_currency_a'
                elif combined_signal < -self.trade_threshold:
                    ensemble_direction = 'sell_currency_a'
                else:
                    ensemble_direction = 'no_trade'
                if ensemble_direction != 'no_trade':
                    self.num_trades['ensemble'] += 1
                    f_i = self.kelly_criterion('ensemble')
                    profit = self.calculate_profit('ensemble', ensemble_direction, curr_ratio, f_i, use_kelly)
                    self.total_profit_or_loss['ensemble'] += profit
                    if profit > 0:
                        self.num_wins['ensemble'] += 1
                        self.total_gains['ensemble'] += abs(profit)
                    elif profit < 0:
                        self.num_losses['ensemble'] += 1
                        self.total_losses['ensemble'] += abs(profit)
        
        # Close all open positions at the end
        for strategy in self.open_positions:
            if self.open_positions[strategy]['type'] is not None:
                final_ratio = actual_rates[-1]
                profit = self.close_position(strategy, final_ratio)
                self.total_profit_or_loss[strategy] += profit

        # Display results
        self.display_total_profit()
        self.display_final_wallet_amount()
        self.display_profit_per_trade()

   