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
        self.wallet_a = {'mean_reversion': wallet_a, 'trend': wallet_a, 'pure_forcasting': wallet_a, 'hybrid_mean_reversion': wallet_a, 'hybrid_trend': wallet_a, 'ensemble': wallet_a}
        self.wallet_b = {'mean_reversion': wallet_b, 'trend': wallet_b, 'pure_forcasting': wallet_b, 'hybrid_mean_reversion': wallet_b, 'hybrid_trend': wallet_b, 'ensemble': wallet_b}
        # Track profit/loss, wins/losses, and total gains/losses for each strategy
        self.total_profit_or_loss = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensemble': 0}
        self.num_trades = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensemble': 0}
        self.num_wins = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensemble': 0}
        self.num_losses = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensemble': 0}
        self.total_gains = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensemble': 0}
        self.total_losses = {'mean_reversion': 0, 'trend': 0, 'pure_forcasting': 0, 'hybrid_mean_reversion': 0, 'hybrid_trend': 0, 'ensemble': 0}

        # New: Track open positions
        self.open_positions = {
            'mean_reversion': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'trend': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'pure_forcasting': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'hybrid_mean_reversion': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'hybrid_trend': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0},
            'ensemble': {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0}
        }

        self.min_trades_for_full_kelly = 50  # Minimum trades before using full Kelly
        self.fixed_position_size = 1000  # Fixed position size for training
        
        # Initialize linear regression model
        self.linear_model = LinearRegression()
        self.trained = False

        self.trend_coeff = 0
        self.forecasting_coeff = 0
        self.spread = 0.0005

    def calculate_profit_for_signals(self, curr_ratio, next_ratio):
        """Calculate potential profit for given signals using fixed position size."""
        max_profit = float('-inf')
        best_direction = 'no_trade'
        
        # Try both possible trade directions
        for direction in ['buy_currency_a', 'sell_currency_a']:
            # Calculate profit for this direction
            if direction == 'buy_currency_a':
                # You open (buy) at (curr_ratio + spread/2),  
                # you close (sell) at (next_ratio - spread/2).
                open_price = curr_ratio + self.spread / 2
                close_price = next_ratio - self.spread / 2
                
                # Avoid zero or negative
                if close_price <= 0 or open_price <= 0:
                    continue
                
                # Example P/L in terms of currency B:
                profit = self.fixed_position_size * (close_price - open_price) / close_price

            else:  # 'sell_currency_a'
                # You open (sell) at (curr_ratio - spread/2),
                # you close (buy) at (next_ratio + spread/2).
                open_price = curr_ratio - self.spread / 2
                close_price = next_ratio + self.spread / 2
                
                if close_price <= 0 or open_price <= 0:
                    continue
                
                profit = self.fixed_position_size * (open_price - close_price) / close_price
            
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
        """Calculate profit/loss and handle position management"""        
        profit_in_base_currency = 0
        
        # First, close any open position
        if self.open_positions[strategy_name]['type'] is not None:
            profit_in_base_currency += self.close_position(strategy_name, curr_ratio)
        
        # Then open new position if there's a trade signal
        if trade_direction != 'no_trade':
            # Calculate total portfolio value in currency A
            total_value_in_a = self.wallet_a[strategy_name] + (self.wallet_b[strategy_name] / curr_ratio)

            if(use_kelly):
                base_bet_size_a = f_i * total_value_in_a
            else:
                base_bet_size_a = 1000
            
            if trade_direction == 'buy_currency_a':
                bet_size_a = min(base_bet_size_a, self.wallet_a[strategy_name])
                bet_size_b = bet_size_a * (curr_ratio + self.spread / 2)
                
                # Check if we have enough B
                if bet_size_b <= self.wallet_b[strategy_name]:
                    self.wallet_b[strategy_name] -= bet_size_b
                    self.wallet_a[strategy_name] += bet_size_a
                    
                    self.open_positions[strategy_name] = {
                        'type': 'long',
                        'size_a': bet_size_a,
                        'size_b': bet_size_b,
                        'entry_ratio': curr_ratio + self.spread / 2
                    }
                
            elif trade_direction == 'sell_currency_a':
                bet_size_a = min(base_bet_size_a, self.wallet_a[strategy_name])
                bet_size_b = bet_size_a * (curr_ratio - self.spread / 2)
                
                if bet_size_a <= self.wallet_a[strategy_name]:
                    self.wallet_a[strategy_name] -= bet_size_a
                    self.wallet_b[strategy_name] += bet_size_b
                    
                    self.open_positions[strategy_name] = {
                        'type': 'short',
                        'size_a': bet_size_a,
                        'size_b': bet_size_b,
                        'entry_ratio': curr_ratio - self.spread / 2
                    }
        
        return profit_in_base_currency
    
    def close_position(self, strategy_name, curr_ratio):
        """Close an open position and calculate profit/loss"""
        position = self.open_positions[strategy_name]
        profit = 0
        
        if position['type'] == 'long':
            # Close long position (sell currency A)
            self.wallet_a[strategy_name] -= position['size_a']
            self.wallet_b[strategy_name] += position['size_a'] * (curr_ratio - self.spread / 2)
            # Calculate profit
            profit = position['size_a'] * ((curr_ratio - self.spread / 2) - position['entry_ratio']) / (curr_ratio - self.spread / 2)
            
        elif position['type'] == 'short':
            # Close short position (buy currency A)
            self.wallet_b[strategy_name] -= position['size_b']
            self.wallet_a[strategy_name] += position['size_b'] / (curr_ratio + self.spread / 2)
            # Calculate profit
            profit = position['size_a'] * (position['entry_ratio'] - (curr_ratio + self.spread / 2)) / (curr_ratio + self.spread / 2)
        
        # Reset position tracking
        self.open_positions[strategy_name] = {'type': None, 'size_a': 0, 'size_b': 0, 'entry_ratio': 0}
        
        return profit

    def determine_trade_direction(self, strategy_name, base_ratio_change, predicted_ratio_change, curr_ratio):
        """Determine the trade direction based on strategy and ratio changes."""
        trade_direction = 'no_trade'

        # Safeguard against curr_ratio == 0
        #if curr_ratio != 0:
        #    spread_as_pct = (self.spread / curr_ratio) * 100
        #else:
        #    spread_as_pct = 0.0  # or skip trading logic, etc.
        effective_threshold = self.trade_threshold + self.spread / 2
        #print(f"base_ratio_change={base_ratio_change:.4f}, effective_threshold={effective_threshold:.4f}")

        if(strategy_name == 'mean_reversion'):
            # Mean reversion strategy: trade against significant ratio changes
            if base_ratio_change > effective_threshold:
                trade_direction = 'sell_currency_a'
            elif base_ratio_change < -effective_threshold:
                trade_direction = 'buy_currency_a'
                
        elif(strategy_name == 'trend'):
            # Trend strategy: trade towards significant ratio changes
            if base_ratio_change > effective_threshold:
                trade_direction = 'buy_currency_a'
            elif base_ratio_change < -effective_threshold:
                trade_direction = 'sell_currency_a'
                
        elif(strategy_name == 'pure_forcasting'):
            # Pure forecasting strategy: trade based on predicted future ratio changes
            if predicted_ratio_change > effective_threshold:
                trade_direction = 'buy_currency_a'
            elif predicted_ratio_change < -effective_threshold:
                trade_direction = 'sell_currency_a'
                
        elif(strategy_name == 'hybrid_mean_reversion'):
            # Hybrid strategy: combine mean reversion and pure forecasting signals
            if base_ratio_change < -effective_threshold and predicted_ratio_change > effective_threshold:
                trade_direction = 'buy_currency_a'
            elif base_ratio_change > effective_threshold and predicted_ratio_change < -effective_threshold:
                trade_direction = 'sell_currency_a'
                
        elif(strategy_name == 'hybrid_trend'):
            # Hybrid strategy: combine trend and pure forecasting signals
            if base_ratio_change > effective_threshold and predicted_ratio_change > effective_threshold:
                trade_direction = 'buy_currency_a'
            elif base_ratio_change < -effective_threshold and predicted_ratio_change < -effective_threshold:
                trade_direction = 'sell_currency_a'
            
        return trade_direction
    
    def get_strategy_signals(self, base_ratio_change, predicted_ratio_change, curr_ratio, next_ratio):
        """Get the signals (+1, 0, -1) for each strategy."""
        signals = {}
        for strategy_name in ['trend', 'pure_forcasting']:
            trade_direction = self.determine_trade_direction(strategy_name, base_ratio_change, predicted_ratio_change, curr_ratio)
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
            strategy_names = ['trend', 'pure_forcasting']
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

    def simulate_trading_with_strategies(self, actual_rates, predicted_rates, use_kelly=True, horizon=50):
        """
        Simulate trading over a series of exchange rates using different strategies,
        but now using a multi-step horizon (looking 'horizon' steps ahead).
        We'll define 'actual_next_ratio' as the max over the next 'horizon' bars.
        """
        strategy_name = "ensemble"
        
        historical_data = []
        split_idx = len(actual_rates)//2
        print(f"split_idx : {split_idx}")
        
        actual_rates_first_half = actual_rates[:split_idx]
        actual_rates_second_half = actual_rates[split_idx:]
        
        trend_cumulative_profit = 0
        forecast_cumulative_profit = 0
        max_cumulative_profit = 0

        print("Training loop range = ", 2, "to", len(actual_rates_first_half) - horizon)
        print("Testing loop range = ", 2, "to", len(actual_rates_second_half) - horizon)


        # ---------------------- TRAINING PHASE (FIRST HALF) ----------------------
        for i in range(2, len(actual_rates_first_half) - horizon):
            curr_ratio = actual_rates_first_half[i - 1]
            prev_ratio = actual_rates_first_half[i - 2]
            predicted_next_ratio = predicted_rates[i]  # If you like, or you can skip

            # Grab the future window of size 'horizon'
            future_window = actual_rates_first_half[i : i + horizon]
            future_max = max(future_window)  # Or np.mean(future_window), min(...), etc.
            actual_next_ratio = future_max
            
            # Avoid division by zero
            if prev_ratio != 0 and curr_ratio != 0:
                predicted_percentage_increase = ((predicted_next_ratio - curr_ratio) / curr_ratio) * 100
                actual_percentage_increase = ((actual_next_ratio - curr_ratio) / curr_ratio) * 100
                base_percentage_increase = ((curr_ratio - prev_ratio) / prev_ratio) * 100
            else:
                print("Skipping iteration due to zero division risk.")
                continue

            # Evaluate signals for your "trend" and "pure_forcasting"
            signals = self.get_strategy_signals(
                base_percentage_increase,
                predicted_percentage_increase,
                curr_ratio,
                actual_next_ratio
            )
            trend_cumulative_profit += signals['trend']
            forecast_cumulative_profit += signals['pure_forcasting']

            # For "max_cumulative_profit," we do a hypothetical single‐trade profit:
            profit, _ = self.calculate_profit_for_signals(curr_ratio, actual_next_ratio)
            max_cumulative_profit += profit

            # Store that as a training sample for the linear regression
            feature_vector = [trend_cumulative_profit, forecast_cumulative_profit]
            historical_data.append((feature_vector, max_cumulative_profit))

        # Train the linear regression model
        self.train_linear_regression(historical_data)
        if self.trained:
            self.trend_coeff = self.linear_model.coef_[0]
            self.forecasting_coeff = self.linear_model.coef_[1]
        else:
            self.trend_coeff = 0
            self.forecasting_coeff = 0

        # Decide which strategy to use for the "ensemble"
        if (self.trend_coeff >= self.forecasting_coeff):
            trade_direction_strategy = "trend"
        else:
            trade_direction_strategy = "pure_forcasting"

        # ---------------------- TESTING/ACTUAL TRADING PHASE (SECOND HALF) ----------------------
        # We'll do the same multi-step approach, but now we actually open/close trades in self.calculate_profit().
        for i in range(2, len(actual_rates_second_half) - horizon):
            # We'll define j for indexing predicted_rates if needed:
            # j = i + split_idx  # If your predicted_rates aligns with the original indices

            curr_ratio = actual_rates_second_half[i - 1]
            prev_ratio = actual_rates_second_half[i - 2]
            predicted_next_ratio = predicted_rates[i + split_idx]  # if predictions have same length as actual_rates

            # Grab horizon future bars
            future_window = actual_rates_second_half[i : i + horizon]
            future_max = max(future_window)
            actual_next_ratio = future_max

            # Avoid division by zero
            if prev_ratio != 0 and curr_ratio != 0:
                predicted_percentage_increase = ((predicted_next_ratio - curr_ratio) / curr_ratio) * 100
                actual_percentage_increase = ((actual_next_ratio - curr_ratio) / curr_ratio) * 100
                base_percentage_increase = ((curr_ratio - prev_ratio) / prev_ratio) * 100
            else:
                print("Skipping iteration in second half due to zero division risk.")
                continue

            # Determine the trade direction for the "ensemble" strategy
            trade_direction = self.determine_trade_direction(
                trade_direction_strategy,
                base_percentage_increase,
                predicted_percentage_increase,
                curr_ratio
            )

            if trade_direction != "no_trade":
                self.num_trades[strategy_name] += 1

            # Calculate the Kelly fraction and open/close trades
            f_i = self.kelly_criterion(strategy_name)
            profit = self.calculate_profit(strategy_name, trade_direction, curr_ratio, f_i, use_kelly)
            self.total_profit_or_loss[strategy_name] += profit

            # Track win/loss stats
            if profit > 0:
                self.num_wins[strategy_name] += 1
                self.total_gains[strategy_name] += abs(profit)
            elif profit < 0:
                self.num_losses[strategy_name] += 1
                self.total_losses[strategy_name] += abs(profit)

        # If there's an open "ensemble" position at the end, close it
        if self.open_positions[strategy_name]['type'] is not None:
            final_ratio = actual_rates[-1]
            profit = self.close_position(strategy_name, final_ratio)
            self.total_profit_or_loss[strategy_name] += profit

        # ---------------------- OTHER STRATEGIES (MEAN, TREND, HYBRID, ETC.) ----------------------
        # For each of your other strategies, you can do the exact same loop logic:
        strategy_names = ['mean_reversion', 'trend', 'pure_forcasting', 'hybrid_mean_reversion', 'hybrid_trend']
        for strat in strategy_names:
            for i in range(2, len(actual_rates_second_half) - horizon):
                curr_ratio = actual_rates_second_half[i - 1]
                prev_ratio = actual_rates_second_half[i - 2]
                # Possibly use predicted_rates[i + split_idx] if needed

                # multi-step horizon
                future_window = actual_rates_second_half[i : i + horizon]
                future_max = max(future_window)
                actual_next_ratio = future_max

                if prev_ratio != 0 and curr_ratio != 0:
                    base_percentage_increase = ((curr_ratio - prev_ratio) / prev_ratio) * 100
                    # If you have predictions for these strategies, define them, or keep them zero
                    predicted_percentage_increase = 0  # or define properly
                else:
                    continue

                trade_direction = self.determine_trade_direction(
                    strat,
                    base_percentage_increase,
                    predicted_percentage_increase,
                    curr_ratio
                )

                if trade_direction != "no_trade":
                    self.num_trades[strat] += 1

                f_i = self.kelly_criterion(strat)
                profit = self.calculate_profit(strat, trade_direction, curr_ratio, f_i, use_kelly)
                self.total_profit_or_loss[strat] += profit

                if profit > 0:
                    self.num_wins[strat] += 1
                    self.total_gains[strat] += abs(profit)
                elif profit < 0:
                    self.num_losses[strat] += 1
                    self.total_losses[strat] += abs(profit)

        # Close any leftover positions for the other strategies
        for strat in strategy_names:
            if self.open_positions[strat]['type'] is not None:
                final_ratio = actual_rates[-1]
                profit = self.close_position(strat, final_ratio)
                self.total_profit_or_loss[strat] += profit

        # Print final results
        self.display_total_profit()
        self.display_final_wallet_amount()
        self.display_profit_per_trade()
