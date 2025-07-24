import pandas as pd
import numpy as np

# Read the exchange rate data and news article data from Excel files
fx_df = pd.read_excel("dataset/USD-CNH-2024.xlsx")
news_xls = pd.ExcelFile("dataset/NEWs.xlsx")
sheet_names = news_xls.sheet_names
news_all_sheets = pd.read_excel("dataset/NEWs.xlsx", sheet_name=[sheet for sheet in sheet_names if sheet.lower() != "exchange_rate"])
news_df = pd.concat(news_all_sheets.values(), ignore_index=True)

# Convert the relevant timestamp columns to datetime objects
fx_df['timestamp'] = pd.to_datetime(fx_df['Date'])
news_df['timestamp'] = pd.to_datetime(news_df['Time'])

# Sort both DataFrames by timestamp to ensure proper time ordering
fx_df.sort_values('timestamp', inplace=True)
news_df.sort_values('timestamp', inplace=True)

# Lists to hold the resulting values for each news article
labels = []
trading_periods = []
window_starts = []
window_ends = []

# Process each news article one by one
for idx, news in news_df.iterrows():
    T = news['timestamp']
    window_end = T + pd.Timedelta(minutes=2)
    
    # Record the window start and end times
    window_starts.append(T)
    window_ends.append(window_end)
    
    # Get FX data for the 2-minute observation window [T, T + 2 minutes]
    fx_window = fx_df[(fx_df['timestamp'] >= T) & (fx_df['timestamp'] <= window_end)]
    
    # Debug print (optional)
    print(f"Processing article at {T} (window ends at {window_end}), found {len(fx_window)} FX records")
    
    if fx_window.empty:
        labels.append("no_data")
        trading_periods.append(np.nan)
        continue

    # Use the first available FX record in the window as the initial data point
    initial_record = fx_window.iloc[0]
    initial_rate = initial_record['Rate']
    
    # Here we use a fixed spread value of 0.0005; adjust if needed.
    threshold = 3*0.0005  

    # Get the FX rate at the end of the window and compute the rate change
    final_record = fx_window.iloc[-1]
    final_rate = final_record['Rate']
    rate_change = final_rate - initial_rate

    # If the absolute rate change is less than the threshold, label the article as neutral
    if abs(rate_change) < threshold:
        labels.append("neutral")
        trading_periods.append(np.nan)
        continue

    # Determine the label:
    # "positive" if the rate decreased (i.e. CNH strengthens, so the USD weakens)
    # "negative" if the rate increased (i.e. CNH weakens, so the USD strengthens)
    label = "positive" if rate_change < 0 else "negative"
    
    # Find the first timestamp within the window where the rate change meets/exceeds the threshold
    tp = np.nan  # initialize trading period (in seconds)
    for _, row in fx_window.iterrows():
        current_rate = row['Rate']
        if abs(current_rate - initial_rate) >= threshold:
            tp = (row['timestamp'] - T).total_seconds()
            break

    labels.append(label)
    trading_periods.append(tp)
    # Optional: print out the result for each article
    print(f"Article at {T} labeled {label} with trading period {tp} seconds")

# Append the results as new columns to the original news DataFrame (if desired)
news_df['Label'] = labels
news_df['Trading_Period_sec'] = trading_periods

# Create a new DataFrame with only the desired columns:
# "Time" (the article timestamp), "Content", "Window_Start", "Window_End",
# "Trading_Period_sec", and "Label"
new_columns = {
    'Time': news_df['Time'],
    'Content': news_df['Content'],
    'Window_Start': window_starts,
    'Window_End': window_ends,
    'Trading_Period_sec': trading_periods,
    'Label': labels
}
new_df = pd.DataFrame(new_columns)

# Save the new DataFrame to a new Excel file
output_path = "news_with_labels.xlsx"
new_df.to_excel(output_path, index=False)
print(f"New Excel file '{output_path}' has been created with the required columns.")
