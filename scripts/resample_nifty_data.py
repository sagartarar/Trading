
import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Data Directories
SOURCE_DIR = "data/nifty_200_history"
TARGET_DIRS = {
    "5min": "data/nifty_200_5min",
    "15min": "data/nifty_200_15min",
    "30min": "data/nifty_200_30min",
    "45min": "data/nifty_200_45min",
    "60min": "data/nifty_200_60min"
}

# Ensure target directories exist
for directory in TARGET_DIRS.values():
    os.makedirs(directory, exist_ok=True)

# Define the market open time for aligment (09:15 AM)
# using a random date, pandas only cares about the time part for alignment if expected correctly
MARKET_OPEN_ORIGIN = pd.Timestamp("2000-01-01 09:15:00+05:30")

def resample_dataframe(df, interval, origin=MARKET_OPEN_ORIGIN):
    """
    Resample dataframe to given interval.
    interval: pandas offset string (e.g., '5min', '15min')
    origin: Timestamp to align the bins.
    """
    try:
        # Resample logic
        # Open: first, High: max, Low: min, Close: last, Volume: sum
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample with specific origin to align to 9:15 AM
        # closed='left', label='left' is standard (9:15-9:20 is labelled 9:15)
        resampled_df = df.resample(interval, origin=origin, closed='left', label='left').agg(agg_dict)
        
        # Drop rows with NaN (where no data existed in that bucket)
        resampled_df.dropna(inplace=True)
        
        return resampled_df
    except Exception as e:
        raise e

def save_dataframe(df, target_dir, symbol, interval_name):
    filename = f"{target_dir}/{symbol}_{interval_name}.csv"
    df.to_csv(filename)

def process_file(filepath):
    filename = os.path.basename(filepath)
    symbol = filename.replace("_minute.csv", "")
    
    logging.info(f"Processing {symbol}...")
    
    try:
        # Load 1-minute data
        df_1min = pd.read_csv(filepath)
        
        # Parse date column
        df_1min['date'] = pd.to_datetime(df_1min['date'])
        
        # Set date as index
        df_1min.set_index('date', inplace=True)
        df_1min.sort_index(inplace=True)
        
        # --- Step 1: Resample 1m -> 5m (Base for others) ---
        df_5min = resample_dataframe(df_1min, '5min')
        save_dataframe(df_5min, TARGET_DIRS["5min"], symbol, "5min")
        
        # --- Step 2: Resample 5m -> Higher Timeframes ---
        # Note: We use df_5min as source. 
        # Since 5min is already aligned to 9:15, resampling it to 15, 30, etc. 
        # will preserve that alignment if we stick to the same origin logic.
        
        higher_tf_configs = [
            ("15min", TARGET_DIRS["15min"]),
            ("30min", TARGET_DIRS["30min"]),
            ("45min", TARGET_DIRS["45min"]),
            ("60min", TARGET_DIRS["60min"])
        ]
        
        for interval, target_dir in higher_tf_configs:
            # Resampling from 5min data is faster
            df_resampled = resample_dataframe(df_5min, interval)
            
            # Special case for filename: 60min is sometimes called 1h but we want '60min'
            # The config list already has '60min' as interval string which pandas understands (or '60T')
            # '60min' is valid in recent pandas.
            
            save_dataframe(df_resampled, target_dir, symbol, interval)
            
    except Exception as e:
        logging.error(f"Failed to process {filename}: {e}")


def main():
    if not os.path.exists(SOURCE_DIR):
        logging.error(f"Source directory {SOURCE_DIR} does not exist.")
        return

    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith("_minute.csv")]
    logging.info(f"Found {len(files)} files to process.")
    
    count = 0
    for f in files:
        process_file(os.path.join(SOURCE_DIR, f))
        count += 1
        if count % 10 == 0:
            logging.info(f"Processed {count}/{len(files)} files.")

    logging.info("Resampling complete.")

if __name__ == "__main__":
    main()
