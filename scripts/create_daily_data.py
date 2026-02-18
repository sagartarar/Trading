import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

SOURCE_DIR = "data/nifty_200_history"
TARGET_DIR = "data/nifty_200_daily"

os.makedirs(TARGET_DIR, exist_ok=True)

def process_file(filepath):
    filename = os.path.basename(filepath)
    symbol = filename.replace("_minute.csv", "")
    
    try:
        # Load 1-minute data
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # Resample to Daily
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # 'D' for generic Daily. 
        # Note: Indian market closes at 15:30. '1D' might create rows indexed at midnight or similar.
        # We just want the date as the index.
        df_daily = df.resample('D').agg(agg_dict)
        
        # Drop NaN rows (weekends, holidays)
        df_daily.dropna(inplace=True)
        
        # Save
        target_file = os.path.join(TARGET_DIR, f"{symbol}_daily.csv")
        df_daily.to_csv(target_file)
        # logging.info(f"Generated {target_file}") # Commented out to reduce noise
        return True

    except Exception as e:
        logging.error(f"Error processing {symbol}: {e}")
        return False

def main():
    if not os.path.exists(SOURCE_DIR):
        logging.error(f"Source directory {SOURCE_DIR} missing.")
        return

    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith("_minute.csv")]
    logging.info(f"Found {len(files)} 1-minute files. Converting to Daily...")
    
    count = 0
    for f in files:
        if process_file(os.path.join(SOURCE_DIR, f)):
            count += 1
        if count % 20 == 0:
            logging.info(f"Converted {count} files...")

    logging.info(f"Done. {count} files converted to Daily format in {TARGET_DIR}.")

if __name__ == "__main__":
    main()
