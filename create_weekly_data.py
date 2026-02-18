import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def create_weekly_data(input_dir='data/nifty_200_daily', output_dir='data/nifty_200_weekly'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    files = [f for f in os.listdir(input_dir) if f.endswith('_daily.csv')]
    
    for f in files:
        file_path = os.path.join(input_dir, f)
        try:
            df = pd.read_csv(file_path, index_col='date', parse_dates=True)
            
            # Resample to Weekly (W-FRI, week ending Friday)
            # aggregator logic:
            # open: first
            # high: max
            # low: min
            # close: last
            # volume: sum
            
            weekly_df = df.resample('W-FRI').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # Drop rows with NaN (incomplete weeks at start/end if any, usually resample handles it but good to be safe)
            weekly_df.dropna(inplace=True)
            
            output_filename = f.replace('_daily.csv', '_weekly.csv')
            output_path = os.path.join(output_dir, output_filename)
            weekly_df.to_csv(output_path)
            # logging.info(f"Processed {output_filename}")
            
        except Exception as e:
            logging.error(f"Error processing {f}: {e}")

    logging.info(f"Weekly data generation complete. Processed {len(files)} files.")

if __name__ == "__main__":
    create_weekly_data()
