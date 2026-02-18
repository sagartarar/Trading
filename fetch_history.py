import logging
import os
import time
import pandas as pd
from datetime import datetime, timedelta
import kiteconnect
from dateutil.relativedelta import relativedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load credentials
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("KITE_API_KEY")

if not api_key:
    logging.error("KITE_API_KEY must be set in .env file")
    exit(1)

# Load access token
try:
    with open("access_token.txt", "r") as f:
        access_token = f.read().strip()
except FileNotFoundError:
    logging.error("access_token.txt not found. Please run auth.py first.")
    exit(1)

# Initialize KiteConnect
kite = kiteconnect.KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Data Directory
DATA_DIR = "data/history"
os.makedirs(DATA_DIR, exist_ok=True)


def get_instrument_token(symbol, exchange="NSE"):
    """Fetch instrument token for a given symbol."""
    instruments = kite.instruments(exchange)
    for instrument in instruments:
        if instrument['tradingsymbol'] == symbol:
            return instrument['instrument_token']
    return None

def fetch_historical_data(instrument_token, create_date, end_date, interval="minute"):
    """
    Fetch historical data for a given instrument token and interval.
    Handles rate limiting and large data requests by chunking.
    """
    all_data = []
    current_start = create_date
    
    # Define chunk size based on interval to stay safe within limits
    # Minute data limit is ~60 days, we use 30 days for safety.
    # Day data limit is ~365 days, we use 360 days.
    if interval == "minute":
        chunk_days = 30
    elif interval == "day":
        chunk_days = 360  # Approx 1 year
    else:
        chunk_days = 90 # Safe default for other intervals like 3minute, 5minute etc.

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
        
        logging.info(f"Fetching data from {current_start.date()} to {current_end.date()}...")
        
        try:
            data = kite.historical_data(instrument_token, current_start, current_end, interval)
            if data:
                all_data.extend(data)
                logging.info(f"Fetched {len(data)} records.")
            else:
                logging.warning(f"No data returned for period {current_start.date()} to {current_end.date()}")

        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            # simple retry logic or just continue? heavily depends on error type.
            # for now, log and continue, maybe add a retry loop later if needed.

        # Update start date for next chunk (avoid overlap issues if API includes end date, usually it does)
        # To avoid duplicate records on boundary, we can start next chunk from current_end + 1 minute/day
        # Kite API generally includes both start and end timestamps in range.
        # But fetching 'from 2023-01-01 to 2023-01-02' gets data for both days if available (interval dependent)
        # Let's simply advance. Duplicates can be dropped later.
        current_start = current_end + timedelta(days=1) # naive progression, might skip a day if end is 23:59 and start is 00:00 next day?
        # Actually for minute data, start=end usually returns nothing or just that minute.
        # Safer to overlap slightly or ensure precise timestamps.
        # A simple robust way: `current_start = current_end` but then we get duplicates.
        # Let's stick to `current_start` moving forward. If `current_end` was `2024-01-30`, next start is `2024-01-31`.

        # Rate Limiting: 3 req/sec max. Sleep 0.5s to be safe.
        time.sleep(0.5)

    return all_data

def save_to_csv(data, symbol, interval):
    if not data:
        logging.warning("No data to save.")
        return

    df = pd.DataFrame(data)
    filename = f"{DATA_DIR}/{symbol}_{interval}.csv"
    df.to_csv(filename, index=False)
    logging.info(f"Saved {len(df)} records to {filename}")


if __name__ == "__main__":
    symbol = input("Enter Stock Symbol (e.g., INFY): ").upper()
    interval = input("Enter Interval (minute, day, 3minute, 5minute...): ").lower()
    start_date_str = input("Enter Start Date (YYYY-MM-DD): ")
    end_date_str = input("Enter End Date (YYYY-MM-DD): ")

    try:
        from_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        to_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        # Ensure to_date covers the full day
        to_date = to_date.replace(hour=23, minute=59, second=59)

        token = get_instrument_token(symbol)
        if token:
            logging.info(f"Instrument Token for {symbol}: {token}")
            history = fetch_historical_data(token, from_date, to_date, interval)
            save_to_csv(history, symbol, interval)
        else:
            logging.error(f"Instrument token not found for {symbol}")

    except ValueError as e:
        logging.error(f"Invalid date format: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
