import logging
import os
import time
import pandas as pd
from datetime import datetime, timedelta
import kiteconnect
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[
                        logging.FileHandler("fetch_log_2min.txt"),
                        logging.StreamHandler()
                    ])

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
DATA_DIR = "data/nifty_200_2min"
os.makedirs(DATA_DIR, exist_ok=True)

def get_nifty_200_symbols():
    try:
        df = pd.read_csv("nifty_200_raw.csv")
        return df['Symbol'].tolist()
    except Exception as e:
        logging.error(f"Error reading Nifty 200 list: {e}")
        return []

def get_instrument_token(symbol, exchange="NSE"):
    """Fetch instrument token for a given symbol."""
    try:
        instruments = kite.instruments(exchange)
        for instrument in instruments:
            if instrument['tradingsymbol'] == symbol:
                return instrument['instrument_token']
    except Exception as e:
        logging.error(f"Error fetching instrument token for {symbol}: {e}")
    return None

def get_listing_date(symbol):
    """
    Find the listing date using yfinance.
    Returns datetime object or None if failed.
    """
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        # specific period 'max' gets full history
        hist = ticker.history(period="max")
        if not hist.empty:
            listing_date = hist.index[0].to_pydatetime()
            logging.info(f"{symbol} listing date found: {listing_date.date()}")
            return listing_date
        else:
            logging.warning(f"No history found for {symbol} on yfinance.")
            return None
    except Exception as e:
        logging.error(f"Error fetching listing date for {symbol}: {e}")
        return None


def fetch_and_save_data(symbol):
    filename = f"{DATA_DIR}/{symbol}_2minute.csv"
    if os.path.exists(filename):
        logging.info(f"Data for {symbol} already exists. Skipping.")
        return

    # 1. Get Instrument Token
    token = get_instrument_token(symbol)
    if not token:
        logging.error(f"Could not find instrument token for {symbol}. Skipping.")
        return

    # 2. Get Listing Date
    start_date = get_listing_date(symbol)
    if not start_date:
        logging.warning(f"Could not determine listing date for {symbol}. Defaulting to 2010-01-01.")
        start_date = datetime(2010, 1, 1)

    # 3. Monthly Loop
    end_date = datetime.now()
    # Ensure start_date is timezone-naive or handled correctly
    if start_date.tzinfo:
        start_date = start_date.replace(tzinfo=None)
    
    current_start = start_date
    all_data = []
    
    # Define chunk size - Monthly (30 days) to be safe with limits
    CHUNK_DAYS = 30
    
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS), end_date)
        
        # Adjust end date to avoid future calls if exactly today/tomorrow
        if current_end > end_date:
            current_end = end_date

        logging.info(f"  Fetching {symbol} from {current_start.date()} to {current_end.date()}...")
        
        try:
            # kite.historical_data expects limits. minute data limit is 60 days.
            # We are using 30 days which is safe.
            data = kite.historical_data(token, current_start, current_end, "2minute")
            if data:
                all_data.extend(data)
                # logging.info(f"    Fetched {len(data)} records.") # Verbose
            
        except Exception as e:
            logging.error(f"    Error fetching chunk {current_start.date()} - {current_end.date()}: {e}")
            # Continue to next chunk? Or retry? 
            # For now, continue. Gaps might occur.

        # Move to next chunk
        current_start = current_end + timedelta(days=1) # Standard logic
        
        # Rate Limiting
        time.sleep(0.5) # 2 requests per second max roughly

    # 4. Save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        # filename already defined above
        df.to_csv(filename, index=False)
        logging.info(f"Saved {len(df)} records for {symbol} to {filename}")
    else:
        logging.warning(f"No data fetched for {symbol}.")


if __name__ == "__main__":
    symbols = get_nifty_200_symbols()
    logging.info(f"Found {len(symbols)} symbols in Nifty 200 list.")
    
    for symbol in symbols:
        fetch_and_save_data(symbol)
        # Optional: Sleep between symbols to be extra safe?
        time.sleep(1)
