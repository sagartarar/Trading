import logging
import os
import kiteconnect

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load credentials from .env
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

try:
    # Fetch Equity Holdings
    print("\n--- Equity Holdings ---")
    holdings = kite.holdings()
    if holdings:
        for holding in holdings:
            print(f"{holding['tradingsymbol']}: {holding['quantity']} qty @ {holding['average_price']} (LTP: {holding['last_price']}) - P&L: {holding['pnl']}")
    else:
        print("No equity holdings found.")

    # Fetch Mutual Fund Holdings
    print("\n--- Mutual Fund Holdings ---")
    mf_holdings = kite.mf_holdings()
    if mf_holdings:
        for mf in mf_holdings:
            print(f"{mf['fund']} ({mf['tradingsymbol']}): {mf['quantity']} units @ {mf['average_price']} (NAV: {mf['last_price']}) - P&L: {mf['pnl']}")
    else:
        print("No mutual fund holdings found.")

except Exception as e:
    logging.error(f"Error fetching holdings: {e}")
