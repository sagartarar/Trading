import logging
import os
from kiteconnect import KiteConnect

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load credentials from .env file (simple manual parsing for now to avoid extra dependencies if possible, or assume user will install python-dotenv)
# Ideally we should use python-dotenv, let's assume we will install it.
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("KITE_API_KEY")
api_secret = os.getenv("KITE_API_SECRET")

if not api_key or not api_secret:
    logging.error("KITE_API_KEY and KITE_API_SECRET must be set in .env file")
    exit(1)

kite = KiteConnect(api_key=api_key)

print("1. Go to this URL to login:", kite.login_url())
request_token = input("2. Enter the request_token from the redirected URL: ")

try:
    data = kite.generate_session(request_token, api_secret=api_secret)
    kite.set_access_token(data["access_token"])
    print(f"Authentication successful! Access token: {data['access_token']}")
    
    # Save access token to file
    with open("access_token.txt", "w") as f:
        f.write(data["access_token"])
    print("Access token saved to access_token.txt")

except Exception as e:
    logging.error(f"Authentication failed: {e}")
