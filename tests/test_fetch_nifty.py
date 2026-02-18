import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.fetch_nifty_200_data import fetch_and_save_data
import logging

# Configure logging to console
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    test_symbols = ["INFY", "TCS"]
    print(f"Testing data fetch for: {test_symbols}")
    
    for symbol in test_symbols:
        fetch_and_save_data(symbol)
        
    print("Test completed.")
