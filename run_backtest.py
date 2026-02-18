
from backtest_engine import Backtester
from strategies import SMACrossover, RSIReversion
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run_strategy(name, strategy_class, symbol, interval="5min"):
    logging.info(f"--- Running {name} for {symbol} ({interval}) ---")
    backtester = Backtester(data_dir="data")
    results = backtester.run(strategy_class, symbol, interval)
    
    if results:
        print(f"Results for {symbol}:")
        for k, v in results.items():
            print(f"  {k}: {v}")
        
        # Print Trade Logs
        print("\n  Trade Logs:")
        trades = backtester.trades
        if not trades:
            print("  No trades executed.")
        else:
            # Print first 5 and last 5 trades if too many
            if len(trades) > 10:
                for t in trades[:5]:
                    print(f"    {t}")
                print("    ... (skipping intermediate trades) ...")
                for t in trades[-5:]:
                    print(f"    {t}")
            else:
                for t in trades:
                    print(f"    {t}")
    else:
        print(f"No results for {symbol}")
    print("-" * 30)

def main():
    # List of symbols to test
    # Just picking a few for demonstration
    symbols = ["RELIANCE", "INFY", "HDFCBANK", "TCS"] 
    # Note: TCS might not be in the Nifty 200 list or file might not be present if fetch didn't reach it.
    # Let's check what we have.
    
    # We can also scan the directory to see which files are available
    available_files = os.listdir("data/nifty_200_5min")
    available_symbols = [f.replace("_5min.csv", "") for f in available_files if f.endswith("_5min.csv")]
    
    # Pick top 3 available symbols
    test_symbols = available_symbols[:3]
    if not test_symbols:
        logging.error("No 5min data found to test.")
        return

    logging.info(f"Testing on symbols: {test_symbols}")

    for symbol in test_symbols:
        # Run SMA Crossover
        run_strategy("SMA Crossover", SMACrossover, symbol, "5min")
        
        # Run RSI Reversion
        run_strategy("RSI Reversion", RSIReversion, symbol, "5min")

if __name__ == "__main__":
    main()
