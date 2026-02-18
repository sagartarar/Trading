import subprocess
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Define configurations
# Format: (Timeframe Name, Data Directory, File Suffix, Output Filename)
CONFIGS = [
    ("Weekly", "data/nifty_200_weekly", "_weekly.csv", "minervini_trades_weekly.csv"),
    ("Daily", "data/nifty_200_daily", "_daily.csv", "minervini_trades_daily.csv"),
    ("60 Min", "data/nifty_200_60min", "_60min.csv", "minervini_trades_60min.csv"),
    ("45 Min", "data/nifty_200_45min", "_45min.csv", "minervini_trades_45min.csv"),
    ("30 Min", "data/nifty_200_30min", "_30min.csv", "minervini_trades_30min.csv"),
    ("15 Min", "data/nifty_200_15min", "_15min.csv", "minervini_trades_15min.csv"),
    ("5 Min", "data/nifty_200_5min", "_5min.csv", "minervini_trades_5min.csv"),
    ("3 Min", "data/nifty_200_3min", "_3minute.csv", "minervini_trades_3min.csv"),
    ("2 Min", "data/nifty_200_2min", "_2minute.csv", "minervini_trades_2min.csv"),
    # 1 Min Data is in a different structure (history folder) with different suffix
    ("1 Min", "data/nifty_200_history", "_minute.csv", "minervini_trades_1min.csv") 
]

# Value Area / Market Profile backtests (separate script)
VALUE_AREA_CONFIGS = [
    ("Value Area 30 Min", "data/nifty_200_30min", "_30min.csv", "value_area_trades_30min.csv"),
]

def run_backtests():
    logging.info("Starting Batch Backtesting...")
    
    start_time = time.time()
    
    for name, data_dir, suffix, output_file in CONFIGS:
        logging.info(f"--- Running {name} Backtest ---")
        
        if not os.path.exists(data_dir):
            logging.warning(f"Data directory {data_dir} not found. Skipping.")
            continue
            
        cmd = [
            "python", "trading/strategies/minervini.py",
            "--data-dir", data_dir,
            "--suffix", suffix,
            "--output", output_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logging.info(f"Completed Backtest {name}. Output: {output_file}")
            
            # Run Portfolio Simulation
            logging.info(f"Running Portfolio Simulation for {name}...")
            sim_cmd = ["python", "scripts/simulate_portfolio.py", output_file]
            subprocess.run(sim_cmd, check=True)
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to run {name}: {e}")
            
    elapsed = time.time() - start_time
    logging.info(f"Batch Backtesting Completed in {elapsed:.2f} seconds.")

    # --- Run Value Area / Market Profile Backtests ---
    logging.info("Starting Value Area Backtests...")
    for name, data_dir, suffix, output_file in VALUE_AREA_CONFIGS:
        logging.info(f"--- Running {name} Backtest ---")
        if not os.path.exists(data_dir):
            logging.warning(f"Data directory {data_dir} not found. Skipping.")
            continue
        cmd = [
            "python", "trading/strategies/value_area.py",
            "--data-dir", data_dir,
            "--suffix", suffix,
            "--output", output_file
        ]
        try:
            subprocess.run(cmd, check=True)
            logging.info(f"Completed {name}. Output: {output_file}")
            sim_cmd = ["python", "scripts/simulate_portfolio.py", output_file]
            subprocess.run(sim_cmd, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to run {name}: {e}")

if __name__ == "__main__":
    run_backtests()
