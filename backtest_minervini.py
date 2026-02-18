import pandas as pd
import numpy as np
import os
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class MinerviniBacktester:
    def __init__(self, data_dir, file_suffix, output_file):
        self.data_dir = data_dir
        self.file_suffix = file_suffix
        self.output_file = output_file
        self.tickers = []
        self.data = {}
        self.results = []
        
    def load_data(self):
        logging.info(f"Loading data from {self.data_dir} with suffix '{self.file_suffix}'...")
        if not os.path.exists(self.data_dir):
            logging.error(f"Data directory {self.data_dir} does not exist.")
            return

        files = [f for f in os.listdir(self.data_dir) if f.endswith(self.file_suffix)]
        self.tickers = [f.replace(self.file_suffix, "") for f in files]
        
        all_closes = {}
        
        for f in files:
            symbol = f.replace(self.file_suffix, "")
            try:
                df = pd.read_csv(os.path.join(self.data_dir, f), index_col='date', parse_dates=True)
                # Ensure index is datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                if len(df) < 250: # Need at least ~250 bars for 200 SMA (approx 1 year daily, or ~2 weeks hourly)
                    # For hourly, 200 bars is just ~8-10 days depending on trading hours. 
                    # So 250 is a safe minimum for any timeframe to calculate 200 SMA.
                    continue
                self.data[symbol] = df
                all_closes[symbol] = df['close']
            except Exception as e:
                logging.warning(f"Could not load {f}: {e}")
            
        self.close_df = pd.DataFrame(all_closes)
        logging.info(f"Loaded data for {len(self.data)} stocks.")

    def calculate_indicators(self):
        logging.info("Calculating indicators...")
        
        # Calculate RS Rating Proxy (ROC percentile)
        # For Daily: 3 months ~ 63 days.
        # For Hourly: 3 months ~ 63 * 7 = 440 hours? 
        # To keep it simple and consistent logic wise (Trend Template uses 200 period MA),
        # we need a relative strength metric compatible with the timeframe.
        # Let's use ROC(63) as "Period Momentum". On daily it's 3-month. On hourly it's ~1 week.
        # This scales logically: "Is this stock moving faster than others over the last X bars?"
        
        if not self.close_df.empty:
            rs_raw = self.close_df.pct_change(63)
            self.rs_rank = rs_raw.rank(axis=1, pct=True) * 100
        else:
            self.rs_rank = pd.DataFrame()
        
        for symbol, df in self.data.items():
            # Trend Template Indicators
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_150'] = df['close'].rolling(window=150).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            # 52-Week High/Low equivalent (252 bars)
            df['52_week_low'] = df['low'].rolling(window=252).min()
            df['52_week_high'] = df['high'].rolling(window=252).max()
            
            df['avg_vol_50'] = df['volume'].rolling(window=50).mean()
            
            # VCP Indicators
            # Range over last 20 bars
            df['20d_high'] = df['high'].rolling(window=20).max()
            df['20d_low'] = df['low'].rolling(window=20).min()
            df['consolidation_range'] = (df['20d_high'] - df['20d_low']) / df['close']
            
            # Breakout Signal Helper
            df['prev_20d_high'] = df['20d_high'].shift(1)
            
            # SMA 200 Trend (Check vs 20 bars ago)
            df['sma_200_1m_ago'] = df['sma_200'].shift(20)
            
            # Add RS Rank
            if symbol in self.rs_rank.columns:
                df['rs_rating'] = self.rs_rank[symbol]
            else:
                df['rs_rating'] = 0

    def check_setup(self, df):
        # Vectorized check for speed
        c1 = (df['close'] > df['sma_50']) & (df['sma_50'] > df['sma_150']) & (df['sma_150'] > df['sma_200'])
        c2 = df['sma_200'] > df['sma_200_1m_ago']
        c3 = df['close'] > (1.25 * df['52_week_low'])
        c4 = df['close'] > (0.75 * df['52_week_high'])
        c5 = df['rs_rating'] >= 70
        return c1 & c2 & c3 & c4 & c5

    def run_simulation(self):
        logging.info("Running simulation...")
        trades = []
        
        for symbol, df in self.data.items():
            # Pre-calculate setup condition
            df['setup_met'] = self.check_setup(df)
            
            in_position = False
            entry_price = 0
            stop_loss = 0
            entry_date = None
            
            for date, row in df.iterrows():
                if not in_position:
                    if row['setup_met']:
                        # Check VCP tightness (< 15% range)
                        is_tight = row['consolidation_range'] < 0.15
                        
                        # Check Breakout
                        is_breakout = (row['close'] > row['prev_20d_high']) and \
                                      (row['volume'] > 1.5 * row['avg_vol_50'])
                        
                        if is_tight and is_breakout:
                            # BUY SIGNAL
                            in_position = True
                            entry_price = row['close']
                            entry_date = date
                            stop_loss = entry_price * 0.92 # 8% initial stop
                            
                else:
                    # Manage Position
                    current_price = row['close']
                    
                    # 1. Stop Loss Hit?
                    if current_price < stop_loss:
                        in_position = False
                        # Estimating exit price
                        if row['open'] < stop_loss:
                            exit_price = row['open']
                        else:
                            exit_price = stop_loss
                            
                        pnl = (exit_price - entry_price) / entry_price
                        trades.append({
                            'Symbol': symbol,
                            'Entry Date': entry_date,
                            'Entry Price': entry_price,
                            'Exit Date': date,
                            'Exit Price': exit_price,
                            'PnL': pnl,
                            'Reason': 'Stop Loss'
                        })
                        continue

                    # 2. Trailing Stop (Close < SMA 50)
                    if current_price < row['sma_50']:
                        in_position = False
                        exit_price = current_price
                        pnl = (exit_price - entry_price) / entry_price
                        trades.append({
                            'Symbol': symbol,
                            'Entry Date': entry_date,
                            'Entry Price': entry_price,
                            'Exit Date': date,
                            'Exit Price': exit_price,
                            'PnL': pnl,
                            'Reason': 'Trailing Stop (SMA 50)'
                        })
                        continue
                        
                    # 3. Raise Stop Loss (Breakeven)
                    if current_price > 1.2 * entry_price and stop_loss < entry_price:
                        stop_loss = entry_price

        self.results = pd.DataFrame(trades)
        if not self.results.empty:
            logging.info(f"Simulation complete. {len(self.results)} trades generated.")
            self.results.to_csv(self.output_file, index=False)
            
            # Calculate metrics
            win_rate = len(self.results[self.results['PnL'] > 0]) / len(self.results)
            avg_pnl = self.results['PnL'].mean()
            
            print(f"--- Backtest Results ({self.file_suffix}) ---")
            print(f"Total Trades: {len(self.results)}")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Avg PnL per Trade: {avg_pnl:.2%}")
            print(f"Output saved to {self.output_file}")
            
        else:
            logging.info("No trades generated.")


def main():
    parser = argparse.ArgumentParser(description='Minervini Strategy Backtester')
    parser.add_argument('--data-dir', type=str, default='data/nifty_200_daily', help='Directory containing stock data CSVs')
    parser.add_argument('--suffix', type=str, default='_daily.csv', help='Filename suffix (e.g., _daily.csv, _60min.csv)')
    parser.add_argument('--output', type=str, default='minervini_trades.csv', help='Output CSV file for trades')
    
    args = parser.parse_args()
    
    backtester = MinerviniBacktester(args.data_dir, args.suffix, args.output)
    backtester.load_data()
    if backtester.data:
        backtester.calculate_indicators()
        backtester.run_simulation()
    else:
        logging.error("No data loaded. Check directory and suffix.")

if __name__ == "__main__":
    main()
