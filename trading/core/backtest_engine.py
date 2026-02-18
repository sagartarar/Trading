
import pandas as pd
import os
import logging
import numpy as np

class Strategy:
    def __init__(self):
        self.position = 0  # 0: Flat, 1: Long, -1: Short (not supported yet)
        self.data = None
        self.trades = []

    def init(self, df):
        """
        Initialize indicators and data.
        df: Pandas DataFrame with OHLCV data.
        """
        self.data = df
        
    def next(self, i):
        """
        Process the candle at index i.
        Returns: 1 (Buy), -1 (Sell), 0 (Hold)
        """
        pass

class Backtester:
    def __init__(self, data_dir, initial_capital=100000):
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        self.equity_curve = []

    def load_data(self, symbol, interval="5min"):
        """
        Load data for a symbol and interval.
        """
        # Determine subdirectory based on interval
        # Interval expected as "5min", "15min", etc.
        # The directory naming convention was "nifty_200_5min"
        
        # We need to find the correct folder. 
        # Assuming data_dir is the root "data" folder.
        target_folder = f"nifty_200_{interval}"
        filepath = os.path.join(self.data_dir, target_folder, f"{symbol}_{interval}.csv")
        
        if not os.path.exists(filepath):
            logging.error(f"Data file not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error loading data for {symbol}: {e}")
            return None

    def run(self, strategy_class, symbol, interval="5min"):
        df = self.load_data(symbol, interval)
        if df is None:
            return None

        strategy = strategy_class()
        strategy.init(df)
        
        self.capital = self.initial_capital
        self.trades = []
        position = 0 # 0: flat, >0: shares held
        entry_price = 0
        
        # Iterate through candles
        # Start from index 1 (or sufficient lookback for indicators)
        # We'll just iterate index integers
        for i in range(len(df)):
            # Update equity (mark to market)
            current_price = df['close'].iloc[i]
            current_equity = self.capital + (position * current_price)
            self.equity_curve.append(current_equity)

            # Ask strategy for decision
            signal = strategy.next(i)
            
            # Simple execution: Buy/Sell all capital
            if signal == 1: # Buy Signal
                if position == 0:
                    # Buy Logic
                    # Calculate quantity
                    quantity = int(self.capital / current_price)
                    if quantity > 0:
                        cost = quantity * current_price
                        self.capital -= cost
                        position = quantity
                        entry_price = current_price
                        self.trades.append({
                            'type': 'BUY',
                            'date': df.index[i],
                            'price': current_price,
                            'quantity': quantity
                        })
            
            elif signal == -1: # Sell Signal
                if position > 0:
                    # Sell Logic
                    revenue = position * current_price
                    profit = revenue - (position * entry_price)
                    self.capital += revenue
                    
                    self.trades.append({
                        'type': 'SELL',
                        'date': df.index[i],
                        'price': current_price,
                        'quantity': position,
                        'pnl': profit
                    })
                    position = 0
                    entry_price = 0
        
        # Close any remaining position at end
        if position > 0:
             current_price = df['close'].iloc[-1]
             revenue = position * current_price
             profit = revenue - (position * entry_price)
             self.capital += revenue
             self.trades.append({
                'type': 'SELL (EOD)',
                'date': df.index[-1],
                'price': current_price,
                'quantity': position,
                'pnl': profit
            })

        return self.calculate_performance()

    def calculate_performance(self):
        if not self.trades:
            return {
                "Total Return %": 0.0,
                "Win Rate %": 0.0,
                "Total Trades": 0
            }

        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) <= 0 and t['type'].startswith('SELL')]
        total_closed_trades = len(winning_trades) + len(losing_trades)
        
        win_rate = (len(winning_trades) / total_closed_trades * 100) if total_closed_trades > 0 else 0
        
        return {
            "Total Return %": round(total_return, 2),
            "Win Rate %": round(win_rate, 2),
            "Total Trades": total_closed_trades,
            "Final Capital": round(self.capital, 2)
        }
