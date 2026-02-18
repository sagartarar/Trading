
from backtest_engine import Strategy
import pandas as pd
import numpy as np

class SMACrossover(Strategy):
    def __init__(self, short_window=50, long_window=200):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window

    def init(self, df):
        super().init(df)
        # Calculate Indicators
        self.data['short_mavg'] = self.data['close'].rolling(window=self.short_window, min_periods=1).mean()
        self.data['long_mavg'] = self.data['close'].rolling(window=self.long_window, min_periods=1).mean()
        
    def next(self, i):
        if i < self.long_window:
            return 0
        
        short_mavg = self.data['short_mavg'].iloc[i]
        long_mavg = self.data['long_mavg'].iloc[i]
        
        # Previous values to detect crossover
        prev_short = self.data['short_mavg'].iloc[i-1]
        prev_long = self.data['long_mavg'].iloc[i-1]
        
        # Buy Signal: Short crosses above Long
        if prev_short <= prev_long and short_mavg > long_mavg:
            return 1
            
        # Sell Signal: Short crosses below Long
        elif prev_short >= prev_long and short_mavg < long_mavg:
            return -1
            
        return 0

class RSIReversion(Strategy):
    def __init__(self, period=14, buy_threshold=30, sell_threshold=70):
        super().__init__()
        self.period = period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def calculate_rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def init(self, df):
        super().init(df)
        self.data['rsi'] = self.calculate_rsi(self.data['close'], self.period)

    def next(self, i):
        if i < self.period:
            return 0

        rsi = self.data['rsi'].iloc[i]

        # Buy Signal: RSI < 30 (Oversold)
        if rsi < self.buy_threshold:
            return 1
        
        # Sell Signal: RSI > 70 (Overbought)
        elif rsi > self.sell_threshold:
            return -1
            
        return 0
