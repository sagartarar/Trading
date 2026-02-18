"""
Auction-Based Contextual Backtester
====================================
Backtests the Market Profile strategy from "Mind Over Markets" and
"Steidlmayer on Markets" across the Nifty 200 universe.

Uses 30-minute OHLCV data to build daily Market Profiles and generates
trades from five setups:
    1. Value Area Rule
    2. Failed Range Extension
    3. Single Print Retracement
    4. Go-With Breakout
    5. Parallel Activity Fade

Output is a CSV trade log compatible with simulate_portfolio.py and
the Streamlit dashboard.

Usage:
    python backtest_value_area.py \\
        --data-dir data/nifty_200_30min \\
        --suffix _30min.csv \\
        --output value_area_trades_30min.csv
"""

import pandas as pd
import numpy as np
import os
import logging
import argparse
from datetime import datetime
from typing import List, Optional, Dict

from trading.core.market_profile import (
    build_daily_profiles,
    detect_market_step,
    estimate_overnight_inventory,
    compute_volume_direction,
    DailyProfile,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Risk parameters
MAX_RISK_PER_TRADE_PCT = 0.02   # 2% risk per trade (unused for structural stops but useful for sizing)
MIN_REWARD_RISK = 1.5           # Minimum R:R to take a trade


class ValueAreaBacktester:
    """
    Self-contained backtester for the Auction-Based Contextual Strategy.
    Follows the same pattern as MinerviniBacktester for compatibility.
    """

    def __init__(self, data_dir: str, file_suffix: str, output_file: str):
        self.data_dir = data_dir
        self.file_suffix = file_suffix
        self.output_file = output_file
        self.data: Dict[str, pd.DataFrame] = {}
        self.results = pd.DataFrame()

    # -------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------
    def load_data(self):
        logging.info(f"Loading data from {self.data_dir} with suffix '{self.file_suffix}'...")
        if not os.path.exists(self.data_dir):
            logging.error(f"Data directory {self.data_dir} does not exist.")
            return

        files = [f for f in os.listdir(self.data_dir) if f.endswith(self.file_suffix)]
        for f in files:
            symbol = f.replace(self.file_suffix, "")
            try:
                df = pd.read_csv(
                    os.path.join(self.data_dir, f),
                    index_col="date",
                    parse_dates=True,
                )
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                # Need at least 10 trading days to build profiles
                unique_dates = df.index.normalize().unique()
                if len(unique_dates) < 10:
                    continue
                self.data[symbol] = df
            except Exception as e:
                logging.warning(f"Could not load {f}: {e}")

        logging.info(f"Loaded data for {len(self.data)} stocks.")

    # -------------------------------------------------------------------
    # Run Simulation
    # -------------------------------------------------------------------
    def run_simulation(self):
        logging.info("Running auction-based backtest...")
        all_trades = []

        for idx, (symbol, df) in enumerate(self.data.items()):
            if (idx + 1) % 25 == 0:
                logging.info(f"  Processing {idx + 1}/{len(self.data)}: {symbol}")

            try:
                trades = self._backtest_symbol(symbol, df)
                all_trades.extend(trades)
            except Exception as e:
                logging.warning(f"Error backtesting {symbol}: {e}")

        self.results = pd.DataFrame(all_trades)

        if not self.results.empty:
            logging.info(f"Simulation complete. {len(self.results)} trades generated.")
            self.results.to_csv(self.output_file, index=False)

            # Summary metrics
            win_rate = len(self.results[self.results["PnL"] > 0]) / len(self.results)
            avg_pnl = self.results["PnL"].mean()

            print(f"\n--- Auction Strategy Results ({self.file_suffix}) ---")
            print(f"Total Trades: {len(self.results)}")
            print(f"Win Rate:     {win_rate:.2%}")
            print(f"Avg PnL/Trade:{avg_pnl:.2%}")
            print(f"Output saved: {self.output_file}")

            # Breakdown by setup
            if "Setup" in self.results.columns:
                print("\n--- By Setup ---")
                for setup, grp in self.results.groupby("Setup"):
                    w = len(grp[grp["PnL"] > 0]) / len(grp) if len(grp) > 0 else 0
                    print(f"  {setup}: {len(grp)} trades, Win {w:.1%}, Avg {grp['PnL'].mean():.2%}")
        else:
            logging.info("No trades generated.")

    # -------------------------------------------------------------------
    # Per-Symbol Backtest
    # -------------------------------------------------------------------
    def _backtest_symbol(self, symbol: str, df: pd.DataFrame) -> list:
        """Run all five setups on one stock's data."""
        # Build daily profiles
        profiles = build_daily_profiles(df, tick_size=self._estimate_tick(df))

        if len(profiles) < 10:
            return []

        trades = []

        # We iterate day-by-day; for each day we check setups based on
        # the previous day(s)' profiles and today's intraday data.
        for day_idx in range(5, len(profiles)):
            today = profiles[day_idx]
            yesterday = profiles[day_idx - 1]
            recent = profiles[max(0, day_idx - 5): day_idx]
            market_step = detect_market_step(profiles[:day_idx], lookback=5)
            inventory = estimate_overnight_inventory(today, yesterday.close)

            # Get today's intraday candles for execution
            today_candles = df[df.index.date == today.date.date()].sort_index()
            if today_candles.empty or len(today_candles) < 3:
                continue

            # Compute volume direction for today
            vol_dir = compute_volume_direction(today_candles)

            # === BALANCED MARKET SETUPS (Step 3 / Step 4) ===
            if market_step in ("Step 3", "Step 4"):
                # Setup 1: Value Area Rule
                trade = self._setup_value_area_rule(
                    symbol, today, yesterday, today_candles
                )
                if trade:
                    trades.append(trade)
                    continue  # One trade per day per stock

                # Setup 2: Failed Range Extension
                trade = self._setup_failed_range_extension(
                    symbol, today, today_candles
                )
                if trade:
                    trades.append(trade)
                    continue

                # Setup 3: Parallel Activity Fade
                trade = self._setup_parallel_activity(
                    symbol, today, yesterday, today_candles, vol_dir
                )
                if trade:
                    trades.append(trade)
                    continue

            # === IMBALANCED MARKET SETUPS (Step 1 / Step 2) ===
            if market_step in ("Step 1", "Step 2"):
                # Setup 4: Single Print Retracement
                trade = self._setup_single_print_retracement(
                    symbol, today, yesterday, today_candles
                )
                if trade:
                    trades.append(trade)
                    continue

                # Setup 5: Go-With Breakout
                trade = self._setup_go_with_breakout(
                    symbol, today, yesterday, recent, today_candles, vol_dir
                )
                if trade:
                    trades.append(trade)
                    continue

        return trades

    # -------------------------------------------------------------------
    # Setup 1: Value Area Rule
    # -------------------------------------------------------------------
    def _setup_value_area_rule(
        self,
        symbol: str,
        today: DailyProfile,
        yesterday: DailyProfile,
        candles: pd.DataFrame,
    ) -> Optional[dict]:
        """
        If price opens outside yesterday's value area but is accepted
        back inside (2+ TPOs), it will likely traverse the entire VA.
        """
        open_price = today.open_price
        prev_vah = yesterday.vah
        prev_val = yesterday.val
        prev_va_range = prev_vah - prev_val

        if prev_va_range <= 0:
            return None

        # Check: open must be OUTSIDE previous VA
        open_above = open_price > prev_vah
        open_below = open_price < prev_val

        if not (open_above or open_below):
            return None

        # Check for acceptance back inside VA:
        # We look at the first few candles to see if price trades back
        # inside the VA and stays there for at least 2 periods.
        periods_inside = 0
        entry_candle_idx = None

        for i in range(len(candles)):
            row = candles.iloc[i]
            price = row["close"]

            if prev_val <= price <= prev_vah:
                periods_inside += 1
                if periods_inside >= 2 and entry_candle_idx is None:
                    entry_candle_idx = i
            else:
                periods_inside = 0

        if entry_candle_idx is None:
            return None

        entry_row = candles.iloc[entry_candle_idx]
        entry_price = entry_row["close"]
        entry_date = candles.index[entry_candle_idx]

        # Direction and target
        if open_above:
            # Opened above VA, accepted back in → expect traverse to VAL
            target = prev_val
            stop = prev_vah + prev_va_range * 0.15  # structural stop above VAH
            direction = "Short"
        else:
            # Opened below VA, accepted back in → expect traverse to VAH
            target = prev_vah
            stop = prev_val - prev_va_range * 0.15  # structural stop below VAL
            direction = "Long"

        # Check R:R
        if direction == "Long":
            risk = entry_price - stop
            reward = target - entry_price
        else:
            risk = stop - entry_price
            reward = entry_price - target

        if risk <= 0 or reward / risk < MIN_REWARD_RISK:
            return None

        # Simulate exit: scan remaining candles
        exit_price, exit_date, reason = self._simulate_exit(
            candles, entry_candle_idx + 1, direction, target, stop
        )

        if direction == "Long":
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        return {
            "Symbol": symbol,
            "Entry Date": entry_date,
            "Entry Price": entry_price,
            "Exit Date": exit_date,
            "Exit Price": exit_price,
            "PnL": pnl,
            "Reason": reason,
            "Setup": "Value Area Rule",
            "Direction": direction,
        }

    # -------------------------------------------------------------------
    # Setup 2: Failed Range Extension
    # -------------------------------------------------------------------
    def _setup_failed_range_extension(
        self,
        symbol: str,
        today: DailyProfile,
        candles: pd.DataFrame,
    ) -> Optional[dict]:
        """
        If the market pushes outside the initial balance but fails to
        hold, fade the failed extension.
        """
        ib_high = today.ib_high
        ib_low = today.ib_low
        ib_range = ib_high - ib_low

        if ib_range <= 0:
            return None

        # We need IB to be established first (first 2 periods).
        # Then look for extension that fails.
        if len(candles) < 4:
            return None

        # Start scanning from 3rd candle onward (after IB)
        extended_up = False
        extended_down = False
        extension_price = None
        extension_idx = None

        for i in range(2, len(candles)):
            row = candles.iloc[i]

            if row["high"] > ib_high + ib_range * 0.05:
                extended_up = True
                extension_price = row["high"]
                extension_idx = i
            elif row["low"] < ib_low - ib_range * 0.05:
                extended_down = True
                extension_price = row["low"]
                extension_idx = i

            if extended_up or extended_down:
                break

        if extension_idx is None:
            return None

        # Check for FAILURE: price comes back inside IB within 2 periods
        failed = False
        entry_idx = None

        scan_end = min(extension_idx + 3, len(candles))
        for j in range(extension_idx + 1, scan_end):
            row = candles.iloc[j]
            if extended_up and row["close"] < ib_high:
                failed = True
                entry_idx = j
                break
            elif extended_down and row["close"] > ib_low:
                failed = True
                entry_idx = j
                break

        if not failed or entry_idx is None:
            return None

        entry_row = candles.iloc[entry_idx]
        entry_price = entry_row["close"]
        entry_date = candles.index[entry_idx]

        ib_mid = (ib_high + ib_low) / 2

        if extended_up:
            # Failed upside extension → Short
            direction = "Short"
            target = ib_mid
            stop = extension_price + ib_range * 0.1
        else:
            # Failed downside extension → Long
            direction = "Long"
            target = ib_mid
            stop = extension_price - ib_range * 0.1

        # Check R:R
        if direction == "Long":
            risk = entry_price - stop
            reward = target - entry_price
        else:
            risk = stop - entry_price
            reward = entry_price - target

        if risk <= 0 or reward / risk < 1.0:
            return None

        exit_price, exit_date, reason = self._simulate_exit(
            candles, entry_idx + 1, direction, target, stop
        )

        if direction == "Long":
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        return {
            "Symbol": symbol,
            "Entry Date": entry_date,
            "Entry Price": entry_price,
            "Exit Date": exit_date,
            "Exit Price": exit_price,
            "PnL": pnl,
            "Reason": reason,
            "Setup": "Failed Range Extension",
            "Direction": direction,
        }

    # -------------------------------------------------------------------
    # Setup 3: Parallel Activity Fade
    # -------------------------------------------------------------------
    def _setup_parallel_activity(
        self,
        symbol: str,
        today: DailyProfile,
        yesterday: DailyProfile,
        candles: pd.DataFrame,
        vol_dir: float,
    ) -> Optional[dict]:
        """
        In a bracketing market, today's range ≈ yesterday's range.
        Fade when price reaches the projected range limit without
        strong initiative volume.
        """
        prev_range = yesterday.high - yesterday.low
        if prev_range <= 0:
            return None

        # Project today's range from the open
        projected_high = today.open_price + prev_range / 2
        projected_low = today.open_price - prev_range / 2

        # Don't fade if volume is strongly directional
        if vol_dir > 1.3 or vol_dir < 0.7:
            return None

        # Scan for price reaching projected limits
        entry_idx = None
        direction = None

        for i in range(2, len(candles)):
            row = candles.iloc[i]

            # Price near projected high → Short fade
            if row["high"] >= projected_high * 0.998:
                entry_idx = i
                direction = "Short"
                break

            # Price near projected low → Long fade
            if row["low"] <= projected_low * 1.002:
                entry_idx = i
                direction = "Long"
                break

        if entry_idx is None:
            return None

        entry_row = candles.iloc[entry_idx]
        entry_price = entry_row["close"]
        entry_date = candles.index[entry_idx]

        if direction == "Short":
            target = today.open_price  # Fade back to open
            stop = projected_high + prev_range * 0.15
        else:
            target = today.open_price
            stop = projected_low - prev_range * 0.15

        # Check R:R
        if direction == "Long":
            risk = entry_price - stop
            reward = target - entry_price
        else:
            risk = stop - entry_price
            reward = entry_price - target

        if risk <= 0 or reward / risk < 1.0:
            return None

        exit_price, exit_date, reason = self._simulate_exit(
            candles, entry_idx + 1, direction, target, stop
        )

        if direction == "Long":
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        return {
            "Symbol": symbol,
            "Entry Date": entry_date,
            "Entry Price": entry_price,
            "Exit Date": exit_date,
            "Exit Price": exit_price,
            "PnL": pnl,
            "Reason": reason,
            "Setup": "Parallel Activity",
            "Direction": direction,
        }

    # -------------------------------------------------------------------
    # Setup 4: Single Print Retracement
    # -------------------------------------------------------------------
    def _setup_single_print_retracement(
        self,
        symbol: str,
        today: DailyProfile,
        yesterday: DailyProfile,
        candles: pd.DataFrame,
    ) -> Optional[dict]:
        """
        In a trend, if price retraces shallowly into yesterday's single prints
        and holds, enter with the trend. If single prints are filled → trend over.
        """
        if not yesterday.single_prints:
            return None

        # Use the first (most significant) single print range
        sp_low, sp_high = yesterday.single_prints[0]

        # Determine trend direction from yesterday's day type
        # If yesterday's close > open → uptrend → buy on retracement into SPs
        if yesterday.close > yesterday.open_price:
            trend = "Up"
        elif yesterday.close < yesterday.open_price:
            trend = "Down"
        else:
            return None

        entry_idx = None

        for i in range(len(candles)):
            row = candles.iloc[i]

            if trend == "Up":
                # Retracement into single prints from above
                if row["low"] <= sp_high and row["close"] > sp_low:
                    entry_idx = i
                    break
            else:
                # Retracement into single prints from below
                if row["high"] >= sp_low and row["close"] < sp_high:
                    entry_idx = i
                    break

        if entry_idx is None:
            return None

        entry_row = candles.iloc[entry_idx]
        entry_price = entry_row["close"]
        entry_date = candles.index[entry_idx]

        if trend == "Up":
            direction = "Long"
            target = yesterday.high + (sp_high - sp_low)  # Extend by SP range
            stop = sp_low - (sp_high - sp_low) * 0.5  # Below single prints
        else:
            direction = "Short"
            target = yesterday.low - (sp_high - sp_low)
            stop = sp_high + (sp_high - sp_low) * 0.5

        # Check R:R
        if direction == "Long":
            risk = entry_price - stop
            reward = target - entry_price
        else:
            risk = stop - entry_price
            reward = entry_price - target

        if risk <= 0 or reward / risk < MIN_REWARD_RISK:
            return None

        exit_price, exit_date, reason = self._simulate_exit(
            candles, entry_idx + 1, direction, target, stop
        )

        if direction == "Long":
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        return {
            "Symbol": symbol,
            "Entry Date": entry_date,
            "Entry Price": entry_price,
            "Exit Date": exit_date,
            "Exit Price": exit_price,
            "PnL": pnl,
            "Reason": reason,
            "Setup": "Single Print Retracement",
            "Direction": direction,
        }

    # -------------------------------------------------------------------
    # Setup 5: Go-With Breakout
    # -------------------------------------------------------------------
    def _setup_go_with_breakout(
        self,
        symbol: str,
        today: DailyProfile,
        yesterday: DailyProfile,
        recent: List[DailyProfile],
        candles: pd.DataFrame,
        vol_dir: float,
    ) -> Optional[dict]:
        """
        If the market breaks out of a balance area after sufficient
        development (high TPO count), go with the breakout.
        Confirmation: volume must increase on breakout.
        """
        # Check internal time clock: need maturity (high TPO width)
        if len(recent) < 3:
            return None

        avg_tpo = np.mean([p.tpo_count_max for p in recent])
        if avg_tpo < 7:  # Not enough development (scaled for 13-TPO day)
            return None

        # Define the balance range from recent profiles
        balance_high = max(p.vah for p in recent)
        balance_low = min(p.val for p in recent)
        balance_range = balance_high - balance_low

        if balance_range <= 0:
            return None

        # Check for breakout today
        entry_idx = None
        direction = None

        for i in range(1, len(candles)):
            row = candles.iloc[i]
            prev_row = candles.iloc[i - 1]

            # Volume confirmation: current candle volume > 1.5x average
            avg_vol = candles["volume"].iloc[:i].mean() if i > 0 else row["volume"]
            vol_surge = row["volume"] > 1.5 * avg_vol

            if not vol_surge:
                continue

            # Upside breakout
            if row["close"] > balance_high and prev_row["close"] <= balance_high:
                direction = "Long"
                entry_idx = i
                break

            # Downside breakout
            if row["close"] < balance_low and prev_row["close"] >= balance_low:
                direction = "Short"
                entry_idx = i
                break

        if entry_idx is None:
            return None

        entry_row = candles.iloc[entry_idx]
        entry_price = entry_row["close"]
        entry_date = candles.index[entry_idx]

        if direction == "Long":
            target = balance_high + balance_range  # Measured move
            stop = balance_high - balance_range * 0.2  # Re-entry into balance = failed
        else:
            target = balance_low - balance_range
            stop = balance_low + balance_range * 0.2

        # Check R:R
        if direction == "Long":
            risk = entry_price - stop
            reward = target - entry_price
        else:
            risk = stop - entry_price
            reward = entry_price - target

        if risk <= 0 or reward / risk < MIN_REWARD_RISK:
            return None

        # Use one-timeframing exit for trend trades
        exit_price, exit_date, reason = self._simulate_exit_one_timeframing(
            candles, entry_idx + 1, direction, target, stop
        )

        if direction == "Long":
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        return {
            "Symbol": symbol,
            "Entry Date": entry_date,
            "Entry Price": entry_price,
            "Exit Date": exit_date,
            "Exit Price": exit_price,
            "PnL": pnl,
            "Reason": reason,
            "Setup": "Go-With Breakout",
            "Direction": direction,
        }

    # -------------------------------------------------------------------
    # Exit Simulation Helpers
    # -------------------------------------------------------------------
    def _simulate_exit(
        self,
        candles: pd.DataFrame,
        start_idx: int,
        direction: str,
        target: float,
        stop: float,
    ) -> tuple:
        """
        Simulate exit scanning remaining candles.
        Returns (exit_price, exit_date, reason).
        """
        for i in range(start_idx, len(candles)):
            row = candles.iloc[i]

            if direction == "Long":
                # Stop hit
                if row["low"] <= stop:
                    exit_price = max(row["open"], stop) if row["open"] > stop else stop
                    return exit_price, candles.index[i], "Stop Loss"
                # Target hit
                if row["high"] >= target:
                    return target, candles.index[i], "Target"
            else:  # Short
                if row["high"] >= stop:
                    exit_price = min(row["open"], stop) if row["open"] < stop else stop
                    return exit_price, candles.index[i], "Stop Loss"
                if row["low"] <= target:
                    return target, candles.index[i], "Target"

        # End of day — close at last candle
        last = candles.iloc[-1]
        return last["close"], candles.index[-1], "EOD Close"

    def _simulate_exit_one_timeframing(
        self,
        candles: pd.DataFrame,
        start_idx: int,
        direction: str,
        target: float,
        stop: float,
    ) -> tuple:
        """
        One-timeframing exit: in an uptrend, exit if the current candle
        trades below the previous candle's low (and vice versa for downtrend).
        """
        prev_low = None
        prev_high = None

        for i in range(start_idx, len(candles)):
            row = candles.iloc[i]

            if direction == "Long":
                # Hard stop
                if row["low"] <= stop:
                    exit_price = max(row["open"], stop) if row["open"] > stop else stop
                    return exit_price, candles.index[i], "Stop Loss"
                # Target
                if row["high"] >= target:
                    return target, candles.index[i], "Target"
                # One-timeframing violation
                if prev_low is not None and row["low"] < prev_low:
                    return row["close"], candles.index[i], "One-Timeframe Violation"
                prev_low = row["low"]

            else:  # Short
                if row["high"] >= stop:
                    exit_price = min(row["open"], stop) if row["open"] < stop else stop
                    return exit_price, candles.index[i], "Stop Loss"
                if row["low"] <= target:
                    return target, candles.index[i], "Target"
                if prev_high is not None and row["high"] > prev_high:
                    return row["close"], candles.index[i], "One-Timeframe Violation"
                prev_high = row["high"]

        last = candles.iloc[-1]
        return last["close"], candles.index[-1], "EOD Close"

    # -------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------
    def _estimate_tick(self, df: pd.DataFrame) -> float:
        """
        Estimate a reasonable tick size for the stock based on its price level.
        Higher-priced stocks get larger tick buckets.
        """
        median_price = df["close"].median()
        if median_price > 5000:
            return 10.0
        elif median_price > 1000:
            return 5.0
        elif median_price > 500:
            return 2.0
        elif median_price > 100:
            return 1.0
        else:
            return 0.5


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Auction-Based Contextual Strategy Backtester"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/nifty_200_30min",
        help="Directory containing stock data CSVs",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_30min.csv",
        help="Filename suffix (e.g., _30min.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="value_area_trades_30min.csv",
        help="Output CSV file for trades",
    )

    args = parser.parse_args()

    backtester = ValueAreaBacktester(args.data_dir, args.suffix, args.output)
    backtester.load_data()

    if backtester.data:
        backtester.run_simulation()
    else:
        logging.error("No data loaded. Check directory and suffix.")


if __name__ == "__main__":
    main()
