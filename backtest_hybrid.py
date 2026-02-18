"""
Hybrid Backtester: Minervini + Market Profile Context
======================================================
Combines Minervini's Trend Template + VCP breakout with Market Profile
context (day type, open type, value area, IB range) to improve entry
timing, stop placement, and exit logic.

Usage:
    python backtest_hybrid.py \
        --daily-dir data/nifty_200_daily \
        --intraday-dir data/nifty_200_30min \
        --output hybrid_trades.csv
"""

import pandas as pd
import numpy as np
import os
import logging
import argparse
from typing import Dict, List, Optional, Tuple
from market_profile import build_daily_profiles, DailyProfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# ---------------------------------------------------------------------------
# Hybrid Backtester
# ---------------------------------------------------------------------------
class HybridBacktester:
    """
    Minervini trend-following entries filtered and enhanced by Market Profile.

    Entry:  Minervini Trend Template + VCP breakout
            PLUS Market Profile confirmation (day type, open type, VA position)
    Stop:   Structural (VA / IB based), never worse than 8%
    Trail:  SMA-50 with MP enhancements (widen on Trend Days, tighten on
            Non-Trend deterioration)
    Exit:   POC rejection on high volume, or trailing stop
    """

    # --- Configurable parameters ---
    FIXED_STOP_PCT = 0.08          # Max stop = 8% below entry (fallback)
    BREAKEVEN_TRIGGER = 0.20       # Move stop to breakeven at +20%
    VCP_RANGE_THRESHOLD = 0.15     # Max consolidation range for VCP
    VOLUME_SURGE_MULT = 1.5        # Volume > 1.5x avg for breakout
    RS_RANK_MIN = 70               # Minimum RS percentile rank
    IB_RANGE_MIN_PCT = 0.005       # Minimum IB range (0.5% of price)
    NON_TREND_DETERIORATION = 3    # Consecutive Non-Trend days → tighten

    # --- Day types that BLOCK entries ---
    BLOCKED_DAY_TYPES = {"Non-Trend", "Neutral"}

    # --- Open types that are PREFERRED (bonus, not required) ---
    PREFERRED_OPENS = {"Open-Drive", "Open-Test-Drive"}

    def __init__(self, daily_dir: str, intraday_dir: str, output_file: str,
                 daily_suffix: str = "_daily.csv",
                 intraday_suffix: str = "_30min.csv"):
        self.daily_dir = daily_dir
        self.intraday_dir = intraday_dir
        self.output_file = output_file
        self.daily_suffix = daily_suffix
        self.intraday_suffix = intraday_suffix

        self.daily_data: Dict[str, pd.DataFrame] = {}
        self.intraday_data: Dict[str, pd.DataFrame] = {}
        self.profiles: Dict[str, List[DailyProfile]] = {}
        self.results = pd.DataFrame()

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------
    def load_data(self):
        """Load daily + 30-min data for each stock, build Market Profiles."""

        # --- Daily data ---
        logging.info(f"Loading daily data from {self.daily_dir}...")
        if not os.path.exists(self.daily_dir):
            logging.error(f"Daily dir {self.daily_dir} not found.")
            return

        daily_files = [f for f in os.listdir(self.daily_dir)
                       if f.endswith(self.daily_suffix)]
        all_closes = {}

        for f in daily_files:
            symbol = f.replace(self.daily_suffix, "")
            try:
                df = pd.read_csv(os.path.join(self.daily_dir, f),
                                 index_col="date", parse_dates=True)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                if len(df) < 250:
                    continue
                self.daily_data[symbol] = df
                all_closes[symbol] = df["close"]
            except Exception as e:
                logging.warning(f"Could not load daily {f}: {e}")

        self.close_df = pd.DataFrame(all_closes)
        logging.info(f"Loaded daily data for {len(self.daily_data)} stocks.")

        # --- Intraday data → Market Profiles ---
        logging.info(f"Loading 30-min data from {self.intraday_dir}...")
        if not os.path.exists(self.intraday_dir):
            logging.warning("No intraday data dir; MP filters disabled.")
            return

        intraday_files = [f for f in os.listdir(self.intraday_dir)
                          if f.endswith(self.intraday_suffix)]
        loaded = 0
        for f in intraday_files:
            symbol = f.replace(self.intraday_suffix, "")
            if symbol not in self.daily_data:
                continue  # Only build profiles for stocks with daily data
            try:
                df = pd.read_csv(os.path.join(self.intraday_dir, f),
                                 index_col="date", parse_dates=True)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                if len(df) < 20:
                    continue
                self.intraday_data[symbol] = df
                profiles = build_daily_profiles(df)
                self.profiles[symbol] = {
                    p.date: p for p in profiles
                }
                loaded += 1
            except Exception as e:
                logging.warning(f"Could not load intraday {f}: {e}")

        logging.info(f"Built Market Profiles for {loaded} stocks.")

    # ------------------------------------------------------------------
    # Minervini Indicators (same as original)
    # ------------------------------------------------------------------
    def calculate_indicators(self):
        logging.info("Calculating Minervini indicators...")

        # RS Rank
        if not self.close_df.empty:
            rs_raw = self.close_df.pct_change(63)
            self.rs_rank = rs_raw.rank(axis=1, pct=True) * 100
        else:
            self.rs_rank = pd.DataFrame()

        for symbol, df in self.daily_data.items():
            df["sma_50"] = df["close"].rolling(50).mean()
            df["sma_150"] = df["close"].rolling(150).mean()
            df["sma_200"] = df["close"].rolling(200).mean()
            df["52_week_low"] = df["low"].rolling(252).min()
            df["52_week_high"] = df["high"].rolling(252).max()
            df["avg_vol_50"] = df["volume"].rolling(50).mean()
            df["20d_high"] = df["high"].rolling(20).max()
            df["20d_low"] = df["low"].rolling(20).min()
            df["consolidation_range"] = (
                (df["20d_high"] - df["20d_low"]) / df["close"]
            )
            df["prev_20d_high"] = df["20d_high"].shift(1)
            df["sma_200_1m_ago"] = df["sma_200"].shift(20)

            if symbol in self.rs_rank.columns:
                df["rs_rating"] = self.rs_rank[symbol]
            else:
                df["rs_rating"] = 0

    # ------------------------------------------------------------------
    # Trend Template Check (vectorised)
    # ------------------------------------------------------------------
    @staticmethod
    def _check_trend_template(df: pd.DataFrame) -> pd.Series:
        c1 = ((df["close"] > df["sma_50"])
              & (df["sma_50"] > df["sma_150"])
              & (df["sma_150"] > df["sma_200"]))
        c2 = df["sma_200"] > df["sma_200_1m_ago"]
        c3 = df["close"] > 1.25 * df["52_week_low"]
        c4 = df["close"] > 0.75 * df["52_week_high"]
        c5 = df["rs_rating"] >= 70
        return c1 & c2 & c3 & c4 & c5

    # ------------------------------------------------------------------
    # Market Profile Helpers
    # ------------------------------------------------------------------
    def _get_profile(self, symbol: str,
                     date: pd.Timestamp) -> Optional[DailyProfile]:
        """Get the DailyProfile for a symbol on a given date."""
        sym_profiles = self.profiles.get(symbol)
        if not sym_profiles:
            return None
        # Normalise date to midnight
        day = pd.Timestamp(date.date())
        return sym_profiles.get(day)

    def _get_prev_profile(self, symbol: str,
                          date: pd.Timestamp) -> Optional[DailyProfile]:
        """Get the most recent profile BEFORE the given date."""
        sym_profiles = self.profiles.get(symbol)
        if not sym_profiles:
            return None
        day = pd.Timestamp(date.date())
        candidates = [d for d in sym_profiles if d < day]
        if not candidates:
            return None
        return sym_profiles[max(candidates)]

    def _count_recent_non_trend(self, symbol: str,
                                date: pd.Timestamp, n: int = 5) -> int:
        """Count Non-Trend/Neutral days in the last n trading days."""
        sym_profiles = self.profiles.get(symbol)
        if not sym_profiles:
            return 0
        day = pd.Timestamp(date.date())
        recent = sorted([d for d in sym_profiles if d < day], reverse=True)[:n]
        return sum(
            1 for d in recent
            if sym_profiles[d].day_type in self.BLOCKED_DAY_TYPES
        )

    # ------------------------------------------------------------------
    # Entry Filter: Market Profile Context
    # ------------------------------------------------------------------
    def _mp_allows_entry(self, symbol: str, date: pd.Timestamp,
                         entry_price: float) -> Tuple[bool, str]:
        """
        Check Market Profile conditions for entry.
        Returns (allowed, reason).
        """
        today = self._get_profile(symbol, date)
        yesterday = self._get_prev_profile(symbol, date)

        # If no MP data, allow entry (fall back to pure Minervini)
        if today is None and yesterday is None:
            return True, "no_mp_data"

        reasons = []

        # --- Filter 1: Day type ---
        if today and today.day_type in self.BLOCKED_DAY_TYPES:
            return False, f"blocked_day_type:{today.day_type}"

        # --- Filter 2: IB Range too narrow ---
        if today and today.ib_high > 0 and today.ib_low > 0:
            ib_range_pct = (today.ib_high - today.ib_low) / entry_price
            if ib_range_pct < self.IB_RANGE_MIN_PCT:
                # Check if volume is surging — if so, coiled spring → allow
                if today.total_volume < 1.2 * (yesterday.total_volume if yesterday else 1):
                    return False, "narrow_ib_low_volume"

        # --- Filter 3: Price below yesterday's VAL ---
        if yesterday and yesterday.val > 0:
            if entry_price < yesterday.val:
                return False, "below_prev_val"

        # --- Bonus: Open type (not a hard filter, just tracked) ---
        if today and today.open_type in self.PREFERRED_OPENS:
            reasons.append("preferred_open")

        return True, "|".join(reasons) if reasons else "mp_confirmed"

    # ------------------------------------------------------------------
    # Stop Placement: Market Profile Enhanced
    # ------------------------------------------------------------------
    def _compute_stop(self, symbol: str, date: pd.Timestamp,
                      entry_price: float) -> float:
        """
        Structural stop based on MP levels, never worse than 8%.
        """
        max_stop = entry_price * (1 - self.FIXED_STOP_PCT)

        yesterday = self._get_prev_profile(symbol, date)
        today = self._get_profile(symbol, date)

        candidates = [max_stop]

        # Use yesterday's VAL as support
        if yesterday and yesterday.val > 0:
            # Place stop just below VAL (with a small buffer)
            val_stop = yesterday.val * 0.998
            if val_stop > max_stop:  # Only if tighter than 8%
                candidates.append(val_stop)

        # Use today's IB Low as support
        if today and today.ib_low > 0:
            ib_stop = today.ib_low * 0.998
            if ib_stop > max_stop:
                candidates.append(ib_stop)

        # Use yesterday's POC as support
        if yesterday and yesterday.poc > 0:
            poc_stop = yesterday.poc * 0.998
            if poc_stop > max_stop:
                candidates.append(poc_stop)

        # Take the tightest valid stop (highest price below entry)
        valid = [s for s in candidates if s < entry_price]
        return max(valid) if valid else max_stop

    # ------------------------------------------------------------------
    # Exit Logic: Market Profile Enhanced
    # ------------------------------------------------------------------
    def _check_mp_exit(self, symbol: str, date: pd.Timestamp,
                       current_price: float, entry_price: float,
                       row: pd.Series) -> Optional[str]:
        """
        Check for MP-based exit signals.
        Returns exit reason or None.
        """
        today = self._get_profile(symbol, date)
        if not today:
            return None

        # --- POC Rejection ---
        # Price closes below today's POC with above-average volume
        if today.poc > 0 and current_price < today.poc:
            if row["volume"] > 1.3 * row.get("avg_vol_50", row["volume"]):
                # Only if we're in profit (don't exit losers on POC rejection)
                if current_price > entry_price:
                    return "POC Rejection"

        return None

    def _should_widen_trail(self, symbol: str,
                            date: pd.Timestamp) -> bool:
        """On Trend Days, widen trailing stop to let winners run."""
        today = self._get_profile(symbol, date)
        if today and today.day_type == "Trend":
            return True
        return False

    def _should_tighten_stop(self, symbol: str,
                             date: pd.Timestamp) -> bool:
        """If recent days are Non-Trend, tighten to breakeven."""
        non_trend_count = self._count_recent_non_trend(symbol, date)
        return non_trend_count >= self.NON_TREND_DETERIORATION

    # ------------------------------------------------------------------
    # Main Simulation
    # ------------------------------------------------------------------
    def run_simulation(self):
        logging.info("Running hybrid simulation...")
        trades = []
        total = len(self.daily_data)

        for idx, (symbol, df) in enumerate(self.daily_data.items()):
            if (idx + 1) % 25 == 0:
                logging.info(f"  Processing {idx+1}/{total}: {symbol}")

            # Pre-compute Trend Template
            df["setup_met"] = self._check_trend_template(df)

            in_position = False
            entry_price = 0.0
            stop_loss = 0.0
            entry_date = None
            mp_entry_reason = ""

            for date, row in df.iterrows():
                if not in_position:
                    if not row["setup_met"]:
                        continue

                    # --- VCP + Breakout check ---
                    is_tight = row["consolidation_range"] < self.VCP_RANGE_THRESHOLD
                    is_breakout = (
                        row["close"] > row["prev_20d_high"]
                        and row["volume"] > self.VOLUME_SURGE_MULT * row["avg_vol_50"]
                    )

                    if not (is_tight and is_breakout):
                        continue

                    # --- Market Profile Entry Filter ---
                    allowed, reason = self._mp_allows_entry(
                        symbol, date, row["close"]
                    )
                    if not allowed:
                        continue

                    # --- ENTER ---
                    in_position = True
                    entry_price = row["close"]
                    entry_date = date
                    mp_entry_reason = reason
                    stop_loss = self._compute_stop(
                        symbol, date, entry_price
                    )

                else:
                    # --- MANAGE POSITION ---
                    current_price = row["close"]

                    # 1. Hard Stop
                    if current_price < stop_loss:
                        in_position = False
                        exit_price = (row["open"]
                                      if row["open"] < stop_loss
                                      else stop_loss)
                        pnl = (exit_price - entry_price) / entry_price
                        trades.append({
                            "Symbol": symbol,
                            "Entry Date": entry_date,
                            "Entry Price": entry_price,
                            "Exit Date": date,
                            "Exit Price": exit_price,
                            "PnL": pnl,
                            "Reason": "Stop Loss",
                            "MP Context": mp_entry_reason,
                        })
                        continue

                    # 2. MP Exit: POC Rejection
                    mp_exit = self._check_mp_exit(
                        symbol, date, current_price, entry_price, row
                    )
                    if mp_exit:
                        in_position = False
                        pnl = (current_price - entry_price) / entry_price
                        trades.append({
                            "Symbol": symbol,
                            "Entry Date": entry_date,
                            "Entry Price": entry_price,
                            "Exit Date": date,
                            "Exit Price": current_price,
                            "PnL": pnl,
                            "Reason": mp_exit,
                            "MP Context": mp_entry_reason,
                        })
                        continue

                    # 3. Trailing Stop — SMA 50 with MP adjustments
                    use_sma_trail = True

                    # On Trend Days → give more room (use SMA 150 instead)
                    if self._should_widen_trail(symbol, date):
                        trail_level = row.get("sma_150", row["sma_50"])
                        if current_price < trail_level:
                            # Even on trend days, respect a wider trail
                            in_position = False
                            pnl = (current_price - entry_price) / entry_price
                            trades.append({
                                "Symbol": symbol,
                                "Entry Date": entry_date,
                                "Entry Price": entry_price,
                                "Exit Date": date,
                                "Exit Price": current_price,
                                "PnL": pnl,
                                "Reason": "Trailing Stop (SMA 150 - Trend Widen)",
                                "MP Context": mp_entry_reason,
                            })
                            continue
                        use_sma_trail = False  # Don't check SMA50 on trend days

                    if use_sma_trail and current_price < row["sma_50"]:
                        in_position = False
                        pnl = (current_price - entry_price) / entry_price
                        trades.append({
                            "Symbol": symbol,
                            "Entry Date": entry_date,
                            "Entry Price": entry_price,
                            "Exit Date": date,
                            "Exit Price": current_price,
                            "PnL": pnl,
                            "Reason": "Trailing Stop (SMA 50)",
                            "MP Context": mp_entry_reason,
                        })
                        continue

                    # 4. Raise stop to breakeven
                    if (current_price > (1 + self.BREAKEVEN_TRIGGER) * entry_price
                            and stop_loss < entry_price):
                        stop_loss = entry_price

                    # 5. Tighten on deterioration
                    if self._should_tighten_stop(symbol, date):
                        if stop_loss < entry_price and current_price > entry_price:
                            stop_loss = entry_price  # Move to breakeven

        # --- Save results ---
        self.results = pd.DataFrame(trades)
        if not self.results.empty:
            logging.info(
                f"Simulation complete. {len(self.results)} trades generated."
            )
            self.results.to_csv(self.output_file, index=False)

            win_rate = (self.results["PnL"] > 0).mean()
            avg_pnl = self.results["PnL"].mean()
            median_pnl = self.results["PnL"].median()

            print(f"\n--- Hybrid Strategy Results ---")
            print(f"Total Trades: {len(self.results)}")
            print(f"Win Rate:     {win_rate:.2%}")
            print(f"Avg PnL:      {avg_pnl:.2%}")
            print(f"Median PnL:   {median_pnl:.2%}")
            print(f"Output saved: {self.output_file}")

            # By MP context
            print(f"\n--- By MP Entry Context ---")
            for ctx, grp in self.results.groupby("MP Context"):
                wr = (grp["PnL"] > 0).mean()
                avg = grp["PnL"].mean()
                print(f"  {ctx}: {len(grp)} trades, "
                      f"Win {wr:.1%}, Avg {avg:.2%}")

            # By exit reason
            print(f"\n--- By Exit Reason ---")
            for reason, grp in self.results.groupby("Reason"):
                wr = (grp["PnL"] > 0).mean()
                avg = grp["PnL"].mean()
                print(f"  {reason}: {len(grp)} trades, "
                      f"Win {wr:.1%}, Avg {avg:.2%}")
        else:
            logging.info("No trades generated.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Minervini + Market Profile Backtester"
    )
    parser.add_argument(
        "--daily-dir", type=str, default="data/nifty_200_daily",
        help="Directory with daily CSVs"
    )
    parser.add_argument(
        "--intraday-dir", type=str, default="data/nifty_200_30min",
        help="Directory with 30-min CSVs"
    )
    parser.add_argument(
        "--daily-suffix", type=str, default="_daily.csv",
        help="Suffix for daily files"
    )
    parser.add_argument(
        "--intraday-suffix", type=str, default="_30min.csv",
        help="Suffix for 30-min files"
    )
    parser.add_argument(
        "--output", type=str, default="hybrid_trades.csv",
        help="Output CSV"
    )

    args = parser.parse_args()

    bt = HybridBacktester(
        daily_dir=args.daily_dir,
        intraday_dir=args.intraday_dir,
        output_file=args.output,
        daily_suffix=args.daily_suffix,
        intraday_suffix=args.intraday_suffix,
    )
    bt.load_data()
    if bt.daily_data:
        bt.calculate_indicators()
        bt.run_simulation()
    else:
        logging.error("No data loaded.")


if __name__ == "__main__":
    main()
