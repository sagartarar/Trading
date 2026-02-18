"""
Market Profile Engine
=====================
Constructs TPO (Time-Price Opportunity) profiles from intraday OHLCV data.
Computes Value Area (POC, VAH, VAL), Initial Balance, Single Prints,
Internal Time Clock, Open/Day-Type classification, and Market Steps.

Based on concepts from:
- "Mind Over Markets" by James Dalton
- "Steidlmayer on Markets" by J. Peter Steidlmayer

Convention: Each 30-minute candle = 1 TPO period (letters A-N).
Indian market session: 09:15 - 15:30 IST = 12.5 half-hours → 13 TPO periods (A-M).
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# IST trading hours mapped to TPO letters
# A = 09:15, B = 09:45, C = 10:15, ... M = 15:15
TPO_LETTERS = "ABCDEFGHIJKLMN"

# Tick size for building the price grid (in rupees).
# We discretise prices into this bucket width so that TPO letters
# aggregate at each price *level* rather than each exact price.
DEFAULT_TICK_SIZE = 1.0  # 1 rupee buckets — works well for most Nifty 200 stocks

# Value area width (% of total TPOs)
VALUE_AREA_PCT = 0.70


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class DailyProfile:
    """Represents a single day's Market Profile."""
    date: pd.Timestamp
    open_price: float
    high: float
    low: float
    close: float
    total_volume: int

    # TPO distribution: price_level → list of TPO letters
    tpo_map: Dict[float, List[str]] = field(default_factory=dict)

    # Derived metrics (populated after construction)
    poc: float = 0.0           # Point of Control
    poc_volume: int = 0
    vah: float = 0.0           # Value Area High
    val: float = 0.0           # Value Area Low
    ib_high: float = 0.0       # Initial Balance High
    ib_low: float = 0.0        # Initial Balance Low
    single_prints: List[Tuple[float, float]] = field(default_factory=list)
    tpo_count_max: int = 0     # Internal Time Clock (max TPO width)
    day_type: str = ""
    open_type: str = ""
    range_ext_up: bool = False
    range_ext_down: bool = False


# ---------------------------------------------------------------------------
# Profile Construction
# ---------------------------------------------------------------------------
def _discretise_price(price: float, tick: float) -> float:
    """Round a price down to its tick bucket."""
    return round(np.floor(price / tick) * tick, 4)


def build_daily_profiles(
    df: pd.DataFrame,
    tick_size: float = DEFAULT_TICK_SIZE,
) -> List[DailyProfile]:
    """
    Build a list of DailyProfile objects from a DataFrame of 30-min candles.

    Parameters
    ----------
    df : DataFrame with columns [open, high, low, close, volume]
         and a DatetimeIndex named 'date'.
    tick_size : price bucket width for the TPO grid.

    Returns
    -------
    List of DailyProfile, one per trading day, sorted chronologically.
    """
    profiles: List[DailyProfile] = []

    # Group candles by calendar date
    df = df.copy()
    df["_trade_date"] = df.index.date
    grouped = df.groupby("_trade_date")

    for trade_date, day_df in grouped:
        day_df = day_df.sort_index()

        # Assign TPO letters in chronological order
        tpo_map: Dict[float, List[str]] = defaultdict(list)
        volume_map: Dict[float, int] = defaultdict(int)

        for idx, (ts, row) in enumerate(day_df.iterrows()):
            letter = TPO_LETTERS[idx] if idx < len(TPO_LETTERS) else TPO_LETTERS[-1]

            # Discretise high-low range into price levels
            lo = _discretise_price(row["low"], tick_size)
            hi = _discretise_price(row["high"], tick_size)

            level = lo
            while level <= hi + tick_size * 0.1:  # small epsilon for float
                tpo_map[level].append(letter)
                volume_map[level] += int(row["volume"])
                level = round(level + tick_size, 4)

        if not tpo_map:
            continue

        profile = DailyProfile(
            date=pd.Timestamp(trade_date),
            open_price=day_df.iloc[0]["open"],
            high=day_df["high"].max(),
            low=day_df["low"].min(),
            close=day_df.iloc[-1]["close"],
            total_volume=int(day_df["volume"].sum()),
            tpo_map=dict(tpo_map),
        )

        # --- Compute derived metrics ---
        _compute_poc(profile, volume_map)
        _compute_value_area(profile)
        _compute_initial_balance(profile)
        _detect_single_prints(profile)
        _compute_tpo_width(profile)
        _detect_range_extensions(profile)

        profiles.append(profile)

    # Sort chronologically
    profiles.sort(key=lambda p: p.date)

    # Day-type and open-type need previous profile context
    for i, p in enumerate(profiles):
        prev = profiles[i - 1] if i > 0 else None
        p.open_type = classify_open(p, prev)
        p.day_type = classify_day_type(p)

    return profiles


# ---------------------------------------------------------------------------
# Value Area Computation
# ---------------------------------------------------------------------------
def _compute_poc(profile: DailyProfile, volume_map: Dict[float, int]):
    """Find Point of Control — price level with the most TPOs (tie-break: volume)."""
    best_level = None
    best_tpo_count = 0
    best_volume = 0

    for level, letters in profile.tpo_map.items():
        count = len(letters)
        vol = volume_map.get(level, 0)
        if count > best_tpo_count or (count == best_tpo_count and vol > best_volume):
            best_level = level
            best_tpo_count = count
            best_volume = vol

    profile.poc = best_level if best_level is not None else profile.close
    profile.poc_volume = best_volume


def _compute_value_area(profile: DailyProfile):
    """
    Value Area = the price range covering ~70% of total TPOs,
    expanding symmetrically from the POC.
    """
    if not profile.tpo_map:
        profile.vah = profile.high
        profile.val = profile.low
        return

    total_tpos = sum(len(v) for v in profile.tpo_map.values())
    target = int(np.ceil(total_tpos * VALUE_AREA_PCT))

    sorted_levels = sorted(profile.tpo_map.keys())
    poc_idx = None
    for i, lvl in enumerate(sorted_levels):
        if lvl == profile.poc:
            poc_idx = i
            break

    if poc_idx is None:
        profile.vah = profile.high
        profile.val = profile.low
        return

    # Start with POC
    accumulated = len(profile.tpo_map[sorted_levels[poc_idx]])
    lo_idx = poc_idx
    hi_idx = poc_idx

    while accumulated < target:
        # Look one step above and below
        above_count = 0
        below_count = 0

        if hi_idx + 1 < len(sorted_levels):
            above_count = len(profile.tpo_map[sorted_levels[hi_idx + 1]])
        if lo_idx - 1 >= 0:
            below_count = len(profile.tpo_map[sorted_levels[lo_idx - 1]])

        if above_count == 0 and below_count == 0:
            break

        # Expand toward the side with more TPOs (tie → expand both)
        if above_count >= below_count:
            hi_idx += 1
            accumulated += above_count
        else:
            lo_idx -= 1
            accumulated += below_count

    profile.val = sorted_levels[lo_idx]
    profile.vah = sorted_levels[hi_idx]


# ---------------------------------------------------------------------------
# Initial Balance
# ---------------------------------------------------------------------------
def _compute_initial_balance(profile: DailyProfile):
    """IB = price range covered by the first two TPO periods (A + B)."""
    ib_levels = []
    for level, letters in profile.tpo_map.items():
        if "A" in letters or "B" in letters:
            ib_levels.append(level)

    if ib_levels:
        profile.ib_low = min(ib_levels)
        profile.ib_high = max(ib_levels)
    else:
        profile.ib_high = profile.high
        profile.ib_low = profile.low


# ---------------------------------------------------------------------------
# Single Prints (Minus Development)
# ---------------------------------------------------------------------------
def _detect_single_prints(profile: DailyProfile):
    """
    Single prints = price levels with exactly 1 TPO.
    Group consecutive single-print levels into ranges.
    Exclude the extreme high/low (tails are expected to be thin).
    """
    sorted_levels = sorted(profile.tpo_map.keys())
    if len(sorted_levels) < 3:
        return

    singles = []
    # Skip the very top and bottom levels (tails)
    for level in sorted_levels[1:-1]:
        if len(profile.tpo_map[level]) == 1:
            singles.append(level)

    if not singles:
        return

    # Group consecutive levels into ranges
    ranges = []
    start = singles[0]
    prev = singles[0]
    tick = sorted_levels[1] - sorted_levels[0] if len(sorted_levels) > 1 else 1.0

    for s in singles[1:]:
        if abs(s - prev - tick) < tick * 0.1:
            prev = s
        else:
            if prev > start:  # At least 2 consecutive levels to count
                ranges.append((start, prev))
            start = s
            prev = s

    if prev > start:
        ranges.append((start, prev))

    profile.single_prints = ranges


# ---------------------------------------------------------------------------
# Internal Time Clock
# ---------------------------------------------------------------------------
def _compute_tpo_width(profile: DailyProfile):
    """Max TPO count across all price levels."""
    if profile.tpo_map:
        profile.tpo_count_max = max(len(v) for v in profile.tpo_map.values())
    else:
        profile.tpo_count_max = 0


# ---------------------------------------------------------------------------
# Range Extensions
# ---------------------------------------------------------------------------
def _detect_range_extensions(profile: DailyProfile):
    """Did the market extend beyond the Initial Balance?"""
    profile.range_ext_up = profile.high > profile.ib_high
    profile.range_ext_down = profile.low < profile.ib_low


# ---------------------------------------------------------------------------
# Open-Type Classification
# ---------------------------------------------------------------------------
def classify_open(profile: DailyProfile, prev_profile: Optional[DailyProfile]) -> str:
    """
    Classify the open into one of four types based on the first two TPO periods.

    Open-Drive:             Opens at/near extreme and drives away aggressively.
    Open-Test-Drive:        Opens, tests a level, reverses, and drives.
    Open-Rejection-Reverse: Opens one way, gets pushed back.
    Open-Auction:           Random rotation, no clear direction.
    """
    if not prev_profile:
        return "Open-Auction"

    ib_range = profile.ib_high - profile.ib_low
    if ib_range <= 0:
        return "Open-Auction"

    prev_range = prev_profile.high - prev_profile.low
    if prev_range <= 0:
        return "Open-Auction"

    open_price = profile.open_price
    ib_mid = (profile.ib_high + profile.ib_low) / 2

    # Open-Drive: open is at the extreme of IB and IB is wide
    # (open is within 10% of one IB extreme and close of A period is directional)
    open_near_ib_high = (profile.ib_high - open_price) < 0.1 * ib_range
    open_near_ib_low = (open_price - profile.ib_low) < 0.1 * ib_range

    # Check if open is outside previous value area
    open_outside_prev_va = (open_price > prev_profile.vah) or (open_price < prev_profile.val)

    if open_near_ib_low and profile.close > ib_mid:
        if open_outside_prev_va:
            return "Open-Test-Drive"
        return "Open-Drive"
    elif open_near_ib_high and profile.close < ib_mid:
        if open_outside_prev_va:
            return "Open-Test-Drive"
        return "Open-Drive"

    # Open-Rejection-Reverse: gap open that reverses
    if open_outside_prev_va:
        if open_price > prev_profile.vah and profile.close < prev_profile.vah:
            return "Open-Rejection-Reverse"
        if open_price < prev_profile.val and profile.close > prev_profile.val:
            return "Open-Rejection-Reverse"

    return "Open-Auction"


# ---------------------------------------------------------------------------
# Day-Type Classification
# ---------------------------------------------------------------------------
def classify_day_type(profile: DailyProfile) -> str:
    """
    Classify the day based on profile shape:
    - Trend Day:            Wide range, single prints, strong directional move.
    - Normal Day:           IB = ~day range, balanced bell curve.
    - Normal Variation:     Moderate range extension (one side only).
    - Double Distribution:  Two distinct POC areas.
    - Neutral Day:          Range extensions on both sides, closes near middle.
    - Non-Trend Day:       Very narrow range, low volume.
    """
    ib_range = profile.ib_high - profile.ib_low
    day_range = profile.high - profile.low

    if day_range <= 0:
        return "Non-Trend"

    ib_ratio = ib_range / day_range if day_range > 0 else 1.0
    has_single_prints = len(profile.single_prints) > 0
    both_extensions = profile.range_ext_up and profile.range_ext_down

    # Trend Day: wide range, single prints present, IB is small part of range
    if ib_ratio < 0.35 and has_single_prints:
        return "Trend"

    # Double Distribution: both extensions AND wide range
    if both_extensions and ib_ratio < 0.5:
        return "Double Distribution"

    # Neutral Day: both extensions but closes near middle
    mid = (profile.high + profile.low) / 2
    close_near_mid = abs(profile.close - mid) < 0.25 * day_range
    if both_extensions and close_near_mid:
        return "Neutral"

    # Normal Variation: one-sided extension
    if (profile.range_ext_up or profile.range_ext_down) and not both_extensions:
        return "Normal Variation"

    # Non-Trend: very narrow range relative to profile width
    if profile.tpo_count_max >= 8 and ib_ratio > 0.85:
        return "Non-Trend"

    return "Normal"


# ---------------------------------------------------------------------------
# Market Step Detection (Steidlmayer's Four Steps)
# ---------------------------------------------------------------------------
def detect_market_step(profiles: List[DailyProfile], lookback: int = 5) -> str:
    """
    Determine which "Step" the market is in based on the last `lookback` profiles.

    Step 1 (Distribution):  Trending — directional, value shifting in one direction.
    Step 2 (Stopping):      Trend slows — large volume at extreme, narrowing range.
    Step 3 (Development):   Bracketing — value area overlaps, mean reversion.
    Step 4 (Efficiency):    Retracement toward center of vertical range.

    Returns one of: "Step 1", "Step 2", "Step 3", "Step 4"
    """
    if len(profiles) < lookback:
        return "Step 3"  # Default: development (balanced)

    recent = profiles[-lookback:]

    # Check value area migration
    va_mids = [(p.vah + p.val) / 2 for p in recent]
    va_changes = [va_mids[i] - va_mids[i - 1] for i in range(1, len(va_mids))]

    # All positive or all negative = trending
    all_up = all(c > 0 for c in va_changes)
    all_down = all(c < 0 for c in va_changes)

    if all_up or all_down:
        # Check if latest day shows stopping behavior
        latest = recent[-1]
        prev = recent[-2]

        # Stopping: range narrows significantly, high volume at extreme
        range_contraction = (latest.high - latest.low) < 0.6 * (prev.high - prev.low)
        if range_contraction:
            return "Step 2"
        return "Step 1"

    # Check for bracketing (overlapping value areas)
    va_overlaps = 0
    for i in range(1, len(recent)):
        prev_va = (recent[i - 1].val, recent[i - 1].vah)
        curr_va = (recent[i].val, recent[i].vah)
        # Overlap exists if max(lows) < min(highs)
        if max(prev_va[0], curr_va[0]) < min(prev_va[1], curr_va[1]):
            va_overlaps += 1

    if va_overlaps >= lookback - 2:
        # Heavy overlap = development (Step 3)
        # Check internal time clock for maturity
        tpo_widths = [p.tpo_count_max for p in recent]
        avg_tpo = np.mean(tpo_widths)

        # If overdeveloped, expect efficiency move
        if avg_tpo > 10:  # Scaled for 30-min TPOs (13 max)
            return "Step 4"
        return "Step 3"

    # Mixed: likely transitioning (Step 4 — retracement/efficiency)
    return "Step 4"


# ---------------------------------------------------------------------------
# Overnight Inventory
# ---------------------------------------------------------------------------
def estimate_overnight_inventory(
    profile: DailyProfile, prev_close: float
) -> str:
    """
    Estimate whether overnight traders are net long or short.

    If open > prev_close → overnight longs accumulated → expect potential
    selling pressure (inventory adjustment).

    Returns: "Long", "Short", or "Neutral"
    """
    if prev_close <= 0:
        return "Neutral"

    gap_pct = (profile.open_price - prev_close) / prev_close

    if gap_pct > 0.003:   # > 0.3% gap up
        return "Long"
    elif gap_pct < -0.003:  # > 0.3% gap down
        return "Short"
    return "Neutral"


# ---------------------------------------------------------------------------
# Volume Analysis Helpers (OFI approximation)
# ---------------------------------------------------------------------------
def compute_volume_direction(day_df: pd.DataFrame) -> float:
    """
    Approximate buying/selling pressure from OHLCV data.
    Uses the close-to-range ratio per candle to classify volume.

    Returns a ratio: > 1.0 = buying pressure, < 1.0 = selling pressure.
    """
    if day_df.empty:
        return 1.0

    buy_vol = 0
    sell_vol = 0

    for _, row in day_df.iterrows():
        candle_range = row["high"] - row["low"]
        if candle_range <= 0:
            continue
        # Close relative position within the candle
        close_ratio = (row["close"] - row["low"]) / candle_range
        buy_vol += row["volume"] * close_ratio
        sell_vol += row["volume"] * (1 - close_ratio)

    if sell_vol <= 0:
        return 2.0
    return buy_vol / sell_vol


# ---------------------------------------------------------------------------
# Convenience: get profile metrics as a DataFrame
# ---------------------------------------------------------------------------
def profiles_to_dataframe(profiles: List[DailyProfile]) -> pd.DataFrame:
    """Convert a list of DailyProfile objects into a summary DataFrame."""
    records = []
    for p in profiles:
        records.append({
            "date": p.date,
            "open": p.open_price,
            "high": p.high,
            "low": p.low,
            "close": p.close,
            "volume": p.total_volume,
            "poc": p.poc,
            "vah": p.vah,
            "val": p.val,
            "ib_high": p.ib_high,
            "ib_low": p.ib_low,
            "ib_range": p.ib_high - p.ib_low,
            "day_range": p.high - p.low,
            "tpo_width": p.tpo_count_max,
            "single_print_count": len(p.single_prints),
            "day_type": p.day_type,
            "open_type": p.open_type,
            "range_ext_up": p.range_ext_up,
            "range_ext_down": p.range_ext_down,
        })
    return pd.DataFrame(records)
