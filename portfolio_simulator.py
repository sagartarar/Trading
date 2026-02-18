"""
Portfolio Simulator with Kelly's Criterion
============================================
Realistic portfolio simulation that processes raw trade signals from
any backtester and applies:
  - Kelly's Criterion position sizing (half-Kelly, bounded 2-25%)
  - Zerodha charge model (brokerage, STT, DP, stamp, GST, API fees)
  - Capital management (max concurrent positions, cash reserve)
  - Trade frequency caps
  - Date range filtering

Usage:
    python portfolio_simulator.py \
        --trades minervini_trades_daily.csv \
        --mode delivery \
        --start-date 2025-02-17 \
        --end-date 2026-02-17
"""

import pandas as pd
import numpy as np
import argparse
import logging
import os
from typing import Tuple, List, Dict
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Zerodha Charge Model (2024-2025 rates)
# ─────────────────────────────────────────────────────────

@dataclass
class ZerodhaCharges:
    """Complete Zerodha charge model."""
    api_monthly_fee: float = 500.0    # ₹500/month for Kite Connect API

    @staticmethod
    def delivery(buy_value: float, sell_value: float) -> Dict[str, float]:
        """Equity delivery (CNC) charges."""
        turnover = buy_value + sell_value
        brokerage = 0.0                          # Free for delivery
        stt = sell_value * 0.001                 # 0.1% on sell
        txn_charge = turnover * 0.0000297        # NSE transaction charge
        gst = (brokerage + txn_charge) * 0.18    # 18% GST
        sebi = turnover * 0.000001               # ₹10 per crore
        stamp = buy_value * 0.00015              # 0.015% on buy
        dp_charge = 15.93 if sell_value > 0 else 0  # DP charge per stock on sell
        total = brokerage + stt + txn_charge + gst + sebi + stamp + dp_charge
        return {
            "brokerage": brokerage, "stt": stt, "txn": txn_charge,
            "gst": gst, "sebi": sebi, "stamp": stamp, "dp": dp_charge,
            "total": total,
        }

    @staticmethod
    def intraday(buy_value: float, sell_value: float) -> Dict[str, float]:
        """Equity intraday (MIS) charges."""
        turnover = buy_value + sell_value
        brokerage = min(40.0, turnover * 0.0003) # ₹20/order × 2 or 0.03%
        stt = sell_value * 0.00025               # 0.025% on sell
        txn_charge = turnover * 0.0000297
        gst = (brokerage + txn_charge) * 0.18
        sebi = turnover * 0.000001
        stamp = buy_value * 0.00003              # 0.003% on buy
        dp_charge = 0.0                          # No DP for intraday
        total = brokerage + stt + txn_charge + gst + sebi + stamp
        return {
            "brokerage": brokerage, "stt": stt, "txn": txn_charge,
            "gst": gst, "sebi": sebi, "stamp": stamp, "dp": dp_charge,
            "total": total,
        }


# ─────────────────────────────────────────────────────────
# Kelly's Criterion
# ─────────────────────────────────────────────────────────

class KellySizer:
    """
    Rolling Half-Kelly position sizer.
    
    Kelly % = W - (1-W)/R
    where W = win rate, R = avg_win / |avg_loss|
    
    We use Half-Kelly (Kelly/2) for reduced variance.
    Bounded between min_size and max_size.
    """

    def __init__(self, lookback: int = 20, min_size: float = 0.02,
                 max_size: float = 0.25, initial_kelly: float = 0.10):
        self.lookback = lookback
        self.min_size = min_size
        self.max_size = max_size
        self.initial_kelly = initial_kelly
        self.trade_results: List[float] = []  # list of PnL %

    def record_trade(self, pnl_pct: float):
        """Record a completed trade result."""
        self.trade_results.append(pnl_pct)

    def get_fraction(self) -> Tuple[float, Dict]:
        """
        Calculate Half-Kelly fraction for next trade.
        Returns (fraction, debug_info).
        """
        if len(self.trade_results) < self.lookback:
            return self.initial_kelly, {
                "method": "initial_default",
                "n_trades": len(self.trade_results),
                "kelly_raw": self.initial_kelly,
            }

        recent = self.trade_results[-self.lookback:]
        wins = [r for r in recent if r > 0]
        losses = [r for r in recent if r <= 0]

        if not wins or not losses:
            # All wins or all losses — use conservative sizing
            frac = self.min_size if not wins else self.max_size * 0.5
            return frac, {"method": "edge_case", "all_wins": not losses}

        W = len(wins) / len(recent)          # win probability
        avg_win = np.mean(wins)              # average win
        avg_loss = abs(np.mean(losses))      # average loss (positive)
        R = avg_win / avg_loss if avg_loss > 0 else 1.0  # payoff ratio

        kelly_raw = W - (1 - W) / R
        half_kelly = kelly_raw / 2.0

        # Bound it
        fraction = max(self.min_size, min(self.max_size, half_kelly))

        # If Kelly is negative, strategy has negative edge — minimum size
        if kelly_raw <= 0:
            fraction = self.min_size

        return fraction, {
            "method": "half_kelly",
            "W": W, "R": R,
            "kelly_raw": kelly_raw,
            "half_kelly": half_kelly,
            "fraction_bounded": fraction,
        }


# ─────────────────────────────────────────────────────────
# Portfolio Simulator
# ─────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    size_rupees: float
    shares: int
    kelly_fraction: float


@dataclass
class CompletedTrade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    position_size: float
    gross_pnl: float
    charges: float
    net_pnl: float
    pnl_pct: float
    net_pnl_pct: float
    kelly_fraction: float
    equity_at_entry: float
    reason: str
    charge_breakdown: Dict


class PortfolioSimulator:
    """
    Realistic portfolio simulator.

    Processes trade signals chronologically and manages:
    - Position sizing via Kelly's Criterion
    - Max concurrent positions
    - Trade frequency limits
    - Zerodha charges
    - Monthly API fees
    """

    def __init__(self, initial_capital: float = 100_000,
                 max_positions: int = 5,
                 max_trades_per_day: int = 3,
                 cash_reserve_pct: float = 0.10,
                 mode: str = "delivery",
                 kelly_lookback: int = 20,
                 kelly_min: float = 0.02,
                 kelly_max: float = 0.25,
                 api_monthly_fee: float = 500.0):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.max_positions = max_positions
        self.max_trades_per_day = max_trades_per_day
        self.cash_reserve_pct = cash_reserve_pct
        self.mode = mode
        self.api_monthly_fee = api_monthly_fee

        self.charge_fn = ZerodhaCharges.delivery if mode == "delivery" else ZerodhaCharges.intraday

        self.kelly = KellySizer(
            lookback=kelly_lookback, min_size=kelly_min, max_size=kelly_max
        )

        # State
        self.open_positions: List[Position] = []
        self.completed_trades: List[CompletedTrade] = []
        self.daily_trade_count: Dict[str, int] = {}  # date_str -> count
        self.equity_history: List[Dict] = []
        self.months_billed: set = set()

        # Accumulators
        self.total_charges = 0.0
        self.total_api_fees = 0.0
        self.total_brokerage = 0.0
        self.total_stt = 0.0
        self.skipped_signals = 0
        self.skip_reasons: Dict[str, int] = {}

    def _available_capital(self) -> float:
        """Capital available for new positions (after cash reserve)."""
        reserved = self.equity * self.cash_reserve_pct
        allocated = sum(p.size_rupees for p in self.open_positions)
        return max(0, self.equity - reserved - allocated)

    def _deduct_api_fee(self, current_date: pd.Timestamp):
        """Deduct monthly API charge if not yet billed this month."""
        month_key = current_date.strftime("%Y-%m")
        if month_key not in self.months_billed:
            self.months_billed.add(month_key)
            self.equity -= self.api_monthly_fee
            self.total_api_fees += self.api_monthly_fee

    def _skip(self, reason: str):
        self.skipped_signals += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def _can_trade_today(self, date: pd.Timestamp) -> bool:
        date_key = date.strftime("%Y-%m-%d")
        count = self.daily_trade_count.get(date_key, 0)
        return count < self.max_trades_per_day

    def _record_daily_trade(self, date: pd.Timestamp):
        date_key = date.strftime("%Y-%m-%d")
        self.daily_trade_count[date_key] = self.daily_trade_count.get(date_key, 0) + 1

    def process_trade(self, trade: pd.Series) -> bool:
        """
        Process a single trade signal. Returns True if trade was taken.
        """
        entry_date = trade["Entry Date"]
        exit_date = trade["Exit Date"]
        symbol = trade["Symbol"]
        entry_price = trade["Entry Price"]
        exit_price = trade["Exit Price"]
        raw_pnl = trade["PnL"]
        reason = trade.get("Reason", "")

        # ── Gate checks ──
        # 1. Daily trade limit
        if not self._can_trade_today(entry_date):
            self._skip("daily_limit")
            return False

        # 2. Max positions
        if len(self.open_positions) >= self.max_positions:
            self._skip("max_positions")
            return False

        # 3. No duplicate symbol
        if any(p.symbol == symbol for p in self.open_positions):
            self._skip("duplicate_symbol")
            return False

        # 4. Monthly API fee
        self._deduct_api_fee(entry_date)

        # 5. Kelly sizing
        kelly_frac, kelly_debug = self.kelly.get_fraction()
        position_budget = self.equity * kelly_frac

        # 6. Enough capital?
        available = self._available_capital()
        position_size = min(position_budget, available)

        if position_size < 1000:  # Min ₹1000 position
            self._skip("insufficient_capital")
            return False

        # ── Execute trade ──
        shares = int(position_size / entry_price)
        if shares < 1:
            self._skip("price_too_high")
            return False

        actual_buy_value = shares * entry_price
        actual_sell_value = shares * exit_price
        gross_pnl = actual_sell_value - actual_buy_value

        # Charges
        charges = self.charge_fn(actual_buy_value, actual_sell_value)
        charge_total = charges["total"]

        net_pnl = gross_pnl - charge_total
        net_pnl_pct = net_pnl / actual_buy_value if actual_buy_value > 0 else 0

        # Update equity
        self.equity += net_pnl
        self.total_charges += charge_total
        self.total_brokerage += charges["brokerage"]
        self.total_stt += charges["stt"]

        # Record
        self.kelly.record_trade(net_pnl_pct)
        self._record_daily_trade(entry_date)

        ct = CompletedTrade(
            symbol=symbol,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            shares=shares,
            position_size=actual_buy_value,
            gross_pnl=gross_pnl,
            charges=charge_total,
            net_pnl=net_pnl,
            pnl_pct=raw_pnl,
            net_pnl_pct=net_pnl_pct,
            kelly_fraction=kelly_frac,
            equity_at_entry=self.equity - net_pnl,
            reason=reason,
            charge_breakdown=charges,
        )
        self.completed_trades.append(ct)

        # Equity snapshot
        self.equity_history.append({
            "Date": exit_date,
            "Equity": self.equity,
            "Trade #": len(self.completed_trades),
            "Kelly %": kelly_frac * 100,
        })

        return True

    def run(self, trades_df: pd.DataFrame,
            start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Run simulation on a DataFrame of trade signals.
        """
        df = trades_df.copy()

        # Parse dates
        for col in ["Entry Date", "Exit Date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        # Date filter
        if start_date:
            sd = pd.Timestamp(start_date, tz="UTC")
            df = df[df["Entry Date"] >= sd]
        if end_date:
            ed = pd.Timestamp(end_date, tz="UTC")
            df = df[df["Entry Date"] <= ed]

        # Sort chronologically
        df = df.sort_values("Entry Date").reset_index(drop=True)

        log.info(f"Processing {len(df):,} trade signals "
                 f"({df['Entry Date'].min()} → {df['Entry Date'].max()})")

        # Initial equity snapshot
        self.equity_history.append({
            "Date": df["Entry Date"].iloc[0] if len(df) > 0 else pd.Timestamp.now(),
            "Equity": self.equity,
            "Trade #": 0,
            "Kelly %": 10.0,
        })

        taken = 0
        for _, trade in df.iterrows():
            if self.equity <= 1000:
                log.warning("Account blown up. Stopping.")
                break
            if self.process_trade(trade):
                taken += 1

        log.info(f"Trades taken: {taken:,} / {len(df):,} signals "
                 f"(skipped: {self.skipped_signals:,})")

        return self._build_results()

    def _build_results(self) -> pd.DataFrame:
        """Build results DataFrame."""
        if not self.completed_trades:
            return pd.DataFrame()

        records = []
        for t in self.completed_trades:
            records.append({
                "Symbol": t.symbol,
                "Entry Date": t.entry_date,
                "Exit Date": t.exit_date,
                "Entry Price": t.entry_price,
                "Exit Price": t.exit_price,
                "Shares": t.shares,
                "Position Size": t.position_size,
                "Gross PnL": t.gross_pnl,
                "Charges": t.charges,
                "Net PnL": t.net_pnl,
                "Raw PnL %": t.pnl_pct,
                "Net PnL %": t.net_pnl_pct,
                "Kelly %": t.kelly_fraction * 100,
                "Equity": t.equity_at_entry + t.net_pnl,
                "Reason": t.reason,
            })
        return pd.DataFrame(records)

    def print_summary(self, label: str = "Strategy"):
        """Print comprehensive summary."""
        if not self.completed_trades:
            print("No trades executed.")
            return

        trades = self.completed_trades
        n = len(trades)
        net_pnls = [t.net_pnl_pct for t in trades]
        R = np.array(net_pnls)

        # Time span
        start = trades[0].entry_date
        end = trades[-1].exit_date
        days = (end - start).days
        years = days / 365.25
        months = days / 30.44

        # Win/Loss
        wins = sum(1 for r in R if r > 0)
        losses = n - wins
        wr = wins / n

        # Expected value
        ev = R.mean()

        # Sharpe (annualized by trades/year)
        tpy = n / years if years > 0 else n
        sharpe = R.mean() / R.std() * np.sqrt(min(tpy, 252)) if R.std() > 0 else 0

        # Sortino
        down = R[R < 0]
        sortino = R.mean() / down.std() * np.sqrt(min(tpy, 252)) if len(down) > 1 and down.std() > 0 else 0

        # Profit Factor
        gw = R[R > 0].sum()
        gl = abs(R[R <= 0].sum())
        pf = gw / gl if gl > 0 else float('inf')

        # Max Drawdown (on equity curve)
        eq_hist = [self.initial_capital] + [t.equity_at_entry + t.net_pnl for t in trades]
        eq_arr = np.array(eq_hist)
        peak = np.maximum.accumulate(eq_arr)
        dd = (peak - eq_arr) / peak
        max_dd = dd.max()
        max_dd_rupee = (peak - eq_arr).max()

        # CAGR
        final_eq = self.equity
        cagr = (final_eq / self.initial_capital) ** (1 / years) - 1 if years > 0 and final_eq > 0 else -1

        # Calmar
        calmar = cagr / max_dd if max_dd > 0 else float('inf')

        # Max consecutive losses
        streak = pd.Series([1 if r <= 0 else 0 for r in R])
        grp = streak.groupby((streak != streak.shift()).cumsum()).sum()
        max_consec = int(grp.max()) if not grp.empty else 0

        # Kelly stats
        kelly_fracs = [t.kelly_fraction for t in trades]

        # Average position size
        avg_pos = np.mean([t.position_size for t in trades])

        # Total charges breakdown
        total_brk = sum(t.charge_breakdown["brokerage"] for t in trades)
        total_stt = sum(t.charge_breakdown["stt"] for t in trades)
        total_dp = sum(t.charge_breakdown.get("dp", 0) for t in trades)

        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"  ₹{self.initial_capital:,.0f} starting capital | Zerodha {self.mode.upper()}")
        print(f"{'='*70}")
        print(f"  Period: {start.strftime('%d %b %Y')} → {end.strftime('%d %b %Y')} ({months:.0f} months)")
        print(f"  Signals received: {n + self.skipped_signals:,}")
        print(f"  Trades executed:  {n:,}")
        print(f"  Signals skipped:  {self.skipped_signals:,}")
        if self.skip_reasons:
            for reason, count in sorted(self.skip_reasons.items(), key=lambda x: -x[1]):
                print(f"    - {reason}: {count:,}")
        print()
        print(f"  ┌─── RETURNS {'─'*50}┐")
        print(f"  │ Initial Capital:       ₹{self.initial_capital:>12,.0f}          │")
        print(f"  │ Final Equity:          ₹{final_eq:>12,.0f}          │")
        print(f"  │ Net Profit:            ₹{final_eq - self.initial_capital:>+12,.0f}          │")
        print(f"  │ Total Return:          {(final_eq/self.initial_capital-1)*100:>+11.1f}%          │")
        if years >= 1:
            print(f"  │ CAGR:                  {cagr*100:>+11.1f}%          │")
        print(f"  └{'─'*65}┘")
        print()
        print(f"  ┌─── RISK METRICS {'─'*44}┐")
        print(f"  │ Sharpe Ratio:          {sharpe:>12.2f}            │")
        print(f"  │ Sortino Ratio:         {sortino:>12.2f}            │")
        print(f"  │ Profit Factor:         {pf:>12.2f}            │")
        if years >= 1:
            print(f"  │ Calmar Ratio:          {calmar:>12.2f}            │")
        print(f"  │ Max Drawdown:          {max_dd*100:>12.1f}%          │")
        print(f"  │ Max Drawdown (₹):      ₹{max_dd_rupee:>12,.0f}          │")
        print(f"  │ Max Consec Losses:     {max_consec:>12}            │")
        print(f"  └{'─'*65}┘")
        print()
        print(f"  ┌─── TRADE QUALITY {'─'*43}┐")
        print(f"  │ Win Rate (net):        {wr*100:>12.1f}%          │")
        print(f"  │ E.V. per Trade:        {ev*100:>12.3f}%          │")
        print(f"  │ Avg Win:               {R[R>0].mean()*100:>12.2f}%          │")
        print(f"  │ Avg Loss:              {R[R<=0].mean()*100:>12.2f}%          │")
        print(f"  │ Best Trade:            {R.max()*100:>12.2f}%          │")
        print(f"  │ Worst Trade:           {R.min()*100:>12.2f}%          │")
        print(f"  └{'─'*65}┘")
        print()
        print(f"  ┌─── KELLY SIZING {'─'*44}┐")
        print(f"  │ Avg Kelly Fraction:    {np.mean(kelly_fracs)*100:>12.1f}%          │")
        print(f"  │ Min Kelly Fraction:    {np.min(kelly_fracs)*100:>12.1f}%          │")
        print(f"  │ Max Kelly Fraction:    {np.max(kelly_fracs)*100:>12.1f}%          │")
        print(f"  │ Avg Position Size:     ₹{avg_pos:>12,.0f}          │")
        print(f"  └{'─'*65}┘")
        print()
        print(f"  ┌─── ZERODHA CHARGES {'─'*41}┐")
        print(f"  │ Total Charges:         ₹{self.total_charges:>12,.0f}          │")
        print(f"  │   Brokerage:           ₹{total_brk:>12,.0f}          │")
        print(f"  │   STT:                 ₹{total_stt:>12,.0f}          │")
        print(f"  │   DP Charges:          ₹{total_dp:>12,.0f}          │")
        print(f"  │   Other (txn/GST/etc): ₹{self.total_charges-total_brk-total_stt-total_dp:>12,.0f}          │")
        print(f"  │ API Fees ({len(self.months_billed)} months):  ₹{self.total_api_fees:>12,.0f}          │")
        gross_pnl = sum(t.gross_pnl for t in trades)
        all_costs = self.total_charges + self.total_api_fees
        print(f"  │ Total All Costs:       ₹{all_costs:>12,.0f}          │")
        if gross_pnl > 0:
            print(f"  │ Costs % of Gross PnL:  {all_costs/gross_pnl*100:>12.1f}%          │")
        print(f"  └{'─'*65}┘")
        print(f"{'='*70}")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Realistic Portfolio Simulator")
    parser.add_argument("--trades", required=True, nargs="+",
                        help="Trade CSV file(s) to process")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each strategy (same order as --trades)")
    parser.add_argument("--mode", choices=["delivery", "intraday"], default="delivery",
                        help="Zerodha order type (affects charges)")
    parser.add_argument("--capital", type=float, default=100_000,
                        help="Starting capital in ₹ (default: 100000)")
    parser.add_argument("--max-positions", type=int, default=5,
                        help="Max concurrent positions (default: 5)")
    parser.add_argument("--max-daily-trades", type=int, default=3,
                        help="Max trades per day (default: 3)")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV for executed trades")
    parser.add_argument("--kelly-lookback", type=int, default=20)
    parser.add_argument("--kelly-min", type=float, default=0.02)
    parser.add_argument("--kelly-max", type=float, default=0.25)
    parser.add_argument("--api-fee", type=float, default=500.0,
                        help="Monthly API fee in ₹ (default: 500)")
    args = parser.parse_args()

    modes_map = {}
    for f in args.trades:
        # Auto-detect mode: intraday for value_area/PA, delivery for rest
        if "value_area" in f.lower():
            modes_map[f] = "intraday"
        else:
            modes_map[f] = args.mode

    labels = args.labels or [os.path.basename(f).replace(".csv","").replace("_trades","").replace("_"," ").title() for f in args.trades]

    for i, trades_file in enumerate(args.trades):
        label = labels[i] if i < len(labels) else os.path.basename(trades_file)
        mode = modes_map.get(trades_file, args.mode)

        log.info(f"\n{'#'*70}")
        log.info(f"# Running: {label} (mode={mode})")
        log.info(f"{'#'*70}")

        df = pd.read_csv(trades_file)

        sim = PortfolioSimulator(
            initial_capital=args.capital,
            max_positions=args.max_positions,
            max_trades_per_day=args.max_daily_trades,
            mode=mode,
            kelly_lookback=args.kelly_lookback,
            kelly_min=args.kelly_min,
            kelly_max=args.kelly_max,
            api_monthly_fee=args.api_fee,
        )

        results = sim.run(df, start_date=args.start_date, end_date=args.end_date)
        sim.print_summary(label)

        if args.output:
            out_name = args.output if len(args.trades) == 1 else f"{label.lower().replace(' ','_')}_{args.output}"
            results.to_csv(out_name, index=False)
            log.info(f"Saved results to {out_name}")


if __name__ == "__main__":
    main()
