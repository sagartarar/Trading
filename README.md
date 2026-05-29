# Trading — Dip-Buy Backtesting Engine

A high-performance **Rust backtesting engine** that runs a dip-buying + profit-sweeping strategy across the entire Nifty 200 stock universe using local Zerodha market data. Includes an interactive web dashboard for visualizing results.

## Architecture

```
Trading/
├── backtester/          # Rust CLI backtesting engine
│   ├── src/
│   │   ├── main.rs      # CLI orchestrator + JSON export
│   │   ├── data.rs      # Zerodha CSV parser
│   │   ├── composite.rs # Nifty 50 equal-weight index builder
│   │   ├── strategy.rs  # Core dip-buy + profit-sweep engine
│   │   ├── xirr.rs      # Newton-Raphson XIRR calculator
│   │   └── report.rs    # Terminal output formatting
│   └── Cargo.toml
├── dashboard/           # Web visualization dashboard
│   ├── index.html
│   ├── style.css
│   └── app.js
└── data/                # Zerodha CSV data (not tracked in git)
    └── nifty_200_daily/ # 181 stock daily OHLCV files
```

## Strategy

**Continuous Linear Dip-Buying + Cost-Basis Profit Sweeping**

- **Capital:** ₹20L initial + monthly SIPs (₹50K pre-2021, ₹1.5L post-2021)
- **Dip Buys:** When price drops ≥1% from 60-day rolling high, buy ₹20K–₹1L (linearly scaled by drop severity)
- **Profit Sweep:** When market price exceeds average buy price by 15%, sell 10% of holdings back to Liquid BeES
- **Idle Cash:** Earns 6% p.a. in Liquid BeES

## Quick Start

### 1. Run the Backtester
```bash
cd backtester
cargo run --release
```

### 2. Launch the Dashboard
```bash
cd dashboard
python3 -m http.server 8080
# Open http://localhost:8080
```

## Performance

- **0.06 seconds** to backtest all 178 stocks (462,800 simulated trading days)
- Parallel execution via Rayon across all CPU cores

## Key Results

| Metric | Value |
|---|---|
| Stocks Analyzed | 178 |
| Average XIRR | 7.81% |
| Best Performer | ETERNAL (17.78%) |
| Profitable Stocks | 169 / 174 (97%) |
| Above 10% XIRR | 26 stocks |

## Data

Place your Zerodha daily CSV files in `data/nifty_200_daily/` with the format:
```
date,open,high,low,close,volume
2015-02-02 00:00:00+05:30,216.45,217.15,213.8,214.4,18650614
```

## Tech Stack

- **Engine:** Rust 1.92 with Rayon (parallel processing)
- **Dashboard:** Vanilla HTML/CSS/JS + Chart.js
- **Data:** Zerodha Kite historical exports
