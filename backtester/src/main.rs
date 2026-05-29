mod composite;
mod data;
mod report;
mod strategy;
mod xirr;

use rayon::prelude::*;
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// JSON output structure for the dashboard.
#[derive(Serialize)]
struct DashboardOutput {
    config: ConfigSummary,
    composite: Option<strategy::StrategyResult>,
    stocks: Vec<strategy::StrategyResult>,
    summary: SummaryStats,
    execution_time_ms: u64,
}

#[derive(Serialize)]
struct ConfigSummary {
    initial_capital: f64,
    initial_deployment: f64,
    lookback_days: usize,
    cooldown_days: u32,
    min_trade_size: f64,
    max_trade_size: f64,
    profit_sweep_target_pct: f64,
    profit_sweep_fraction_pct: f64,
    liquid_bees_rate_pct: f64,
    transaction_cost_pct: f64,
}

#[derive(Serialize)]
struct SummaryStats {
    total_stocks_analyzed: usize,
    valid_results: usize,
    avg_xirr: f64,
    best_xirr: f64,
    best_symbol: String,
    worst_xirr: f64,
    worst_symbol: String,
    positive_xirr_count: usize,
    above_10_count: usize,
    above_15_count: usize,
    above_20_count: usize,
}

fn main() {
    let total_start = Instant::now();

    let data_dir = Path::new("../data/nifty_200_daily");

    if !data_dir.exists() {
        eprintln!("ERROR: Data directory not found at: {}", data_dir.display());
        eprintln!("Expected Zerodha CSV files at: ../data/nifty_200_daily/");
        std::process::exit(1);
    }

    let config = strategy::StrategyConfig::default();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║        RUST BACKTESTING ENGINE v0.1.0                   ║");
    println!("║        Dip-Buy + Profit-Sweep Strategy                  ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("  Data Directory : {}", data_dir.display());
    println!("  Strategy       : Continuous Linear Scaling + Cost-Basis Sweep");
    println!("  Capital        : ₹20,00,000 initial + monthly SIPs");
    println!("  Lookback       : {} days", config.lookback);
    println!("  Dip Range      : ₹{:.0} (1% dip) → ₹{:.0} (5%+ dip)", config.min_trade_size, config.max_trade_size);
    println!("  Profit Sweep   : Sell {:.0}% when price >{:.0}% above avg cost", config.profit_sweep_fraction * 100.0, config.profit_sweep_target * 100.0);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 1: Nifty 50 Composite Index
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PHASE 1: Building Nifty 50 Composite Index...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let phase1_start = Instant::now();
    let mut composite_result: Option<strategy::StrategyResult> = None;

    match composite::build_nifty50_composite(data_dir) {
        Ok((composite_bars, loaded, skipped)) => {
            println!("  Loaded {} / 50 constituent stocks", loaded.len());
            if !skipped.is_empty() {
                println!("  Skipped: {}", skipped.join(", "));
            }
            println!("  Composite index: {} trading days ({} → {})",
                composite_bars.len(),
                composite_bars.first().map(|b| b.date.to_string()).unwrap_or_default(),
                composite_bars.last().map(|b| b.date.to_string()).unwrap_or_default(),
            );

            if let Some(result) = strategy::run_strategy("NIFTY50-EW", &composite_bars, &config) {
                report::print_detailed_report("NIFTY 50 EQUAL-WEIGHT COMPOSITE", &result);
                composite_result = Some(result);
            } else {
                println!("  WARNING: Strategy could not run on composite index.");
            }
        }
        Err(e) => {
            eprintln!("  ERROR building composite: {}", e);
        }
    }

    println!("  Phase 1 completed in {:.2}s", phase1_start.elapsed().as_secs_f64());

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2: Individual Stock Scan (Parallel)
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PHASE 2: Scanning All Individual Stocks...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let phase2_start = Instant::now();

    let symbols = data::list_available_symbols(data_dir);
    println!("  Found {} stock symbols", symbols.len());
    println!("  Running strategy on all stocks in parallel...");

    let mut results: Vec<strategy::StrategyResult> = symbols
        .par_iter()
        .filter_map(|symbol| {
            let bars = data::load_stock(data_dir, symbol).ok()?;
            if bars.len() < 252 {
                return None;
            }
            strategy::run_strategy(symbol, &bars, &config)
        })
        .collect();

    // Sort by XIRR descending (NaN goes to the bottom)
    results.sort_by(|a, b| {
        match (a.xirr_pct.is_nan(), b.xirr_pct.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => b.xirr_pct.partial_cmp(&a.xirr_pct).unwrap_or(std::cmp::Ordering::Equal),
        }
    });

    println!("  Successfully backtested {} stocks", results.len());
    println!("  Phase 2 completed in {:.2}s", phase2_start.elapsed().as_secs_f64());

    // Print leaderboard
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  INDIVIDUAL STOCK LEADERBOARD (Ranked by XIRR)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    report::print_leaderboard(&results);
    report::print_summary(&results);

    // Print top 10 detailed results
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  TOP 10 STOCKS — DETAILED BREAKDOWN");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    for (i, result) in results.iter().take(10).enumerate() {
        if result.xirr_pct.is_nan() {
            continue;
        }
        let title = format!("#{} — {}", i + 1, result.symbol);
        report::print_detailed_report(&title, result);
    }

    let execution_time_ms = total_start.elapsed().as_millis() as u64;

    // ═══════════════════════════════════════════════════════════════
    // PHASE 3: Export JSON for Dashboard
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PHASE 3: Exporting JSON for Dashboard...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let valid: Vec<&strategy::StrategyResult> = results.iter().filter(|r| !r.xirr_pct.is_nan()).collect();

    let summary = if valid.is_empty() {
        SummaryStats {
            total_stocks_analyzed: results.len(),
            valid_results: 0,
            avg_xirr: 0.0,
            best_xirr: 0.0,
            best_symbol: String::new(),
            worst_xirr: 0.0,
            worst_symbol: String::new(),
            positive_xirr_count: 0,
            above_10_count: 0,
            above_15_count: 0,
            above_20_count: 0,
        }
    } else {
        let best = valid.iter().max_by(|a, b| a.xirr_pct.partial_cmp(&b.xirr_pct).unwrap()).unwrap();
        let worst = valid.iter().min_by(|a, b| a.xirr_pct.partial_cmp(&b.xirr_pct).unwrap()).unwrap();
        SummaryStats {
            total_stocks_analyzed: results.len(),
            valid_results: valid.len(),
            avg_xirr: valid.iter().map(|r| r.xirr_pct).sum::<f64>() / valid.len() as f64,
            best_xirr: best.xirr_pct,
            best_symbol: best.symbol.clone(),
            worst_xirr: worst.xirr_pct,
            worst_symbol: worst.symbol.clone(),
            positive_xirr_count: valid.iter().filter(|r| r.xirr_pct > 0.0).count(),
            above_10_count: valid.iter().filter(|r| r.xirr_pct > 10.0).count(),
            above_15_count: valid.iter().filter(|r| r.xirr_pct > 15.0).count(),
            above_20_count: valid.iter().filter(|r| r.xirr_pct > 20.0).count(),
        }
    };

    let output = DashboardOutput {
        config: ConfigSummary {
            initial_capital: config.initial_total_capital,
            initial_deployment: config.initial_deployment,
            lookback_days: config.lookback,
            cooldown_days: config.cooldown_days,
            min_trade_size: config.min_trade_size,
            max_trade_size: config.max_trade_size,
            profit_sweep_target_pct: config.profit_sweep_target * 100.0,
            profit_sweep_fraction_pct: config.profit_sweep_fraction * 100.0,
            liquid_bees_rate_pct: config.liquid_bees_rate * 100.0,
            transaction_cost_pct: config.transaction_cost_rate * 100.0,
        },
        composite: composite_result,
        stocks: results,
        summary,
        execution_time_ms,
    };

    // Write JSON to dashboard directory
    let dashboard_dir = Path::new("../dashboard");
    fs::create_dir_all(dashboard_dir).ok();
    let json_path = dashboard_dir.join("results.json");

    match serde_json::to_string_pretty(&output) {
        Ok(json) => {
            if let Err(e) = fs::write(&json_path, &json) {
                eprintln!("  ERROR writing JSON: {}", e);
            } else {
                println!("  ✓ JSON exported to: {}", json_path.display());
                println!("  ✓ {} stock results serialized", output.stocks.len());
            }
        }
        Err(e) => {
            eprintln!("  ERROR serializing JSON: {}", e);
        }
    }

    // Total execution time
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Total execution time: {:.2}s", total_start.elapsed().as_secs_f64());
    println!("  Dashboard: cd ../dashboard && python3 -m http.server 8080");
    println!("═══════════════════════════════════════════════════════════");
}
