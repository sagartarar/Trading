use crate::data::{self, DailyBar};
use chrono::NaiveDate;
use std::collections::BTreeMap;
use std::path::Path;

/// Current Nifty 50 constituent symbols mapped to Zerodha file naming.
const NIFTY_50_SYMBOLS: &[&str] = &[
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
    "HINDUNILVR",
    "ITC",
    "SBIN",
    "BHARTIARTL",
    "KOTAKBANK",
    "LT",
    "AXISBANK",
    "ASIANPAINT",
    "MARUTI",
    "SUNPHARMA",
    "TITAN",
    "BAJFINANCE",
    "HCLTECH",
    "ADANIENT",
    "ADANIPORTS",
    "NTPC",
    "POWERGRID",
    "M&M",
    "TATASTEEL",
    "JSWSTEEL",
    "TECHM",
    "INDUSINDBK",
    "HINDALCO",
    "CIPLA",
    "GRASIM",
    "DRREDDY",
    "EICHERMOT",
    "DIVISLAB",
    "BAJAJFINSV",
    "COALINDIA",
    "BPCL",
    "BRITANNIA",
    "HEROMOTOCO",
    "NESTLEIND",
    "ONGC",
    "BAJAJ-AUTO",
    "APOLLOHOSP",
    "SHRIRAMFIN",
    "TATACONSUM",
    "BEL",
    "ETERNAL",
    "TATAPOWER",
    "BHEL",
    "IOC",
    "GAIL",
];

/// Build a synthetic equal-weight Nifty 50 composite index.
///
/// Method:
/// 1. Load daily data for all available Nifty 50 constituents
/// 2. Find the common date range (intersection of all dates)
/// 3. Normalize each stock's close to Day 1 = 100
/// 4. Equal-weight average per day
/// 5. Return as a Vec<DailyBar> with the composite "price"
pub fn build_nifty50_composite(data_dir: &Path) -> Result<(Vec<DailyBar>, Vec<String>, Vec<String>), String> {
    let mut loaded_stocks: Vec<(String, BTreeMap<NaiveDate, f64>)> = Vec::new();
    let mut loaded_symbols: Vec<String> = Vec::new();
    let mut skipped_symbols: Vec<String> = Vec::new();

    for &symbol in NIFTY_50_SYMBOLS {
        match data::load_stock(data_dir, symbol) {
            Ok(bars) => {
                if bars.len() < 252 {
                    skipped_symbols.push(format!("{} (only {} days)", symbol, bars.len()));
                    continue;
                }
                let mut date_map = BTreeMap::new();
                for bar in &bars {
                    date_map.insert(bar.date, bar.close);
                }
                loaded_symbols.push(symbol.to_string());
                loaded_stocks.push((symbol.to_string(), date_map));
            }
            Err(_) => {
                skipped_symbols.push(format!("{} (file not found)", symbol));
            }
        }
    }

    if loaded_stocks.is_empty() {
        return Err("No Nifty 50 constituent data could be loaded".to_string());
    }

    // Collect ALL dates from ALL stocks (union, not intersection)
    let mut all_dates: BTreeMap<NaiveDate, usize> = BTreeMap::new();
    for (_, date_map) in &loaded_stocks {
        for &date in date_map.keys() {
            *all_dates.entry(date).or_insert(0) += 1;
        }
    }

    // Use any date where at least 30 stocks have data (majority threshold)
    let min_stocks_required = 30.min(loaded_stocks.len());
    let sorted_dates: Vec<NaiveDate> = all_dates
        .iter()
        .filter(|(_, &count)| count >= min_stocks_required)
        .map(|(&date, _)| date)
        .collect();

    if sorted_dates.is_empty() {
        return Err("No trading dates found with sufficient stock coverage".to_string());
    }

    let first_date = sorted_dates[0];

    // Normalize each stock: first available day's close = 100
    // Use the first date in sorted_dates where each stock has data
    let base_prices: Vec<f64> = loaded_stocks
        .iter()
        .map(|(_, dm)| {
            for &d in &sorted_dates {
                if let Some(&p) = dm.get(&d) {
                    return p;
                }
            }
            1.0 // fallback, should not happen
        })
        .collect();

    // Build composite using available stocks per day
    let mut composite_bars: Vec<DailyBar> = Vec::with_capacity(sorted_dates.len());

    for &date in &sorted_dates {
        let mut sum_normalized = 0.0;
        let mut count = 0;

        for (i, (_, date_map)) in loaded_stocks.iter().enumerate() {
            if let Some(&close) = date_map.get(&date) {
                let normalized = (close / base_prices[i]) * 100.0;
                sum_normalized += normalized;
                count += 1;
            }
        }

        if count > 0 {
            let avg_price = sum_normalized / count as f64;
            composite_bars.push(DailyBar {
                date,
                open: avg_price,
                high: avg_price,
                low: avg_price,
                close: avg_price,
                volume: 0,
            });
        }
    }

    Ok((composite_bars, loaded_symbols, skipped_symbols))
}
