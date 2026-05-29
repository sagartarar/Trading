use chrono::NaiveDate;
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// A single daily OHLCV bar from Zerodha CSV data.
#[derive(Debug, Clone)]
pub struct DailyBar {
    pub date: NaiveDate,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
}

/// Raw CSV row — intermediate deserialization target.
#[derive(Debug, Deserialize)]
struct CsvRow {
    date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: u64,
}

/// Parse a Zerodha date string like "2015-02-02 00:00:00+05:30" into NaiveDate.
fn parse_zerodha_date(s: &str) -> Option<NaiveDate> {
    // Take only the date portion "2015-02-02"
    let date_part = s.split_whitespace().next()?;
    NaiveDate::parse_from_str(date_part, "%Y-%m-%d").ok()
}

/// Load a single stock's daily data from a CSV file.
/// Returns bars sorted by date ascending.
pub fn load_stock(data_dir: &Path, symbol: &str) -> Result<Vec<DailyBar>, String> {
    let filename = format!("{}_daily.csv", symbol);
    let filepath = data_dir.join(&filename);

    if !filepath.exists() {
        return Err(format!("File not found: {}", filepath.display()));
    }

    let mut reader = csv::Reader::from_path(&filepath)
        .map_err(|e| format!("Failed to open {}: {}", filename, e))?;

    let mut bars: Vec<DailyBar> = Vec::new();

    for result in reader.deserialize() {
        let row: CsvRow = result.map_err(|e| format!("Parse error in {}: {}", filename, e))?;

        let date = parse_zerodha_date(&row.date)
            .ok_or_else(|| format!("Invalid date '{}' in {}", row.date, filename))?;

        bars.push(DailyBar {
            date,
            open: row.open,
            high: row.high,
            low: row.low,
            close: row.close,
            volume: row.volume,
        });
    }

    // Ensure chronological order
    bars.sort_by_key(|b| b.date);

    Ok(bars)
}

/// List all available stock symbols in the data directory.
/// Looks for files matching the pattern `{SYMBOL}_daily.csv`.
pub fn list_available_symbols(data_dir: &Path) -> Vec<String> {
    let mut symbols = Vec::new();

    if let Ok(entries) = fs::read_dir(data_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(symbol) = name.strip_suffix("_daily.csv") {
                symbols.push(symbol.to_string());
            }
        }
    }

    symbols.sort();
    symbols
}
