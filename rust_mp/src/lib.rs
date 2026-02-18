/// Market Profile Engine (Rust + PyO3)
/// =====================================
/// High-performance TPO map construction, Value Area (POC/VAH/VAL),
/// Initial Balance, Single Prints, Day-Type and Open-Type classification.
///
/// Called from Python via `import rust_mp`.

use pyo3::prelude::*;
use std::collections::BTreeMap;

// ─── Constants ────────────────────────────────────────────────────────────
const TPO_LETTERS: &[u8] = b"ABCDEFGHIJKLMN";
const VALUE_AREA_PCT: f64 = 0.70;

// ─── DailyProfile result returned to Python ──────────────────────────────
#[pyclass]
#[derive(Clone, Debug)]
pub struct DailyProfile {
    #[pyo3(get)]
    pub date: String,
    #[pyo3(get)]
    pub open_price: f64,
    #[pyo3(get)]
    pub high: f64,
    #[pyo3(get)]
    pub low: f64,
    #[pyo3(get)]
    pub close: f64,
    #[pyo3(get)]
    pub total_volume: i64,
    #[pyo3(get)]
    pub poc: f64,
    #[pyo3(get)]
    pub poc_volume: i64,
    #[pyo3(get)]
    pub vah: f64,
    #[pyo3(get)]
    pub val: f64,
    #[pyo3(get)]
    pub ib_high: f64,
    #[pyo3(get)]
    pub ib_low: f64,
    #[pyo3(get)]
    pub tpo_count_max: usize,
    #[pyo3(get)]
    pub single_print_count: usize,
    #[pyo3(get)]
    pub day_type: String,
    #[pyo3(get)]
    pub open_type: String,
    #[pyo3(get)]
    pub range_ext_up: bool,
    #[pyo3(get)]
    pub range_ext_down: bool,
}

// ─── Internal bar representation ─────────────────────────────────────────
struct Bar {
    date_str: String,  // "YYYY-MM-DD"
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: i64,
}

// ─── Internal working profile ────────────────────────────────────────────
struct WorkingProfile {
    date_str: String,
    open_price: f64,
    high: f64,
    low: f64,
    close: f64,
    total_volume: i64,
    // price_level → list of TPO letter indices
    tpo_map: BTreeMap<i64, Vec<u8>>,
    volume_map: BTreeMap<i64, i64>,
    tick_size: f64,
    // derived
    poc: f64,
    poc_volume: i64,
    vah: f64,
    val: f64,
    ib_high: f64,
    ib_low: f64,
    tpo_count_max: usize,
    single_print_count: usize,
    day_type: String,
    open_type: String,
    range_ext_up: bool,
    range_ext_down: bool,
}

#[inline]
fn discretise(price: f64, tick: f64) -> i64 {
    (price / tick).floor() as i64
}

#[inline]
fn level_to_price(level: i64, tick: f64) -> f64 {
    (level as f64) * tick
}

impl WorkingProfile {
    fn new(date_str: String, tick_size: f64) -> Self {
        Self {
            date_str,
            open_price: 0.0,
            high: f64::MIN,
            low: f64::MAX,
            close: 0.0,
            total_volume: 0,
            tpo_map: BTreeMap::new(),
            volume_map: BTreeMap::new(),
            tick_size,
            poc: 0.0,
            poc_volume: 0,
            vah: 0.0,
            val: 0.0,
            ib_high: 0.0,
            ib_low: 0.0,
            tpo_count_max: 0,
            single_print_count: 0,
            day_type: String::new(),
            open_type: String::new(),
            range_ext_up: false,
            range_ext_down: false,
        }
    }

    fn add_bar(&mut self, bar_idx: usize, bar: &Bar) {
        let letter = if bar_idx < TPO_LETTERS.len() {
            TPO_LETTERS[bar_idx]
        } else {
            TPO_LETTERS[TPO_LETTERS.len() - 1]
        };

        if bar_idx == 0 {
            self.open_price = bar.open;
        }
        self.close = bar.close;
        if bar.high > self.high {
            self.high = bar.high;
        }
        if bar.low < self.low {
            self.low = bar.low;
        }
        self.total_volume += bar.volume;

        let lo = discretise(bar.low, self.tick_size);
        let hi = discretise(bar.high, self.tick_size);

        for level in lo..=hi {
            self.tpo_map.entry(level).or_default().push(letter);
            *self.volume_map.entry(level).or_insert(0) += bar.volume;
        }
    }

    fn compute_poc(&mut self) {
        let mut best_level: i64 = 0;
        let mut best_count: usize = 0;
        let mut best_vol: i64 = 0;

        for (&level, letters) in &self.tpo_map {
            let count = letters.len();
            let vol = *self.volume_map.get(&level).unwrap_or(&0);
            if count > best_count || (count == best_count && vol > best_vol) {
                best_level = level;
                best_count = count;
                best_vol = vol;
            }
        }

        self.poc = level_to_price(best_level, self.tick_size);
        self.poc_volume = best_vol;
    }

    fn compute_value_area(&mut self) {
        if self.tpo_map.is_empty() {
            self.vah = self.high;
            self.val = self.low;
            return;
        }

        let total_tpos: usize = self.tpo_map.values().map(|v| v.len()).sum();
        let target = ((total_tpos as f64) * VALUE_AREA_PCT).ceil() as usize;

        let levels: Vec<i64> = self.tpo_map.keys().copied().collect();
        let poc_discrete = discretise(self.poc, self.tick_size);

        let poc_idx = match levels.iter().position(|&l| l == poc_discrete) {
            Some(i) => i,
            None => {
                self.vah = self.high;
                self.val = self.low;
                return;
            }
        };

        let mut accumulated = self.tpo_map[&levels[poc_idx]].len();
        let mut lo_idx = poc_idx;
        let mut hi_idx = poc_idx;

        while accumulated < target {
            let above_count = if hi_idx + 1 < levels.len() {
                self.tpo_map[&levels[hi_idx + 1]].len()
            } else {
                0
            };
            let below_count = if lo_idx > 0 {
                self.tpo_map[&levels[lo_idx - 1]].len()
            } else {
                0
            };

            if above_count == 0 && below_count == 0 {
                break;
            }

            if above_count >= below_count {
                hi_idx += 1;
                accumulated += above_count;
            } else {
                lo_idx -= 1;
                accumulated += below_count;
            }
        }

        self.val = level_to_price(levels[lo_idx], self.tick_size);
        self.vah = level_to_price(levels[hi_idx], self.tick_size);
    }

    fn compute_initial_balance(&mut self) {
        let mut ib_levels: Vec<i64> = Vec::new();

        for (&level, letters) in &self.tpo_map {
            if letters.contains(&b'A') || letters.contains(&b'B') {
                ib_levels.push(level);
            }
        }

        if ib_levels.is_empty() {
            self.ib_high = self.high;
            self.ib_low = self.low;
        } else {
            self.ib_low = level_to_price(*ib_levels.iter().min().unwrap(), self.tick_size);
            self.ib_high = level_to_price(*ib_levels.iter().max().unwrap(), self.tick_size);
        }
    }

    fn detect_single_prints(&mut self) {
        let levels: Vec<i64> = self.tpo_map.keys().copied().collect();
        if levels.len() < 3 {
            self.single_print_count = 0;
            return;
        }

        let mut singles: Vec<i64> = Vec::new();
        // Skip top and bottom levels (tails)
        for &level in &levels[1..levels.len() - 1] {
            if self.tpo_map[&level].len() == 1 {
                singles.push(level);
            }
        }

        if singles.is_empty() {
            self.single_print_count = 0;
            return;
        }

        // Group consecutive levels into ranges
        let mut count = 0;
        let mut start = singles[0];
        let mut prev = singles[0];

        for &s in &singles[1..] {
            if s - prev <= 1 {
                // consecutive
                prev = s;
            } else {
                if prev > start {
                    count += 1;
                }
                start = s;
                prev = s;
            }
        }
        if prev > start {
            count += 1;
        }

        self.single_print_count = count;
    }

    fn compute_tpo_width(&mut self) {
        self.tpo_count_max = self.tpo_map.values().map(|v| v.len()).max().unwrap_or(0);
    }

    fn detect_range_extensions(&mut self) {
        self.range_ext_up = self.high > self.ib_high;
        self.range_ext_down = self.low < self.ib_low;
    }

    fn classify_day_type(&mut self) {
        let ib_range = self.ib_high - self.ib_low;
        let day_range = self.high - self.low;

        if day_range <= 0.0 {
            self.day_type = "Non-Trend".to_string();
            return;
        }

        let ib_ratio = ib_range / day_range;
        let has_single_prints = self.single_print_count > 0;
        let both_ext = self.range_ext_up && self.range_ext_down;

        if ib_ratio < 0.35 && has_single_prints {
            self.day_type = "Trend".to_string();
            return;
        }

        if both_ext && ib_ratio < 0.5 {
            self.day_type = "Double Distribution".to_string();
            return;
        }

        let mid = (self.high + self.low) / 2.0;
        let close_near_mid = (self.close - mid).abs() < 0.25 * day_range;
        if both_ext && close_near_mid {
            self.day_type = "Neutral".to_string();
            return;
        }

        if (self.range_ext_up || self.range_ext_down) && !both_ext {
            self.day_type = "Normal Variation".to_string();
            return;
        }

        if self.tpo_count_max >= 8 && ib_ratio > 0.85 {
            self.day_type = "Non-Trend".to_string();
            return;
        }

        self.day_type = "Normal".to_string();
    }

    fn classify_open_type(&mut self, prev: Option<&WorkingProfile>) {
        let prev = match prev {
            Some(p) => p,
            None => {
                self.open_type = "Open-Auction".to_string();
                return;
            }
        };

        let ib_range = self.ib_high - self.ib_low;
        if ib_range <= 0.0 {
            self.open_type = "Open-Auction".to_string();
            return;
        }

        let prev_range = prev.high - prev.low;
        if prev_range <= 0.0 {
            self.open_type = "Open-Auction".to_string();
            return;
        }

        let open_price = self.open_price;
        let ib_mid = (self.ib_high + self.ib_low) / 2.0;

        let open_near_ib_high = (self.ib_high - open_price) < 0.1 * ib_range;
        let open_near_ib_low = (open_price - self.ib_low) < 0.1 * ib_range;
        let open_outside_prev_va = open_price > prev.vah || open_price < prev.val;

        if open_near_ib_low && self.close > ib_mid {
            if open_outside_prev_va {
                self.open_type = "Open-Test-Drive".to_string();
            } else {
                self.open_type = "Open-Drive".to_string();
            }
            return;
        }
        if open_near_ib_high && self.close < ib_mid {
            if open_outside_prev_va {
                self.open_type = "Open-Test-Drive".to_string();
            } else {
                self.open_type = "Open-Drive".to_string();
            }
            return;
        }

        if open_outside_prev_va {
            if open_price > prev.vah && self.close < prev.vah {
                self.open_type = "Open-Rejection-Reverse".to_string();
                return;
            }
            if open_price < prev.val && self.close > prev.val {
                self.open_type = "Open-Rejection-Reverse".to_string();
                return;
            }
        }

        self.open_type = "Open-Auction".to_string();
    }

    fn finalize(&mut self, prev: Option<&WorkingProfile>) {
        self.compute_poc();
        self.compute_value_area();
        self.compute_initial_balance();
        self.detect_single_prints();
        self.compute_tpo_width();
        self.detect_range_extensions();
        self.classify_day_type();
        self.classify_open_type(prev);
    }

    fn to_daily_profile(&self) -> DailyProfile {
        DailyProfile {
            date: self.date_str.clone(),
            open_price: self.open_price,
            high: self.high,
            low: self.low,
            close: self.close,
            total_volume: self.total_volume,
            poc: self.poc,
            poc_volume: self.poc_volume,
            vah: self.vah,
            val: self.val,
            ib_high: self.ib_high,
            ib_low: self.ib_low,
            tpo_count_max: self.tpo_count_max,
            single_print_count: self.single_print_count,
            day_type: self.day_type.clone(),
            open_type: self.open_type.clone(),
            range_ext_up: self.range_ext_up,
            range_ext_down: self.range_ext_down,
        }
    }
}

// ─── Python-exposed functions ────────────────────────────────────────────

/// Build daily Market Profiles from arrays of 30-min OHLCV data.
///
/// Parameters:
///     dates: list of date strings "YYYY-MM-DD HH:MM:SS" (one per bar)
///     opens, highs, lows, closes: list of f64
///     volumes: list of i64
///     tick_size: price bucket width (default 1.0)
///
/// Returns: list of DailyProfile objects
#[pyfunction]
#[pyo3(signature = (dates, opens, highs, lows, closes, volumes, tick_size=1.0))]
fn build_profiles(
    dates: Vec<String>,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<i64>,
    tick_size: f64,
) -> PyResult<Vec<DailyProfile>> {
    let n = dates.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Parse bars and group by date
    let mut day_bars: BTreeMap<String, Vec<Bar>> = BTreeMap::new();

    for i in 0..n {
        // Extract date portion "YYYY-MM-DD" from datetime string
        let date_str = if dates[i].len() >= 10 {
            dates[i][..10].to_string()
        } else {
            dates[i].clone()
        };

        let bar = Bar {
            date_str: date_str.clone(),
            open: opens[i],
            high: highs[i],
            low: lows[i],
            close: closes[i],
            volume: volumes[i],
        };

        day_bars.entry(date_str).or_default().push(bar);
    }

    // Build profiles
    let mut profiles: Vec<WorkingProfile> = Vec::new();

    for (date_str, bars) in &day_bars {
        let mut wp = WorkingProfile::new(date_str.clone(), tick_size);

        for (idx, bar) in bars.iter().enumerate() {
            wp.add_bar(idx, bar);
        }

        if wp.tpo_map.is_empty() {
            continue;
        }

        profiles.push(wp);
    }

    // Finalize: compute derived metrics (need previous profile for open type)
    for i in 0..profiles.len() {
        // Split the slice to get mutable current and immutable previous
        let (prev_slice, current_slice) = profiles.split_at_mut(i);
        let prev = prev_slice.last();
        current_slice[0].finalize(prev);
    }

    // Convert to Python-friendly DailyProfile
    let results: Vec<DailyProfile> = profiles.iter().map(|wp| wp.to_daily_profile()).collect();

    Ok(results)
}

/// Build profiles for multiple symbols in parallel.
///
/// symbols: list of symbol names
/// all_dates, all_opens, ... : flattened arrays with a parallel symbol_indices array
/// symbol_indices: for each bar, the index into the symbols array
#[pyfunction]
#[pyo3(signature = (symbols, symbol_indices, dates, opens, highs, lows, closes, volumes, tick_size=1.0))]
fn build_profiles_batch(
    symbols: Vec<String>,
    symbol_indices: Vec<usize>,
    dates: Vec<String>,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<i64>,
    tick_size: f64,
) -> PyResult<Vec<(String, Vec<DailyProfile>)>> {
    let n = dates.len();

    // Group bars by symbol
    let mut symbol_bars: Vec<Vec<usize>> = vec![vec![]; symbols.len()];
    for i in 0..n {
        let sym_idx = symbol_indices[i];
        if sym_idx < symbols.len() {
            symbol_bars[sym_idx].push(i);
        }
    }

    let mut results: Vec<(String, Vec<DailyProfile>)> = Vec::new();

    for (sym_idx, bar_indices) in symbol_bars.iter().enumerate() {
        if bar_indices.is_empty() {
            continue;
        }

        // Group by date
        let mut day_bars: BTreeMap<String, Vec<Bar>> = BTreeMap::new();
        for &i in bar_indices {
            let date_str = if dates[i].len() >= 10 {
                dates[i][..10].to_string()
            } else {
                dates[i].clone()
            };
            let bar = Bar {
                date_str: date_str.clone(),
                open: opens[i],
                high: highs[i],
                low: lows[i],
                close: closes[i],
                volume: volumes[i],
            };
            day_bars.entry(date_str).or_default().push(bar);
        }

        let mut profiles: Vec<WorkingProfile> = Vec::new();
        for (date_str, bars) in &day_bars {
            let mut wp = WorkingProfile::new(date_str.clone(), tick_size);
            for (idx, bar) in bars.iter().enumerate() {
                wp.add_bar(idx, bar);
            }
            if !wp.tpo_map.is_empty() {
                profiles.push(wp);
            }
        }

        for i in 0..profiles.len() {
            let (prev_slice, current_slice) = profiles.split_at_mut(i);
            let prev = prev_slice.last();
            current_slice[0].finalize(prev);
        }

        let sym_profiles: Vec<DailyProfile> =
            profiles.iter().map(|wp| wp.to_daily_profile()).collect();
        results.push((symbols[sym_idx].clone(), sym_profiles));
    }

    Ok(results)
}

// ─── Python module ───────────────────────────────────────────────────────
#[pymodule]
fn rust_mp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_profiles, m)?)?;
    m.add_function(wrap_pyfunction!(build_profiles_batch, m)?)?;
    m.add_class::<DailyProfile>()?;
    Ok(())
}
