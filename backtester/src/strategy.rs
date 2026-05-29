use crate::data::DailyBar;
use crate::xirr;
use chrono::{Datelike, NaiveDate};

/// Strategy configuration parameters.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    pub initial_total_capital: f64,
    pub initial_deployment: f64,
    pub cooldown_days: u32,
    pub transaction_cost_rate: f64,
    pub liquid_bees_rate: f64,       // Annual rate (e.g., 0.06 for 6%)
    pub lookback: usize,             // Rolling high window in trading days
    pub min_trade_size: f64,
    pub max_trade_size: f64,
    pub profit_sweep_target: f64,    // e.g., 0.15 for 15% above avg cost
    pub profit_sweep_fraction: f64,  // e.g., 0.10 to sell 10% of units
    pub savings_pre_2021: f64,
    pub savings_post_2021: f64,
    pub savings_cycle_days: u32,     // Trading days between savings injections
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            initial_total_capital: 2_000_000.0,
            initial_deployment: 100_000.0,
            cooldown_days: 7,
            transaction_cost_rate: 0.0015,
            liquid_bees_rate: 0.06,
            lookback: 60,
            min_trade_size: 20_000.0,
            max_trade_size: 100_000.0,
            profit_sweep_target: 0.15,
            profit_sweep_fraction: 0.10,
            savings_pre_2021: 50_000.0,
            savings_post_2021: 150_000.0,
            savings_cycle_days: 21,
        }
    }
}

/// Results from running the strategy on a single instrument.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StrategyResult {
    pub symbol: String,
    pub xirr_pct: f64,
    pub final_portfolio: f64,
    pub equity_value: f64,
    pub liquid_cash: f64,
    pub total_dip_buys: u32,
    pub avg_trade_size: f64,
    pub profit_sweeps: u32,
    pub data_start: NaiveDate,
    pub data_end: NaiveDate,
    pub total_days: usize,
}

/// Run the full dip-buying + profit-sweep strategy on a price series.
pub fn run_strategy(symbol: &str, bars: &[DailyBar], config: &StrategyConfig) -> Option<StrategyResult> {
    if bars.len() <= config.lookback {
        return None;
    }

    // Pre-compute rolling highs
    let rolling_highs = compute_rolling_highs(bars, config.lookback);

    // Initialize state
    let mut liquid_cash = config.initial_total_capital - config.initial_deployment;
    let mut units_owned: f64 = 0.0;
    let mut total_equity_spent: f64 = 0.0;
    let mut cooldown_counter: u32 = 0;
    let mut total_dip_buys: u32 = 0;
    let mut total_capital_deployed: f64 = 0.0;
    let mut profit_sweeps: u32 = 0;

    let start_idx = config.lookback;
    let initial_price = bars[start_idx].close;
    let start_date = bars[start_idx].date;

    // Seed initial deployment
    let cost = config.initial_deployment * config.transaction_cost_rate;
    let net_initial = config.initial_deployment - cost;
    units_owned += net_initial / initial_price;
    total_equity_spent += net_initial;

    let mut cashflows: Vec<(NaiveDate, f64)> = vec![(start_date, -config.initial_total_capital)];
    let mut days_since_savings: u32 = 0;

    // Main simulation loop
    for i in (start_idx + 1)..bars.len() {
        let current_date = bars[i].date;
        let price = bars[i].close;
        let rolling_high = rolling_highs[i];

        // Determine monthly savings rate
        let current_savings = if current_date.year() >= 2021 {
            config.savings_post_2021
        } else {
            config.savings_pre_2021
        };

        // Compound daily Liquid BeES interest
        liquid_cash *= 1.0 + config.liquid_bees_rate / 252.0;

        // Monthly cash injection
        days_since_savings += 1;
        if days_since_savings >= config.savings_cycle_days {
            liquid_cash += current_savings;
            cashflows.push((current_date, -current_savings));
            days_since_savings = 0;
        }

        // Calculate drop from rolling high
        let drop = if rolling_high > 0.0 {
            (rolling_high - price) / rolling_high
        } else {
            0.0
        };

        // Calculate average buy price (cost basis)
        let avg_buy_price = if units_owned > 0.0 {
            total_equity_spent / units_owned
        } else {
            0.0
        };

        if cooldown_counter > 0 {
            cooldown_counter -= 1;
        } else {
            // PROFIT SWEEP: Price is 15% above average buy price
            if avg_buy_price > 0.0
                && price >= avg_buy_price * (1.0 + config.profit_sweep_target)
                && units_owned > 0.0
            {
                let units_to_sell = units_owned * config.profit_sweep_fraction;
                let gross_revenue = units_to_sell * price;
                let sell_cost = gross_revenue * config.transaction_cost_rate;

                liquid_cash += gross_revenue - sell_cost;
                units_owned -= units_to_sell;

                // Reduce cost basis proportionally
                total_equity_spent -= units_to_sell * avg_buy_price;

                profit_sweeps += 1;
                cooldown_counter = config.cooldown_days;
            }
            // DIP BUY: Continuous linear scaling
            else if drop >= 0.01 {
                let calculated_size =
                    config.min_trade_size + (drop - 0.01) * 2_000_000.0;
                let trade_size = calculated_size.min(config.max_trade_size);

                if liquid_cash >= trade_size {
                    let trade_cost = trade_size * config.transaction_cost_rate;
                    let net_trade = trade_size - trade_cost;

                    units_owned += net_trade / price;
                    total_equity_spent += net_trade;
                    liquid_cash -= trade_size;

                    cooldown_counter = config.cooldown_days;
                    total_dip_buys += 1;
                    total_capital_deployed += trade_size;
                }
            }
        }
    }

    // Final evaluation
    let final_price = bars.last()?.close;
    let final_date = bars.last()?.date;
    let equity_value = units_owned * final_price;
    let final_portfolio = equity_value + liquid_cash;

    cashflows.push((final_date, final_portfolio));

    let xirr_pct = xirr::calculate_xirr(&cashflows).unwrap_or(f64::NAN);

    let avg_trade_size = if total_dip_buys > 0 {
        total_capital_deployed / total_dip_buys as f64
    } else {
        0.0
    };

    Some(StrategyResult {
        symbol: symbol.to_string(),
        xirr_pct,
        final_portfolio,
        equity_value,
        liquid_cash,
        total_dip_buys,
        avg_trade_size,
        profit_sweeps,
        data_start: start_date,
        data_end: final_date,
        total_days: bars.len(),
    })
}

/// Compute rolling maximum of close prices over a given window.
/// Returns a Vec of the same length as bars, where rolling_highs[i]
/// is the max close in bars[max(0, i-window+1)..=i].
fn compute_rolling_highs(bars: &[DailyBar], window: usize) -> Vec<f64> {
    let n = bars.len();
    let mut highs = vec![0.0_f64; n];

    for i in 0..n {
        let start = if i >= window { i - window + 1 } else { 0 };
        let mut max_close = f64::NEG_INFINITY;
        for j in start..=i {
            if bars[j].close > max_close {
                max_close = bars[j].close;
            }
        }
        highs[i] = max_close;
    }

    highs
}
