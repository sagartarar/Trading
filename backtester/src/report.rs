use crate::strategy::StrategyResult;

/// Print a detailed report for a single strategy result (used for Nifty 50 composite).
pub fn print_detailed_report(title: &str, result: &StrategyResult) {
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  {}  ║", center_text(title, 54));
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  True Strategy XIRR       : {:>8.2}%                   ║", result.xirr_pct);
    println!("║  Final Portfolio Value     : ₹ {:>14}            ║", format_inr(result.final_portfolio));
    println!("║    └─ Equity Value         : ₹ {:>14}            ║", format_inr(result.equity_value));
    println!("║    └─ Liquid BeES Cash     : ₹ {:>14}            ║", format_inr(result.liquid_cash));
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Total Dip Buys            : {:>6}                     ║", result.total_dip_buys);
    println!("║  Average Trade Size        : ₹ {:>14}            ║", format_inr(result.avg_trade_size));
    println!("║  Profit Sweeps Executed    : {:>6}                     ║", result.profit_sweeps);
    println!("║  Data Range                : {} → {}       ║", result.data_start, result.data_end);
    println!("║  Trading Days              : {:>6}                     ║", result.total_days);
    println!("╚══════════════════════════════════════════════════════════╝");
}

/// Print the full stock leaderboard sorted by XIRR.
pub fn print_leaderboard(results: &[StrategyResult]) {
    println!();
    println!("┌──────┬──────────────┬──────────┬──────────────────┬──────────┬────────┬─────────────────────────┐");
    println!("│ Rank │ Symbol       │  XIRR %  │  Final Portfolio │ Dip Buys │ Sweeps │ Data Range              │");
    println!("├──────┼──────────────┼──────────┼──────────────────┼──────────┼────────┼─────────────────────────┤");

    for (i, r) in results.iter().enumerate() {
        let rank = i + 1;
        let xirr_str = if r.xirr_pct.is_nan() {
            "  N/A   ".to_string()
        } else {
            format!("{:>7.2}%", r.xirr_pct)
        };

        println!(
            "│ {:>4} │ {:<12} │ {} │ ₹{:>14} │ {:>8} │ {:>6} │ {} → {} │",
            rank,
            r.symbol,
            xirr_str,
            format_inr(r.final_portfolio),
            r.total_dip_buys,
            r.profit_sweeps,
            r.data_start,
            r.data_end,
        );
    }

    println!("└──────┴──────────────┴──────────┴──────────────────┴──────────┴────────┴─────────────────────────┘");
}

/// Print summary statistics for the leaderboard.
pub fn print_summary(results: &[StrategyResult]) {
    let valid: Vec<&StrategyResult> = results.iter().filter(|r| !r.xirr_pct.is_nan()).collect();

    if valid.is_empty() {
        println!("\n  No valid results to summarize.");
        return;
    }

    let avg_xirr: f64 = valid.iter().map(|r| r.xirr_pct).sum::<f64>() / valid.len() as f64;
    let max_xirr = valid.iter().map(|r| r.xirr_pct).fold(f64::NEG_INFINITY, f64::max);
    let min_xirr = valid.iter().map(|r| r.xirr_pct).fold(f64::INFINITY, f64::min);

    let best = valid.iter().max_by(|a, b| a.xirr_pct.partial_cmp(&b.xirr_pct).unwrap()).unwrap();
    let worst = valid.iter().min_by(|a, b| a.xirr_pct.partial_cmp(&b.xirr_pct).unwrap()).unwrap();

    let positive = valid.iter().filter(|r| r.xirr_pct > 0.0).count();
    let above_10 = valid.iter().filter(|r| r.xirr_pct > 10.0).count();
    let above_15 = valid.iter().filter(|r| r.xirr_pct > 15.0).count();
    let above_20 = valid.iter().filter(|r| r.xirr_pct > 20.0).count();

    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║              LEADERBOARD SUMMARY STATISTICS             ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Stocks Analyzed           : {:>6}                     ║", results.len());
    println!("║  Valid XIRR Results        : {:>6}                     ║", valid.len());
    println!("║  Average XIRR             : {:>8.2}%                   ║", avg_xirr);
    println!("║  Best  XIRR               : {:>8.2}% ({:<12})    ║", max_xirr, best.symbol);
    println!("║  Worst XIRR               : {:>8.2}% ({:<12})    ║", min_xirr, worst.symbol);
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Positive XIRR (>0%)       : {:>6}                     ║", positive);
    println!("║  Above 10% XIRR           : {:>6}                     ║", above_10);
    println!("║  Above 15% XIRR           : {:>6}                     ║", above_15);
    println!("║  Above 20% XIRR           : {:>6}                     ║", above_20);
    println!("╚══════════════════════════════════════════════════════════╝");
}

/// Format a number as Indian Rupee notation (with commas).
fn format_inr(value: f64) -> String {
    let is_negative = value < 0.0;
    let abs_val = value.abs();
    let whole = abs_val as u64;
    let frac = ((abs_val - whole as f64) * 100.0).round() as u64;

    let whole_str = format!("{}", whole);
    let chars: Vec<char> = whole_str.chars().collect();
    let len = chars.len();

    let mut formatted = String::new();

    if len <= 3 {
        formatted = whole_str;
    } else {
        // Last 3 digits
        let last_three: String = chars[len - 3..].iter().collect();
        let remaining: Vec<char> = chars[..len - 3].to_vec();

        // Group remaining in pairs from right
        let mut groups: Vec<String> = Vec::new();
        let mut i = remaining.len();
        while i > 0 {
            let start = if i >= 2 { i - 2 } else { 0 };
            let group: String = remaining[start..i].iter().collect();
            groups.push(group);
            i = start;
        }
        groups.reverse();

        formatted = groups.join(",");
        formatted.push(',');
        formatted.push_str(&last_three);
    }

    let result = format!("{}.{:02}", formatted, frac);
    if is_negative {
        format!("-{}", result)
    } else {
        result
    }
}

/// Center text within a given width.
fn center_text(text: &str, width: usize) -> String {
    let text_len = text.len();
    if text_len >= width {
        return text[..width].to_string();
    }
    let left_pad = (width - text_len) / 2;
    let right_pad = width - text_len - left_pad;
    format!("{}{}{}", " ".repeat(left_pad), text, " ".repeat(right_pad))
}
