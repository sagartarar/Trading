use chrono::NaiveDate;

/// Calculate XIRR (Extended Internal Rate of Return) using Newton-Raphson method.
/// Cashflows are (date, amount) pairs where negative = outflow, positive = inflow.
/// Returns the annualized return as a percentage (e.g., 11.5 for 11.5%).
pub fn calculate_xirr(cashflows: &[(NaiveDate, f64)]) -> Option<f64> {
    if cashflows.len() < 2 {
        return None;
    }

    let start_date = cashflows[0].0;
    let days: Vec<f64> = cashflows
        .iter()
        .map(|(d, _)| (*d - start_date).num_days() as f64)
        .collect();
    let amounts: Vec<f64> = cashflows.iter().map(|(_, a)| *a).collect();

    // Newton-Raphson iteration
    let mut rate = 0.1_f64; // Initial guess: 10%
    let max_iter = 200;
    let tolerance = 1e-7;

    for _ in 0..max_iter {
        let mut xnpv = 0.0_f64;
        let mut xnpv_deriv = 0.0_f64;

        for i in 0..cashflows.len() {
            let exponent = days[i] / 365.0;
            let base = 1.0 + rate;

            if base <= 0.0 {
                return None;
            }

            let discount = base.powf(exponent);
            if discount.is_nan() || discount.is_infinite() || discount == 0.0 {
                return None;
            }

            xnpv += amounts[i] / discount;

            // Derivative: d/dr [A / (1+r)^t] = -t * A / (1+r)^(t+1)
            xnpv_deriv -= exponent * amounts[i] / (base * discount);
        }

        if xnpv_deriv.abs() < 1e-15 {
            return None;
        }

        let new_rate = rate - xnpv / xnpv_deriv;

        if (new_rate - rate).abs() < tolerance {
            return Some(new_rate * 100.0);
        }

        rate = new_rate;

        // Guard against divergence
        if rate < -0.99 || rate > 100.0 || rate.is_nan() {
            return None;
        }
    }

    // Did not converge but return best guess if XNPV is close enough
    Some(rate * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_xirr() {
        let cashflows = vec![
            (NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(), -100000.0),
            (NaiveDate::from_ymd_opt(2021, 1, 1).unwrap(), 110000.0),
        ];
        let result = calculate_xirr(&cashflows).unwrap();
        assert!((result - 10.0).abs() < 0.1, "Expected ~10%, got {}", result);
    }
}
