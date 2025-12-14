//! Return calculation functions

/// Calculate daily returns from prices
///
/// # Arguments
/// * `prices` - Slice of price values
///
/// # Returns
/// Vector of Option<f64> where first element is None (no previous price)
pub fn daily_returns(prices: &[f64]) -> Vec<Option<f64>> {
    if prices.is_empty() {
        return vec![];
    }

    let mut returns = Vec::with_capacity(prices.len());
    returns.push(None);

    for window in prices.windows(2) {
        let (prev, curr) = (window[0], window[1]);
        if prev > 0.0 && prev.is_finite() && curr.is_finite() {
            returns.push(Some(curr / prev - 1.0));
        } else {
            returns.push(None);
        }
    }

    returns
}

/// Calculate cumulative returns from daily returns (starting at 1.0)
///
/// # Arguments
/// * `returns` - Slice of daily returns (Option<f64>)
///
/// # Returns
/// Vector of cumulative return values
pub fn cumulative_returns(returns: &[Option<f64>]) -> Vec<f64> {
    let mut cum = 1.0;
    returns
        .iter()
        .map(|r| {
            if let Some(ret) = r {
                cum *= 1.0 + ret;
            }
            cum
        })
        .collect()
}

/// Calculate weighted portfolio return for a single period
///
/// # Arguments
/// * `weights` - Portfolio weights for each asset
/// * `returns` - Returns for each asset
///
/// # Returns
/// Weighted sum of returns
pub fn portfolio_return(weights: &[f64], returns: &[f64]) -> f64 {
    weights
        .iter()
        .zip(returns.iter())
        .map(|(w, r)| w * r)
        .sum()
}

/// Calculate equal weights for given signals
///
/// # Arguments
/// * `signals` - Boolean signals (true = hold)
///
/// # Returns
/// Vector of weights (sum = 1.0 for true signals, 0.0 for false)
pub fn equal_weights(signals: &[bool]) -> Vec<f64> {
    let count = signals.iter().filter(|&&s| s).count();
    let weight = if count > 0 { 1.0 / count as f64 } else { 0.0 };

    signals
        .iter()
        .map(|&s| if s { weight } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daily_returns_basic() {
        let prices = vec![100.0, 105.0, 103.0, 110.0];
        let returns = daily_returns(&prices);

        assert_eq!(returns.len(), 4);
        assert!(returns[0].is_none());
        assert!((returns[1].unwrap() - 0.05).abs() < 1e-10);
        assert!((returns[2].unwrap() - (-2.0 / 105.0)).abs() < 1e-10);
        assert!((returns[3].unwrap() - (7.0 / 103.0)).abs() < 1e-10);
    }

    #[test]
    fn test_daily_returns_empty() {
        let returns = daily_returns(&[]);
        assert!(returns.is_empty());
    }

    #[test]
    fn test_daily_returns_single() {
        let returns = daily_returns(&[100.0]);
        assert_eq!(returns.len(), 1);
        assert!(returns[0].is_none());
    }

    #[test]
    fn test_daily_returns_with_zero() {
        let prices = vec![100.0, 0.0, 110.0];
        let returns = daily_returns(&prices);

        assert!(returns[0].is_none());
        assert!(returns[1].is_some()); // 0/100 = -1
        assert!(returns[2].is_none()); // prev is 0, invalid
    }

    #[test]
    fn test_cumulative_returns() {
        let returns = vec![None, Some(0.01), Some(0.02), Some(-0.01)];
        let creturn = cumulative_returns(&returns);

        assert_eq!(creturn.len(), 4);
        assert!((creturn[0] - 1.0).abs() < 1e-10);
        assert!((creturn[1] - 1.01).abs() < 1e-10);
        assert!((creturn[2] - 1.01 * 1.02).abs() < 1e-10);
        assert!((creturn[3] - 1.01 * 1.02 * 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_return() {
        let weights = vec![0.5, 0.3, 0.2];
        let returns = vec![0.02, -0.01, 0.05];

        let port_ret = portfolio_return(&weights, &returns);
        let expected = 0.5 * 0.02 + 0.3 * (-0.01) + 0.2 * 0.05;

        assert!((port_ret - expected).abs() < 1e-10);
    }

    #[test]
    fn test_equal_weights() {
        let signals = vec![true, false, true, true, false];
        let weights = equal_weights(&signals);

        assert_eq!(weights.len(), 5);
        let expected_weight = 1.0 / 3.0;
        assert!((weights[0] - expected_weight).abs() < 1e-10);
        assert!((weights[1] - 0.0).abs() < 1e-10);
        assert!((weights[2] - expected_weight).abs() < 1e-10);
        assert!((weights[3] - expected_weight).abs() < 1e-10);
        assert!((weights[4] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_equal_weights_all_false() {
        let signals = vec![false, false, false];
        let weights = equal_weights(&signals);

        assert!(weights.iter().all(|&w| w == 0.0));
    }
}
