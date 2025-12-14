//! Portfolio management and position handling

use crate::returns::{daily_returns, equal_weights, portfolio_return};

/// Portfolio state at a point in time
#[derive(Debug, Clone)]
pub struct PortfolioState {
    pub weights: Vec<f64>,
    pub value: f64,
}

/// Calculate portfolio returns over time
///
/// # Arguments
/// * `prices` - 2D array of prices [time x assets]
/// * `signals` - 2D array of signals [time x assets]
/// * `rebalance_indices` - Indices where rebalancing occurs
///
/// # Returns
/// Vector of portfolio returns at each time step
pub fn calculate_portfolio_returns(
    prices: &[Vec<f64>],
    signals: &[Vec<bool>],
    rebalance_indices: &[usize],
) -> Vec<f64> {
    if prices.is_empty() || signals.is_empty() {
        return vec![];
    }

    let n_times = prices.len();
    let n_assets = prices[0].len();

    // Calculate daily returns for each asset
    let asset_returns: Vec<Vec<Option<f64>>> = (0..n_assets)
        .map(|asset_idx| {
            let asset_prices: Vec<f64> = prices.iter().map(|row| row[asset_idx]).collect();
            daily_returns(&asset_prices)
        })
        .collect();

    let mut portfolio_returns = Vec::with_capacity(n_times);
    let mut current_weights = vec![0.0; n_assets];
    let mut rebalance_iter = rebalance_indices.iter().peekable();

    for t in 0..n_times {
        // Check if we need to rebalance
        if rebalance_iter.peek() == Some(&&t) {
            // Get signals at this time and calculate new weights
            if t < signals.len() {
                current_weights = equal_weights(&signals[t]);
            }
            rebalance_iter.next();
        }

        // Calculate portfolio return at time t
        let returns_at_t: Vec<f64> = asset_returns
            .iter()
            .map(|asset_ret| asset_ret.get(t).and_then(|&r| r).unwrap_or(0.0))
            .collect();

        let port_ret = portfolio_return(&current_weights, &returns_at_t);
        portfolio_returns.push(port_ret);
    }

    portfolio_returns
}

/// Normalize weights to sum to target (default 1.0)
pub fn normalize_weights(weights: &[f64], target: f64) -> Vec<f64> {
    let sum: f64 = weights.iter().sum();
    if sum > 0.0 {
        weights.iter().map(|w| w * target / sum).collect()
    } else {
        vec![0.0; weights.len()]
    }
}

/// Apply position limit to weights
///
/// Iteratively caps weights and renormalizes until all weights are within limit.
pub fn apply_position_limit(weights: &[f64], limit: f64) -> Vec<f64> {
    let mut result = weights.to_vec();
    let mut max_iterations = 100;

    while max_iterations > 0 {
        let mut needs_cap = false;
        for w in result.iter_mut() {
            if *w > limit {
                *w = limit;
                needs_cap = true;
            }
        }

        if !needs_cap {
            break;
        }

        result = normalize_weights(&result, 1.0);
        max_iterations -= 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_weights() {
        let weights = vec![1.0, 2.0, 2.0];
        let normalized = normalize_weights(&weights, 1.0);

        assert!((normalized[0] - 0.2).abs() < 1e-10);
        assert!((normalized[1] - 0.4).abs() < 1e-10);
        assert!((normalized[2] - 0.4).abs() < 1e-10);

        let sum: f64 = normalized.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_position_limit() {
        let weights = vec![0.5, 0.3, 0.2];
        let limited = apply_position_limit(&weights, 0.4);

        // 0.5 should be capped to 0.4, then renormalized
        assert!(limited[0] <= 0.4 + 1e-10);
        let sum: f64 = limited.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
