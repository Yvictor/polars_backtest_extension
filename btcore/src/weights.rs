//! Weight normalization and signal conversion utilities
//!
//! This module provides the `IntoWeights` trait for converting signals to target weights,
//! along with weight normalization functions that match Finlab's behavior.

/// Trait for converting signals to target weights
///
/// This trait allows `run_backtest` to accept both boolean signals and float weights.
///
/// # Implementations
///
/// - `Vec<bool>`: Converts to equal weights (sum = 1.0, fully invested)
/// - `Vec<f64>`: Uses Finlab-style normalization (allows partial investment)
pub trait IntoWeights {
    /// Convert signal/weight to normalized target weights
    fn into_weights(&self, stopped_stocks: &[bool], position_limit: f64) -> Vec<f64>;
}

impl IntoWeights for Vec<bool> {
    /// Convert boolean signals to equal weights (fully invested)
    ///
    /// All `true` signals get equal weight, total sum = 1.0
    fn into_weights(&self, stopped_stocks: &[bool], position_limit: f64) -> Vec<f64> {
        calculate_target_weights(self, stopped_stocks, position_limit)
    }
}

impl IntoWeights for Vec<f64> {
    /// Normalize float weights using Finlab-style rules
    ///
    /// - If sum > 1.0: divide by sum (normalize)
    /// - If sum <= 1.0: keep as-is (allows partial investment)
    fn into_weights(&self, stopped_stocks: &[bool], position_limit: f64) -> Vec<f64> {
        normalize_weights_finlab(self, stopped_stocks, position_limit)
    }
}

/// Normalize weights like Finlab does
///
/// Finlab's normalization:
/// ```python
/// total_weight = position.abs().sum(axis=1).clip(1, None)
/// position = position.astype(float).div(total_weight, axis=0).fillna(0)
///            .clip(-abs(position_limit), abs(position_limit))
/// ```
pub fn normalize_weights_finlab(
    weights: &[f64],
    stopped_stocks: &[bool],
    position_limit: f64,
) -> Vec<f64> {
    let mut result = Vec::with_capacity(weights.len());

    // Finlab behavior: First set stopped stocks to 0, THEN calculate ratio
    // This ensures remaining stocks are re-normalized to maintain full investment
    //
    // Finlab code flow:
    // 1. pos_values[pos_id, abs(sid)] = 0  # Set stopped to 0
    // 2. rebalance(pos, pos_values[pos_id], ...)
    // 3. ratio = balance / max(abs(newp).sum(), 1)  # Sum excludes stopped!
    //
    // Key insight: Finlab uses RAW weights (1 for each target) not pre-normalized.
    // When a stock is stopped, the sum decreases (e.g., 30→29), and ratio adjusts.
    // Our pre-normalized weights (1/30 each) don't work the same way.
    //
    // Fix: After excluding stopped stocks, re-normalize remaining to maintain
    // the same investment level (sum = original sum, clipped to min 1.0).

    // Calculate original total absolute weight (before excluding stopped)
    let original_abs_weight: f64 = weights.iter().map(|w| w.abs()).sum();

    // Calculate remaining weight after excluding stopped
    let remaining_abs_weight: f64 = weights
        .iter()
        .enumerate()
        .filter(|(i, _)| !stopped_stocks.get(*i).copied().unwrap_or(false))
        .map(|(_, w)| w.abs())
        .sum();

    // If all remaining stocks are stopped, return zeros
    if remaining_abs_weight < 1e-10 {
        return vec![0.0; weights.len()];
    }

    // Calculate scale factor: maintain original investment level
    // Scale up remaining weights to compensate for stopped stocks
    let scale_factor = if original_abs_weight > 1e-10 {
        original_abs_weight / remaining_abs_weight
    } else {
        1.0
    };

    // Apply Finlab's normalization: divide by max(sum, 1.0)
    // After scaling, the sum equals original_abs_weight
    let divisor = original_abs_weight.max(1.0);

    // Normalize and apply stops
    for (i, &w) in weights.iter().enumerate() {
        let stopped = stopped_stocks.get(i).copied().unwrap_or(false);
        if stopped {
            result.push(0.0);
        } else {
            // Scale up to compensate for stopped, then normalize
            let scaled = w * scale_factor;
            let normalized = scaled / divisor;
            // Clip to position limit
            let clipped = normalized.clamp(-position_limit, position_limit);
            result.push(clipped);
        }
    }

    result
}

/// Calculate target weights from boolean signals
///
/// Converts boolean signals to equal weights where all `true` signals
/// receive equal allocation summing to 1.0 (fully invested).
pub fn calculate_target_weights(
    signals: &[bool],
    stopped_stocks: &[bool],
    position_limit: f64,
) -> Vec<f64> {
    let mut weights = Vec::with_capacity(signals.len());

    // Count active signals (excluding stopped stocks)
    // Use .get(i).unwrap_or(&false) to handle empty stopped_stocks
    let active_count: usize = signals
        .iter()
        .enumerate()
        .filter(|(i, &sig)| sig && !stopped_stocks.get(*i).copied().unwrap_or(false))
        .count();

    if active_count == 0 {
        return vec![0.0; signals.len()];
    }

    // Equal weight
    let weight = (1.0 / active_count as f64).min(position_limit);

    for (i, &sig) in signals.iter().enumerate() {
        let stopped = stopped_stocks.get(i).copied().unwrap_or(false);
        if sig && !stopped {
            weights.push(weight);
        } else {
            weights.push(0.0);
        }
    }

    // Normalize if position limit reduced total below 1.0
    let total: f64 = weights.iter().sum();
    if total > 0.0 && total < 1.0 {
        for w in weights.iter_mut() {
            *w /= total;
        }
    }

    // Re-apply position limit
    apply_position_limit(&mut weights, position_limit);

    weights
}

/// Apply position limit iteratively
///
/// Ensures no single position exceeds the limit while maintaining
/// total allocation as close to 1.0 as possible.
pub fn apply_position_limit(weights: &mut [f64], limit: f64) {
    for _ in 0..100 {
        let mut needs_cap = false;
        for w in weights.iter_mut() {
            if *w > limit {
                *w = limit;
                needs_cap = true;
            }
        }

        if !needs_cap {
            break;
        }

        let total: f64 = weights.iter().sum();
        if total > 0.0 {
            for w in weights.iter_mut() {
                *w /= total;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_weights_finlab_basic() {
        // Weights that sum to 1.0 should stay the same
        let weights = vec![0.4, 0.3, 0.3];
        let stopped = vec![false, false, false];
        let result = normalize_weights_finlab(&weights, &stopped, 1.0);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.4).abs() < 1e-10);
        assert!((result[1] - 0.3).abs() < 1e-10);
        assert!((result[2] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_weights_finlab_sum_greater_than_one() {
        // Weights that sum to > 1.0 should be divided by sum
        let weights = vec![0.6, 0.6, 0.4]; // sum = 1.6
        let stopped = vec![false, false, false];
        let result = normalize_weights_finlab(&weights, &stopped, 1.0);

        // Divided by 1.6
        assert!((result[0] - 0.6 / 1.6).abs() < 1e-10);
        assert!((result[1] - 0.6 / 1.6).abs() < 1e-10);
        assert!((result[2] - 0.4 / 1.6).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_weights_finlab_sum_less_than_one() {
        // Weights that sum to < 1.0 should NOT be normalized (divisor clipped to 1.0)
        // This matches Finlab's behavior: total_weight.clip(1, None)
        let weights = vec![0.2, 0.3]; // sum = 0.5, but divisor clipped to 1.0
        let stopped = vec![false, false];
        let result = normalize_weights_finlab(&weights, &stopped, 1.0);

        // Should stay the same (divided by max(0.5, 1.0) = 1.0)
        assert!((result[0] - 0.2).abs() < 1e-10);
        assert!((result[1] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_weights_finlab_with_position_limit() {
        // Weights should be clipped to position_limit
        let weights = vec![0.8, 0.4]; // sum = 1.2
        let stopped = vec![false, false];
        let result = normalize_weights_finlab(&weights, &stopped, 0.5);

        // First: 0.8 / 1.2 = 0.667, clipped to 0.5
        // Second: 0.4 / 1.2 = 0.333, stays
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.4 / 1.2).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_weights_finlab_with_stopped_stocks() {
        // Stopped stocks should get weight 0, and remaining stocks are SCALED UP
        // to maintain full investment (same as Finlab's raw weight behavior)
        let weights = vec![0.5, 0.5, 0.5]; // sum = 1.5
        let stopped = vec![false, true, false]; // stock 1 stopped
        let result = normalize_weights_finlab(&weights, &stopped, 1.0);

        // Original sum = 1.5, remaining sum = 1.0
        // Scale factor = 1.5 / 1.0 = 1.5
        // divisor = max(1.5, 1.0) = 1.5
        // Each remaining: 0.5 * 1.5 / 1.5 = 0.5 (maintains original proportion)
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10); // stopped
        assert!((result[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_weights_finlab_with_stopped_rescales() {
        // When one stock is stopped from a fully-invested portfolio,
        // remaining stocks should be scaled up to maintain 100% investment
        let weights = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]; // sum = 1.0
        let stopped = vec![false, true, false]; // stock 1 stopped
        let result = normalize_weights_finlab(&weights, &stopped, 1.0);

        // Original sum = 1.0, remaining sum = 2/3
        // Scale factor = 1.0 / (2/3) = 1.5
        // divisor = max(1.0, 1.0) = 1.0
        // Each remaining: (1/3) * 1.5 / 1.0 = 0.5 (full 100% investment)
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10); // stopped
        assert!((result[2] - 0.5).abs() < 1e-10);
        // Total should be 1.0 (100% invested)
        let total: f64 = result.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_weights_finlab_negative_weights() {
        // Negative weights (short positions) should be handled
        let weights = vec![0.5, -0.3]; // abs sum = 0.8, clipped to 1.0
        let stopped = vec![false, false];
        let result = normalize_weights_finlab(&weights, &stopped, 1.0);

        // Divided by max(0.8, 1.0) = 1.0, so stays same
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - (-0.3)).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_target_weights_basic() {
        let signals = vec![true, true, false];
        let stopped = vec![false, false, false];
        let result = calculate_target_weights(&signals, &stopped, 1.0);

        // Two active signals, each gets 0.5
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.5).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_target_weights_with_stopped() {
        let signals = vec![true, true, true];
        let stopped = vec![false, true, false]; // stock 1 stopped
        let result = calculate_target_weights(&signals, &stopped, 1.0);

        // Two active signals (stock 1 stopped), each gets 0.5
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10); // stopped
        assert!((result[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_apply_position_limit() {
        let mut weights = vec![0.6, 0.3, 0.2, 0.1];
        apply_position_limit(&mut weights, 0.3);

        // All weights should be <= 0.3
        for w in &weights {
            assert!(*w <= 0.3 + 1e-10);
        }

        // Sum should be ~1.0
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
