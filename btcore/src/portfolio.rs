//! Portfolio state management for backtest simulation
//!
//! This module provides the PortfolioState struct that tracks cash and positions
//! during simulation, along with helper methods for balance calculation and
//! position updates.

use std::collections::HashMap;

use crate::is_valid_price;
use crate::position::Position;

/// Portfolio state during simulation
///
/// Tracks cash balance and all open positions. Used as the core state
/// container throughout the backtest simulation.
#[derive(Debug)]
pub struct PortfolioState {
    /// Cash balance (starts at 1.0, representing 100% of initial capital)
    pub cash: f64,
    /// Map of stock_id -> Position
    pub positions: HashMap<usize, Position>,
}

impl PortfolioState {
    /// Create a new portfolio with initial cash of 1.0
    pub fn new() -> Self {
        Self {
            cash: 1.0,
            positions: HashMap::new(),
        }
    }

    /// Calculate total portfolio value (cash + positions)
    /// In standard mode: cash + sum(current_value)
    pub fn balance(&self) -> f64 {
        let pos_value: f64 = self.positions.values().map(|p| p.value).sum();
        self.cash + pos_value
    }

    /// Calculate portfolio balance in Finlab mode
    ///
    /// Finlab formula: cash + Σ(pos[sid] * close / price)
    /// But when close == price (both adj_close), this simplifies to cash + Σ(pos[sid])
    /// We use last_market_value which is updated via cumulative multiplication (pos *= r)
    /// This matches Finlab's floating point behavior exactly
    #[allow(dead_code)]
    pub fn balance_finlab(&self, _prices: &[f64]) -> f64 {
        // Finlab: balance = cash + Σ(pos[sid])
        // where pos[sid] is updated daily via pos *= r (cumulative multiplication)
        // We track this as last_market_value, updated in update_max_prices
        let pos_value: f64 = self
            .positions
            .values()
            .map(|p| p.last_market_value)
            .sum();
        self.cash + pos_value
    }

    /// Calculate total cost basis (used for Finlab rebalance calculation)
    ///
    /// Finlab rebalance uses sum(cost_basis), NOT market value
    pub fn total_cost_basis(&self) -> f64 {
        let pos_value: f64 = self.positions.values().map(|p| p.value).sum();
        self.cash + pos_value
    }

    /// Update market values for all positions using cumulative multiplication
    ///
    /// Finlab: pos[sid] *= r, where r = price / previous_price
    /// This matches Finlab's floating point behavior exactly
    #[allow(dead_code)]
    pub fn update_market_values(&mut self, prices: &[f64]) {
        for (&stock_id, pos) in self.positions.iter_mut() {
            if stock_id < prices.len() {
                let close_price = prices[stock_id];
                if is_valid_price(close_price) && pos.previous_price > 0.0 {
                    // Valid price: use cumulative multiplication (Finlab: pos *= r)
                    let r = close_price / pos.previous_price;
                    pos.last_market_value *= r;
                }
                // If NaN or first day, keep last_market_value unchanged (like Finlab's pos[sid] *= 1)
            }
        }
    }

    /// Update max_price and cr for stop tracking
    ///
    /// This function updates:
    /// - max_price: for trailing stop detection
    /// - cr: cumulative return ratio using Finlab's multiplication method (cr *= r)
    /// - last_market_value: position value updated with daily return
    /// - maxcr: maximum cumulative return for trailing stop
    ///
    /// Finlab's cr calculation (line 319 of restored_backtest_core.pyx):
    /// ```python
    /// r = price_values[d, sidprice] / previous_price[sidprice]
    /// cr[sid] *= r
    /// ```
    ///
    /// NOTE: Does NOT update previous_price - call update_previous_prices separately.
    pub fn update_max_prices(&mut self, prices: &[f64]) {
        for (&stock_id, pos) in self.positions.iter_mut() {
            if stock_id < prices.len() {
                let current_price = prices[stock_id];

                // Skip if price is invalid
                if !is_valid_price(current_price) {
                    continue;
                }

                // Update max_price for trailing stop
                if current_price > pos.max_price {
                    pos.max_price = current_price;
                }

                // Update cr, last_market_value using Finlab's cumulative multiplication
                // Finlab line 304-320: r = price / previous_price; pos *= r; cr *= r; maxcr = max(maxcr, cr)
                if pos.previous_price > 0.0 {
                    let r = current_price / pos.previous_price;
                    pos.cr *= r;
                    pos.last_market_value *= r; // Finlab: pos[sid] *= r
                }

                // Update maxcr using Finlab's cumulative max
                // maxcr = max(maxcr, cr)  (lines 319-320 of restored_backtest_core.pyx)
                pos.maxcr = pos.maxcr.max(pos.cr);

                // NOTE: previous_price is NOT updated here.
                // Call update_previous_prices after detect_touched_exit.
            }
        }
    }

    /// Update previous_price for all positions after touched_exit detection
    ///
    /// This should be called AFTER update_max_prices and detect_touched_exit
    /// to properly track price history for the next day's calculations.
    pub fn update_previous_prices(&mut self, prices: &[f64]) {
        for (&stock_id, pos) in self.positions.iter_mut() {
            if stock_id < prices.len() {
                let current_price = prices[stock_id];
                if is_valid_price(current_price) {
                    pos.previous_price = current_price;
                }
            }
        }
    }

    /// Calculate portfolio daily return using fixed target weights (Finlab style)
    ///
    /// Finlab uses: portfolio_return = sum(target_weight_i * stock_return_i)
    /// This is conceptually like daily rebalancing to maintain target weights
    #[allow(dead_code)]
    pub fn daily_return_finlab(
        &self,
        current_prices: &[f64],
        prev_prices: &[f64],
        target_weights: &[f64],
    ) -> f64 {
        let mut total_return = 0.0;
        let mut total_weight = 0.0;

        for (stock_id, &target_weight) in target_weights.iter().enumerate() {
            if target_weight == 0.0 {
                continue;
            }
            if stock_id >= current_prices.len() || stock_id >= prev_prices.len() {
                continue;
            }

            let curr_price = current_prices[stock_id];
            let prev_price = prev_prices[stock_id];

            // Skip if either price is invalid or NaN
            let curr_valid = is_valid_price(curr_price);
            let prev_valid = is_valid_price(prev_price);

            if curr_valid && prev_valid {
                let stock_return = (curr_price - prev_price) / prev_price;
                total_return += target_weight * stock_return;
                total_weight += target_weight;
            } else {
                // If price is NaN, this stock contributes 0% return
                // but we still count its weight
                total_weight += target_weight;
            }
        }

        // If no valid stocks, return 0
        if total_weight == 0.0 {
            0.0
        } else {
            // Normalize by actual weight (in case some stocks are missing)
            total_return / total_weight * total_weight.min(1.0)
        }
    }
}

impl Default for PortfolioState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Weight utility functions (kept from original portfolio.rs)
// ============================================================================

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
    fn test_portfolio_new() {
        let portfolio = PortfolioState::new();
        assert!((portfolio.cash - 1.0).abs() < 1e-10);
        assert!(portfolio.positions.is_empty());
    }

    #[test]
    fn test_portfolio_balance_empty() {
        let portfolio = PortfolioState::new();
        assert!((portfolio.balance() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_balance_with_positions() {
        let mut portfolio = PortfolioState::new();
        portfolio.cash = 0.5;
        portfolio.positions.insert(0, Position::new(0.3, 100.0));
        portfolio.positions.insert(1, Position::new(0.2, 50.0));

        assert!((portfolio.balance() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_total_cost_basis() {
        let mut portfolio = PortfolioState::new();
        portfolio.cash = 0.4;
        portfolio.positions.insert(0, Position::new(0.6, 100.0));

        assert!((portfolio.total_cost_basis() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_update_max_prices() {
        let mut portfolio = PortfolioState::new();
        portfolio.cash = 0.0;

        let mut pos = Position::new(1.0, 100.0);
        pos.previous_price = 100.0;
        portfolio.positions.insert(0, pos);

        // Price goes up 10%
        let prices = vec![110.0];
        portfolio.update_max_prices(&prices);

        let updated_pos = portfolio.positions.get(&0).unwrap();
        assert!((updated_pos.max_price - 110.0).abs() < 1e-10);
        assert!((updated_pos.cr - 1.1).abs() < 1e-10);
        assert!((updated_pos.last_market_value - 1.1).abs() < 1e-10);
    }

    #[test]
    fn test_update_previous_prices() {
        let mut portfolio = PortfolioState::new();
        portfolio.cash = 0.0;

        let mut pos = Position::new(1.0, 100.0);
        pos.previous_price = 100.0;
        portfolio.positions.insert(0, pos);

        let prices = vec![105.0];
        portfolio.update_previous_prices(&prices);

        let updated_pos = portfolio.positions.get(&0).unwrap();
        assert!((updated_pos.previous_price - 105.0).abs() < 1e-10);
    }

    #[test]
    fn test_update_previous_prices_skips_nan() {
        let mut portfolio = PortfolioState::new();

        let mut pos = Position::new(1.0, 100.0);
        pos.previous_price = 100.0;
        portfolio.positions.insert(0, pos);

        let prices = vec![f64::NAN];
        portfolio.update_previous_prices(&prices);

        // previous_price should remain unchanged
        let updated_pos = portfolio.positions.get(&0).unwrap();
        assert!((updated_pos.previous_price - 100.0).abs() < 1e-10);
    }

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
