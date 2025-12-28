//! Long format backtest simulation engine
//!
//! This module implements backtest simulation for long format data (date, symbol, price, weight),
//! avoiding the overhead of pivot/partition_by operations.
//!
//! # Performance Advantage
//!
//! Long format processes only active stocks per day O(k) instead of O(n_stocks) for wide format.
//! For sparse data (10% active), this can be 5-10x faster.

use std::collections::HashMap;

use crate::config::BacktestConfig;
use crate::portfolio::PortfolioState;
use crate::position::Position;
use crate::stops::detect_stops_finlab;
use crate::tracker::BacktestResult;
use crate::weights::normalize_weights_finlab;

/// Long format backtest input data
///
/// All arrays must have the same length and be sorted by date_idx.
#[derive(Debug)]
pub struct LongFormatInput<'a> {
    /// Date indices (0-based, sorted ascending)
    pub date_indices: &'a [u32],
    /// Symbol IDs (0-based)
    pub symbol_ids: &'a [u32],
    /// Close prices for each (date, symbol) pair
    pub prices: &'a [f64],
    /// Target weights for each (date, symbol) pair
    pub weights: &'a [f64],
}

/// Run backtest on long format data
///
/// # Arguments
/// * `input` - Long format input data (must be sorted by date_idx)
/// * `n_dates` - Total number of unique dates
/// * `n_symbols` - Total number of unique symbols
/// * `rebalance_mask` - Boolean mask indicating rebalance days (length = n_dates)
/// * `config` - Backtest configuration
///
/// # Returns
/// BacktestResult containing cumulative returns (length = n_dates)
pub fn backtest_long(
    input: &LongFormatInput,
    n_dates: usize,
    n_symbols: usize,
    rebalance_mask: &[bool],
    config: &BacktestConfig,
) -> BacktestResult {
    if input.date_indices.is_empty() || n_dates == 0 {
        return BacktestResult {
            creturn: vec![],
            trades: vec![],
        };
    }

    // Initialize portfolio
    let mut portfolio = PortfolioState::new();

    // Cumulative return array
    let mut creturn = Vec::with_capacity(n_dates);

    // Track which stocks to skip due to stop loss/take profit
    let mut stopped_stocks: Vec<bool> = vec![false; n_symbols];

    // Pending weights to execute on next day (T+1 execution)
    let mut pending_weights: Option<HashMap<usize, f64>> = None;

    // Pending stop exits to execute on next day (T+1 execution)
    let mut pending_stop_exits: Vec<usize> = Vec::new();

    // Index into input arrays
    let mut i = 0;

    // Process each date
    for date_idx in 0..n_dates {
        // Collect today's data (consecutive rows with same date_idx)
        let day_start = i;
        while i < input.date_indices.len() && input.date_indices[i] == date_idx as u32 {
            i += 1;
        }
        let day_end = i;

        // Build today's price map (sparse)
        let today_prices: HashMap<usize, f64> = (day_start..day_end)
            .map(|j| (input.symbol_ids[j] as usize, input.prices[j]))
            .collect();

        // Build today's weight map (sparse)
        let today_weights: HashMap<usize, f64> = (day_start..day_end)
            .map(|j| (input.symbol_ids[j] as usize, input.weights[j]))
            .collect();

        if date_idx > 0 {
            // ===== STEP 1: Update positions with today's prices =====
            update_positions_sparse(&mut portfolio, &today_prices);

            // ===== STEP 2: Detect stops for T+1 execution =====
            let today_stops = detect_stops_sparse(&portfolio, &today_prices, config);

            // ===== STEP 3: Execute pending stop exits (yesterday's detection) =====
            if !pending_stop_exits.is_empty() {
                execute_pending_stops_sparse(
                    &mut portfolio,
                    &pending_stop_exits,
                    &today_prices,
                    &mut stopped_stocks,
                    &pending_weights,
                    config,
                );
                pending_stop_exits.clear();
            }

            // Transfer today's stops to pending
            pending_stop_exits.extend(today_stops);

            // ===== STEP 4: Execute pending rebalance =====
            if let Some(target_weights) = pending_weights.take() {
                execute_rebalance_sparse(
                    &mut portfolio,
                    &target_weights,
                    &today_prices,
                    &stopped_stocks,
                    n_symbols,
                    config,
                );

                stopped_stocks = vec![false; n_symbols];
            }
        }

        // ===== STEP 5: Check if this is a rebalance day =====
        let should_rebalance = date_idx < rebalance_mask.len() && rebalance_mask[date_idx];

        if should_rebalance {
            // Normalize and store weights for T+1 execution
            let normalized = normalize_weights_sparse(
                &today_weights,
                &stopped_stocks,
                config.position_limit,
            );
            pending_weights = Some(normalized);
        }

        // ===== STEP 6: Record cumulative return =====
        creturn.push(balance_sparse(&portfolio, &today_prices));
    }

    BacktestResult {
        creturn,
        trades: vec![], // Trade tracking not implemented for long format yet
    }
}

/// Update positions with daily returns (sparse version)
fn update_positions_sparse(portfolio: &mut PortfolioState, prices: &HashMap<usize, f64>) {
    for (&stock_id, pos) in portfolio.positions.iter_mut() {
        if let Some(&current_price) = prices.get(&stock_id) {
            if current_price.is_nan() || current_price <= 0.0 {
                continue;
            }

            // Update max_price for trailing stop
            if current_price > pos.max_price {
                pos.max_price = current_price;
            }

            // Update cr, last_market_value using cumulative multiplication
            if pos.previous_price > 0.0 {
                let r = current_price / pos.previous_price;
                pos.cr *= r;
                pos.last_market_value *= r;
            }

            // Update maxcr
            pos.maxcr = pos.maxcr.max(pos.cr);

            // Update previous_price for next day
            pos.previous_price = current_price;
        }
    }
}

/// Detect stops for positions (sparse version)
fn detect_stops_sparse(
    portfolio: &PortfolioState,
    prices: &HashMap<usize, f64>,
    config: &BacktestConfig,
) -> Vec<usize> {
    // Convert sparse prices to dense for compatibility with detect_stops_finlab
    // This is slightly inefficient but reuses existing tested logic
    let max_id = portfolio
        .positions
        .keys()
        .chain(prices.keys())
        .copied()
        .max()
        .unwrap_or(0);

    let dense_prices: Vec<f64> = (0..=max_id)
        .map(|id| prices.get(&id).copied().unwrap_or(f64::NAN))
        .collect();

    detect_stops_finlab(&portfolio.positions, &dense_prices, config)
}

/// Execute pending stop exits (sparse version)
fn execute_pending_stops_sparse(
    portfolio: &mut PortfolioState,
    pending_stops: &[usize],
    _prices: &HashMap<usize, f64>,
    stopped_stocks: &mut [bool],
    pending_weights: &Option<HashMap<usize, f64>>,
    config: &BacktestConfig,
) {
    for &stock_id in pending_stops {
        // Check will_be_set_by_rebalance
        let should_process = if let Some(ref weights) = pending_weights {
            let has_nonzero_weight = weights
                .get(&stock_id)
                .map(|w| w.abs() > 1e-10)
                .unwrap_or(false);
            if config.stop_trading_next_period {
                true
            } else {
                !has_nonzero_weight
            }
        } else {
            true
        };

        if !should_process {
            continue;
        }

        if let Some(pos) = portfolio.positions.remove(&stock_id) {
            let market_value = pos.last_market_value;
            let sell_value =
                market_value - market_value.abs() * (config.fee_ratio + config.tax_ratio);
            portfolio.cash += sell_value;

            if config.stop_trading_next_period && stock_id < stopped_stocks.len() {
                stopped_stocks[stock_id] = true;
            }
        }
    }
}

/// Execute rebalance (sparse version)
fn execute_rebalance_sparse(
    portfolio: &mut PortfolioState,
    target_weights: &HashMap<usize, f64>,
    prices: &HashMap<usize, f64>,
    stopped_stocks: &[bool],
    _n_symbols: usize,
    config: &BacktestConfig,
) {
    // Apply stopped stocks filter
    let mut filtered_weights = target_weights.clone();
    if config.stop_trading_next_period {
        let original_sum: f64 = filtered_weights.values().map(|w| w.abs()).sum();

        for (stock_id, weight) in filtered_weights.iter_mut() {
            if *stock_id < stopped_stocks.len() && stopped_stocks[*stock_id] {
                *weight = 0.0;
            }
        }

        // Re-normalize
        let remaining_sum: f64 = filtered_weights.values().map(|w| w.abs()).sum();
        if remaining_sum > 0.0 && remaining_sum < original_sum {
            let scale_factor = original_sum / remaining_sum;
            for weight in filtered_weights.values_mut() {
                *weight *= scale_factor;
            }
        }
    }

    // Update existing positions to market value
    // Important: Always update pos.value = pos.last_market_value to match Finlab behavior
    // This ensures balance calculation uses the latest market value, not old cost basis
    for (stock_id, pos) in portfolio.positions.iter_mut() {
        // Always update value to last_market_value (matches Finlab's pos[sid] *= r behavior)
        pos.value = pos.last_market_value;

        // Update entry_price only if we have a valid price
        if let Some(&price) = prices.get(stock_id) {
            if price > 0.0 && !price.is_nan() {
                pos.entry_price = price;
            }
        }
    }

    // Calculate current balance
    let balance = portfolio.total_cost_basis();
    let total_target_weight: f64 = filtered_weights.values().map(|w| w.abs()).sum();

    if total_target_weight == 0.0 || balance <= 0.0 {
        // Exit all positions
        let all_positions: Vec<usize> = portfolio.positions.keys().copied().collect();
        for stock_id in all_positions {
            if let Some(pos) = portfolio.positions.remove(&stock_id) {
                let sell_value = pos.value - pos.value.abs() * (config.fee_ratio + config.tax_ratio);
                portfolio.cash += sell_value;
            }
        }
        return;
    }

    let ratio = balance / total_target_weight.max(1.0);

    // Store old positions
    let old_positions: HashMap<usize, f64> = portfolio
        .positions
        .iter()
        .map(|(&k, v)| (k, v.value))
        .collect();

    // Clear and rebuild
    portfolio.positions.clear();
    let mut cash = portfolio.cash;

    for (&stock_id, &target_weight) in filtered_weights.iter() {
        if target_weight.abs() < 1e-10 {
            // Exit position
            if let Some(&old_value) = old_positions.get(&stock_id) {
                if old_value.abs() > 1e-10 {
                    let sell_fee = old_value.abs() * (config.fee_ratio + config.tax_ratio);
                    cash += old_value - sell_fee;
                }
            }
            continue;
        }

        let price = prices.get(&stock_id).copied().unwrap_or(f64::NAN);
        let price_valid = price > 0.0 && !price.is_nan();

        if !price_valid {
            continue;
        }

        let target_value = target_weight * ratio;
        let current_value = old_positions.get(&stock_id).copied().unwrap_or(0.0);
        let amount = target_value - current_value;

        let is_buy = amount > 0.0;
        let is_entry =
            (target_value >= 0.0 && amount > 0.0) || (target_value <= 0.0 && amount < 0.0);
        let cost = if is_entry {
            amount.abs() * config.fee_ratio
        } else {
            amount.abs() * (config.fee_ratio + config.tax_ratio)
        };

        let new_position_value = if is_buy {
            cash -= amount;
            current_value + amount - cost
        } else {
            let sell_amount = amount.abs();
            cash += sell_amount - cost;
            current_value - sell_amount
        };

        if new_position_value.abs() > 1e-10 {
            portfolio.positions.insert(
                stock_id,
                Position {
                    value: new_position_value,
                    entry_price: price,
                    stop_entry_price: price,
                    max_price: price,
                    last_market_value: new_position_value,
                    cr: 1.0,
                    maxcr: 1.0,
                    previous_price: price,
                },
            );
        }
    }

    // Handle positions outside target_weights
    for (&stock_id, &old_value) in old_positions.iter() {
        if !filtered_weights.contains_key(&stock_id) && old_value.abs() > 1e-10 {
            let sell_fee = old_value.abs() * (config.fee_ratio + config.tax_ratio);
            cash += old_value - sell_fee;
        }
    }

    portfolio.cash = cash;
}

/// Calculate portfolio balance (sparse version)
fn balance_sparse(portfolio: &PortfolioState, prices: &HashMap<usize, f64>) -> f64 {
    let pos_value: f64 = portfolio
        .positions
        .iter()
        .map(|(&stock_id, p)| {
            if let Some(&price) = prices.get(&stock_id) {
                if price > 0.0 && !price.is_nan() {
                    p.last_market_value
                } else {
                    p.last_market_value
                }
            } else {
                p.last_market_value
            }
        })
        .sum();
    portfolio.cash + pos_value
}

/// Normalize weights (sparse version)
fn normalize_weights_sparse(
    weights: &HashMap<usize, f64>,
    stopped_stocks: &[bool],
    position_limit: f64,
) -> HashMap<usize, f64> {
    // Convert to dense, normalize, convert back to sparse
    let max_id = weights.keys().copied().max().unwrap_or(0);
    let dense: Vec<f64> = (0..=max_id)
        .map(|id| weights.get(&id).copied().unwrap_or(0.0))
        .collect();

    let stopped: Vec<bool> = (0..=max_id)
        .map(|id| {
            if id < stopped_stocks.len() {
                stopped_stocks[id]
            } else {
                false
            }
        })
        .collect();

    let normalized = normalize_weights_finlab(&dense, &stopped, position_limit);

    normalized
        .into_iter()
        .enumerate()
        .filter(|(_, w)| w.abs() > 1e-10)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_long_empty() {
        let input = LongFormatInput {
            date_indices: &[],
            symbol_ids: &[],
            prices: &[],
            weights: &[],
        };
        let result = backtest_long(&input, 0, 0, &[], &BacktestConfig::default());
        assert!(result.creturn.is_empty());
    }

    #[test]
    fn test_backtest_long_single_stock() {
        // 4 days, 1 stock
        // Day 0: price=100, weight=1.0 (signal)
        // Day 1: price=100 (entry at T+1)
        // Day 2: price=110 (+10%)
        // Day 3: price=121 (+10%)
        let input = LongFormatInput {
            date_indices: &[0, 1, 2, 3],
            symbol_ids: &[0, 0, 0, 0],
            prices: &[100.0, 100.0, 110.0, 121.0],
            weights: &[1.0, 0.0, 0.0, 0.0], // Only signal on day 0
        };
        let rebalance_mask = vec![true, false, false, false];
        let config = BacktestConfig {
            fee_ratio: 0.0,
            tax_ratio: 0.0,
            finlab_mode: true,
            ..Default::default()
        };

        let result = backtest_long(&input, 4, 1, &rebalance_mask, &config);

        assert_eq!(result.creturn.len(), 4);
        // Day 0: No position yet = 1.0
        assert!(
            (result.creturn[0] - 1.0).abs() < 1e-10,
            "Day 0: expected 1.0, got {}",
            result.creturn[0]
        );
        // Day 1: Entry at 100, no return yet = 1.0
        assert!(
            (result.creturn[1] - 1.0).abs() < 1e-10,
            "Day 1: expected 1.0, got {}",
            result.creturn[1]
        );
        // Day 2: +10% = 1.1
        assert!(
            (result.creturn[2] - 1.1).abs() < 1e-10,
            "Day 2: expected 1.1, got {}",
            result.creturn[2]
        );
        // Day 3: +10% more = 1.21
        assert!(
            (result.creturn[3] - 1.21).abs() < 1e-10,
            "Day 3: expected 1.21, got {}",
            result.creturn[3]
        );
    }

    #[test]
    fn test_backtest_long_two_stocks() {
        // 3 days, 2 stocks with equal weight
        // Day 0: signal for both stocks
        // Day 1: entry
        // Day 2: Stock 0 +10%, Stock 1 -10%
        let input = LongFormatInput {
            date_indices: &[0, 0, 1, 1, 2, 2],
            symbol_ids: &[0, 1, 0, 1, 0, 1],
            prices: &[100.0, 100.0, 100.0, 100.0, 110.0, 90.0],
            weights: &[0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        };
        let rebalance_mask = vec![true, false, false];
        let config = BacktestConfig {
            fee_ratio: 0.0,
            tax_ratio: 0.0,
            finlab_mode: true,
            ..Default::default()
        };

        let result = backtest_long(&input, 3, 2, &rebalance_mask, &config);

        assert_eq!(result.creturn.len(), 3);
        // Day 0: 1.0
        assert!(
            (result.creturn[0] - 1.0).abs() < 1e-10,
            "Day 0: expected 1.0, got {}",
            result.creturn[0]
        );
        // Day 1: 1.0
        assert!(
            (result.creturn[1] - 1.0).abs() < 1e-10,
            "Day 1: expected 1.0, got {}",
            result.creturn[1]
        );
        // Day 2: 0.5 * 1.1 + 0.5 * 0.9 = 1.0
        assert!(
            (result.creturn[2] - 1.0).abs() < 1e-10,
            "Day 2: expected 1.0, got {}",
            result.creturn[2]
        );
    }

    #[test]
    fn test_backtest_long_with_fees() {
        // Single stock with entry fee
        let input = LongFormatInput {
            date_indices: &[0, 1, 2],
            symbol_ids: &[0, 0, 0],
            prices: &[100.0, 100.0, 100.0],
            weights: &[1.0, 0.0, 0.0],
        };
        let rebalance_mask = vec![true, false, false];
        let fee_ratio = 0.01; // 1% fee
        let config = BacktestConfig {
            fee_ratio,
            tax_ratio: 0.0,
            finlab_mode: true,
            ..Default::default()
        };

        let result = backtest_long(&input, 3, 1, &rebalance_mask, &config);

        // Day 1: Entry with 1% fee = 0.99
        let expected_day1 = 1.0 - fee_ratio;
        assert!(
            (result.creturn[1] - expected_day1).abs() < 1e-6,
            "Day 1: expected {}, got {}",
            expected_day1,
            result.creturn[1]
        );
    }

    #[test]
    fn test_backtest_long_stop_loss() {
        // Stock drops 15%, then recovers - stop loss should have exited
        // Add Day 5 where price recovers to verify position was closed
        let input = LongFormatInput {
            date_indices: &[0, 1, 2, 3, 4, 5],
            symbol_ids: &[0, 0, 0, 0, 0, 0],
            prices: &[100.0, 100.0, 95.0, 85.0, 80.0, 100.0],
            weights: &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let rebalance_mask = vec![true, false, false, false, false, false];
        let config = BacktestConfig {
            fee_ratio: 0.0,
            tax_ratio: 0.0,
            stop_loss: 0.10, // 10% stop loss
            finlab_mode: true,
            ..Default::default()
        };

        let result = backtest_long(&input, 6, 1, &rebalance_mask, &config);

        // Day 0: 1.0 (no position)
        // Day 1: 1.0 (entry at T+1)
        // Day 2: 0.95 (-5%, cr=0.95, no stop)
        // Day 3: 0.85 (-15%, cr=0.85 < 0.90, triggers stop for T+1)
        // Day 4: 0.80 (exit executed at day 4's price, now all cash)
        // Day 5: 0.80 (still all cash, price recovery doesn't affect us)

        assert_eq!(result.creturn.len(), 6);
        assert!(
            (result.creturn[0] - 1.0).abs() < 1e-10,
            "Day 0: expected 1.0, got {}",
            result.creturn[0]
        );
        assert!(
            (result.creturn[1] - 1.0).abs() < 1e-10,
            "Day 1: expected 1.0, got {}",
            result.creturn[1]
        );
        assert!(
            (result.creturn[2] - 0.95).abs() < 1e-10,
            "Day 2: expected 0.95, got {}",
            result.creturn[2]
        );
        assert!(
            (result.creturn[3] - 0.85).abs() < 1e-10,
            "Day 3: expected 0.85, got {}",
            result.creturn[3]
        );
        // Day 4: Stop exit executed at price=80, so value = 0.80
        assert!(
            (result.creturn[4] - 0.80).abs() < 1e-10,
            "Day 4: expected 0.80, got {}",
            result.creturn[4]
        );
        // Day 5: Position was closed, price recovery to 100 doesn't affect portfolio
        // Should still be 0.80 (all cash)
        assert!(
            (result.creturn[5] - 0.80).abs() < 1e-10,
            "Day 5: expected 0.80 (flat), got {}",
            result.creturn[5]
        );
    }
}
