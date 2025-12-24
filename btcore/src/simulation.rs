//! Core backtest simulation engine
//!
//! This module implements the main backtest simulation loop that matches
//! Finlab's backtest_core Cython implementation.
//!
//! # Weight Modes
//!
//! Supports two input modes:
//! 1. **Boolean signals** - Converted to equal weights (like Finlab with bool positions)
//! 2. **Float weights** - Custom weights, normalized to sum=1 (like Finlab with float positions)

use std::collections::HashMap;

/// A single trade record matching Finlab's trades DataFrame structure
///
/// Fields use original prices (not adjusted) for entry/exit prices,
/// matching Finlab's actual trading record format.
#[derive(Debug, Clone)]
pub struct TradeRecord {
    /// Stock ID (column index in price matrix)
    pub stock_id: usize,
    /// Actual entry date (row index in price matrix, T+1 after signal)
    pub entry_index: usize,
    /// Actual exit date (row index in price matrix)
    pub exit_index: Option<usize>,
    /// Signal date for entry (row index in price matrix)
    pub entry_sig_index: usize,
    /// Signal date for exit (row index in price matrix)
    pub exit_sig_index: Option<usize>,
    /// Position weight at entry
    pub position_weight: f64,
    /// Entry price (ORIGINAL price, not adjusted)
    pub entry_price: f64,
    /// Exit price (ORIGINAL price, not adjusted)
    pub exit_price: Option<f64>,
    /// Trade return (calculated using original prices with fees)
    pub trade_return: Option<f64>,
}

impl TradeRecord {
    /// Calculate holding period in days
    pub fn holding_period(&self) -> Option<usize> {
        self.exit_index.map(|exit| exit - self.entry_index)
    }

    /// Calculate trade return with fees
    ///
    /// Finlab formula:
    /// trade_return = (1 - fee_ratio) * (exit_price / entry_price) * (1 - tax_ratio - fee_ratio) - 1
    pub fn calculate_return(&self, fee_ratio: f64, tax_ratio: f64) -> Option<f64> {
        self.exit_price.map(|exit_price| {
            (1.0 - fee_ratio) * (exit_price / self.entry_price) * (1.0 - tax_ratio - fee_ratio) - 1.0
        })
    }
}

/// Open position tracking for trades generation
#[derive(Debug, Clone)]
struct OpenTrade {
    /// Stock ID
    #[allow(dead_code)]
    stock_id: usize,
    /// Entry index (actual entry date, T+1)
    entry_index: usize,
    /// Signal index (entry signal date)
    entry_sig_index: usize,
    /// Position weight
    weight: f64,
    /// Entry price (original, not adjusted)
    entry_price: f64,
}

/// Result of a backtest simulation including trades
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Cumulative returns at each time step
    pub creturn: Vec<f64>,
    /// List of completed trades
    pub trades: Vec<TradeRecord>,
}

// ============================================================================
// Trade Tracker Trait - abstracts trade tracking for zero-cost abstraction
// ============================================================================

/// Trait for tracking trades during simulation
///
/// This allows `run_backtest` to use `NoopTracker` (zero overhead)
/// while `run_backtest_with_trades` uses `RealTracker` (full tracking).
trait TradeTracker {
    /// Create a new tracker
    fn new() -> Self;

    /// Record opening a new trade
    fn open_trade(
        &mut self,
        stock_id: usize,
        entry_index: usize,
        signal_index: usize,
        entry_price: f64,
        weight: f64,
    );

    /// Record closing a trade (rebalance or stop)
    fn close_trade(
        &mut self,
        stock_id: usize,
        exit_index: usize,
        exit_sig_index: Option<usize>,
        exit_price: f64,
        fee_ratio: f64,
        tax_ratio: f64,
    );

    /// Check if a trade is open for a stock
    fn has_open_trade(&self, stock_id: usize) -> bool;

    /// Get entry price for an open trade (needed for return calculation)
    #[allow(dead_code)]
    fn get_entry_price(&self, stock_id: usize) -> Option<f64>;

    /// Finalize all open trades at the end of simulation
    fn finalize(
        self,
        last_index: usize,
        trade_prices: &[f64],
        fee_ratio: f64,
        tax_ratio: f64,
    ) -> Vec<TradeRecord>;
}

/// No-op trade tracker - zero overhead for simple backtest
struct NoopTracker;

impl TradeTracker for NoopTracker {
    #[inline]
    fn new() -> Self {
        Self
    }

    #[inline]
    fn open_trade(&mut self, _: usize, _: usize, _: usize, _: f64, _: f64) {}

    #[inline]
    fn close_trade(&mut self, _: usize, _: usize, _: Option<usize>, _: f64, _: f64, _: f64) {}

    #[inline]
    fn has_open_trade(&self, _: usize) -> bool {
        false
    }

    #[inline]
    fn get_entry_price(&self, _: usize) -> Option<f64> {
        None
    }

    #[inline]
    fn finalize(self, _: usize, _: &[f64], _: f64, _: f64) -> Vec<TradeRecord> {
        vec![]
    }
}

/// Real trade tracker - tracks all trades for full reporting
struct RealTracker {
    open_trades: HashMap<usize, OpenTrade>,
    completed_trades: Vec<TradeRecord>,
}

impl TradeTracker for RealTracker {
    fn new() -> Self {
        Self {
            open_trades: HashMap::new(),
            completed_trades: Vec::new(),
        }
    }

    fn open_trade(
        &mut self,
        stock_id: usize,
        entry_index: usize,
        signal_index: usize,
        entry_price: f64,
        weight: f64,
    ) {
        self.open_trades.insert(
            stock_id,
            OpenTrade {
                stock_id,
                entry_index,
                entry_sig_index: signal_index,
                weight,
                entry_price,
            },
        );
    }

    fn close_trade(
        &mut self,
        stock_id: usize,
        exit_index: usize,
        exit_sig_index: Option<usize>,
        exit_price: f64,
        fee_ratio: f64,
        tax_ratio: f64,
    ) {
        if let Some(open_trade) = self.open_trades.remove(&stock_id) {
            let trade = TradeRecord {
                stock_id,
                entry_index: open_trade.entry_index,
                exit_index: Some(exit_index),
                entry_sig_index: open_trade.entry_sig_index,
                exit_sig_index,
                position_weight: open_trade.weight,
                entry_price: open_trade.entry_price,
                exit_price: Some(exit_price),
                trade_return: None,
            };
            let trade_return = trade.calculate_return(fee_ratio, tax_ratio);

            self.completed_trades.push(TradeRecord {
                trade_return,
                ..trade
            });
        }
    }

    fn has_open_trade(&self, stock_id: usize) -> bool {
        self.open_trades.contains_key(&stock_id)
    }

    fn get_entry_price(&self, stock_id: usize) -> Option<f64> {
        self.open_trades.get(&stock_id).map(|t| t.entry_price)
    }

    fn finalize(
        mut self,
        last_index: usize,
        trade_prices: &[f64],
        fee_ratio: f64,
        tax_ratio: f64,
    ) -> Vec<TradeRecord> {
        // Close all remaining open trades
        for (stock_id, open_trade) in self.open_trades.drain() {
            let exit_price = if stock_id < trade_prices.len() {
                trade_prices[stock_id]
            } else {
                open_trade.entry_price
            };

            let trade = TradeRecord {
                stock_id,
                entry_index: open_trade.entry_index,
                exit_index: Some(last_index),
                entry_sig_index: open_trade.entry_sig_index,
                exit_sig_index: None,
                position_weight: open_trade.weight,
                entry_price: open_trade.entry_price,
                exit_price: Some(exit_price),
                trade_return: None,
            };
            let trade_return = trade.calculate_return(fee_ratio, tax_ratio);

            self.completed_trades.push(TradeRecord {
                trade_return,
                ..trade
            });
        }

        self.completed_trades
    }
}

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Transaction fee ratio (default: 0.001425 for Taiwan stocks)
    pub fee_ratio: f64,
    /// Transaction tax ratio (default: 0.003 for Taiwan stocks)
    pub tax_ratio: f64,
    /// Stop loss threshold (1.0 = disabled)
    pub stop_loss: f64,
    /// Take profit threshold (f64::INFINITY = disabled)
    pub take_profit: f64,
    /// Trailing stop threshold (f64::INFINITY = disabled)
    pub trail_stop: f64,
    /// Maximum weight per stock (default: 1.0)
    pub position_limit: f64,
    /// Retain cost when rebalancing (default: false)
    pub retain_cost_when_rebalance: bool,
    /// Stop trading next period after stop loss/take profit (default: true)
    pub stop_trading_next_period: bool,
    /// Use Finlab-compatible calculation mode (default: false)
    ///
    /// When enabled:
    /// - Positions track cost_basis + entry_price (not current_value)
    /// - Balance = cash + Σ(cost_basis * close_price / entry_price)
    /// - Rebalance uses Σ(cost_basis) as base (not market value)
    ///
    /// This mode exactly replicates Finlab's backtest_core.pyx calculation.
    pub finlab_mode: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            fee_ratio: 0.001425,
            tax_ratio: 0.003,
            stop_loss: 1.0,              // disabled
            take_profit: f64::INFINITY,  // disabled
            trail_stop: f64::INFINITY,   // disabled
            position_limit: 1.0,
            retain_cost_when_rebalance: false,
            stop_trading_next_period: true,
            finlab_mode: false,          // Use standard calculation by default
        }
    }
}

// Re-export IntoWeights from weights module
pub use crate::weights::IntoWeights;

use crate::weights::normalize_weights_finlab;

/// Position in a single stock
#[derive(Debug, Clone)]
struct Position {
    /// Current value of the position (as fraction of portfolio)
    /// In standard mode: updated daily with returns
    /// In Finlab mode: this is the cost_basis (constant after entry)
    value: f64,
    /// Entry price for value calculation in Finlab mode
    /// In Finlab mode: used to calculate market value = cost_basis * close / entry_price
    /// This is reset on rebalance when retain_cost_when_rebalance=false
    entry_price: f64,
    /// Original entry price for stop loss/take profit calculation
    /// This is NEVER reset during rebalance - always the original entry price
    stop_entry_price: f64,
    /// Maximum price since entry (for trailing stop)
    max_price: f64,
    /// Last valid market value (Finlab: pos[sid] updated daily)
    /// Used when current price is NaN
    last_market_value: f64,
    /// Cumulative return ratio for stop detection (Finlab: cr[sid] *= r)
    /// This uses cumulative multiplication to match Finlab's floating point behavior
    cr: f64,
    /// Previous price for daily return calculation (r = current / previous)
    previous_price: f64,
}

/// Portfolio state during simulation
#[derive(Debug)]
struct PortfolioState {
    /// Cash balance (starts at 1.0)
    cash: f64,
    /// Map of stock_id -> Position
    positions: HashMap<usize, Position>,
}

impl PortfolioState {
    fn new() -> Self {
        Self {
            cash: 1.0,
            positions: HashMap::new(),
        }
    }

    /// Calculate total portfolio value (cash + positions)
    /// In standard mode: cash + sum(current_value)
    fn balance(&self) -> f64 {
        let pos_value: f64 = self.positions.values().map(|p| p.value).sum();
        self.cash + pos_value
    }

    /// Calculate portfolio balance in Finlab mode
    /// Finlab formula: cash + Σ(cost_basis * close_price / entry_price)
    /// When price is NaN, Finlab uses the last valid market value (from pos[sid] *= r updates)
    #[allow(dead_code)]
    fn balance_finlab(&self, prices: &[f64]) -> f64 {
        let pos_value: f64 = self.positions.iter()
            .map(|(&stock_id, p)| {
                if stock_id < prices.len() {
                    let close_price = prices[stock_id];
                    let entry_price = p.entry_price;
                    if entry_price > 0.0 && close_price > 0.0 && !close_price.is_nan() {
                        // Valid case: cost_basis * close / entry
                        p.value * close_price / entry_price
                    } else {
                        // NaN case: use last valid market value (Finlab line 473)
                        p.last_market_value
                    }
                } else {
                    p.last_market_value
                }
            })
            .sum();
        self.cash + pos_value
    }

    /// Update last_market_value for all positions (Finlab: pos[sid] *= r)
    /// Call this daily to track market values for NaN handling
    fn update_market_values(&mut self, prices: &[f64]) {
        for (&stock_id, pos) in self.positions.iter_mut() {
            if stock_id < prices.len() {
                let close_price = prices[stock_id];
                if pos.entry_price > 0.0 && close_price > 0.0 && !close_price.is_nan() {
                    // Valid price: update market value
                    pos.last_market_value = pos.value * close_price / pos.entry_price;
                }
                // If NaN, keep last_market_value unchanged (like Finlab's pos[sid] *= 1)
            }
        }
    }

    /// Calculate portfolio daily return using fixed target weights (Finlab style)
    /// Finlab uses: portfolio_return = sum(target_weight_i * stock_return_i)
    /// This is conceptually like daily rebalancing to maintain target weights
    #[allow(dead_code)]
    fn daily_return_finlab(
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
            let curr_valid = curr_price > 0.0 && !curr_price.is_nan();
            let prev_valid = prev_price > 0.0 && !prev_price.is_nan();

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

    /// Calculate total cost basis (used for Finlab rebalance calculation)
    /// Finlab rebalance uses sum(cost_basis), NOT market value
    fn total_cost_basis(&self) -> f64 {
        let pos_value: f64 = self.positions.values().map(|p| p.value).sum();
        self.cash + pos_value
    }
}

/// Run backtest simulation (T+1 execution mode, like Finlab)
///
/// This is a unified backtest function that accepts both boolean signals and float weights.
///
// ============================================================================
// Unified Simulation Engine
// ============================================================================

/// Core simulation loop with generic trade tracking
///
/// This is the unified internal implementation that both `run_backtest`
/// and `run_backtest_with_trades` use. The `TradeTracker` trait allows
/// for zero-cost abstraction when trade tracking is not needed.
fn simulate_backtest<T: TradeTracker>(
    close_prices: &[Vec<f64>],
    trade_prices: &[Vec<f64>],
    weights: &[Vec<f64>],
    rebalance_indices: &[usize],
    config: &BacktestConfig,
    tracker: &mut T,
) -> Vec<f64> {
    if close_prices.is_empty() {
        return vec![];
    }

    let n_times = close_prices.len();
    let n_assets = close_prices[0].len();

    // Initialize portfolio
    let mut portfolio = PortfolioState::new();

    // Map rebalance index -> weight index
    let mut weight_idx = 0;

    // Previous prices for return calculation
    let mut prev_prices = close_prices[0].clone();

    // Cumulative return array
    let mut creturn = Vec::with_capacity(n_times);

    // Track which stocks to skip due to stop loss/take profit
    let mut stopped_stocks: Vec<bool> = vec![false; n_assets];

    // Pending weights to execute on next day (T+1 execution)
    let mut pending_weights: Option<Vec<f64>> = None;
    let mut pending_signal_index: Option<usize> = None;

    // Pending stop exits to execute on next day (T+1 execution)
    let mut pending_stop_exits: Vec<usize> = Vec::new();

    // Finlab mode: track active weights for stop loss handling
    let mut active_weights: Vec<f64> = vec![0.0; n_assets];

    for t in 0..n_times {
        if config.finlab_mode {
            // ====== FINLAB MODE ======
            if t > 0 {
                // ===== STEP 1: Execute pending stop exits BEFORE rebalance =====
                // Finlab processes exit_stocks before rebalance (lines 432-451 of restored_backtest_core.pyx)
                if !pending_stop_exits.is_empty() {
                    // Check will_be_set_by_rebalance (Finlab line 434):
                    // will_be_set = should_rebalance and not stop_trading_next_period and pos_values[pos_id, abssid] != 0
                    // Skip stop exit ONLY if all three conditions are met
                    let exits_to_process: Vec<usize> = pending_stop_exits
                        .iter()
                        .filter(|&&stock_id| {
                            if let Some(ref weights) = pending_weights {
                                // Finlab: skip exit if will_be_set_by_rebalance
                                // will_be_set = should_rebalance AND (not stop_trading_next_period) AND (weight != 0)
                                // We skip exit if all conditions are true
                                // So we KEEP (process) exit if any condition is false
                                let has_nonzero_weight = stock_id < weights.len() && weights[stock_id].abs() > 1e-10;
                                // If stop_trading_next_period is true, always process the exit (don't skip)
                                // because will_be_set_by_rebalance = False when stop_trading_next_period = True
                                if config.stop_trading_next_period {
                                    true // Always process exit when stop_trading_next_period is true
                                } else {
                                    !has_nonzero_weight // Skip only if has non-zero weight
                                }
                            } else {
                                true // No pending rebalance, process all exits
                            }
                        })
                        .copied()
                        .collect();

                    for stock_id in exits_to_process {
                        if let Some(pos) = portfolio.positions.remove(&stock_id) {
                            // Calculate market value for fee calculation
                            let market_value = if stock_id < close_prices[t].len() && pos.entry_price > 0.0 {
                                let close_price = close_prices[t][stock_id];
                                if close_price > 0.0 {
                                    pos.value * close_price / pos.entry_price
                                } else {
                                    pos.value
                                }
                            } else {
                                pos.value
                            };
                            let sell_value =
                                market_value - market_value.abs() * (config.fee_ratio + config.tax_ratio);
                            portfolio.cash += sell_value;

                            if config.stop_trading_next_period && stock_id < stopped_stocks.len() {
                                stopped_stocks[stock_id] = true;
                            }

                            // Clear weight for stopped stock (for daily return calculation)
                            if stock_id < active_weights.len() {
                                active_weights[stock_id] = 0.0;
                            }

                            // Record trade close
                            let exit_price = if stock_id < trade_prices[t].len() {
                                trade_prices[t][stock_id]
                            } else {
                                1.0
                            };
                            tracker.close_trade(
                                stock_id,
                                t,
                                None,
                                exit_price,
                                config.fee_ratio,
                                config.tax_ratio,
                            );
                        }
                    }
                    pending_stop_exits.clear();
                }

                // ===== STEP 2: Execute rebalance =====
                if let Some(target_weights) = pending_weights.take() {
                    let signal_index = pending_signal_index.take().unwrap_or(t - 1);

                    // Close ALL trades during rebalance (Finlab behavior: sell all, rebuy all)
                    // This matches Finlab's trades where each rebalance creates new trade records
                    let open_positions: Vec<usize> = portfolio.positions.keys().copied().collect();
                    for stock_id in open_positions {
                        let exit_price = if stock_id < trade_prices[t].len() {
                            trade_prices[t][stock_id]
                        } else {
                            1.0
                        };
                        tracker.close_trade(
                            stock_id,
                            t,
                            Some(signal_index),
                            exit_price,
                            config.fee_ratio,
                            config.tax_ratio,
                        );
                    }

                    // Execute rebalance at current prices (Finlab style: sell all, rebuy all)
                    execute_finlab_rebalance(&mut portfolio, &target_weights, &close_prices[t], config);

                    // Update active weights for daily return calculation
                    active_weights = target_weights.clone();

                    // Open trades for ALL new positions (after sell-all-rebuy-all)
                    for (stock_id, &target_weight) in target_weights.iter().enumerate() {
                        if target_weight != 0.0 && portfolio.positions.contains_key(&stock_id) {
                            let entry_price = if stock_id < trade_prices[t].len() {
                                trade_prices[t][stock_id]
                            } else {
                                1.0
                            };
                            tracker.open_trade(stock_id, t, signal_index, entry_price, target_weight);
                        }
                    }

                    stopped_stocks = vec![false; n_assets];
                }

                // ===== STEP 3: Update max_prices and detect new stops =====
                update_max_prices(&mut portfolio, &close_prices[t]);

                // Detect stops for T+1 execution
                let new_stops = detect_stops_finlab(&portfolio, &close_prices[t], config);
                pending_stop_exits.extend(new_stops);

                // Update entry prices for positions recovering from NaN
                // This ensures 0% return when a stock comes back from NaN (like Finlab)
                update_entry_prices_after_nan(&mut portfolio, &close_prices[t], &prev_prices);
            }

            // Check if this is a rebalance day
            let should_rebalance = rebalance_indices.contains(&t) && weight_idx < weights.len();

            if should_rebalance {
                let target_weights =
                    normalize_weights_finlab(&weights[weight_idx], &stopped_stocks, config.position_limit);
                pending_weights = Some(target_weights);
                pending_signal_index = Some(t);
                weight_idx += 1;
            }

            // Update market values for NaN handling (Finlab: pos[sid] *= r)
            // This tracks the last valid market value for each position
            portfolio.update_market_values(&close_prices[t]);

            // Record cumulative return using Finlab balance formula
            // Finlab uses: balance = cash + Σ(cost_basis * close / entry_price)
            // When price is NaN, uses last_market_value instead
            creturn.push(portfolio.balance_finlab(&close_prices[t]));
        } else {
            // ====== STANDARD MODE ======
            if t > 0 {
                if let Some(target_weights) = pending_weights.take() {
                    let signal_index = pending_signal_index.take().unwrap_or(t - 1);

                    // Close trades for exiting positions
                    for (stock_id, _) in portfolio.positions.iter() {
                        let stock_id = *stock_id;
                        if stock_id < target_weights.len() && target_weights[stock_id] == 0.0 {
                            let exit_price = if stock_id < trade_prices[t].len() {
                                trade_prices[t][stock_id]
                            } else {
                                1.0
                            };
                            tracker.close_trade(
                                stock_id,
                                t,
                                Some(signal_index),
                                exit_price,
                                config.fee_ratio,
                                config.tax_ratio,
                            );
                        }
                    }

                    // T+1 execution: rebalance at prev_prices, then experience today's return
                    execute_t1_rebalance(
                        &mut portfolio,
                        &target_weights,
                        &prev_prices,
                        &close_prices[t],
                        config,
                    );

                    // Open trades for new positions
                    for (stock_id, &target_weight) in target_weights.iter().enumerate() {
                        if target_weight != 0.0
                            && portfolio.positions.contains_key(&stock_id)
                            && !tracker.has_open_trade(stock_id)
                        {
                            let entry_price = if stock_id < trade_prices[t].len() {
                                trade_prices[t][stock_id]
                            } else {
                                1.0
                            };
                            tracker.open_trade(stock_id, t, signal_index, entry_price, target_weight);
                        }
                    }

                    stopped_stocks = vec![false; n_assets];
                } else {
                    // No pending weights, just update position values
                    update_position_values(&mut portfolio, &close_prices[t], &prev_prices);
                }

                // Execute pending stop exits
                if !pending_stop_exits.is_empty() {
                    for &stock_id in &pending_stop_exits {
                        if let Some(pos) = portfolio.positions.remove(&stock_id) {
                            let sell_value = pos.value * (1.0 - config.fee_ratio - config.tax_ratio);
                            portfolio.cash += sell_value;

                            if config.stop_trading_next_period && stock_id < stopped_stocks.len() {
                                stopped_stocks[stock_id] = true;
                            }

                            // Record trade close
                            let exit_price = if stock_id < trade_prices[t].len() {
                                trade_prices[t][stock_id]
                            } else {
                                1.0
                            };
                            tracker.close_trade(
                                stock_id,
                                t,
                                None,
                                exit_price,
                                config.fee_ratio,
                                config.tax_ratio,
                            );
                        }
                    }
                    pending_stop_exits.clear();
                }

                // Detect stops for T+1 execution
                let new_stops = detect_stops(&portfolio, &close_prices[t], config);
                pending_stop_exits.extend(new_stops);
            }

            // Check if this is a rebalance day
            let should_rebalance = rebalance_indices.contains(&t) && weight_idx < weights.len();

            if should_rebalance {
                let target_weights =
                    normalize_weights_finlab(&weights[weight_idx], &stopped_stocks, config.position_limit);
                pending_weights = Some(target_weights);
                pending_signal_index = Some(t);
                weight_idx += 1;
            }

            // Record cumulative return
            creturn.push(portfolio.balance());
        }

        // Update previous prices
        prev_prices = close_prices[t].clone();
    }

    creturn
}

/// Run backtest simulation
///
/// # Arguments
/// * `prices` - 2D array of prices [n_times x n_assets]
/// * `signals` - 2D array of signals [n_rebalance_times x n_assets]
///   - `Vec<Vec<bool>>`: Boolean signals, converted to equal weights (fully invested)
///   - `Vec<Vec<f64>>`: Float weights, normalized using Finlab rules (allows partial investment)
/// * `rebalance_indices` - Indices in price array where rebalancing occurs
/// * `config` - Backtest configuration
///
/// # Returns
/// Vector of cumulative returns at each time step
///
/// # Execution Model (T+1, Finlab-compatible)
///
/// This function uses T+1 execution to match Finlab's behavior:
/// - Signal on Day T
/// - Trade executes at Day T+1's close price
/// - Entry fee applied on Day T+1
/// - First price return for new entries starts Day T+2
/// - Existing positions experience Day T→T+1 return on Day T+1
///
/// # Examples
///
/// ```ignore
/// // Using boolean signals (equal weight, fully invested)
/// let signals: Vec<Vec<bool>> = vec![vec![true, false, true]];
/// let result = run_backtest(&prices, &signals, &indices, &config);
///
/// // Using float weights (custom allocation)
/// let weights: Vec<Vec<f64>> = vec![vec![0.5, 0.0, 0.3]];
/// let result = run_backtest(&prices, &weights, &indices, &config);
/// ```
pub fn run_backtest<S: IntoWeights>(
    prices: &[Vec<f64>],
    signals: &[S],
    rebalance_indices: &[usize],
    config: &BacktestConfig,
) -> Vec<f64> {
    // Convert signals to weights (empty stopped_stocks is fine now)
    let weights: Vec<Vec<f64>> = signals
        .iter()
        .map(|s| s.into_weights(&[], config.position_limit))
        .collect();

    // Use NoopTracker for zero-overhead simulation
    let mut tracker = NoopTracker::new();
    simulate_backtest(prices, prices, &weights, rebalance_indices, config, &mut tracker)
}

/// Update position values based on price changes
fn update_position_values(
    portfolio: &mut PortfolioState,
    current_prices: &[f64],
    prev_prices: &[f64],
) {
    for (&stock_id, pos) in portfolio.positions.iter_mut() {
        if stock_id >= current_prices.len() || stock_id >= prev_prices.len() {
            continue;
        }

        let prev_price = prev_prices[stock_id];
        let curr_price = current_prices[stock_id];

        if prev_price > 0.0 && curr_price > 0.0 {
            // Update value based on price change
            let return_pct = (curr_price - prev_price) / prev_price;
            pos.value *= 1.0 + return_pct;

            // Update max price for trailing stop
            if curr_price > pos.max_price {
                pos.max_price = curr_price;
            }
        }
    }
}

/// Update entry prices for positions that are recovering from NaN (Finlab mode)
///
/// Two cases are handled:
/// 1. When a stock's previous price was NaN but current price is valid
/// 2. When a position was entered with NaN price (entry_price = 0) and current price is valid
///
/// In both cases, we update entry_price to current price so the position value stays constant.
/// This matches Finlab's behavior: NaN price means 0% return for that stock.
fn update_entry_prices_after_nan(
    portfolio: &mut PortfolioState,
    current_prices: &[f64],
    prev_prices: &[f64],
) {
    for (&stock_id, pos) in portfolio.positions.iter_mut() {
        if stock_id >= current_prices.len() {
            continue;
        }

        let curr_price = current_prices[stock_id];
        let curr_is_valid = !curr_price.is_nan() && curr_price > 0.0;

        if !curr_is_valid {
            continue;
        }

        // Case 1: Position was entered with NaN price (entry_price = 0)
        // Set entry_price to current price so market_value = cost_basis * curr / curr = cost_basis
        if pos.entry_price <= 0.0 {
            pos.entry_price = curr_price;
            pos.stop_entry_price = curr_price;
            pos.max_price = curr_price;
            continue;
        }

        // Case 2: Previous day's price was NaN but current is valid
        // DO NOT update entry_price here!
        // Finlab behavior: previous_price is NOT updated during NaN days
        // So when price recovers, the return includes the "hidden" price change during NaN
        // Our model: balance = pos.value * close_price / entry_price
        // If entry_price stays at pre-NaN value, the calculation correctly includes the return
        //
        // Example:
        // Day 1: price=100, entry_price=100, balance = 1.0 * 100/100 = 1.0
        // Day 2: price=NaN, uses last_market_value = 1.0
        // Day 3: price=120, balance = 1.0 * 120/100 = 1.2 (correctly captures 20% return)
        //
        // If we updated entry_price to 120 on Day 3, balance would be 1.0 * 120/120 = 1.0 (WRONG!)
    }
}

/// Detect positions that should be closed due to stop conditions (T+1 preparation)
/// Returns a list of stock IDs that should be closed on the next trading day
fn detect_stops(
    portfolio: &PortfolioState,
    prices: &[f64],
    config: &BacktestConfig,
) -> Vec<usize> {
    portfolio
        .positions
        .iter()
        .filter_map(|(&stock_id, pos)| {
            if stock_id >= prices.len() {
                return None;
            }

            let current_price = prices[stock_id];
            // Use stop_entry_price for stop loss calculation (original entry, never reset)
            let stop_entry = pos.stop_entry_price;

            if stop_entry <= 0.0 {
                return None;
            }

            let return_since_entry = (current_price - stop_entry) / stop_entry;

            // Check stop loss
            if config.stop_loss < 1.0 && return_since_entry <= -config.stop_loss {
                return Some(stock_id);
            }

            // Check take profit
            if config.take_profit < f64::INFINITY && return_since_entry >= config.take_profit {
                return Some(stock_id);
            }

            // Check trailing stop
            // Finlab formula: triggers when (max_price - current_price) / entry_price >= trail_stop
            if config.trail_stop < f64::INFINITY {
                let drawdown_from_entry = (pos.max_price - current_price) / stop_entry;
                if drawdown_from_entry >= config.trail_stop {
                    return Some(stock_id);
                }
            }

            None
        })
        .collect()
}

/// Execute pending stop exits (T+1 execution)
/// Closes positions that were marked for exit on the previous day
#[allow(dead_code)]
fn execute_pending_stops(
    portfolio: &mut PortfolioState,
    pending_stops: &[usize],
    stopped_stocks: &mut [bool],
    config: &BacktestConfig,
) {
    for &stock_id in pending_stops {
        if let Some(pos) = portfolio.positions.remove(&stock_id) {
            let sell_value = pos.value * (1.0 - config.fee_ratio - config.tax_ratio);
            portfolio.cash += sell_value;

            if config.stop_trading_next_period && stock_id < stopped_stocks.len() {
                stopped_stocks[stock_id] = true;
            }
        }
    }
}

/// Check stop loss, take profit, and trailing stop conditions (DEPRECATED - use detect_stops + execute_pending_stops for T+1)
#[allow(dead_code)]
fn check_stops(
    portfolio: &mut PortfolioState,
    prices: &[f64],
    stopped_stocks: &mut [bool],
    config: &BacktestConfig,
) {
    let positions_to_close = detect_stops(portfolio, prices, config);

    for stock_id in positions_to_close {
        if let Some(pos) = portfolio.positions.remove(&stock_id) {
            let sell_value = pos.value * (1.0 - config.fee_ratio - config.tax_ratio);
            portfolio.cash += sell_value;

            if config.stop_trading_next_period && stock_id < stopped_stocks.len() {
                stopped_stocks[stock_id] = true;
            }
        }
    }
}

/// Detect positions that should be closed due to stop conditions in Finlab mode (T+1 preparation)
///
/// Uses Finlab's cr/maxcr formula for stop detection:
/// - cr is updated daily via cumulative multiplication: cr *= r (where r = today/yesterday)
///   This matches Finlab's floating point behavior exactly (line 319 of restored_backtest_core.pyx)
/// - maxcr = max_price / stop_entry_price (max cumulative return)
/// - max_r = 1 + take_profit
/// - min_r = max(1 - stop_loss, maxcr - trail_stop)
/// - Trigger: cr >= max_r (take profit) or cr < min_r (stop loss/trail)
fn detect_stops_finlab(
    portfolio: &PortfolioState,
    prices: &[f64],
    config: &BacktestConfig,
) -> Vec<usize> {
    portfolio
        .positions
        .iter()
        .filter_map(|(&stock_id, pos)| {
            if stock_id >= prices.len() {
                return None;
            }

            let current_price = prices[stock_id];
            let stop_entry = pos.stop_entry_price;

            if stop_entry <= 0.0 || current_price.is_nan() || current_price <= 0.0 {
                return None;
            }

            // Use cumulative cr from Position (Finlab: cr *= r)
            // This matches Finlab's floating point behavior exactly
            // Note: cr is updated by update_max_prices() before this function is called
            let cr = pos.cr;

            // maxcr = max_price / stop_entry (Finlab's max cumulative return)
            // Note: We still use direct division for maxcr since Finlab tracks it similarly
            let maxcr = pos.max_price / stop_entry;

            // Finlab stop conditions (lines 326-393 of restored_backtest_core.pyx):
            // For long positions:
            //   max_r = 1 + take_profit
            //   min_r = max(1 - stop_loss, maxcr - trail_stop)
            // Trigger: cr >= max_r (take profit) or cr < min_r (stop loss/trail)

            // Check take profit: cr >= 1 + take_profit
            if config.take_profit < f64::INFINITY && cr >= 1.0 + config.take_profit {
                return Some(stock_id);
            }

            // Calculate min_r using Finlab formula
            let stop_threshold = 1.0 - config.stop_loss;
            let trail_threshold = if config.trail_stop < f64::INFINITY {
                maxcr - config.trail_stop
            } else {
                f64::NEG_INFINITY
            };
            let min_r = stop_threshold.max(trail_threshold);

            // Check stop loss / trail stop: cr < min_r (Finlab uses < not <=)
            if cr < min_r {
                return Some(stock_id);
            }

            None
        })
        .collect()
}

/// Execute pending stop exits in Finlab mode (T+1 execution)
#[allow(dead_code)]
fn execute_pending_stops_finlab(
    portfolio: &mut PortfolioState,
    pending_stops: &[usize],
    prices: &[f64],
    stopped_stocks: &mut [bool],
    config: &BacktestConfig,
) {
    for &stock_id in pending_stops {
        if let Some(pos) = portfolio.positions.remove(&stock_id) {
            // First, update cost_basis to market value (like execute_finlab_rebalance does)
            // This ensures exit fee is calculated on market value, not original cost
            let market_value = if stock_id < prices.len() && pos.entry_price > 0.0 {
                let close_price = prices[stock_id];
                if close_price > 0.0 {
                    pos.value * close_price / pos.entry_price
                } else {
                    pos.value
                }
            } else {
                pos.value
            };

            // Finlab exit formula: cash += market_value - |market_value| * (fee + tax)
            let sell_value = market_value - market_value.abs() * (config.fee_ratio + config.tax_ratio);
            portfolio.cash += sell_value;

            if config.stop_trading_next_period && stock_id < stopped_stocks.len() {
                stopped_stocks[stock_id] = true;
            }
        }
    }
}

/// Update max_price and cr for stop tracking
///
/// This function updates:
/// - max_price: for trailing stop detection
/// - cr: cumulative return ratio using Finlab's multiplication method (cr *= r)
/// - previous_price: for next day's r calculation
///
/// Finlab's cr calculation (line 319 of restored_backtest_core.pyx):
/// ```python
/// r = price_values[d, sidprice] / previous_price[sidprice]
/// cr[sid] *= r
/// ```
fn update_max_prices(portfolio: &mut PortfolioState, prices: &[f64]) {
    for (&stock_id, pos) in portfolio.positions.iter_mut() {
        if stock_id < prices.len() {
            let current_price = prices[stock_id];

            // Skip if price is invalid
            if current_price.is_nan() || current_price <= 0.0 {
                continue;
            }

            // Update max_price for trailing stop
            if current_price > pos.max_price {
                pos.max_price = current_price;
            }

            // Update cr using Finlab's cumulative multiplication
            // cr *= r, where r = current_price / previous_price
            if pos.previous_price > 0.0 {
                let r = current_price / pos.previous_price;
                pos.cr *= r;
            }

            // Update previous_price for next day
            pos.previous_price = current_price;
        }
    }
}

/// Check stop loss / take profit for Finlab mode (DEPRECATED - use detect/execute for T+1)
#[allow(dead_code)]
fn check_stops_finlab(
    portfolio: &mut PortfolioState,
    prices: &[f64],
    stopped_stocks: &mut [bool],
    config: &BacktestConfig,
) {
    let positions_to_close = detect_stops_finlab(portfolio, prices, config);

    for stock_id in positions_to_close {
        if let Some(pos) = portfolio.positions.remove(&stock_id) {
            let sell_value = pos.value - pos.value.abs() * (config.fee_ratio + config.tax_ratio);
            portfolio.cash += sell_value;

            if config.stop_trading_next_period && stock_id < stopped_stocks.len() {
                stopped_stocks[stock_id] = true;
            }
        }
    }

    update_max_prices(portfolio, prices);
}

/// Execute rebalance in Finlab mode
///
/// Finlab uses proportional fee allocation:
/// 1. Calculate total rebalance fee cost (sell fee + buy fee)
/// 2. Spread this cost proportionally across all target positions
/// 3. Each position gets: market_value * target_weight - proportional_fee
///
/// This ensures all positions have exactly equal weight after rebalance.
fn execute_finlab_rebalance(
    portfolio: &mut PortfolioState,
    target_weights: &[f64],
    prices: &[f64],
    config: &BacktestConfig,
) {
    // Finlab uses the DIFFERENCE method for fee calculation:
    // - Continuing positions: keep value, pay fee only on SOLD portion
    // - New positions: pay entry fee on full amount
    // - This results in unequal cost_basis and potentially negative cash

    // Step 1: Update all positions to market value
    // For valid prices: market_value = cost_basis * close / entry_price
    // For NaN prices: use last_market_value (tracked daily via update_market_values)
    // This matches Finlab's pos[sid] which is updated daily with pos *= r (r=1 for NaN)
    //
    // Note: stop_entry_price handling is done in Step 4 when rebuilding positions,
    // because Step 4 clears and rebuilds all positions. The logic there handles
    // both retain_cost_when_rebalance=True and False cases.
    for (stock_id, pos) in portfolio.positions.iter_mut() {
        if *stock_id < prices.len() {
            let close_price = prices[*stock_id];
            let price_valid = close_price > 0.0 && !close_price.is_nan();
            if price_valid && pos.entry_price > 0.0 {
                pos.value = pos.value * close_price / pos.entry_price;
            } else {
                // For NaN prices, use last valid market value (Finlab: pos[sid] *= 1)
                // This is critical! Without this, we'd use cost_basis instead of market value
                pos.value = pos.last_market_value;
            }
            pos.entry_price = close_price;
        }
    }

    // Step 2: Calculate current market value (balance)
    let balance = portfolio.total_cost_basis();

    // Step 3: Calculate ratio for weight scaling
    let total_target_weight: f64 = target_weights.iter().map(|w| w.abs()).sum();
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

    // Step 4: Process each stock using Finlab's set_position logic
    // Store old positions for reference (both cost_basis and last_market_value)
    let old_positions: std::collections::HashMap<usize, f64> = portfolio
        .positions
        .iter()
        .map(|(&k, v)| (k, v.value))
        .collect();
    let old_market_values: std::collections::HashMap<usize, f64> = portfolio
        .positions
        .iter()
        .map(|(&k, v)| (k, v.last_market_value))
        .collect();
    // Store old stop_entry_price for retain_cost_when_rebalance logic
    let old_stop_entries: std::collections::HashMap<usize, f64> = portfolio
        .positions
        .iter()
        .map(|(&k, v)| (k, v.stop_entry_price))
        .collect();
    let old_max_prices: std::collections::HashMap<usize, f64> = portfolio
        .positions
        .iter()
        .map(|(&k, v)| (k, v.max_price))
        .collect();

    // Clear positions, keep track of initial cash, rebuild using Finlab's method
    portfolio.positions.clear();
    // Start with the initial cash from the portfolio (usually 0 for positions,
    // but could be non-zero at the very start or after exits)
    let mut cash = portfolio.cash;

    for (stock_id, &target_weight) in target_weights.iter().enumerate() {
        if stock_id >= prices.len() {
            continue;
        }

        let price = prices[stock_id];
        let price_valid = price > 0.0 && !price.is_nan();

        // Target position value (scaled by ratio)
        let target_value = target_weight * ratio;
        let current_value = old_positions.get(&stock_id).copied().unwrap_or(0.0);

        // Handle NaN price case:
        // Finlab enters positions even when price is NaN!
        // The position value is set, and balance uses the value directly when price is NaN.
        if !price_valid {
            // If target is 0 and we have an old position, sell it using last market value
            if target_weight.abs() < 1e-10 {
                if let Some(&market_value) = old_market_values.get(&stock_id) {
                    if market_value.abs() > 1e-10 {
                        let sell_fee = market_value.abs() * (config.fee_ratio + config.tax_ratio);
                        cash += market_value - sell_fee;
                    }
                }
                continue;
            }

            // Finlab behavior: Enter/modify position even with NaN price
            // Fees are calculated based on monetary amount difference, not price
            // - entry_price = 0 signals NaN entry, balance_finlab will use last_market_value
            // - When price becomes valid, update_entry_prices_after_nan will set entry_price
            if target_value.abs() > 1e-10 {
                // Same fee logic as valid price case
                let amount = target_value - current_value;
                let is_buy = amount > 0.0;
                let is_entry = (target_value >= 0.0 && amount > 0.0) || (target_value <= 0.0 && amount < 0.0);
                let cost = if is_entry {
                    amount.abs() * config.fee_ratio
                } else {
                    amount.abs() * (config.fee_ratio + config.tax_ratio)
                };

                let new_value = if is_buy {
                    cash -= amount;
                    current_value + amount - cost
                } else {
                    let sell_amount = amount.abs();
                    cash += sell_amount - cost;
                    current_value - sell_amount
                };

                portfolio.positions.insert(
                    stock_id,
                    Position {
                        value: new_value,
                        entry_price: 0.0, // Signal that price was NaN at entry
                        stop_entry_price: 0.0,
                        max_price: 0.0,
                        last_market_value: new_value, // Use position value as market value
                        cr: 1.0,
                        previous_price: 0.0,
                    },
                );
            }
            continue;
        }

        // Calculate trade amount (difference method)
        let amount = target_value - current_value;

        if target_value.abs() < 1e-10 {
            // Exit position completely
            if current_value.abs() > 1e-10 {
                // Sell all: cash gets net of exit fee
                let sell_fee = current_value.abs() * (config.fee_ratio + config.tax_ratio);
                cash += current_value - sell_fee;
            }
            continue;
        }

        let is_buy = amount > 0.0;
        // is_entry: buying into a long position OR selling into a short position
        let is_entry = (target_value >= 0.0 && amount > 0.0) || (target_value <= 0.0 && amount < 0.0);
        let cost = if is_entry {
            amount.abs() * config.fee_ratio
        } else {
            amount.abs() * (config.fee_ratio + config.tax_ratio)
        };

        let new_position_value;
        if is_buy {
            // Buying: cash decreases by gross amount, position increases by net amount
            cash -= amount;
            new_position_value = current_value + amount - cost;
        } else {
            // Selling: cash increases by net amount, position decreases by gross amount
            let sell_amount = amount.abs();
            cash += sell_amount - cost;
            new_position_value = current_value - sell_amount;
        }

        if new_position_value.abs() > 1e-10 {
            // Determine stop_entry_price based on retain_cost_when_rebalance
            // Finlab logic (lines 468-478 of restored_backtest_core.pyx):
            // - retain_cost=False: cr.fill(1); maxcr.fill(1); for ALL stocks
            // - retain_cost=True: only reset for NEW positions or DIRECTION CHANGE
            let (stop_entry, max_price_val) = if config.retain_cost_when_rebalance {
                // Check if this is a continuing position with same direction
                // Use target_weight (not target_value) for direction comparison
                let old_value = old_positions.get(&stock_id).copied().unwrap_or(0.0);
                let is_continuing = old_value.abs() > 1e-10 && old_value * target_weight > 0.0;

                if is_continuing {
                    // Preserve old stop tracking for continuing same-direction positions
                    let old_stop = old_stop_entries.get(&stock_id).copied().unwrap_or(price);
                    let old_max = old_max_prices.get(&stock_id).copied().unwrap_or(price);
                    (old_stop, old_max)
                } else {
                    // New position or direction change: reset stop tracking
                    (price, price)
                }
            } else {
                // Default: reset stop tracking for all positions (cr.fill(1); maxcr.fill(1);)
                (price, price)
            };

            portfolio.positions.insert(
                stock_id,
                Position {
                    value: new_position_value,
                    entry_price: price,
                    stop_entry_price: stop_entry,
                    max_price: max_price_val,
                    last_market_value: new_position_value, // Initialize to cost_basis
                    cr: 1.0, // Reset cr on rebalance (Finlab: cr.fill(1))
                    previous_price: price, // Initialize for daily r calculation
                },
            );
        }
    }

    // Step 5: Handle old positions that are OUTSIDE target_weights array
    // (positions within target_weights range are already handled in step 4)
    for (&stock_id, &old_value) in old_positions.iter() {
        if stock_id >= target_weights.len() && old_value.abs() > 1e-10 {
            // This position is outside target_weights and should be sold
            let sell_fee = old_value.abs() * (config.fee_ratio + config.tax_ratio);
            cash += old_value - sell_fee;
        }
    }

    // Step 6: Store the cash (may be negative in Finlab model)
    portfolio.cash = cash;
}

/// Execute T+1 rebalance with Finlab-compatible sequence
///
/// T+1 execution model (matches Finlab):
/// - Signal on Day T
/// - Execute at Day T+1's close (current_prices)
/// - Existing positions experience Day T→T+1 return
/// - New entries are made at Day T+1's close (no return on entry day, just fee)
/// - First price return for new positions starts Day T+2
fn execute_t1_rebalance(
    portfolio: &mut PortfolioState,
    target_weights: &[f64],
    prev_prices: &[f64],
    current_prices: &[f64],
    config: &BacktestConfig,
) {
    // Step 1: Update existing positions to experience Day T→T+1 return
    update_position_values(portfolio, current_prices, prev_prices);

    // Step 2: Rebalance at current_prices (Day T+1's close)
    // This matches Finlab's behavior where entries happen at execution day's close
    rebalance_to_target_weights(portfolio, target_weights, current_prices, config);
}

/// Enter new positions and sell positions that will be exited
///
/// This function:
/// 1. Sells positions that have target_weight = 0 (to free up cash for new entries)
/// 2. Enters new positions at prev_prices
#[allow(dead_code)]
fn enter_new_positions(
    portfolio: &mut PortfolioState,
    target_weights: &[f64],
    prices: &[f64],
    config: &BacktestConfig,
) {
    // First, sell positions that will be exited (target_weight = 0)
    // This frees up cash for new entries at the better entry price
    let positions_to_sell: Vec<usize> = portfolio
        .positions
        .keys()
        .filter(|&&id| id < target_weights.len() && target_weights[id] == 0.0)
        .copied()
        .collect();

    for stock_id in positions_to_sell {
        if let Some(pos) = portfolio.positions.remove(&stock_id) {
            // Sell at entry prices (prev_prices)
            let sell_value = pos.value * (1.0 - config.fee_ratio - config.tax_ratio);
            portfolio.cash += sell_value;
        }
    }

    // Also sell positions not in target_weights
    let extra_positions: Vec<usize> = portfolio
        .positions
        .keys()
        .filter(|&&id| id >= target_weights.len())
        .copied()
        .collect();

    for stock_id in extra_positions {
        if let Some(pos) = portfolio.positions.remove(&stock_id) {
            let sell_value = pos.value * (1.0 - config.fee_ratio - config.tax_ratio);
            portfolio.cash += sell_value;
        }
    }

    // Now enter new positions
    let total_value = portfolio.balance();

    for (stock_id, &target_weight) in target_weights.iter().enumerate() {
        if target_weight > 0.0 && !portfolio.positions.contains_key(&stock_id) {
            // This is a new position
            let target_value = total_value * target_weight;

            // Calculate how much we can buy with available cash
            let max_buy = portfolio.cash / (1.0 + config.fee_ratio);
            let actual_buy = target_value.min(max_buy);

            if actual_buy > 1e-10 {
                let cost = actual_buy * (1.0 + config.fee_ratio);
                portfolio.cash -= cost;
                portfolio.positions.insert(
                    stock_id,
                    Position {
                        value: actual_buy,
                        entry_price: prices[stock_id],
                        stop_entry_price: prices[stock_id],
                        max_price: prices[stock_id],
                        last_market_value: actual_buy,
                        cr: 1.0,
                        previous_price: prices[stock_id],
                    },
                );
            }
        }
    }
}

/// Exit positions and adjust weights (after return has been applied)
#[allow(dead_code)]
fn exit_and_adjust_positions(
    portfolio: &mut PortfolioState,
    target_weights: &[f64],
    prices: &[f64],
    config: &BacktestConfig,
) {
    // First, exit positions that should be closed (target weight = 0)
    let positions_to_close: Vec<usize> = portfolio
        .positions
        .keys()
        .filter(|&&id| id < target_weights.len() && target_weights[id] == 0.0)
        .copied()
        .collect();

    for stock_id in positions_to_close {
        if let Some(pos) = portfolio.positions.remove(&stock_id) {
            let sell_value = pos.value * (1.0 - config.fee_ratio - config.tax_ratio);
            portfolio.cash += sell_value;
        }
    }

    // Also close positions not in target_weights
    let extra_positions: Vec<usize> = portfolio
        .positions
        .keys()
        .filter(|&&id| id >= target_weights.len())
        .copied()
        .collect();

    for stock_id in extra_positions {
        if let Some(pos) = portfolio.positions.remove(&stock_id) {
            let sell_value = pos.value * (1.0 - config.fee_ratio - config.tax_ratio);
            portfolio.cash += sell_value;
        }
    }

    // Now rebalance existing positions to target weights
    let total_value = portfolio.balance();
    let total_target_weight: f64 = target_weights.iter().sum();

    if total_target_weight == 0.0 {
        return;
    }

    // Sell positions that are overweight
    for (stock_id, &target_weight) in target_weights.iter().enumerate() {
        let target_value = total_value * target_weight;
        let current_value = portfolio
            .positions
            .get(&stock_id)
            .map(|p| p.value)
            .unwrap_or(0.0);

        if current_value > target_value + 1e-10 {
            let sell_amount = current_value - target_value;
            if let Some(pos) = portfolio.positions.get_mut(&stock_id) {
                let sell_value = sell_amount * (1.0 - config.fee_ratio - config.tax_ratio);
                pos.value -= sell_amount;
                portfolio.cash += sell_value;
            }
        }
    }

    // Buy positions that are underweight
    for (stock_id, &target_weight) in target_weights.iter().enumerate() {
        if target_weight == 0.0 {
            continue;
        }

        let target_value = total_value * target_weight;
        let current_value = portfolio
            .positions
            .get(&stock_id)
            .map(|p| p.value)
            .unwrap_or(0.0);

        if current_value < target_value - 1e-10 {
            let buy_amount = target_value - current_value;
            let max_buy = portfolio.cash / (1.0 + config.fee_ratio);
            let actual_buy = buy_amount.min(max_buy);

            if actual_buy > 1e-10 {
                let cost = actual_buy * (1.0 + config.fee_ratio);
                portfolio.cash -= cost;
                let entry = portfolio.positions.entry(stock_id).or_insert(Position {
                    value: 0.0,
                    entry_price: prices[stock_id],
                    stop_entry_price: prices[stock_id],
                    max_price: prices[stock_id],
                    last_market_value: 0.0,
                    cr: 1.0,
                    previous_price: prices[stock_id],
                });
                entry.value += actual_buy;
                entry.last_market_value = entry.value;
            }
        }
    }
}

/// Rebalance portfolio to target weights at specified prices
///
/// This function:
/// 1. Sells positions that should be closed (target weight = 0)
/// 2. Adjusts existing positions to match target weights
/// 3. Enters new positions
fn rebalance_to_target_weights(
    portfolio: &mut PortfolioState,
    target_weights: &[f64],
    prices: &[f64],
    config: &BacktestConfig,
) {
    // If retain_cost_when_rebalance is false (default), reset all entry prices
    // This matches Finlab's behavior where cr.fill(1) resets all cumulative returns
    if !config.retain_cost_when_rebalance {
        for (stock_id, pos) in portfolio.positions.iter_mut() {
            if *stock_id < prices.len() {
                pos.entry_price = prices[*stock_id];
                pos.max_price = prices[*stock_id];
                pos.cr = 1.0; // Reset cr (Finlab: cr.fill(1))
                pos.previous_price = prices[*stock_id]; // Reset for daily r calculation
            }
        }
    }

    // First, fully exit positions that should be closed
    let positions_to_close: Vec<usize> = portfolio
        .positions
        .keys()
        .filter(|&&id| id < target_weights.len() && target_weights[id] == 0.0)
        .copied()
        .collect();

    for stock_id in positions_to_close {
        if let Some(pos) = portfolio.positions.remove(&stock_id) {
            let sell_value = pos.value * (1.0 - config.fee_ratio - config.tax_ratio);
            portfolio.cash += sell_value;
        }
    }

    // Also close any positions not in target_weights array
    let extra_positions: Vec<usize> = portfolio
        .positions
        .keys()
        .filter(|&&id| id >= target_weights.len())
        .copied()
        .collect();

    for stock_id in extra_positions {
        if let Some(pos) = portfolio.positions.remove(&stock_id) {
            let sell_value = pos.value * (1.0 - config.fee_ratio - config.tax_ratio);
            portfolio.cash += sell_value;
        }
    }

    // Calculate target allocation based on current portfolio value
    // NOTE: DO NOT normalize weights! If total_weight < 1.0, it means partial allocation
    let total_target_weight: f64 = target_weights.iter().sum();
    if total_target_weight == 0.0 {
        return;
    }

    let total_value = portfolio.balance();

    // First pass: sell positions that need to be reduced or closed
    for (stock_id, &target_weight) in target_weights.iter().enumerate() {
        // Use weight directly, not normalized
        let target_value = total_value * target_weight;
        let current_value = portfolio
            .positions
            .get(&stock_id)
            .map(|p| p.value)
            .unwrap_or(0.0);

        let diff = target_value - current_value;
        if diff < 0.0 {
            let sell_amount = -diff;
            if let Some(pos) = portfolio.positions.get_mut(&stock_id) {
                if pos.value >= sell_amount - 1e-10 {
                    let sell_value = sell_amount * (1.0 - config.fee_ratio - config.tax_ratio);
                    pos.value -= sell_amount;
                    portfolio.cash += sell_value;

                    // Remove position if value is near zero
                    if pos.value < 1e-10 {
                        portfolio.positions.remove(&stock_id);
                    }
                }
            }
        }
    }

    // Second pass: buy positions that need to be increased
    for (stock_id, &target_weight) in target_weights.iter().enumerate() {
        if target_weight == 0.0 {
            continue;
        }

        // Use weight directly, not normalized
        // target_value represents allocation amount (like Finlab)
        let target_allocation = total_value * target_weight;
        let current_value = portfolio
            .positions
            .get(&stock_id)
            .map(|p| p.value)
            .unwrap_or(0.0);

        // How much more allocation is needed
        // Note: We need to figure out how much MORE to spend to reach target position
        // If current_value < target_allocation * (1 - fee), we need to buy more
        let target_position = target_allocation * (1.0 - config.fee_ratio);
        let diff = target_position - current_value;

        if diff > 1e-10 {
            // Finlab-style fee calculation:
            // - Spend `amount` from cash
            // - Position value = amount * (1 - fee_ratio)
            // So to get position_diff, we need to spend position_diff / (1 - fee_ratio)
            let spend_needed = diff / (1.0 - config.fee_ratio);
            let actual_spend = spend_needed.min(portfolio.cash);

            if actual_spend > 1e-10 {
                // Position value after fee deduction (Finlab style)
                let position_value = actual_spend * (1.0 - config.fee_ratio);
                portfolio.cash -= actual_spend;

                let entry = portfolio.positions.entry(stock_id).or_insert(Position {
                    value: 0.0,
                    entry_price: prices[stock_id],
                    stop_entry_price: prices[stock_id],
                    max_price: prices[stock_id],
                    last_market_value: 0.0,
                    cr: 1.0,
                    previous_price: prices[stock_id],
                });
                // Update entry price only for new positions
                if entry.value < 1e-10 {
                    entry.entry_price = prices[stock_id];
                    entry.stop_entry_price = prices[stock_id];
                    entry.max_price = prices[stock_id];
                    entry.cr = 1.0;
                    entry.previous_price = prices[stock_id];
                }
                entry.value += position_value;
                entry.last_market_value = entry.value;
            }
        }
    }
}

/// Price data for backtest simulation
///
/// Contains close and trade prices (required), with optional high/low for touched_exit.
#[derive(Debug, Clone)]
pub struct PriceData<'a> {
    /// Close prices (adjusted) for return calculation [n_times x n_assets]
    pub close: &'a [Vec<f64>],
    /// Trade prices (original) for trade records [n_times x n_assets]
    pub trade: &'a [Vec<f64>],
    /// High prices for touched_exit (optional)
    pub high: Option<&'a [Vec<f64>]>,
    /// Low prices for touched_exit (optional)
    pub low: Option<&'a [Vec<f64>]>,
}

impl<'a> PriceData<'a> {
    /// Create with close and trade prices only
    pub fn new(close: &'a [Vec<f64>], trade: &'a [Vec<f64>]) -> Self {
        Self {
            close,
            trade,
            high: None,
            low: None,
        }
    }

    /// Create with full OHLC data
    pub fn with_ohlc(
        close: &'a [Vec<f64>],
        trade: &'a [Vec<f64>],
        high: &'a [Vec<f64>],
        low: &'a [Vec<f64>],
    ) -> Self {
        Self {
            close,
            trade,
            high: Some(high),
            low: Some(low),
        }
    }
}

/// Run backtest with trades tracking
///
/// This function returns both cumulative returns and trade records.
/// It uses:
/// - `prices.close`: Adjusted prices for return calculation (creturn)
/// - `prices.trade`: Original prices for trade records (entry/exit prices)
/// - `prices.high/low`: Optional, for touched_exit support (not yet implemented)
///
/// The trade records match Finlab's trades DataFrame format, using
/// original prices for entry/exit to match real trading execution.
///
/// # Arguments
/// * `prices` - PriceData containing close/trade prices (and optional high/low)
/// * `signals` - 2D array of bool signals or float weights [n_rebalance_times x n_assets]
/// * `rebalance_indices` - Indices in price array where rebalancing occurs
/// * `config` - Backtest configuration
///
/// # Returns
/// BacktestResult containing creturn and trades list
pub fn run_backtest_with_trades<S: IntoWeights>(
    prices: &PriceData,
    signals: &[S],
    rebalance_indices: &[usize],
    config: &BacktestConfig,
) -> BacktestResult {
    // Convert signals to weights
    let weights: Vec<Vec<f64>> = signals
        .iter()
        .map(|s| s.into_weights(&[], config.position_limit))
        .collect();

    // Use RealTracker for full trade tracking
    let mut tracker = RealTracker::new();
    let creturn = simulate_backtest(
        prices.close,
        prices.trade,
        &weights,
        rebalance_indices,
        config,
        &mut tracker,
    );

    // Finalize trades (close any remaining open positions)
    let last_trade_prices = if !prices.trade.is_empty() {
        &prices.trade[prices.trade.len() - 1]
    } else {
        &vec![]
    };
    let trades = tracker.finalize(
        prices.close.len().saturating_sub(1),
        last_trade_prices,
        config.fee_ratio,
        config.tax_ratio,
    );

    BacktestResult { creturn, trades }
}

// Dead code removed - the following functions were replaced by simulate_backtest:
// - run_backtest_with_trades_internal
// - close_trades_for_rebalance
// - open_trades_for_rebalance
// - check_stops_with_trade_tracking

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_backtest() {
        // 3 days, 2 stocks
        let prices = vec![
            vec![100.0, 200.0],  // Day 0
            vec![102.0, 198.0],  // Day 1: Stock 0 +2%, Stock 1 -1%
            vec![105.0, 200.0],  // Day 2: Stock 0 +2.9%, Stock 1 +1%
        ];

        // Hold both stocks from day 0
        let signals = vec![
            vec![true, true],
        ];

        let rebalance_indices = vec![0];

        let config = BacktestConfig {
            fee_ratio: 0.001425,
            tax_ratio: 0.003,
            ..Default::default()
        };
        let creturn = run_backtest(&prices, &signals, &rebalance_indices, &config);

        assert_eq!(creturn.len(), 3);
        // T+1 mode: Day 0 signal not yet executed = 1.0
        assert!((creturn[0] - 1.0).abs() < 1e-10, "Day 0 should be 1.0, got {}", creturn[0]);
        // Day 1: Executed with entry fee, price movement (+2% - 1%) / 2 = +0.5%, minus fees
        // Result could be above or below 1.0 depending on fee impact
        assert!(creturn[1] > 0.9 && creturn[1] < 1.1, "Day 1 should be reasonable, got {}", creturn[1]);
        assert!(creturn[2] > 0.0);
    }

    #[test]
    fn test_no_positions() {
        let prices = vec![
            vec![100.0, 200.0],
            vec![102.0, 198.0],
        ];

        let signals = vec![
            vec![false, false],
        ];

        let rebalance_indices = vec![0];

        let config = BacktestConfig {
            fee_ratio: 0.001425,
            tax_ratio: 0.003,
            ..Default::default()
        };
        let creturn = run_backtest(&prices, &signals, &rebalance_indices, &config);

        // Should stay at 1.0 with no positions
        assert_eq!(creturn.len(), 2);
        assert!((creturn[0] - 1.0).abs() < 1e-10);
        assert!((creturn[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rebalancing() {
        // 5 days, 2 stocks
        let prices = vec![
            vec![100.0, 100.0],  // Day 0
            vec![100.0, 100.0],  // Day 1 (rebalance)
            vec![110.0, 90.0],   // Day 2
            vec![110.0, 90.0],   // Day 3 (rebalance)
            vec![120.0, 80.0],   // Day 4
        ];

        // First period: hold both, second period: only stock 0
        let signals = vec![
            vec![true, true],   // Day 0
            vec![true, false],  // Day 3
        ];

        let rebalance_indices = vec![0, 3];

        let config = BacktestConfig {
            fee_ratio: 0.001425,
            tax_ratio: 0.003,
            ..Default::default()
        };
        let creturn = run_backtest(&prices, &signals, &rebalance_indices, &config);

        assert_eq!(creturn.len(), 5);
        // Check that rebalancing occurred
        assert!(creturn[4] > 0.0);
    }

    #[test]
    fn test_stop_loss() {
        let prices = vec![
            vec![100.0],
            vec![95.0],   // -5%
            vec![89.0],   // -11% from entry
            vec![85.0],
        ];

        let signals = vec![vec![true]];
        let rebalance_indices = vec![0];

        let config = BacktestConfig {
            fee_ratio: 0.001425,
            tax_ratio: 0.003,
            stop_loss: 0.10,  // 10% stop loss
            ..Default::default()
        };

        let creturn = run_backtest(&prices, &signals, &rebalance_indices, &config);

        assert_eq!(creturn.len(), 4);
        // After stop loss triggers, portfolio should stay flat
        // (position was exited on day 2 when loss exceeded 10%)
    }

    #[test]
    fn test_finlab_mode_stop_exit_uses_market_value() {
        // Verify that stop exit in finlab mode uses market value for fee calculation
        // Bug fixed: was using cost_basis instead of market value
        let prices = vec![
            vec![100.0],  // Day 0: signal
            vec![100.0],  // Day 1: entry at 100
            vec![125.0],  // Day 2: +25%, triggers 20% take profit
            vec![130.0],  // Day 3: execute exit (T+1)
            vec![140.0],  // Day 4: should be flat (already exited)
        ];

        let signals = vec![vec![true]];
        let rebalance_indices = vec![0];

        let config = BacktestConfig {
            fee_ratio: 0.01,  // 1% fee
            tax_ratio: 0.0,
            take_profit: 0.20,
            finlab_mode: true,
            ..Default::default()
        };

        let creturn = run_backtest(&prices, &signals, &rebalance_indices, &config);

        assert_eq!(creturn.len(), 5);

        // After exit, portfolio should be flat
        assert!(
            (creturn[4] - creturn[3]).abs() < 1e-10,
            "Portfolio should be flat after stop exit"
        );

        // Final value should reflect profit from 100 -> 130 (exit price) minus fees
        // If bug existed, would be ~0.99 (only cost_basis returned)
        assert!(
            creturn[4] > 1.2,
            "Final value {} should reflect profit (>1.2)",
            creturn[4]
        );
    }

    // Tests for run_backtest with weights

    #[test]
    fn test_backtest_with_weights_basic() {
        // Same as boolean signals test, but using float weights
        let prices = vec![
            vec![100.0, 200.0],  // Day 0
            vec![102.0, 198.0],  // Day 1
            vec![105.0, 200.0],  // Day 2
        ];

        // Equal weights (0.5, 0.5) should match boolean (true, true)
        let weights = vec![vec![0.5, 0.5]];
        let rebalance_indices = vec![0];

        let config = BacktestConfig {
            fee_ratio: 0.001425,
            tax_ratio: 0.003,
            ..Default::default()
        };

        let creturn = run_backtest(&prices, &weights, &rebalance_indices, &config);

        assert_eq!(creturn.len(), 3);
        // T+1 mode: Day 0 signal not yet executed = 1.0
        assert!((creturn[0] - 1.0).abs() < 1e-10, "Day 0 should be 1.0, got {}", creturn[0]);
        // Day 1: Executed with entry fee, price change could offset
        assert!(creturn[1] > 0.9 && creturn[1] < 1.1, "Day 1 should be reasonable, got {}", creturn[1]);
        assert!(creturn[2] > 0.0);
    }

    #[test]
    fn test_backtest_with_weights_unequal() {
        // 70% stock 0, 30% stock 1
        // Finlab-compatible T+1: Signal Day 0 → Execute Day 1 at Day 1's close → Return on Day 2
        let prices = vec![
            vec![100.0, 100.0],  // Day 0: Signal day
            vec![100.0, 100.0],  // Day 1: Entry day (at Day 1's close), no return yet
            vec![110.0, 100.0],  // Day 2: Stock 0 +10%, Stock 1 flat
        ];

        let weights = vec![vec![0.7, 0.3]];
        let rebalance_indices = vec![0];

        let config = BacktestConfig {
            fee_ratio: 0.0,
            tax_ratio: 0.0,
            ..Default::default()
        };

        let creturn = run_backtest(&prices, &weights, &rebalance_indices, &config);

        assert_eq!(creturn.len(), 3);
        // Day 0: Signal not yet executed = 1.0
        assert!((creturn[0] - 1.0).abs() < 1e-10, "Day 0 should be 1.0, got {}", creturn[0]);
        // Day 1: Entry at Day 1's close, no return (same prices) = 1.0
        assert!((creturn[1] - 1.0).abs() < 1e-10, "Day 1 should be 1.0, got {}", creturn[1]);
        // Day 2: Return = 0.7 * 10% + 0.3 * 0% = 7%
        let expected_day2 = 1.0 + 0.07;
        assert!((creturn[2] - expected_day2).abs() < 0.001,
            "Expected {}, got {}", expected_day2, creturn[2]);
    }

    #[test]
    fn test_backtest_with_weights_overweight() {
        // Weights sum to 1.5, should be normalized
        // Finlab-compatible T+1: Signal Day 0 → Execute Day 1 → Return on Day 2
        let prices = vec![
            vec![100.0, 100.0],  // Day 0: Signal day
            vec![100.0, 100.0],  // Day 1: Entry day
            vec![110.0, 100.0],  // Day 2: Stock 0 +10%
        ];

        let weights = vec![vec![1.0, 0.5]]; // sum = 1.5
        let rebalance_indices = vec![0];

        let config = BacktestConfig {
            fee_ratio: 0.0,
            tax_ratio: 0.0,
            ..Default::default()
        };

        let creturn = run_backtest(&prices, &weights, &rebalance_indices, &config);

        assert_eq!(creturn.len(), 3);
        // Day 0 and Day 1: No return yet
        assert!((creturn[0] - 1.0).abs() < 1e-10);
        assert!((creturn[1] - 1.0).abs() < 1e-10);
        // Day 2: Normalized: 1.0/1.5 = 0.667, 0.5/1.5 = 0.333
        // Return: 0.667 * 10% + 0.333 * 0% = 6.67%
        let expected_day2 = 1.0 + (1.0 / 1.5) * 0.10;
        assert!((creturn[2] - expected_day2).abs() < 0.001,
            "Expected {}, got {}", expected_day2, creturn[2]);
    }

    #[test]
    fn test_backtest_with_weights_underweight() {
        // Weights sum to 0.5, should NOT be normalized (partial allocation)
        // Finlab-compatible T+1: Signal Day 0 → Execute Day 1 → Return on Day 2
        let prices = vec![
            vec![100.0, 100.0],  // Day 0: Signal day
            vec![100.0, 100.0],  // Day 1: Entry day
            vec![110.0, 110.0],  // Day 2: Both +10%
        ];

        let weights = vec![vec![0.25, 0.25]]; // sum = 0.5
        let rebalance_indices = vec![0];

        let config = BacktestConfig {
            fee_ratio: 0.0,
            tax_ratio: 0.0,
            ..Default::default()
        };

        let creturn = run_backtest(&prices, &weights, &rebalance_indices, &config);

        assert_eq!(creturn.len(), 3);
        // Day 0 and Day 1: No return yet
        assert!((creturn[0] - 1.0).abs() < 1e-10);
        assert!((creturn[1] - 1.0).abs() < 1e-10);
        // Day 2: Only 50% invested, so return is 0.5 * 10% = 5%
        let expected_day2 = 1.0 + 0.50 * 0.10;
        assert!((creturn[2] - expected_day2).abs() < 0.001,
            "Expected {}, got {}", expected_day2, creturn[2]);
    }

    #[test]
    fn test_backtest_with_weights_position_limit() {
        // Weight of 0.8 should be clipped to position_limit of 0.4
        // Finlab-compatible T+1: Signal Day 0 → Execute Day 1 → Return on Day 2
        let prices = vec![
            vec![100.0, 100.0],  // Day 0: Signal day
            vec![100.0, 100.0],  // Day 1: Entry day
            vec![110.0, 100.0],  // Day 2: Stock 0 +10%
        ];

        let weights = vec![vec![0.8, 0.2]]; // Stock 0 = 0.8, will be clipped to 0.4
        let rebalance_indices = vec![0];

        let config = BacktestConfig {
            fee_ratio: 0.0,
            tax_ratio: 0.0,
            position_limit: 0.4,
            ..Default::default()
        };

        let creturn = run_backtest(&prices, &weights, &rebalance_indices, &config);

        assert_eq!(creturn.len(), 3);
        // Day 0 and Day 1: No return yet
        assert!((creturn[0] - 1.0).abs() < 1e-10);
        assert!((creturn[1] - 1.0).abs() < 1e-10);
        // Day 2: Normalized: 0.8 / 1.0 = 0.8, clipped to 0.4
        //                   0.2 / 1.0 = 0.2, stays
        // Return: 0.4 * 10% + 0.2 * 0% = 4%
        let expected_day2 = 1.0 + 0.04;
        assert!((creturn[2] - expected_day2).abs() < 0.001,
            "Expected {}, got {}", expected_day2, creturn[2]);
    }

    #[test]
    fn test_backtest_with_weights_matches_signals() {
        // run_backtest([0.5, 0.5]) should match run_backtest([true, true])
        let prices = vec![
            vec![100.0, 100.0],
            vec![110.0, 90.0],   // +10%, -10%
            vec![115.0, 85.0],
        ];

        let signals = vec![vec![true, true]];
        let weights = vec![vec![0.5, 0.5]];
        let rebalance_indices = vec![0];

        let config = BacktestConfig {
            fee_ratio: 0.001425,
            tax_ratio: 0.003,
            ..Default::default()
        };

        let creturn_signals = run_backtest(&prices, &signals, &rebalance_indices, &config);
        let creturn_weights = run_backtest(&prices, &weights, &rebalance_indices, &config);

        assert_eq!(creturn_signals.len(), creturn_weights.len());
        for (cs, cw) in creturn_signals.iter().zip(creturn_weights.iter()) {
            assert!((cs - cw).abs() < 1e-10,
                "Signal result {} != Weight result {}", cs, cw);
        }
    }

    #[test]
    fn test_backtest_with_weights_empty() {
        let prices: Vec<Vec<f64>> = vec![];
        let weights: Vec<Vec<f64>> = vec![];
        let rebalance_indices: Vec<usize> = vec![];

        let config = BacktestConfig::default();
        let creturn = run_backtest(&prices, &weights, &rebalance_indices, &config);

        assert!(creturn.is_empty());
    }

    #[test]
    fn test_backtest_with_weights_all_zero() {
        let prices = vec![
            vec![100.0, 100.0],
            vec![110.0, 110.0],
        ];

        let weights = vec![vec![0.0, 0.0]];
        let rebalance_indices = vec![0];

        let config = BacktestConfig {
            fee_ratio: 0.0,
            tax_ratio: 0.0,
            ..Default::default()
        };

        let creturn = run_backtest(&prices, &weights, &rebalance_indices, &config);

        // With zero weights, should stay at 1.0
        assert!((creturn[0] - 1.0).abs() < 1e-10);
        assert!((creturn[1] - 1.0).abs() < 1e-10);
    }

    // T+1 Execution Mode Tests (default and only mode)
    //
    // All backtests use T+1 execution (Finlab-compatible):
    // - Signal on Day T
    // - Execute at Day T+1's close price
    // - First price return starts Day T+2

    #[test]
    fn test_t1_execution_basic() {
        // Finlab-compatible T+1 execution:
        // Signal Day 0 → Execute Day 1 at Day 1's close → First return on Day 2
        let prices = vec![
            vec![100.0],  // Day 0: signal day
            vec![100.0],  // Day 1: entry day (at Day 1's close)
            vec![105.0],  // Day 2: first return +5%
            vec![110.0],  // Day 3: second return +4.76%
        ];

        let signals = vec![vec![true]];
        let rebalance_indices = vec![0];

        let config = BacktestConfig {
            fee_ratio: 0.0,
            tax_ratio: 0.0,
            ..Default::default()
        };

        let creturn = run_backtest(&prices, &signals, &rebalance_indices, &config);

        assert_eq!(creturn.len(), 4);
        // Day 0: Signal given but not yet executed = 1.0
        assert!((creturn[0] - 1.0).abs() < 1e-10, "Day 0 should be 1.0, got {}", creturn[0]);
        // Day 1: Entry at Day 1's close = 1.0 (no return yet)
        assert!((creturn[1] - 1.0).abs() < 1e-10, "Day 1 should be 1.0, got {}", creturn[1]);
        // Day 2: First return from 100 to 105 = +5%
        let expected_day2 = 1.0 * (105.0 / 100.0);
        assert!((creturn[2] - expected_day2).abs() < 1e-10,
            "Day 2: Expected {}, got {}", expected_day2, creturn[2]);
        // Day 3: From 105 to 110 = +4.76% additional
        let expected_day3 = expected_day2 * (110.0 / 105.0);
        assert!((creturn[3] - expected_day3).abs() < 0.001,
            "Day 3: Expected {}, got {}", expected_day3, creturn[3]);
    }

    #[test]
    fn test_t1_execution_with_fees() {
        // Test that entry fee is applied on T+1 (but entry price is Day T's close)
        let prices = vec![
            vec![100.0],  // Day 0: signal day, entry price
            vec![100.0],  // Day 1: execute with fee, flat price
            vec![100.0],  // Day 2: flat price
        ];

        let signals = vec![vec![true]];
        let rebalance_indices = vec![0];

        let fee_ratio = 0.001425;

        let config = BacktestConfig {
            fee_ratio,
            tax_ratio: 0.0,
            ..Default::default()
        };

        let creturn = run_backtest(&prices, &signals, &rebalance_indices, &config);

        // Day 0: No trade yet = 1.0
        assert!((creturn[0] - 1.0).abs() < 1e-10, "Day 0 should be 1.0, got {}", creturn[0]);

        // Day 1: Entry fee applied. Finlab-style: value = 1 * (1 - fee_ratio)
        let expected_day1 = 1.0 * (1.0 - fee_ratio);
        assert!((creturn[1] - expected_day1).abs() < 1e-6,
            "Day 1: Expected {}, got {}", expected_day1, creturn[1]);
    }

    #[test]
    fn test_t1_execution_weights() {
        // Finlab-compatible T+1 mode with custom weights:
        // Signal Day 0 → Execute Day 1 → First return on Day 2
        let prices = vec![
            vec![100.0, 100.0],  // Day 0: signal day
            vec![100.0, 100.0],  // Day 1: entry day
            vec![110.0, 100.0],  // Day 2: stock 0 +10%
            vec![120.0, 100.0],  // Day 3: stock 0 +9.1% more
        ];

        let weights = vec![vec![0.5, 0.5]];  // Equal weight
        let rebalance_indices = vec![0];

        let config = BacktestConfig {
            fee_ratio: 0.0,
            tax_ratio: 0.0,
            ..Default::default()
        };

        let creturn = run_backtest(&prices, &weights, &rebalance_indices, &config);

        assert_eq!(creturn.len(), 4);
        // Day 0 and Day 1: Signal and entry, no return yet = 1.0
        assert!((creturn[0] - 1.0).abs() < 1e-10);
        assert!((creturn[1] - 1.0).abs() < 1e-10);
        // Day 2: stock 0 rose 10%, 50% weight = +5%
        let expected_day2 = 1.0 + 0.5 * 0.10;
        assert!((creturn[2] - expected_day2).abs() < 0.001,
            "Day 2: Expected {}, got {}", expected_day2, creturn[2]);
    }

    #[test]
    fn test_t1_multiple_rebalances() {
        // Finlab-compatible T+1 mode with multiple rebalance points:
        // Signal 0 on Day 0 → Execute Day 1 at Day 1's close → Return on Day 2
        // Signal 1 on Day 1 → Execute Day 2 at Day 2's close → Return on Day 3
        let prices = vec![
            vec![100.0, 100.0],  // Day 0: signal 1 (stock 0)
            vec![100.0, 100.0],  // Day 1: execute signal 1, signal 2 (switch to stock 1)
            vec![110.0, 100.0],  // Day 2: stock 0 +10%, execute signal 2
            vec![110.0, 100.0],  // Day 3: stock 0 sold, stock 1 bought
            vec![110.0, 110.0],  // Day 4: stock 1 +10%
        ];

        // Signal 1: stock 0 only
        // Signal 2: stock 1 only (switch from stock 0 to stock 1)
        let signals = vec![
            vec![true, false],  // Day 0
            vec![false, true],  // Day 1
        ];
        let rebalance_indices = vec![0, 1];

        let config = BacktestConfig {
            fee_ratio: 0.0,
            tax_ratio: 0.0,
            ..Default::default()
        };

        let creturn = run_backtest(&prices, &signals, &rebalance_indices, &config);

        assert_eq!(creturn.len(), 5);
        // Day 0: Signal, not executed = 1.0
        assert!((creturn[0] - 1.0).abs() < 1e-10, "Day 0 should be 1.0, got {}", creturn[0]);
        // Day 1: Entry at Day 1's close, no return yet = 1.0
        assert!((creturn[1] - 1.0).abs() < 1e-10, "Day 1 should be 1.0, got {}", creturn[1]);
        // Day 2: stock 0 +10%, then switch signal executes at Day 2's close
        assert!((creturn[2] - 1.10).abs() < 0.001, "Day 2: Expected 1.10, got {}", creturn[2]);
        // Day 3: Stock 0 sold at Day 2's close (1.10), stock 1 bought at Day 3's close
        // Value should be ~1.10 (no return on switch day)
        assert!((creturn[3] - 1.10).abs() < 0.01, "Day 3 should be ~1.10, got {}", creturn[3]);
        // Day 4: In stock 1, gains from 100 to 110 = +10% on 1.10 = 1.21
        assert!((creturn[4] - 1.21).abs() < 0.01, "Day 4: Expected ~1.21, got {}", creturn[4]);
    }
}
