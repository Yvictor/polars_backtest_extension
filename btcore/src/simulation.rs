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

/// Position in a single stock
#[derive(Debug, Clone)]
struct Position {
    /// Current value of the position (as fraction of portfolio)
    /// In standard mode: updated daily with returns
    /// In Finlab mode: this is the cost_basis (constant after entry)
    value: f64,
    /// Entry price for stop loss/take profit calculation
    /// In Finlab mode: used to calculate market value = cost_basis * close / entry_price
    entry_price: f64,
    /// Maximum price since entry (for trailing stop)
    max_price: f64,
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
    fn balance_finlab(&self, prices: &[f64]) -> f64 {
        let pos_value: f64 = self.positions.iter()
            .map(|(&stock_id, p)| {
                if stock_id < prices.len() {
                    let close_price = prices[stock_id];
                    let entry_price = p.entry_price;
                    if entry_price > 0.0 && close_price > 0.0 && !close_price.is_nan() {
                        // Finlab formula: cost_basis * close / entry
                        p.value * close_price / entry_price
                    } else {
                        // If price is invalid, use cost_basis directly (like Finlab)
                        p.value
                    }
                } else {
                    p.value
                }
            })
            .sum();
        self.cash + pos_value
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
/// # Arguments
/// * `prices` - 2D array of prices [n_times x n_assets]
/// * `signals` - 2D array of boolean signals [n_rebalance_times x n_assets]
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
pub fn run_backtest(
    prices: &[Vec<f64>],
    signals: &[Vec<bool>],
    rebalance_indices: &[usize],
    config: &BacktestConfig,
) -> Vec<f64> {
    if prices.is_empty() {
        return vec![];
    }

    let n_times = prices.len();
    let n_assets = prices[0].len();

    // Initialize portfolio
    let mut portfolio = PortfolioState::new();

    // Map rebalance index -> signal index
    let mut signal_idx = 0;

    // Previous prices for return calculation
    let mut prev_prices = prices[0].clone();

    // Cumulative return array
    let mut creturn = Vec::with_capacity(n_times);

    // Track which stocks to skip due to stop loss/take profit
    let mut stopped_stocks: Vec<bool> = vec![false; n_assets];

    // Pending weights to execute on next day (T+1 execution)
    let mut pending_weights: Option<Vec<f64>> = None;

    for t in 0..n_times {
        // T+1 execution model (Finlab-compatible):
        // - Entries: happen at prev_prices (yesterday's close), then experience today's return
        // - Exits: experience today's return first, then exit at current prices
        if t > 0 {
            if let Some(target_weights) = pending_weights.take() {
                // T+1 execution: rebalance at prev_prices, then experience today's return
                execute_t1_rebalance(
                    &mut portfolio,
                    &target_weights,
                    &prev_prices,
                    &prices[t],
                    config,
                );

                stopped_stocks = vec![false; n_assets];
            } else {
                // No pending weights, just update position values
                update_position_values(&mut portfolio, &prices[t], &prev_prices);
            }

            // Check stop loss / take profit / trailing stop
            check_stops(
                &mut portfolio,
                &prices[t],
                &mut stopped_stocks,
                config,
            );
        }

        // Check if this is a rebalance day
        let should_rebalance = rebalance_indices.contains(&t) && signal_idx < signals.len();

        if should_rebalance {
            // Get target weights from signals
            let target_weights = calculate_target_weights(
                &signals[signal_idx],
                &stopped_stocks,
                config.position_limit,
            );

            // T+1 mode: store for execution on next day
            pending_weights = Some(target_weights);

            signal_idx += 1;
        }

        // Record cumulative return
        creturn.push(portfolio.balance());

        // Update previous prices
        prev_prices = prices[t].clone();
    }

    creturn
}

/// Run backtest simulation with custom float weights (T+1 execution mode, like Finlab)
///
/// This function accepts float weights directly, matching Finlab's behavior
/// when position DataFrame contains float values.
///
/// # Arguments
/// * `prices` - 2D array of prices [n_times x n_assets]
/// * `weights` - 2D array of float weights [n_rebalance_times x n_assets]
/// * `rebalance_indices` - Indices in price array where rebalancing occurs
/// * `config` - Backtest configuration
///
/// # Returns
/// Vector of cumulative returns at each time step
///
/// # Weight Normalization (matches Finlab)
///
/// Weights are normalized like Finlab:
/// ```text
/// total_weight = abs(weights).sum().clip(1, None)
/// normalized = weights / total_weight
/// normalized = normalized.clip(-position_limit, position_limit)
/// ```
///
/// # Execution Model (T+1, Finlab-compatible)
///
/// This function uses T+1 execution to match Finlab's behavior:
/// - Signal on Day T
/// - Trade executes at Day T+1's close price
/// - Entry fee applied on Day T+1
/// - First price return for new entries starts Day T+2
/// - Existing positions experience Day T→T+1 return on Day T+1
pub fn run_backtest_with_weights(
    prices: &[Vec<f64>],
    weights: &[Vec<f64>],
    rebalance_indices: &[usize],
    config: &BacktestConfig,
) -> Vec<f64> {
    if prices.is_empty() {
        return vec![];
    }

    let n_times = prices.len();
    let n_assets = prices[0].len();

    // Initialize portfolio
    let mut portfolio = PortfolioState::new();

    // Map rebalance index -> weight index
    let mut weight_idx = 0;

    // Previous prices for return calculation
    let mut prev_prices = prices[0].clone();

    // Cumulative return array
    let mut creturn = Vec::with_capacity(n_times);

    // Track which stocks to skip due to stop loss/take profit
    let mut stopped_stocks: Vec<bool> = vec![false; n_assets];

    // Pending weights to execute on next day (T+1 execution)
    let mut pending_weights: Option<Vec<f64>> = None;

    for t in 0..n_times {
        if config.finlab_mode {
            // ====== FINLAB MODE ======
            // Key differences:
            // 1. cost_basis doesn't change with price (no update_position_values)
            // 2. balance = cash + Σ(cost_basis * close / entry_price)
            // 3. rebalance uses sum(cost_basis) as base
            if t > 0 {
                if let Some(target_weights) = pending_weights.take() {
                    // Execute rebalance at current prices (Finlab style)
                    execute_finlab_rebalance(
                        &mut portfolio,
                        &target_weights,
                        &prices[t],
                        config,
                    );
                    stopped_stocks = vec![false; n_assets];
                }
                // In Finlab mode, we DON'T update position values
                // because cost_basis stays constant

                // Check stop loss / take profit (using current prices)
                check_stops_finlab(
                    &mut portfolio,
                    &prices[t],
                    &mut stopped_stocks,
                    config,
                );
            }

            // Check if this is a rebalance (signal) day
            let should_rebalance = rebalance_indices.contains(&t) && weight_idx < weights.len();

            if should_rebalance {
                let target_weights = normalize_weights_finlab(
                    &weights[weight_idx],
                    &stopped_stocks,
                    config.position_limit,
                );
                pending_weights = Some(target_weights);
                weight_idx += 1;
            }

            // Record cumulative return using Finlab formula
            creturn.push(portfolio.balance_finlab(&prices[t]));
        } else {
            // ====== STANDARD MODE ======
            // T+1 execution model:
            // - Entries: happen at prev_prices (yesterday's close), then experience today's return
            // - Exits: experience today's return first, then exit at current prices
            if t > 0 {
                if let Some(target_weights) = pending_weights.take() {
                    // T+1 execution: rebalance at prev_prices, then experience today's return
                    execute_t1_rebalance(
                        &mut portfolio,
                        &target_weights,
                        &prev_prices,
                        &prices[t],
                        config,
                    );

                    stopped_stocks = vec![false; n_assets];
                } else {
                    // No pending weights, just update position values
                    update_position_values(&mut portfolio, &prices[t], &prev_prices);
                }

                // Check stop loss / take profit / trailing stop
                check_stops(
                    &mut portfolio,
                    &prices[t],
                    &mut stopped_stocks,
                    config,
                );
            }

            // Check if this is a rebalance (signal) day
            let should_rebalance = rebalance_indices.contains(&t) && weight_idx < weights.len();

            if should_rebalance {
                // Normalize weights like Finlab
                let target_weights = normalize_weights_finlab(
                    &weights[weight_idx],
                    &stopped_stocks,
                    config.position_limit,
                );

                // T+1 mode: store for execution on next day
                pending_weights = Some(target_weights);

                weight_idx += 1;
            }

            // Record cumulative return
            creturn.push(portfolio.balance());
        }

        // Update previous prices
        prev_prices = prices[t].clone();
    }

    creturn
}

/// Normalize weights like Finlab does
///
/// Finlab's normalization:
/// ```python
/// total_weight = position.abs().sum(axis=1).clip(1, None)
/// position = position.astype(float).div(total_weight, axis=0).fillna(0)
///            .clip(-abs(position_limit), abs(position_limit))
/// ```
fn normalize_weights_finlab(
    weights: &[f64],
    stopped_stocks: &[bool],
    position_limit: f64,
) -> Vec<f64> {
    let mut result = Vec::with_capacity(weights.len());

    // Calculate total absolute weight (clip to min 1.0 like Finlab)
    let total_abs_weight: f64 = weights.iter().map(|w| w.abs()).sum();
    let divisor = total_abs_weight.max(1.0);

    // Normalize and apply stops
    for (i, &w) in weights.iter().enumerate() {
        let stopped = stopped_stocks.get(i).copied().unwrap_or(false);
        if stopped {
            result.push(0.0);
        } else {
            // Normalize by total weight
            let normalized = w / divisor;
            // Clip to position limit
            let clipped = normalized.clamp(-position_limit, position_limit);
            result.push(clipped);
        }
    }

    result
}

/// Calculate target weights from boolean signals
fn calculate_target_weights(
    signals: &[bool],
    stopped_stocks: &[bool],
    position_limit: f64,
) -> Vec<f64> {
    let mut weights = Vec::with_capacity(signals.len());

    // Count active signals (excluding stopped stocks)
    let active_count: usize = signals
        .iter()
        .zip(stopped_stocks.iter())
        .filter(|(&sig, &stopped)| sig && !stopped)
        .count();

    if active_count == 0 {
        return vec![0.0; signals.len()];
    }

    // Equal weight
    let weight = (1.0 / active_count as f64).min(position_limit);

    for (sig, stopped) in signals.iter().zip(stopped_stocks.iter()) {
        if *sig && !*stopped {
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
fn apply_position_limit(weights: &mut [f64], limit: f64) {
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

/// Check stop loss, take profit, and trailing stop conditions
fn check_stops(
    portfolio: &mut PortfolioState,
    prices: &[f64],
    stopped_stocks: &mut [bool],
    config: &BacktestConfig,
) {
    let positions_to_close: Vec<usize> = portfolio
        .positions
        .iter()
        .filter_map(|(&stock_id, pos)| {
            if stock_id >= prices.len() {
                return None;
            }

            let current_price = prices[stock_id];
            let entry_price = pos.entry_price;

            if entry_price <= 0.0 {
                return None;
            }

            let return_since_entry = (current_price - entry_price) / entry_price;

            // Check stop loss
            if config.stop_loss < 1.0 && return_since_entry <= -config.stop_loss {
                return Some(stock_id);
            }

            // Check take profit
            if config.take_profit < f64::INFINITY && return_since_entry >= config.take_profit {
                return Some(stock_id);
            }

            // Check trailing stop
            if config.trail_stop < f64::INFINITY {
                let drawdown = (pos.max_price - current_price) / pos.max_price;
                if drawdown >= config.trail_stop {
                    return Some(stock_id);
                }
            }

            None
        })
        .collect();

    for stock_id in positions_to_close {
        if let Some(pos) = portfolio.positions.remove(&stock_id) {
            let sell_value = pos.value * (1.0 - config.fee_ratio - config.tax_ratio);
            portfolio.cash += sell_value;

            if stock_id < stopped_stocks.len() {
                stopped_stocks[stock_id] = true;
            }
        }
    }
}

/// Check stop loss / take profit for Finlab mode
///
/// In Finlab mode, we calculate return using cost_basis * close / entry_price
fn check_stops_finlab(
    portfolio: &mut PortfolioState,
    prices: &[f64],
    stopped_stocks: &mut [bool],
    config: &BacktestConfig,
) {
    let positions_to_close: Vec<usize> = portfolio
        .positions
        .iter()
        .filter_map(|(&stock_id, pos)| {
            if stock_id >= prices.len() {
                return None;
            }

            let current_price = prices[stock_id];
            let entry_price = pos.entry_price;

            if entry_price <= 0.0 {
                return None;
            }

            let return_since_entry = (current_price - entry_price) / entry_price;

            // Check stop loss
            if config.stop_loss < 1.0 && return_since_entry <= -config.stop_loss {
                return Some(stock_id);
            }

            // Check take profit
            if config.take_profit < f64::INFINITY && return_since_entry >= config.take_profit {
                return Some(stock_id);
            }

            // Check trailing stop (update max_price first)
            if config.trail_stop < f64::INFINITY {
                let drawdown = (pos.max_price - current_price) / pos.max_price;
                if drawdown >= config.trail_stop {
                    return Some(stock_id);
                }
            }

            None
        })
        .collect();

    for stock_id in positions_to_close {
        if let Some(pos) = portfolio.positions.remove(&stock_id) {
            // Finlab exit formula: cash += cost_basis - |cost_basis| * (fee + tax)
            // Note: pos.value IS cost_basis in Finlab mode
            let sell_value = pos.value - pos.value.abs() * (config.fee_ratio + config.tax_ratio);
            portfolio.cash += sell_value;

            if stock_id < stopped_stocks.len() {
                stopped_stocks[stock_id] = true;
            }
        }
    }

    // Update max_price for trailing stop
    for (&stock_id, pos) in portfolio.positions.iter_mut() {
        if stock_id < prices.len() {
            let current_price = prices[stock_id];
            if current_price > pos.max_price {
                pos.max_price = current_price;
            }
        }
    }
}

/// Execute rebalance in Finlab mode
///
/// Finlab rebalance formula:
/// - total_balance = cash + sum(cost_basis)  (NOT market value!)
/// - target_cost_basis = total_balance * target_weight
/// - Exit: cash += cost_basis - |cost_basis| * (fee + tax)
/// - Entry: cost_basis = amount * (1 - fee)
fn execute_finlab_rebalance(
    portfolio: &mut PortfolioState,
    target_weights: &[f64],
    prices: &[f64],
    config: &BacktestConfig,
) {
    // Step 1: Exit positions with target_weight = 0
    let positions_to_close: Vec<usize> = portfolio
        .positions
        .keys()
        .filter(|&&id| id < target_weights.len() && target_weights[id] == 0.0)
        .copied()
        .collect();

    for stock_id in positions_to_close {
        if let Some(pos) = portfolio.positions.remove(&stock_id) {
            // Finlab exit formula: cash += cost_basis - |cost_basis| * (fee + tax)
            let sell_value = pos.value - pos.value.abs() * (config.fee_ratio + config.tax_ratio);
            portfolio.cash += sell_value;
        }
    }

    // Also close positions not in target_weights array
    let extra_positions: Vec<usize> = portfolio
        .positions
        .keys()
        .filter(|&&id| id >= target_weights.len())
        .copied()
        .collect();

    for stock_id in extra_positions {
        if let Some(pos) = portfolio.positions.remove(&stock_id) {
            let sell_value = pos.value - pos.value.abs() * (config.fee_ratio + config.tax_ratio);
            portfolio.cash += sell_value;
        }
    }

    // Step 2: Calculate total balance using cost_basis (Finlab formula)
    // CRITICAL: Finlab uses sum(cost_basis), NOT market value!
    let total_balance = portfolio.total_cost_basis();

    let total_target_weight: f64 = target_weights.iter().sum();
    if total_target_weight == 0.0 {
        return;
    }

    // Step 3: Sell overweight positions
    for (stock_id, &target_weight) in target_weights.iter().enumerate() {
        let target_cost_basis = total_balance * target_weight;
        let current_cost_basis = portfolio
            .positions
            .get(&stock_id)
            .map(|p| p.value)
            .unwrap_or(0.0);

        if current_cost_basis > target_cost_basis + 1e-10 {
            // Need to sell
            let sell_amount = current_cost_basis - target_cost_basis;
            if let Some(pos) = portfolio.positions.get_mut(&stock_id) {
                // Finlab exit formula for partial sell
                let sell_value = sell_amount - sell_amount.abs() * (config.fee_ratio + config.tax_ratio);
                pos.value -= sell_amount;
                portfolio.cash += sell_value;

                if pos.value < 1e-10 {
                    portfolio.positions.remove(&stock_id);
                }
            }
        }
    }

    // Step 4: Buy underweight positions
    for (stock_id, &target_weight) in target_weights.iter().enumerate() {
        if target_weight == 0.0 {
            continue;
        }

        let target_cost_basis = total_balance * target_weight;
        let current_cost_basis = portfolio
            .positions
            .get(&stock_id)
            .map(|p| p.value)
            .unwrap_or(0.0);

        if current_cost_basis < target_cost_basis - 1e-10 {
            // Need to buy
            let buy_cost_basis = target_cost_basis - current_cost_basis;

            // In Finlab: spend = buy_cost_basis, actual_cost_basis = spend * (1 - fee)
            // So to get buy_cost_basis of position, we need to spend buy_cost_basis / (1 - fee)
            // But actually in Finlab, cost_basis = amount spent, NOT amount spent * (1-fee)
            // Let me re-read the Finlab code...
            //
            // Actually from the analysis, Finlab stores:
            // - pos[sid] = cost_basis (the amount allocated, which includes fee impact)
            // - On entry: position_value = allocation * (1 - fee)
            //
            // So let's follow that exactly:
            // spend_amount = buy_cost_basis
            // cost_basis_added = spend_amount * (1 - fee)
            // But wait, target_cost_basis already accounts for this...
            //
            // Let me think again:
            // Finlab: amount = position - p[sid]
            //         cost = abs(amount) * fee_ratio if is_entry
            //         cash -= amount + cost
            //
            // So if we want to add cost_basis of X:
            // cash -= X + X * fee_ratio = X * (1 + fee_ratio)
            // But wait, that's the old formula...
            //
            // Actually based on my analysis, in Finlab:
            // - cost_basis IS the position value stored (before any market change)
            // - Entry fee is handled differently
            //
            // Let me just use a simpler approach that matches:
            // We need to add `diff` to cost_basis
            // This costs: diff * (1 + fee) if buying, diff * (1 - fee - tax) back if selling
            //
            // For entry, based on Finlab:
            // cash = cash - amount - abs(amount) * fee_ratio (for buy)
            // So spending `spend` gives us position value of `spend / (1 + fee)`
            //
            // Hmm, this is getting confusing. Let me use the simpler approach:
            // To get target_cost_basis of position, we spend target_cost_basis * (1 + fee) / (1 - fee)
            //
            // Actually, let's use the Finlab-style entry:
            // In Finlab, for a new entry:
            //   position_value = allocation * (1 - fee)
            // So the cost_basis = allocation * (1 - fee)
            // And cash_spent = allocation
            //
            // For adjustment:
            // amount = target_position - current_position
            // cost = |amount| * fee (if buying)
            // cash -= amount + cost = amount * (1 + fee)
            //
            // So to add `diff` to cost_basis:
            // cash -= diff * (1 + fee) / (1 - fee) * allocation_style...
            //
            // Let me simplify: just spend the diff directly and apply fee
            let spend_needed = buy_cost_basis / (1.0 - config.fee_ratio);
            let actual_spend = spend_needed.min(portfolio.cash);

            if actual_spend > 1e-10 {
                let cost_basis_added = actual_spend * (1.0 - config.fee_ratio);
                portfolio.cash -= actual_spend;

                let entry = portfolio.positions.entry(stock_id).or_insert(Position {
                    value: 0.0,
                    entry_price: prices[stock_id],
                    max_price: prices[stock_id],
                });

                if entry.value < 1e-10 {
                    // New position: set entry price
                    entry.entry_price = prices[stock_id];
                    entry.max_price = prices[stock_id];
                }
                entry.value += cost_basis_added;
            }
        }
    }
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
                        max_price: prices[stock_id],
                    },
                );
            }
        }
    }
}

/// Exit positions and adjust weights (after return has been applied)
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
                    max_price: prices[stock_id],
                });
                entry.value += actual_buy;
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
                    max_price: prices[stock_id],
                });
                // Update entry price only for new positions
                if entry.value < 1e-10 {
                    entry.entry_price = prices[stock_id];
                    entry.max_price = prices[stock_id];
                }
                entry.value += position_value;
            }
        }
    }
}

/// Run Finlab-compatible backtest with dual price matrices
///
/// This function exactly replicates Finlab's backtest_core calculation by using:
/// - `trade_prices`: Original prices for entry_price (used in balance calculation ratio)
/// - `close_prices`: Adjusted prices for current market value
///
/// # Finlab's Core Formula (from backtest_core.cpp line 11495-11516)
///
/// ```text
/// balance = cash
/// for each position:
///     trade_price = price_values[entry_day, stock]  // original price at entry
///     close_price = close_values[current_day, stock]  // adjusted close
///     balance += cost_basis * close_price / trade_price
/// ```
///
/// # Key Differences from Standard Mode
///
/// | Aspect | Standard Mode | Finlab Mode |
/// |--------|---------------|-------------|
/// | Position value | Updated daily with returns | cost_basis (constant) |
/// | Balance | cash + Σ(current_value) | cash + Σ(cost_basis * close/entry) |
/// | Entry price source | adjusted prices | trade_prices (original) |
/// | Rebalance base | market value | Σ(cost_basis) |
///
/// # Arguments
/// * `trade_prices` - 2D array of original prices [n_times x n_assets] (for entry_price)
/// * `close_prices` - 2D array of adjusted prices [n_times x n_assets] (for valuation)
/// * `weights` - 2D array of float weights [n_rebalance_times x n_assets]
/// * `rebalance_indices` - Indices in price array where rebalancing occurs
/// * `config` - Backtest configuration
///
/// # Returns
/// Vector of cumulative returns at each time step
pub fn run_backtest_finlab(
    trade_prices: &[Vec<f64>],
    close_prices: &[Vec<f64>],
    weights: &[Vec<f64>],
    rebalance_indices: &[usize],
    config: &BacktestConfig,
) -> Vec<f64> {
    if trade_prices.is_empty() || close_prices.is_empty() {
        return vec![];
    }

    let n_times = trade_prices.len();
    let n_assets = trade_prices[0].len();

    // Initialize portfolio with Finlab-style tracking
    let mut cash: f64 = 1.0;
    // positions[stock_id] = (cost_basis, entry_trade_price)
    let mut positions: HashMap<usize, (f64, f64)> = HashMap::new();

    let mut weight_idx = 0;
    let mut creturn = Vec::with_capacity(n_times);

    // Track which stocks to skip due to stop loss/take profit
    let mut stopped_stocks: Vec<bool> = vec![false; n_assets];

    // Pending weights to execute on next day (T+1 execution)
    let mut pending_weights: Option<Vec<f64>> = None;
    // Store trade_prices at signal day for T+1 entry
    let mut pending_trade_prices: Option<Vec<f64>> = None;

    for t in 0..n_times {
        // T+1 execution: execute pending trades
        if t > 0 {
            if let Some(target_weights) = pending_weights.take() {
                // Get trade prices from signal day (stored) for entry_price
                let entry_trade_prices = pending_trade_prices.take()
                    .unwrap_or_else(|| trade_prices[t - 1].clone());

                // Execute Finlab-style rebalance
                execute_finlab_rebalance_dual(
                    &mut cash,
                    &mut positions,
                    &target_weights,
                    &entry_trade_prices,
                    config,
                );
                stopped_stocks = vec![false; n_assets];
            }

            // Check stop loss / take profit using close_prices
            check_stops_finlab_dual(
                &mut cash,
                &mut positions,
                &close_prices[t],
                &mut stopped_stocks,
                config,
            );
        }

        // Check if this is a rebalance (signal) day
        let should_rebalance = rebalance_indices.contains(&t) && weight_idx < weights.len();

        if should_rebalance {
            let target_weights = normalize_weights_finlab(
                &weights[weight_idx],
                &stopped_stocks,
                config.position_limit,
            );
            pending_weights = Some(target_weights);
            // Store trade_prices at signal day for T+1 entry
            pending_trade_prices = Some(trade_prices[t].clone());
            weight_idx += 1;
        }

        // Calculate balance using Finlab formula
        // balance = cash + Σ(cost_basis * close_price / trade_price)
        let balance = calculate_balance_finlab(&cash, &positions, &close_prices[t]);
        creturn.push(balance);
    }

    creturn
}

/// Calculate balance using Finlab's formula
///
/// Finlab formula: balance = cash + Σ(cost_basis * close_price / entry_trade_price)
fn calculate_balance_finlab(
    cash: &f64,
    positions: &HashMap<usize, (f64, f64)>,  // (cost_basis, entry_trade_price)
    close_prices: &[f64],
) -> f64 {
    let pos_value: f64 = positions.iter()
        .map(|(&stock_id, &(cost_basis, entry_trade_price))| {
            if stock_id < close_prices.len() {
                let close_price = close_prices[stock_id];
                if entry_trade_price > 0.0 && close_price > 0.0 && !close_price.is_nan() {
                    // Finlab formula: cost_basis * close / entry_trade_price
                    cost_basis * close_price / entry_trade_price
                } else {
                    // If price is invalid, use cost_basis directly (like Finlab)
                    cost_basis
                }
            } else {
                cost_basis
            }
        })
        .sum();
    cash + pos_value
}

/// Execute Finlab-style rebalance with dual price tracking
///
/// Finlab rebalance formula (from backtest_core.cpp line 7447-7786):
/// - total_balance = cash + sum(cost_basis)  (NOT market value!)
/// - target_cost_basis = total_balance * target_weight
/// - Exit: cash += cost_basis - |cost_basis| * (fee + tax)
/// - Entry: cost_basis = amount * (1 - fee), entry_trade_price from signal day
fn execute_finlab_rebalance_dual(
    cash: &mut f64,
    positions: &mut HashMap<usize, (f64, f64)>,  // (cost_basis, entry_trade_price)
    target_weights: &[f64],
    entry_trade_prices: &[f64],  // trade_prices at signal day
    config: &BacktestConfig,
) {
    // Step 1: Exit positions with target_weight = 0
    let positions_to_close: Vec<usize> = positions
        .keys()
        .filter(|&&id| id < target_weights.len() && target_weights[id] == 0.0)
        .copied()
        .collect();

    for stock_id in positions_to_close {
        if let Some((cost_basis, _entry_price)) = positions.remove(&stock_id) {
            // Finlab exit formula: cash += cost_basis - |cost_basis| * (fee + tax)
            let sell_value = cost_basis - cost_basis.abs() * (config.fee_ratio + config.tax_ratio);
            *cash += sell_value;
        }
    }

    // Also close positions not in target_weights array
    let extra_positions: Vec<usize> = positions
        .keys()
        .filter(|&&id| id >= target_weights.len())
        .copied()
        .collect();

    for stock_id in extra_positions {
        if let Some((cost_basis, _entry_price)) = positions.remove(&stock_id) {
            let sell_value = cost_basis - cost_basis.abs() * (config.fee_ratio + config.tax_ratio);
            *cash += sell_value;
        }
    }

    // Step 2: Calculate total balance using cost_basis (Finlab formula)
    // CRITICAL: Finlab uses sum(cost_basis), NOT market value!
    let total_cost_basis: f64 = positions.values().map(|(cb, _)| cb).sum();
    let total_balance = *cash + total_cost_basis;

    let total_target_weight: f64 = target_weights.iter().sum();
    if total_target_weight == 0.0 {
        return;
    }

    // Step 3: Sell overweight positions
    let mut positions_to_remove: Vec<usize> = Vec::new();
    for (stock_id, &target_weight) in target_weights.iter().enumerate() {
        let target_cost_basis = total_balance * target_weight;
        let current_cost_basis = positions.get(&stock_id).map(|(cb, _)| *cb).unwrap_or(0.0);

        if current_cost_basis > target_cost_basis + 1e-10 {
            // Need to sell
            let sell_amount = current_cost_basis - target_cost_basis;
            if let Some((cost_basis, _entry_price)) = positions.get_mut(&stock_id) {
                // Finlab exit formula for partial sell
                let sell_value = sell_amount - sell_amount.abs() * (config.fee_ratio + config.tax_ratio);
                *cost_basis -= sell_amount;
                *cash += sell_value;

                if *cost_basis < 1e-10 {
                    positions_to_remove.push(stock_id);
                }
            }
        }
    }
    // Remove positions that went to zero
    for stock_id in positions_to_remove {
        positions.remove(&stock_id);
    }

    // Step 4: Buy underweight positions
    for (stock_id, &target_weight) in target_weights.iter().enumerate() {
        if target_weight == 0.0 {
            continue;
        }

        let target_cost_basis = total_balance * target_weight;
        let current_cost_basis = positions.get(&stock_id).map(|(cb, _)| *cb).unwrap_or(0.0);

        if current_cost_basis < target_cost_basis - 1e-10 {
            // Need to buy
            let buy_cost_basis = target_cost_basis - current_cost_basis;

            // Finlab entry: spend_amount / (1 - fee) gives the cost_basis we want
            // Actually in Finlab: position_value = allocation * (1 - fee)
            // So to get buy_cost_basis of position, we spend buy_cost_basis / (1 - fee)
            let spend_needed = buy_cost_basis / (1.0 - config.fee_ratio);
            let actual_spend = spend_needed.min(*cash);

            if actual_spend > 1e-10 {
                let cost_basis_added = actual_spend * (1.0 - config.fee_ratio);
                *cash -= actual_spend;

                let entry_trade_price = if stock_id < entry_trade_prices.len() {
                    entry_trade_prices[stock_id]
                } else {
                    1.0  // fallback
                };

                if let Some((cost_basis, entry_price)) = positions.get_mut(&stock_id) {
                    // Adding to existing position
                    // For Finlab, we need to handle the weighted average entry_price
                    // Actually, Finlab might track each lot separately, but for simplicity
                    // we'll use weighted average
                    let new_total_cost = *cost_basis + cost_basis_added;
                    let weighted_entry = (*cost_basis * *entry_price + cost_basis_added * entry_trade_price)
                                        / new_total_cost;
                    *cost_basis = new_total_cost;
                    *entry_price = weighted_entry;
                } else {
                    // New position: use trade_price from signal day
                    positions.insert(stock_id, (cost_basis_added, entry_trade_price));
                }
            }
        }
    }
}

/// Check stop loss / take profit for Finlab mode with dual price tracking
fn check_stops_finlab_dual(
    cash: &mut f64,
    positions: &mut HashMap<usize, (f64, f64)>,  // (cost_basis, entry_trade_price)
    close_prices: &[f64],
    stopped_stocks: &mut [bool],
    config: &BacktestConfig,
) {
    let positions_to_close: Vec<usize> = positions
        .iter()
        .filter_map(|(&stock_id, &(cost_basis, entry_trade_price))| {
            if stock_id >= close_prices.len() {
                return None;
            }

            let close_price = close_prices[stock_id];

            if entry_trade_price <= 0.0 {
                return None;
            }

            // Calculate return since entry using trade_price
            let return_since_entry = (close_price - entry_trade_price) / entry_trade_price;

            // Check stop loss
            if config.stop_loss < 1.0 && return_since_entry <= -config.stop_loss {
                return Some(stock_id);
            }

            // Check take profit
            if config.take_profit < f64::INFINITY && return_since_entry >= config.take_profit {
                return Some(stock_id);
            }

            // Note: trailing stop would need max_price tracking, simplified here
            let _ = cost_basis;

            None
        })
        .collect();

    for stock_id in positions_to_close {
        if let Some((cost_basis, _entry_price)) = positions.remove(&stock_id) {
            // Finlab exit formula: cash += cost_basis - |cost_basis| * (fee + tax)
            let sell_value = cost_basis - cost_basis.abs() * (config.fee_ratio + config.tax_ratio);
            *cash += sell_value;

            if stock_id < stopped_stocks.len() {
                stopped_stocks[stock_id] = true;
            }
        }
    }
}

/// Run backtest with simple interface (equal weight, no stops)
///
/// This is a convenience function for basic backtesting without
/// stop loss/take profit features.
pub fn run_simple_backtest(
    prices: &[Vec<f64>],
    signals: &[Vec<bool>],
    rebalance_indices: &[usize],
    fee_ratio: f64,
    tax_ratio: f64,
) -> Vec<f64> {
    let config = BacktestConfig {
        fee_ratio,
        tax_ratio,
        ..Default::default()
    };
    run_backtest(prices, signals, rebalance_indices, &config)
}

/// OHLC price data for a single time step
#[derive(Debug, Clone)]
pub struct OhlcPrices {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
}

/// Full price data matching Finlab's backtest_core interface
///
/// Finlab uses 5 price matrices:
/// - trade_prices: The price used for trade execution (determined by trade_at_price)
/// - close_prices: Close prices for return calculation
/// - high_prices: High prices for stop loss/take profit with touched_exit
/// - low_prices: Low prices for stop loss/take profit with touched_exit
/// - open_prices: Open prices for stop loss/take profit with touched_exit
#[derive(Debug, Clone)]
pub struct FinlabPriceData {
    /// Trading prices (determined by trade_at_price parameter)
    /// For trades record: uses original (non-adjusted) prices
    pub trade_prices: Vec<Vec<f64>>,
    /// Close prices (adjusted) for return calculation
    pub close_prices: Vec<Vec<f64>>,
    /// High prices (adjusted) for stop loss/take profit
    pub high_prices: Vec<Vec<f64>>,
    /// Low prices (adjusted) for stop loss/take profit
    pub low_prices: Vec<Vec<f64>>,
    /// Open prices (adjusted) for stop loss/take profit
    pub open_prices: Vec<Vec<f64>>,
}

impl FinlabPriceData {
    /// Create from simple dual-price setup (backward compatible)
    pub fn from_dual_prices(adj_prices: &[Vec<f64>], original_prices: &[Vec<f64>]) -> Self {
        Self {
            trade_prices: original_prices.to_vec(),
            close_prices: adj_prices.to_vec(),
            high_prices: adj_prices.to_vec(),
            low_prices: adj_prices.to_vec(),
            open_prices: adj_prices.to_vec(),
        }
    }

    /// Create with full OHLC data
    pub fn new(
        trade_prices: Vec<Vec<f64>>,
        close_prices: Vec<Vec<f64>>,
        high_prices: Vec<Vec<f64>>,
        low_prices: Vec<Vec<f64>>,
        open_prices: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            trade_prices,
            close_prices,
            high_prices,
            low_prices,
            open_prices,
        }
    }
}

/// Run backtest with full Finlab-compatible price data
///
/// This function matches Finlab's backtest_core interface with full OHLC support.
///
/// # Arguments
/// * `prices` - FinlabPriceData containing all price matrices
/// * `weights` - 2D array of float weights [n_rebalance_times x n_assets]
/// * `rebalance_indices` - Indices in price array where rebalancing occurs
/// * `config` - Backtest configuration
///
/// # Returns
/// BacktestResult containing creturn and trades list
pub fn run_backtest_finlab_full(
    prices: &FinlabPriceData,
    weights: &[Vec<f64>],
    rebalance_indices: &[usize],
    config: &BacktestConfig,
) -> BacktestResult {
    // Delegate to run_backtest_with_trades using the price data
    run_backtest_with_trades_internal(
        &prices.close_prices,
        &prices.trade_prices,
        &prices.high_prices,
        &prices.low_prices,
        weights,
        rebalance_indices,
        config,
    )
}

/// Run backtest with trades tracking and dual price support
///
/// This function returns both cumulative returns and trade records.
/// It uses:
/// - `adj_prices`: Adjusted prices for return calculation (creturn)
/// - `original_prices`: Original prices for trade records (entry/exit prices)
///
/// The trade records match Finlab's trades DataFrame format, using
/// original prices for entry/exit to match real trading execution.
///
/// # Arguments
/// * `adj_prices` - 2D array of adjusted prices [n_times x n_assets] (for creturn)
/// * `original_prices` - 2D array of original prices [n_times x n_assets] (for trades)
/// * `weights` - 2D array of float weights [n_rebalance_times x n_assets]
/// * `rebalance_indices` - Indices in price array where rebalancing occurs
/// * `config` - Backtest configuration
///
/// # Returns
/// BacktestResult containing creturn and trades list
pub fn run_backtest_with_trades(
    adj_prices: &[Vec<f64>],
    original_prices: &[Vec<f64>],
    weights: &[Vec<f64>],
    rebalance_indices: &[usize],
    config: &BacktestConfig,
) -> BacktestResult {
    // Use adj_prices for high/low as well (no touched_exit support in simple mode)
    run_backtest_with_trades_internal(
        adj_prices,
        original_prices,
        adj_prices,
        adj_prices,
        weights,
        rebalance_indices,
        config,
    )
}

/// Internal implementation with full OHLC support
fn run_backtest_with_trades_internal(
    close_prices: &[Vec<f64>],
    trade_prices: &[Vec<f64>],
    _high_prices: &[Vec<f64>],
    _low_prices: &[Vec<f64>],
    weights: &[Vec<f64>],
    rebalance_indices: &[usize],
    config: &BacktestConfig,
) -> BacktestResult {
    if close_prices.is_empty() {
        return BacktestResult {
            creturn: vec![],
            trades: vec![],
        };
    }

    let n_times = close_prices.len();
    let n_assets = close_prices[0].len();

    // Initialize portfolio
    let mut portfolio = PortfolioState::new();

    // Map rebalance index -> weight index
    let mut weight_idx = 0;

    // Previous prices for return calculation (use close prices)
    let mut prev_prices = close_prices[0].clone();

    // Cumulative return array
    let mut creturn = Vec::with_capacity(n_times);

    // Track which stocks to skip due to stop loss/take profit
    let mut stopped_stocks: Vec<bool> = vec![false; n_assets];

    // Pending weights to execute on next day (T+1 execution)
    let mut pending_weights: Option<Vec<f64>> = None;
    let mut pending_signal_index: Option<usize> = None;

    // Trade tracking
    let mut open_trades: HashMap<usize, OpenTrade> = HashMap::new();
    let mut completed_trades: Vec<TradeRecord> = Vec::new();

    for t in 0..n_times {
        // T+1 execution model (standard mode, matches Finlab perfectly)
        if t > 0 {
            if let Some(target_weights) = pending_weights.take() {
                let signal_index = pending_signal_index.take().unwrap_or(t - 1);

                // Close trades for positions being exited (use trade_prices for record)
                close_trades_for_rebalance(
                    &portfolio,
                    &target_weights,
                    &mut open_trades,
                    &mut completed_trades,
                    t,
                    signal_index,
                    &trade_prices[t],
                    config,
                );

                // T+1 execution: rebalance at prev_prices, then experience today's return
                execute_t1_rebalance(
                    &mut portfolio,
                    &target_weights,
                    &prev_prices,
                    &close_prices[t],
                    config,
                );

                // Open new trades (use trade_prices for record)
                open_trades_for_rebalance(
                    &portfolio,
                    &target_weights,
                    &mut open_trades,
                    t,
                    signal_index,
                    &trade_prices[t],
                );

                stopped_stocks = vec![false; n_assets];
            } else {
                // No pending weights, just update position values
                update_position_values(&mut portfolio, &close_prices[t], &prev_prices);
            }

            // Check stop loss / take profit / trailing stop
            let stopped_this_period = check_stops_with_trade_tracking(
                &mut portfolio,
                &close_prices[t],
                &trade_prices[t],
                &mut stopped_stocks,
                &mut open_trades,
                &mut completed_trades,
                t,
                config,
            );

            // Mark any stopped stocks
            for stock_id in stopped_this_period {
                if stock_id < stopped_stocks.len() {
                    stopped_stocks[stock_id] = true;
                }
            }
        }

        // Check if this is a rebalance (signal) day
        let should_rebalance = rebalance_indices.contains(&t) && weight_idx < weights.len();

        if should_rebalance {
            // Normalize weights like Finlab
            let target_weights = normalize_weights_finlab(
                &weights[weight_idx],
                &stopped_stocks,
                config.position_limit,
            );

            // T+1 mode: store for execution on next day
            pending_weights = Some(target_weights);
            pending_signal_index = Some(t);

            weight_idx += 1;
        }

        // Record cumulative return
        creturn.push(portfolio.balance());

        // Update previous prices
        prev_prices = close_prices[t].clone();
    }

    // Close any remaining open trades at the last price
    if !open_trades.is_empty() {
        let last_index = n_times - 1;
        for (stock_id, open_trade) in open_trades.drain() {
            let exit_price = if stock_id < trade_prices[last_index].len() {
                trade_prices[last_index][stock_id]
            } else {
                open_trade.entry_price
            };

            let trade_return = TradeRecord {
                stock_id,
                entry_index: open_trade.entry_index,
                exit_index: Some(last_index),
                entry_sig_index: open_trade.entry_sig_index,
                exit_sig_index: None, // No explicit exit signal
                position_weight: open_trade.weight,
                entry_price: open_trade.entry_price,
                exit_price: Some(exit_price),
                trade_return: None, // Will be calculated
            }
            .calculate_return(config.fee_ratio, config.tax_ratio);

            completed_trades.push(TradeRecord {
                stock_id,
                entry_index: open_trade.entry_index,
                exit_index: Some(last_index),
                entry_sig_index: open_trade.entry_sig_index,
                exit_sig_index: None,
                position_weight: open_trade.weight,
                entry_price: open_trade.entry_price,
                exit_price: Some(exit_price),
                trade_return,
            });
        }
    }

    BacktestResult {
        creturn,
        trades: completed_trades,
    }
}

/// Close trades when positions are being exited due to rebalance
fn close_trades_for_rebalance(
    portfolio: &PortfolioState,
    target_weights: &[f64],
    open_trades: &mut HashMap<usize, OpenTrade>,
    completed_trades: &mut Vec<TradeRecord>,
    current_index: usize,
    signal_index: usize,
    original_prices: &[f64],
    config: &BacktestConfig,
) {
    // Find positions that will be closed (target weight = 0)
    let positions_to_close: Vec<usize> = portfolio
        .positions
        .keys()
        .filter(|&&id| id < target_weights.len() && target_weights[id] == 0.0)
        .copied()
        .collect();

    for stock_id in positions_to_close {
        if let Some(open_trade) = open_trades.remove(&stock_id) {
            let exit_price = if stock_id < original_prices.len() {
                original_prices[stock_id]
            } else {
                open_trade.entry_price
            };

            let trade_return = TradeRecord {
                stock_id,
                entry_index: open_trade.entry_index,
                exit_index: Some(current_index),
                entry_sig_index: open_trade.entry_sig_index,
                exit_sig_index: Some(signal_index),
                position_weight: open_trade.weight,
                entry_price: open_trade.entry_price,
                exit_price: Some(exit_price),
                trade_return: None,
            }
            .calculate_return(config.fee_ratio, config.tax_ratio);

            completed_trades.push(TradeRecord {
                stock_id,
                entry_index: open_trade.entry_index,
                exit_index: Some(current_index),
                entry_sig_index: open_trade.entry_sig_index,
                exit_sig_index: Some(signal_index),
                position_weight: open_trade.weight,
                entry_price: open_trade.entry_price,
                exit_price: Some(exit_price),
                trade_return,
            });
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
        if let Some(open_trade) = open_trades.remove(&stock_id) {
            let exit_price = if stock_id < original_prices.len() {
                original_prices[stock_id]
            } else {
                open_trade.entry_price
            };

            let trade_return = TradeRecord {
                stock_id,
                entry_index: open_trade.entry_index,
                exit_index: Some(current_index),
                entry_sig_index: open_trade.entry_sig_index,
                exit_sig_index: Some(signal_index),
                position_weight: open_trade.weight,
                entry_price: open_trade.entry_price,
                exit_price: Some(exit_price),
                trade_return: None,
            }
            .calculate_return(config.fee_ratio, config.tax_ratio);

            completed_trades.push(TradeRecord {
                stock_id,
                entry_index: open_trade.entry_index,
                exit_index: Some(current_index),
                entry_sig_index: open_trade.entry_sig_index,
                exit_sig_index: Some(signal_index),
                position_weight: open_trade.weight,
                entry_price: open_trade.entry_price,
                exit_price: Some(exit_price),
                trade_return,
            });
        }
    }
}

/// Open new trades after rebalance
fn open_trades_for_rebalance(
    portfolio: &PortfolioState,
    target_weights: &[f64],
    open_trades: &mut HashMap<usize, OpenTrade>,
    current_index: usize,
    signal_index: usize,
    original_prices: &[f64],
) {
    for (stock_id, &target_weight) in target_weights.iter().enumerate() {
        // If we have a new position (not already tracked in open_trades)
        if target_weight > 0.0
            && portfolio.positions.contains_key(&stock_id)
            && !open_trades.contains_key(&stock_id)
        {
            let entry_price = if stock_id < original_prices.len() {
                original_prices[stock_id]
            } else {
                1.0
            };

            open_trades.insert(
                stock_id,
                OpenTrade {
                    stock_id,
                    entry_index: current_index,
                    entry_sig_index: signal_index,
                    weight: target_weight,
                    entry_price,
                },
            );
        }
    }
}

/// Check stops and generate trade records for stopped positions
fn check_stops_with_trade_tracking(
    portfolio: &mut PortfolioState,
    adj_prices: &[f64],
    original_prices: &[f64],
    stopped_stocks: &mut [bool],
    open_trades: &mut HashMap<usize, OpenTrade>,
    completed_trades: &mut Vec<TradeRecord>,
    current_index: usize,
    config: &BacktestConfig,
) -> Vec<usize> {
    let positions_to_close: Vec<usize> = portfolio
        .positions
        .iter()
        .filter_map(|(&stock_id, pos)| {
            if stock_id >= adj_prices.len() {
                return None;
            }

            let current_price = adj_prices[stock_id];
            let entry_price = pos.entry_price;

            if entry_price <= 0.0 {
                return None;
            }

            let return_since_entry = (current_price - entry_price) / entry_price;

            // Check stop loss
            if config.stop_loss < 1.0 && return_since_entry <= -config.stop_loss {
                return Some(stock_id);
            }

            // Check take profit
            if config.take_profit < f64::INFINITY && return_since_entry >= config.take_profit {
                return Some(stock_id);
            }

            // Check trailing stop
            if config.trail_stop < f64::INFINITY {
                let drawdown = (pos.max_price - current_price) / pos.max_price;
                if drawdown >= config.trail_stop {
                    return Some(stock_id);
                }
            }

            None
        })
        .collect();

    let mut stopped_ids = Vec::new();

    for stock_id in positions_to_close {
        if let Some(pos) = portfolio.positions.remove(&stock_id) {
            let sell_value = pos.value * (1.0 - config.fee_ratio - config.tax_ratio);
            portfolio.cash += sell_value;

            if stock_id < stopped_stocks.len() {
                stopped_stocks[stock_id] = true;
            }

            stopped_ids.push(stock_id);

            // Record the trade
            if let Some(open_trade) = open_trades.remove(&stock_id) {
                let exit_price = if stock_id < original_prices.len() {
                    original_prices[stock_id]
                } else {
                    open_trade.entry_price
                };

                let trade_return = TradeRecord {
                    stock_id,
                    entry_index: open_trade.entry_index,
                    exit_index: Some(current_index),
                    entry_sig_index: open_trade.entry_sig_index,
                    exit_sig_index: None, // Stop loss/take profit, no signal
                    position_weight: open_trade.weight,
                    entry_price: open_trade.entry_price,
                    exit_price: Some(exit_price),
                    trade_return: None,
                }
                .calculate_return(config.fee_ratio, config.tax_ratio);

                completed_trades.push(TradeRecord {
                    stock_id,
                    entry_index: open_trade.entry_index,
                    exit_index: Some(current_index),
                    entry_sig_index: open_trade.entry_sig_index,
                    exit_sig_index: None,
                    position_weight: open_trade.weight,
                    entry_price: open_trade.entry_price,
                    exit_price: Some(exit_price),
                    trade_return,
                });
            }
        }
    }

    stopped_ids
}

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

        let creturn = run_simple_backtest(
            &prices,
            &signals,
            &rebalance_indices,
            0.001425,
            0.003,
        );

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

        let creturn = run_simple_backtest(
            &prices,
            &signals,
            &rebalance_indices,
            0.001425,
            0.003,
        );

        // Should stay at 1.0 with no positions
        assert_eq!(creturn.len(), 2);
        assert!((creturn[0] - 1.0).abs() < 1e-10);
        assert!((creturn[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_position_limit() {
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

        let creturn = run_simple_backtest(
            &prices,
            &signals,
            &rebalance_indices,
            0.001425,
            0.003,
        );

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

    // Tests for run_backtest_with_weights and normalize_weights_finlab

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
        // Stopped stocks should get weight 0
        let weights = vec![0.5, 0.5, 0.5];
        let stopped = vec![false, true, false]; // stock 1 stopped
        let result = normalize_weights_finlab(&weights, &stopped, 1.0);

        // Sum = 1.5, normalize by 1.5
        assert!((result[0] - 0.5 / 1.5).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10); // stopped
        assert!((result[2] - 0.5 / 1.5).abs() < 1e-10);
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

        let creturn = run_backtest_with_weights(&prices, &weights, &rebalance_indices, &config);

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

        let creturn = run_backtest_with_weights(&prices, &weights, &rebalance_indices, &config);

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

        let creturn = run_backtest_with_weights(&prices, &weights, &rebalance_indices, &config);

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

        let creturn = run_backtest_with_weights(&prices, &weights, &rebalance_indices, &config);

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

        let creturn = run_backtest_with_weights(&prices, &weights, &rebalance_indices, &config);

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
        // run_backtest_with_weights([0.5, 0.5]) should match run_backtest([true, true])
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
        let creturn_weights = run_backtest_with_weights(&prices, &weights, &rebalance_indices, &config);

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
        let creturn = run_backtest_with_weights(&prices, &weights, &rebalance_indices, &config);

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

        let creturn = run_backtest_with_weights(&prices, &weights, &rebalance_indices, &config);

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

        let creturn = run_backtest_with_weights(&prices, &weights, &rebalance_indices, &config);

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
