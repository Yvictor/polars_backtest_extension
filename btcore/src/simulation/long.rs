//! Arrow-based zero-copy long format backtest engine
//!
//! This module provides a backtest implementation that processes Arrow arrays directly,
//! avoiding data encoding/copying overhead. It uses string keys for positions, enabling
//! true zero-copy data access from Polars/Arrow DataFrames.
//!
//! # Performance
//!
//! By using Arrow arrays and string keys directly:
//! - No need to encode dates/symbols to indices
//! - Zero-copy access to price/weight data
//! - Only processes active stocks per day O(k) instead of O(n_stocks)

use std::collections::HashMap;

use arrow::array::{Float64Array, Int32Array, StringViewArray};

use crate::config::BacktestConfig;
use crate::position::Position;
use crate::tracker::{BacktestResult, LongBacktestResult, LongTradeRecord, NoopSymbolTracker, SymbolTracker, TradeTracker};

/// Portfolio with string symbol keys (for zero-copy backtest)
pub struct Portfolio {
    pub cash: f64,
    pub positions: HashMap<String, Position>,
}

impl Portfolio {
    pub fn new() -> Self {
        Self {
            cash: 1.0,
            positions: HashMap::new(),
        }
    }

    /// Calculate total balance (cash + position market values)
    pub fn balance(&self) -> f64 {
        self.cash + self.positions.values().map(|p| p.last_market_value).sum::<f64>()
    }
}

impl Default for Portfolio {
    fn default() -> Self {
        Self::new()
    }
}

/// Arrow-based long format backtest input
///
/// All arrays must have the same length and be sorted by date.
/// Uses i32 for dates (days since epoch) to match Polars Date type.
/// Uses StringViewArray for symbols to match Polars string type (zero-copy from polars-arrow).
pub struct LongFormatArrowInput<'a> {
    /// Date as i32 (days since epoch, sorted ascending)
    pub dates: &'a Int32Array,
    /// Symbol strings (StringViewArray for zero-copy from polars)
    pub symbols: &'a StringViewArray,
    /// Close prices
    pub prices: &'a Float64Array,
    /// Target weights
    pub weights: &'a Float64Array,
}

/// Resample frequency for rebalancing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleFreq {
    /// Daily rebalancing
    Daily,
    /// Weekly rebalancing (end of week)
    Weekly,
    /// Monthly rebalancing (end of month)
    Monthly,
    /// Only rebalance when position changes (Finlab default, resample=None)
    PositionChange,
}

impl ResampleFreq {
    /// Parse from string (like Polars/Pandas resample)
    pub fn from_str(s: Option<&str>) -> Self {
        match s {
            Some("M") | Some("ME") => Self::Monthly,
            Some("W") | Some("W-FRI") => Self::Weekly,
            Some("D") => Self::Daily,
            None => Self::PositionChange,
            _ => Self::Daily,
        }
    }
}

/// Core backtest implementation with closure-based data access and generic tracker
///
/// This is the unified core that all backtest functions use.
/// Uses i32 dates for efficient month-end detection without string parsing.
///
/// # Type Parameters
/// * `T` - TradeTracker implementation (NoopSymbolTracker for no tracking, SymbolTracker for trade tracking)
///
/// # Arguments
/// * `n_rows` - Total number of rows
/// * `get_date` - Closure to get date (days since epoch) at index
/// * `get_symbol` - Closure to get symbol string at index
/// * `get_price` - Closure to get price at index
/// * `get_weight` - Closure to get weight at index (NaN = no signal)
/// * `resample` - Rebalancing frequency
/// * `config` - Backtest configuration
/// * `tracker` - Trade tracker (use NoopSymbolTracker::default() for no tracking)
fn backtest_impl<'a, FD, FS, FP, FW, T>(
    n_rows: usize,
    get_date: FD,
    get_symbol: FS,
    get_price: FP,
    get_weight: FW,
    resample: ResampleFreq,
    config: &BacktestConfig,
    tracker: &mut T,
) -> Vec<f64>
where
    FD: Fn(usize) -> i32,
    FS: Fn(usize) -> &'a str,
    FP: Fn(usize) -> f64,
    FW: Fn(usize) -> f64,
    T: TradeTracker<Key = String, Date = i32, Record = LongTradeRecord>,
{
    if n_rows == 0 {
        return vec![];
    }

    let mut portfolio = Portfolio::new();
    let mut creturn: Vec<f64> = Vec::new();
    let mut stopped_stocks: HashMap<String, bool> = HashMap::new();
    let mut pending_weights: Option<HashMap<String, f64>> = None;
    let mut pending_signal_date: Option<i32> = None;
    let mut pending_stop_exits: Vec<String> = Vec::new();
    let mut active_weights: HashMap<String, f64> = HashMap::new();
    let mut has_first_signal = false;
    let mut position_changed = false;
    let mut current_date: Option<i32> = None;
    let mut today_prices: HashMap<&str, f64> = HashMap::new();
    let mut today_weights: HashMap<&str, f64> = HashMap::new();

    for i in 0..n_rows {
        let date = get_date(i);
        let symbol = get_symbol(i);
        let price = get_price(i);
        let weight = get_weight(i);

        let date_changed = current_date.map_or(true, |d| d != date);

        if date_changed && current_date.is_some() {
            let prev_date = current_date.unwrap();

            // STEP 1: Update positions
            update_positions(&mut portfolio, &today_prices);

            // STEP 2: Execute pending stops (from yesterday's detection)
            execute_pending_stops_impl(
                &mut portfolio,
                &mut pending_stop_exits,
                &mut stopped_stocks,
                &today_prices,
                config,
                prev_date,
                tracker,
            );

            // STEP 2.5: Detect new stops (schedule for tomorrow's execution)
            {
                let new_stops = detect_stops_string(&portfolio, &today_prices, config);
                pending_stop_exits.extend(new_stops);
            }

            // STEP 3: Execute pending rebalance
            if let Some(target_weights) = pending_weights.take() {
                let sig_date = pending_signal_date.take().unwrap_or(prev_date);
                execute_rebalance_impl(
                    &mut portfolio,
                    &target_weights,
                    &today_prices,
                    &stopped_stocks,
                    config,
                    prev_date,
                    sig_date,
                    tracker,
                );
                stopped_stocks.clear();
            }

            // STEP 4: Update active weights based on today's signals
            let normalized = normalize_weights(
                &today_weights,
                &stopped_stocks,
                config.position_limit,
            );

            // Check if prev_date is a rebalance boundary
            let is_month_end = is_month_end_i32(prev_date, date);
            let is_week_end = is_week_end_i32(prev_date, date);

            match resample {
                ResampleFreq::Daily => {
                    if !normalized.is_empty() {
                        has_first_signal = true;
                    }
                    if has_first_signal {
                        active_weights = normalized;
                    }
                }
                ResampleFreq::Weekly => {
                    if is_week_end {
                        let has_signals = !normalized.is_empty();
                        active_weights = normalized;
                        if has_signals {
                            has_first_signal = true;
                        }
                    }
                }
                ResampleFreq::Monthly => {
                    if is_month_end {
                        let has_signals = !normalized.is_empty();
                        active_weights = normalized;
                        if has_signals {
                            has_first_signal = true;
                        }
                    }
                }
                ResampleFreq::PositionChange => {
                    if !normalized.is_empty() {
                        has_first_signal = true;
                    }
                    position_changed = weights_differ(&active_weights, &normalized);
                    active_weights = normalized;
                }
            }

            // STEP 5: Check rebalance (only after first signal)
            if has_first_signal {
                let should_rebalance = match resample {
                    ResampleFreq::Monthly => is_month_end,
                    ResampleFreq::Weekly => is_week_end,
                    ResampleFreq::Daily => true,
                    ResampleFreq::PositionChange => position_changed,
                };

                if should_rebalance {
                    pending_weights = Some(active_weights.clone());
                    pending_signal_date = Some(prev_date);
                }
            }

            // STEP 6: Record creturn
            creturn.push(portfolio.balance());

            today_prices.clear();
            today_weights.clear();
        }

        current_date = Some(date);
        if price > 0.0 && !price.is_nan() {
            today_prices.insert(symbol, price);
        }
        if !weight.is_nan() && weight.abs() > 1e-10 {
            today_weights.insert(symbol, weight);
        }
    }

    // Final day
    if let Some(last_date) = current_date {
        if !today_prices.is_empty() {
            update_positions(&mut portfolio, &today_prices);

            execute_pending_stops_impl(
                &mut portfolio,
                &mut pending_stop_exits,
                &mut stopped_stocks,
                &today_prices,
                config,
                last_date,
                tracker,
            );

            if let Some(target_weights) = pending_weights.take() {
                let sig_date = pending_signal_date.take().unwrap_or(last_date);
                execute_rebalance_impl(
                    &mut portfolio,
                    &target_weights,
                    &today_prices,
                    &stopped_stocks,
                    config,
                    last_date,
                    sig_date,
                    tracker,
                );
            }

            creturn.push(portfolio.balance());
        }
    }

    creturn
}

/// Run backtest with closure-based data access (public API, no trade tracking)
///
/// # Arguments
/// * `n_rows` - Total number of rows
/// * `get_date` - Closure to get date (days since epoch) at index
/// * `get_symbol` - Closure to get symbol string at index
/// * `get_price` - Closure to get price at index
/// * `get_weight` - Closure to get weight at index (NaN = no signal)
/// * `resample` - Rebalancing frequency
/// * `config` - Backtest configuration
pub fn backtest_with_accessor<'a, FD, FS, FP, FW>(
    n_rows: usize,
    get_date: FD,
    get_symbol: FS,
    get_price: FP,
    get_weight: FW,
    resample: ResampleFreq,
    config: &BacktestConfig,
) -> BacktestResult
where
    FD: Fn(usize) -> i32,
    FS: Fn(usize) -> &'a str,
    FP: Fn(usize) -> f64,
    FW: Fn(usize) -> f64,
{
    let mut tracker = NoopSymbolTracker::default();
    let creturn = backtest_impl(
        n_rows,
        get_date,
        get_symbol,
        get_price,
        get_weight,
        resample,
        config,
        &mut tracker,
    );
    BacktestResult {
        creturn,
        trades: vec![],
    }
}

/// Run backtest on Arrow arrays with zero-copy access
///
/// Delegates to `backtest_with_accessor` using Arrow array closures.
///
/// # Arguments
/// * `input` - Arrow arrays containing long format data (must be sorted by date)
/// * `resample` - Rebalancing frequency
/// * `config` - Backtest configuration
///
/// # Returns
/// BacktestResult containing cumulative returns (one per unique date)
pub fn backtest_long_arrow(
    input: &LongFormatArrowInput,
    resample: ResampleFreq,
    config: &BacktestConfig,
) -> BacktestResult {
    backtest_with_accessor(
        input.dates.len(),
        |i| input.dates.value(i),
        |i| input.symbols.value(i),
        |i| input.prices.value(i),
        |i| input.weights.value(i),
        resample,
        config,
    )
}

/// Run backtest on native Rust slices
pub fn backtest_long_slice(
    dates: &[i32],
    symbols: &[&str],
    prices: &[f64],
    weights: &[f64],
    resample: ResampleFreq,
    config: &BacktestConfig,
) -> BacktestResult {
    backtest_with_accessor(
        dates.len(),
        |i| dates[i],
        |i| symbols[i],
        |i| prices[i],
        |i| weights[i],
        resample,
        config,
    )
}

/// Check if prev_date is a month-end using i32 dates (days since 1970-01-01)
///
/// Returns true if prev_date and next_date are in different months
fn is_month_end_i32(prev_days: i32, next_days: i32) -> bool {
    // Convert days since epoch to (year, month)
    let prev_ym = days_to_year_month(prev_days);
    let next_ym = days_to_year_month(next_days);
    prev_ym != next_ym
}

/// Check if prev_date is a week-end using i32 dates (days since 1970-01-01)
///
/// Returns true if prev_date and next_date are in different ISO weeks.
/// Week starts on Monday (ISO 8601).
fn is_week_end_i32(prev_days: i32, next_days: i32) -> bool {
    // 1970-01-01 was a Thursday (weekday 3, where Monday=0)
    // Calculate ISO week number for each date
    let prev_week = (prev_days + 3) / 7; // +3 to shift Thursday to Sunday position
    let next_week = (next_days + 3) / 7;
    prev_week != next_week
}

/// Convert days since 1970-01-01 to (year, month)
#[inline]
fn days_to_year_month(days: i32) -> (i32, u32) {
    // Algorithm from Howard Hinnant's date library
    // https://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468; // shift epoch from 1970-01-01 to 0000-03-01
    let era = if z >= 0 { z / 146097 } else { (z - 146096) / 146097 };
    let doe = (z - era * 146097) as u32; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // year of era [0, 399]
    let y = yoe as i32 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year [0, 365]
    let mp = (5 * doy + 2) / 153; // month offset from March [0, 11]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // actual month [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    (year, m)
}

/// Update positions with daily returns
fn update_positions(portfolio: &mut Portfolio, prices: &HashMap<&str, f64>) {
    for (sym, pos) in portfolio.positions.iter_mut() {
        if let Some(&curr_price) = prices.get(sym.as_str()) {
            if curr_price > 0.0 && !curr_price.is_nan() {
                if pos.previous_price > 0.0 {
                    let r = curr_price / pos.previous_price;
                    pos.cr *= r;
                    pos.last_market_value *= r;
                }
                if curr_price > pos.max_price {
                    pos.max_price = curr_price;
                }
                pos.maxcr = pos.maxcr.max(pos.cr);
                pos.previous_price = curr_price;
            }
        }
    }
}

/// Execute pending stop exits with optional trade tracking
///
/// This is the unified implementation that supports both:
/// - No tracking: Use `NoopSymbolTracker`
/// - Trade tracking: Use `SymbolTracker`
fn execute_pending_stops_impl<T>(
    portfolio: &mut Portfolio,
    pending_stops: &mut Vec<String>,
    stopped_stocks: &mut HashMap<String, bool>,
    today_prices: &HashMap<&str, f64>,
    config: &BacktestConfig,
    current_date: i32,
    tracker: &mut T,
)
where
    T: TradeTracker<Key = String, Date = i32, Record = LongTradeRecord>,
{
    for sym in pending_stops.drain(..) {
        if let Some(pos) = portfolio.positions.remove(&sym) {
            let exit_price = today_prices.get(sym.as_str()).copied().unwrap_or(pos.previous_price);
            tracker.close_trade(
                &sym,
                current_date,
                None, // Stop exit, no signal date
                exit_price,
                config.fee_ratio,
                config.tax_ratio,
            );
            let sell_value =
                pos.last_market_value - pos.last_market_value.abs() * (config.fee_ratio + config.tax_ratio);
            portfolio.cash += sell_value;
            if config.stop_trading_next_period {
                stopped_stocks.insert(sym, true);
            }
        }
    }
}

/// Detect stops for string-keyed positions (Finlab mode)
///
/// Returns list of symbols that should be stopped out.
/// Accepts prices to match Finlab's floating point behavior (cr_at_close = cr * price / price).
fn detect_stops_string(
    portfolio: &Portfolio,
    prices: &HashMap<&str, f64>,
    config: &BacktestConfig,
) -> Vec<String> {
    let mut stopped = Vec::new();

    for (sym, pos) in portfolio.positions.iter() {
        // Get current price for this symbol
        let current_price = prices.get(sym.as_str()).copied().unwrap_or(0.0);

        // Skip if price is invalid
        if current_price <= 0.0 || current_price.is_nan() {
            continue;
        }

        // Use last_market_value to determine long/short (like Wide format)
        let is_long = pos.last_market_value >= 0.0;
        let cr = pos.cr;
        let maxcr = pos.maxcr;

        // Finlab uses cr_at_close = cr * close / price for stop detection (line 387)
        // Even when close == price (both adj_close), the multiply-divide operation
        // affects floating point precision, which matters at exact threshold boundaries.
        let cr_at_close = cr * current_price / current_price;

        if is_long {
            // Long positions:
            //   max_r = 1 + take_profit
            //   min_r = max(1 - stop_loss, maxcr - trail_stop)
            // Trigger: cr >= max_r (take profit) or cr < min_r (stop loss/trail)

            // Check take profit
            if config.take_profit < f64::INFINITY && cr_at_close >= 1.0 + config.take_profit {
                stopped.push(sym.clone());
                continue;
            }

            // Calculate min_r using Finlab formula
            let stop_threshold = 1.0 - config.stop_loss;
            let trail_threshold = if config.trail_stop < f64::INFINITY {
                maxcr - config.trail_stop
            } else {
                f64::NEG_INFINITY
            };
            let min_r = stop_threshold.max(trail_threshold);

            // Check stop loss or trailing stop (Finlab uses < not <=)
            if cr_at_close < min_r {
                stopped.push(sym.clone());
            }
        } else {
            // Short positions:
            //   max_r = min(1 + stop_loss, maxcr + trail_stop)
            //   min_r = 1 - take_profit
            // Trigger: cr >= max_r (stop loss/trail) or cr < min_r (take profit)

            // Calculate max_r for short positions
            let stop_threshold = 1.0 + config.stop_loss;
            let trail_threshold = if config.trail_stop < f64::INFINITY {
                maxcr + config.trail_stop
            } else {
                f64::INFINITY
            };
            let max_r = stop_threshold.min(trail_threshold);

            // Check stop loss or trailing stop
            if cr_at_close >= max_r {
                stopped.push(sym.clone());
                continue;
            }

            // Check take profit for short
            let min_r = 1.0 - config.take_profit;
            if config.take_profit < f64::INFINITY && cr_at_close < min_r {
                stopped.push(sym.clone());
            }
        }
    }

    stopped
}

/// Execute rebalance with string-keyed positions
/// Execute portfolio rebalance with optional trade tracking
///
/// This is the unified implementation that supports both:
/// - No tracking: Use `NoopSymbolTracker`
/// - Trade tracking: Use `SymbolTracker`
fn execute_rebalance_impl<T>(
    portfolio: &mut Portfolio,
    target_weights: &HashMap<String, f64>,
    today_prices: &HashMap<&str, f64>,
    stopped_stocks: &HashMap<String, bool>,
    config: &BacktestConfig,
    current_date: i32,
    signal_date: i32,
    tracker: &mut T,
)
where
    T: TradeTracker<Key = String, Date = i32, Record = LongTradeRecord>,
{
    // Update existing positions to market value
    for (_sym, pos) in portfolio.positions.iter_mut() {
        pos.value = pos.last_market_value;
    }

    // Finlab behavior: Close ALL trades before rebalance (sell all, rebuy all)
    // This matches wide.rs lines 263-279
    let open_positions: Vec<String> = portfolio.positions.keys().cloned().collect();
    for sym in &open_positions {
        let exit_price = today_prices.get(sym.as_str()).copied().unwrap_or(f64::NAN);
        tracker.close_trade(
            sym,
            current_date,
            Some(signal_date),
            exit_price,
            config.fee_ratio,
            config.tax_ratio,
        );
    }

    // Calculate total balance
    let balance = portfolio.balance();

    // Finlab behavior: When stop_trading_next_period is true, exclude stopped stocks
    // from weight calculation and re-normalize remaining weights (like Wide format)
    let (effective_weights, total_target_weight) = if config.stop_trading_next_period {
        // Calculate original sum
        let original_sum: f64 = target_weights.values().map(|w| w.abs()).sum();

        // Filter out stopped stocks
        let filtered: HashMap<String, f64> = target_weights
            .iter()
            .filter(|(sym, _)| !stopped_stocks.get(*sym).copied().unwrap_or(false))
            .map(|(k, &v)| (k.clone(), v))
            .collect();

        // Calculate remaining sum
        let remaining_sum: f64 = filtered.values().map(|w| w.abs()).sum();

        // Re-normalize: scale up remaining weights to maintain full investment
        if remaining_sum > 0.0 && remaining_sum < original_sum {
            let scale_factor = original_sum / remaining_sum;
            let scaled: HashMap<String, f64> = filtered
                .into_iter()
                .map(|(k, v)| (k, v * scale_factor))
                .collect();
            let new_sum: f64 = scaled.values().map(|w| w.abs()).sum();
            (scaled, new_sum)
        } else {
            (filtered, remaining_sum)
        }
    } else {
        (target_weights.clone(), target_weights.values().map(|w| w.abs()).sum())
    };

    if total_target_weight == 0.0 || balance <= 0.0 {
        // Exit all positions
        // Note: close_trade was already called for ALL positions at the start
        let all_positions: Vec<String> = portfolio.positions.keys().cloned().collect();
        for sym in all_positions {
            if let Some(pos) = portfolio.positions.remove(&sym) {
                let sell_value = pos.value - pos.value.abs() * (config.fee_ratio + config.tax_ratio);
                portfolio.cash += sell_value;
            }
        }
        return;
    }

    let ratio = balance / total_target_weight.max(1.0);

    // Store old positions (cost basis, market value, and full Position for retain_cost)
    let old_positions: HashMap<String, f64> = portfolio
        .positions
        .iter()
        .map(|(k, v)| (k.clone(), v.value))
        .collect();
    let old_market_values: HashMap<String, f64> = portfolio
        .positions
        .iter()
        .map(|(k, v)| (k.clone(), v.last_market_value))
        .collect();
    // Store full Position for retain_cost_when_rebalance
    let old_full_positions: HashMap<String, Position> = portfolio
        .positions
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    // Clear and rebuild
    portfolio.positions.clear();
    let mut cash = portfolio.cash;

    for (sym, &target_weight) in &effective_weights {
        // Note: stopped stocks are already filtered out in effective_weights when
        // stop_trading_next_period is true

        // Get price and check validity
        let price_opt = today_prices.get(sym.as_str()).copied();
        let price_valid = price_opt.map_or(false, |p| p > 0.0 && !p.is_nan());

        // Target position value (scaled by ratio)
        let target_value = target_weight * ratio;
        let current_value = old_positions.get(sym).copied().unwrap_or(0.0);

        // Handle NaN price case (match Finlab behavior: enter even with NaN price)
        if !price_valid {
            // If target is 0 and we have an old position, sell it using old market value
            // Note: close_trade was already called at the start for ALL positions
            if target_weight.abs() < 1e-10 {
                if let Some(&old_mv) = old_market_values.get(sym) {
                    if old_mv.abs() > 1e-10 {
                        let sell_fee = old_mv.abs() * (config.fee_ratio + config.tax_ratio);
                        cash += old_mv - sell_fee;
                    }
                }
                continue;
            }

            // Finlab behavior: Enter/modify position even with NaN price
            if target_value.abs() > 1e-10 {
                let amount = target_value - current_value;
                let is_buy = amount > 0.0;
                let is_entry =
                    (target_value >= 0.0 && amount > 0.0) || (target_value <= 0.0 && amount < 0.0);
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

                if new_value.abs() > 1e-10 {
                    // Finlab behavior: Open trade for ALL positions after rebalance
                    tracker.open_trade(sym.clone(), current_date, signal_date, f64::NAN, target_weight);
                    portfolio.positions.insert(
                        sym.clone(),
                        Position::new_with_nan_price(new_value),
                    );
                }
            }
            continue;
        }

        let price = price_opt.unwrap();

        // Valid price case: exit position if target is 0
        // Note: close_trade was already called at the start for ALL positions
        if target_weight.abs() < 1e-10 {
            if current_value.abs() > 1e-10 {
                let sell_fee = current_value.abs() * (config.fee_ratio + config.tax_ratio);
                cash += current_value - sell_fee;
            }
            continue;
        }

        let amount = target_value - current_value;

        let is_buy = amount > 0.0;
        let is_entry = (target_value >= 0.0 && amount > 0.0) || (target_value <= 0.0 && amount < 0.0);
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
            // Finlab behavior: Open trade for ALL positions after rebalance
            // (All trades were closed at the start of rebalance)
            tracker.open_trade(sym.clone(), current_date, signal_date, price, target_weight);

            // Determine if this is a continuing same-direction position
            let old_value = old_positions.get(sym).copied().unwrap_or(0.0);
            let is_continuing = old_value.abs() > 1e-10 && old_value * target_weight > 0.0;

            let new_pos = if config.retain_cost_when_rebalance && is_continuing {
                // Preserve stop tracking for continuing same-direction positions
                if let Some(old_pos) = old_full_positions.get(sym) {
                    Position::new_with_preserved_tracking(new_position_value, price, old_pos)
                } else {
                    Position::new(new_position_value, price)
                }
            } else {
                // New position or direction change or retain_cost=False: reset all
                Position::new(new_position_value, price)
            };

            portfolio.positions.insert(sym.clone(), new_pos);
        }
    }

    // Handle positions outside effective_weights (sell them)
    // Note: close_trade was already called at the start for ALL positions
    for (sym, &old_value) in old_positions.iter() {
        if !effective_weights.contains_key(sym) && old_value.abs() > 1e-10 {
            let sell_fee = old_value.abs() * (config.fee_ratio + config.tax_ratio);
            cash += old_value - sell_fee;
        }
    }

    portfolio.cash = cash;
}

/// Check if two weight maps differ (for PositionChange mode)
///
/// Two weight maps differ if:
/// 1. They have different keys (symbols)
/// 2. Any corresponding weights differ by more than 1e-10
fn weights_differ(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> bool {
    // Different number of symbols
    if a.len() != b.len() {
        return true;
    }

    // Check each symbol
    for (sym, &weight_a) in a.iter() {
        match b.get(sym) {
            Some(&weight_b) => {
                if (weight_a - weight_b).abs() > 1e-10 {
                    return true;
                }
            }
            None => return true, // Symbol in a but not in b
        }
    }

    false
}

/// Normalize weights using Finlab's behavior
fn normalize_weights(
    weights: &HashMap<&str, f64>,
    stopped_stocks: &HashMap<String, bool>,
    position_limit: f64,
) -> HashMap<String, f64> {
    // Filter out stopped stocks and zero weights
    let filtered: Vec<(&str, f64)> = weights
        .iter()
        .filter(|(sym, w)| {
            // sym is &&str, *sym is &str
            // stopped_stocks.get() takes &Q where String: Borrow<Q>
            // String: Borrow<str>, so we need to pass &str
            let is_stopped = stopped_stocks.get::<str>(*sym).copied().unwrap_or(false);
            w.abs() > 1e-10 && !is_stopped
        })
        .map(|(&sym, &w)| (sym, w))
        .collect();

    if filtered.is_empty() {
        return HashMap::new();
    }

    // Finlab normalization: divisor = max(abs_sum, 1.0)
    let abs_sum: f64 = filtered.iter().map(|(_, w)| w.abs()).sum();
    let divisor = abs_sum.max(1.0);

    filtered
        .into_iter()
        .map(|(sym, w)| {
            let norm_w = w / divisor;
            let clipped = norm_w.clamp(-position_limit, position_limit);
            (sym.to_string(), clipped)
        })
        .collect()
}

// ============================================================================
// Backtest with trade tracking
// ============================================================================

/// Run backtest on Arrow arrays with trade tracking
///
/// Delegates to `backtest_impl` with `SymbolTracker` for trade tracking.
/// Same as `backtest_long_arrow` but returns trade records as well.
pub fn backtest_with_trades_long_arrow(
    input: &LongFormatArrowInput,
    resample: ResampleFreq,
    config: &BacktestConfig,
) -> LongBacktestResult {
    let mut tracker = SymbolTracker::new();
    let creturn = backtest_impl(
        input.dates.len(),
        |i| input.dates.value(i),
        |i| input.symbols.value(i),
        |i| input.prices.value(i),
        |i| input.weights.value(i),
        resample,
        config,
        &mut tracker,
    );

    // Finalize trades (include open positions)
    let trades = tracker.finalize(config.fee_ratio, config.tax_ratio);

    LongBacktestResult { creturn, trades }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringViewBuilder;

    /// Convert date string to days since epoch (1970-01-01)
    fn date_to_days(s: &str) -> i32 {
        // Simple parser: YYYY-MM-DD
        let parts: Vec<&str> = s.split('-').collect();
        let year: i32 = parts[0].parse().unwrap();
        let month: u32 = parts[1].parse().unwrap();
        let day: u32 = parts[2].parse().unwrap();

        // Days from 1970-01-01 (simplified, accurate for 2000-2050)
        let days_per_year = 365;
        let mut days = (year - 1970) * days_per_year;
        days += ((year - 1970 + 1) / 4) as i32; // Leap years since 1970
        let days_per_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        days += days_per_month[(month - 1) as usize] as i32;
        if month > 2 && year % 4 == 0 {
            days += 1;
        }
        days += day as i32 - 1;
        days
    }

    fn make_symbols(strs: Vec<&str>) -> StringViewArray {
        let mut builder = StringViewBuilder::new();
        for s in strs {
            builder.append_value(s);
        }
        builder.finish()
    }

    fn make_input<'a>(
        dates: &'a Int32Array,
        symbols: &'a StringViewArray,
        prices: &'a Float64Array,
        weights: &'a Float64Array,
    ) -> LongFormatArrowInput<'a> {
        LongFormatArrowInput {
            dates,
            symbols,
            prices,
            weights,
        }
    }

    #[test]
    fn test_backtest_empty() {
        let dates = Int32Array::from(Vec::<i32>::new());
        let symbols = make_symbols(vec![]);
        let prices = Float64Array::from(Vec::<f64>::new());
        let weights = Float64Array::from(Vec::<f64>::new());

        let input = make_input(&dates, &symbols, &prices, &weights);
        let result = backtest_long_arrow(&input, ResampleFreq::Daily, &BacktestConfig::default());

        assert!(result.creturn.is_empty());
    }

    #[test]
    fn test_backtest_single_stock() {
        // 4 days, 1 stock
        // Day 0: signal weight=1.0
        // Day 1: entry at T+1
        // Day 2: +10%
        // Day 3: +10%
        let dates = Int32Array::from(vec![
            date_to_days("2024-01-01"),
            date_to_days("2024-01-02"),
            date_to_days("2024-01-03"),
            date_to_days("2024-01-04"),
        ]);
        let symbols = make_symbols(vec!["AAPL", "AAPL", "AAPL", "AAPL"]);
        let prices = Float64Array::from(vec![100.0, 100.0, 110.0, 121.0]);
        let weights = Float64Array::from(vec![1.0, 0.0, 0.0, 0.0]);

        let input = make_input(&dates, &symbols, &prices, &weights);
        let config = BacktestConfig {
            fee_ratio: 0.0,
            tax_ratio: 0.0,
            finlab_mode: true,
            ..Default::default()
        };

        let result = backtest_long_arrow(&input, ResampleFreq::Daily, &config);

        assert_eq!(result.creturn.len(), 4);
        assert!((result.creturn[0] - 1.0).abs() < 1e-10, "Day 0: {}", result.creturn[0]);
        assert!((result.creturn[1] - 1.0).abs() < 1e-10, "Day 1: {}", result.creturn[1]);
        assert!((result.creturn[2] - 1.1).abs() < 1e-10, "Day 2: {}", result.creturn[2]);
        assert!((result.creturn[3] - 1.21).abs() < 1e-10, "Day 3: {}", result.creturn[3]);
    }

    #[test]
    fn test_backtest_two_stocks() {
        // 3 days, 2 stocks with equal weight
        let d1 = date_to_days("2024-01-01");
        let d2 = date_to_days("2024-01-02");
        let d3 = date_to_days("2024-01-03");
        let dates = Int32Array::from(vec![d1, d1, d2, d2, d3, d3]);
        let symbols = make_symbols(vec!["AAPL", "GOOG", "AAPL", "GOOG", "AAPL", "GOOG"]);
        let prices = Float64Array::from(vec![100.0, 100.0, 100.0, 100.0, 110.0, 90.0]);
        let weights = Float64Array::from(vec![0.5, 0.5, 0.0, 0.0, 0.0, 0.0]);

        let input = make_input(&dates, &symbols, &prices, &weights);
        let config = BacktestConfig {
            fee_ratio: 0.0,
            tax_ratio: 0.0,
            finlab_mode: true,
            ..Default::default()
        };

        let result = backtest_long_arrow(&input, ResampleFreq::Daily, &config);

        assert_eq!(result.creturn.len(), 3);
        assert!((result.creturn[0] - 1.0).abs() < 1e-10);
        assert!((result.creturn[1] - 1.0).abs() < 1e-10);
        // Day 2: 0.5 * 1.1 + 0.5 * 0.9 = 1.0
        assert!((result.creturn[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_backtest_with_fees() {
        let dates = Int32Array::from(vec![
            date_to_days("2024-01-01"),
            date_to_days("2024-01-02"),
            date_to_days("2024-01-03"),
        ]);
        let symbols = make_symbols(vec!["AAPL", "AAPL", "AAPL"]);
        let prices = Float64Array::from(vec![100.0, 100.0, 100.0]);
        let weights = Float64Array::from(vec![1.0, 0.0, 0.0]);

        let input = make_input(&dates, &symbols, &prices, &weights);
        let config = BacktestConfig {
            fee_ratio: 0.01,
            tax_ratio: 0.0,
            finlab_mode: true,
            ..Default::default()
        };

        let result = backtest_long_arrow(&input, ResampleFreq::Daily, &config);

        // Day 1: Entry with 1% fee = 0.99
        assert!((result.creturn[1] - 0.99).abs() < 1e-6, "Day 1: {}", result.creturn[1]);
    }

    #[test]
    fn test_monthly_rebalance() {
        // Test that monthly rebalance triggers at month boundaries
        let dates = Int32Array::from(vec![
            date_to_days("2024-01-30"),
            date_to_days("2024-01-31"),
            date_to_days("2024-02-01"),
            date_to_days("2024-02-02"),
        ]);
        let symbols = make_symbols(vec!["AAPL", "AAPL", "AAPL", "AAPL"]);
        let prices = Float64Array::from(vec![100.0, 100.0, 100.0, 110.0]);
        let weights = Float64Array::from(vec![0.0, 1.0, 0.0, 0.0]); // Signal on Jan 31

        let input = make_input(&dates, &symbols, &prices, &weights);
        let config = BacktestConfig {
            fee_ratio: 0.0,
            tax_ratio: 0.0,
            finlab_mode: true,
            ..Default::default()
        };

        let result = backtest_long_arrow(&input, ResampleFreq::Monthly, &config);

        assert_eq!(result.creturn.len(), 4);
        // Jan 31 is month-end, so signal triggers entry on Feb 1
        // Feb 1: entry at 100
        // Feb 2: +10% = 1.1
        assert!((result.creturn[3] - 1.1).abs() < 1e-10, "Day 3: {}", result.creturn[3]);
    }

    #[test]
    fn test_slice_interface() {
        let dates = [
            date_to_days("2024-01-01"),
            date_to_days("2024-01-02"),
            date_to_days("2024-01-03"),
        ];
        let symbols = ["AAPL", "AAPL", "AAPL"];
        let prices = [100.0, 100.0, 110.0];
        let weights = [1.0, 0.0, 0.0];

        let config = BacktestConfig {
            fee_ratio: 0.0,
            tax_ratio: 0.0,
            finlab_mode: true,
            ..Default::default()
        };

        let result = backtest_long_slice(&dates, &symbols, &prices, &weights, ResampleFreq::Daily, &config);

        assert_eq!(result.creturn.len(), 3);
        assert!((result.creturn[2] - 1.1).abs() < 1e-10);
    }
}
