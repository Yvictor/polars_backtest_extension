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

use arrow::array::{Array, Float64Array, StringArray};
use polars_arrow::array::{PrimitiveArray, StaticArray, Utf8Array, Utf8ViewArray};

use crate::config::BacktestConfig;
use crate::tracker::BacktestResult;

/// Position with string symbol key (for zero-copy backtest)
#[derive(Debug, Clone)]
pub struct StringPosition {
    /// Cost basis value (weight at entry)
    pub value: f64,
    /// Entry price
    pub entry_price: f64,
    /// Max price since entry (for trailing stop)
    pub max_price: f64,
    /// Last market value (updated daily)
    pub last_market_value: f64,
    /// Cumulative return ratio
    pub cr: f64,
    /// Max cumulative return (for trailing stop)
    pub maxcr: f64,
    /// Previous price for daily return calc
    pub previous_price: f64,
}

impl StringPosition {
    pub fn new(value: f64, price: f64) -> Self {
        Self {
            value,
            entry_price: price,
            max_price: price,
            last_market_value: value,
            cr: 1.0,
            maxcr: 1.0,
            previous_price: price,
        }
    }
}

/// Portfolio with string symbol keys (for zero-copy backtest)
pub struct StringPortfolio {
    pub cash: f64,
    pub positions: HashMap<String, StringPosition>,
}

impl StringPortfolio {
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

impl Default for StringPortfolio {
    fn default() -> Self {
        Self::new()
    }
}

/// Arrow-based long format backtest input
///
/// All arrays must have the same length and be sorted by date.
pub struct LongFormatArrowInput<'a> {
    /// Date strings (sorted ascending)
    pub dates: &'a StringArray,
    /// Symbol strings
    pub symbols: &'a StringArray,
    /// Close prices
    pub prices: &'a Float64Array,
    /// Target weights
    pub weights: &'a Float64Array,
}

/// Run backtest with closure-based data access (for zero-copy from any source)
///
/// This allows the caller to provide data accessors without copying data.
/// The closures are called for each row index to get the data.
///
/// # Arguments
/// * `n_rows` - Total number of rows
/// * `get_date` - Closure to get date string at index
/// * `get_symbol` - Closure to get symbol string at index
/// * `get_price` - Closure to get price at index
/// * `get_weight` - Closure to get weight at index (NaN = no signal)
/// * `resample` - Rebalancing frequency
/// * `config` - Backtest configuration
pub fn backtest_long_with_accessor<'a, FD, FS, FP, FW>(
    n_rows: usize,
    get_date: FD,
    get_symbol: FS,
    get_price: FP,
    get_weight: FW,
    resample: ResampleFreq,
    config: &BacktestConfig,
) -> BacktestResult
where
    FD: Fn(usize) -> &'a str,
    FS: Fn(usize) -> &'a str,
    FP: Fn(usize) -> f64,
    FW: Fn(usize) -> f64,
{
    if n_rows == 0 {
        return BacktestResult {
            creturn: vec![],
            trades: vec![],
        };
    }

    let mut portfolio = StringPortfolio::new();
    let mut creturn: Vec<f64> = Vec::new();
    let mut stopped_stocks: HashMap<String, bool> = HashMap::new();
    let mut pending_weights: Option<HashMap<String, f64>> = None;
    let mut pending_stop_exits: Vec<String> = Vec::new();
    let mut current_date: Option<&str> = None;
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

            // STEP 2: Execute pending stops
            execute_pending_stops(
                &mut portfolio,
                &mut pending_stop_exits,
                &mut stopped_stocks,
                config,
            );

            // STEP 3: Execute pending rebalance
            if let Some(target_weights) = pending_weights.take() {
                execute_rebalance(
                    &mut portfolio,
                    &target_weights,
                    &today_prices,
                    &stopped_stocks,
                    config,
                );
                stopped_stocks.clear();
            }

            // STEP 4: Check rebalance
            let should_rebalance = match resample {
                ResampleFreq::Monthly => is_month_end(prev_date, date),
                ResampleFreq::Weekly => true,
                ResampleFreq::Daily => true,
            };

            if should_rebalance && !today_weights.is_empty() {
                let normalized = normalize_weights(
                    &today_weights,
                    &stopped_stocks,
                    config.position_limit,
                );
                if !normalized.is_empty() {
                    pending_weights = Some(normalized);
                }
            }

            // STEP 5: Record creturn
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
    if current_date.is_some() && !today_prices.is_empty() {
        update_positions(&mut portfolio, &today_prices);

        execute_pending_stops(
            &mut portfolio,
            &mut pending_stop_exits,
            &mut stopped_stocks,
            config,
        );

        if let Some(target_weights) = pending_weights.take() {
            execute_rebalance(
                &mut portfolio,
                &target_weights,
                &today_prices,
                &stopped_stocks,
                config,
            );
        }

        creturn.push(portfolio.balance());
    }

    BacktestResult {
        creturn,
        trades: vec![],
    }
}

/// Run backtest with polars-arrow arrays using Date (i32) and Utf8ViewArray
///
/// This accepts polars-arrow arrays directly from Polars DataFrames.
/// No data copying or conversion required.
///
/// # Arguments
/// * `dates` - PrimitiveArray<i32> of dates (days since epoch, Date32 format)
/// * `symbols` - Utf8ViewArray of symbol strings
/// * `prices` - PrimitiveArray<f64> of close prices
/// * `weights` - PrimitiveArray<f64> of target weights (None = no signal)
/// * `resample` - Rebalancing frequency
/// * `config` - Backtest configuration
pub fn backtest_long_polars_arrow(
    dates: &PrimitiveArray<i32>,
    symbols: &Utf8ViewArray,
    prices: &PrimitiveArray<f64>,
    weights: &PrimitiveArray<f64>,
    resample: ResampleFreq,
    config: &BacktestConfig,
) -> BacktestResult {
    let n_rows = dates.len();
    if n_rows == 0 {
        return BacktestResult {
            creturn: vec![],
            trades: vec![],
        };
    }

    let mut portfolio = StringPortfolio::new();
    let mut creturn: Vec<f64> = Vec::new();
    let mut stopped_stocks: HashMap<String, bool> = HashMap::new();
    let mut pending_weights: Option<HashMap<String, f64>> = None;
    let mut pending_stop_exits: Vec<String> = Vec::new();
    let mut current_date: Option<i32> = None;
    let mut today_prices: HashMap<&str, f64> = HashMap::new();
    let mut today_weights: HashMap<&str, f64> = HashMap::new();

    for i in 0..n_rows {
        // Zero-copy access to polars-arrow arrays
        let date = dates.get(i).unwrap_or(0);
        let symbol = symbols.value(i);
        let price = prices.get(i).unwrap_or(f64::NAN);
        let weight = weights.get(i).unwrap_or(f64::NAN);

        let date_changed = current_date.map_or(true, |d| d != date);

        if date_changed && current_date.is_some() {
            let prev_date = current_date.unwrap();

            // STEP 1: Update positions
            update_positions(&mut portfolio, &today_prices);

            // STEP 2: Execute pending stops
            execute_pending_stops(
                &mut portfolio,
                &mut pending_stop_exits,
                &mut stopped_stocks,
                config,
            );

            // STEP 3: Execute pending rebalance
            if let Some(target_weights) = pending_weights.take() {
                execute_rebalance(
                    &mut portfolio,
                    &target_weights,
                    &today_prices,
                    &stopped_stocks,
                    config,
                );
                stopped_stocks.clear();
            }

            // STEP 4: Check rebalance
            let should_rebalance = match resample {
                ResampleFreq::Monthly => is_month_end_i32(prev_date, date),
                ResampleFreq::Weekly => true,
                ResampleFreq::Daily => true,
            };

            if should_rebalance && !today_weights.is_empty() {
                let normalized = normalize_weights(
                    &today_weights,
                    &stopped_stocks,
                    config.position_limit,
                );
                if !normalized.is_empty() {
                    pending_weights = Some(normalized);
                }
            }

            // STEP 5: Record creturn
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
    if current_date.is_some() && !today_prices.is_empty() {
        update_positions(&mut portfolio, &today_prices);

        execute_pending_stops(
            &mut portfolio,
            &mut pending_stop_exits,
            &mut stopped_stocks,
            config,
        );

        if let Some(target_weights) = pending_weights.take() {
            execute_rebalance(
                &mut portfolio,
                &target_weights,
                &today_prices,
                &stopped_stocks,
                config,
            );
        }

        creturn.push(portfolio.balance());
    }

    BacktestResult {
        creturn,
        trades: vec![],
    }
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
}

impl ResampleFreq {
    /// Parse from string (like Polars/Pandas resample)
    pub fn from_str(s: Option<&str>) -> Self {
        match s {
            Some("M") | Some("ME") => Self::Monthly,
            Some("W") | Some("W-FRI") => Self::Weekly,
            _ => Self::Daily,
        }
    }
}

/// Run backtest on Arrow arrays with zero-copy access
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
    let n_rows = input.dates.len();
    if n_rows == 0 {
        return BacktestResult {
            creturn: vec![],
            trades: vec![],
        };
    }

    let mut portfolio = StringPortfolio::new();
    let mut creturn: Vec<f64> = Vec::new();

    // Track stopped stocks by symbol string
    let mut stopped_stocks: HashMap<String, bool> = HashMap::new();

    // Pending weights for T+1 execution
    let mut pending_weights: Option<HashMap<String, f64>> = None;
    let mut pending_stop_exits: Vec<String> = Vec::new();

    // Track current date
    let mut current_date: Option<&str> = None;

    // Today's prices and weights
    let mut today_prices: HashMap<&str, f64> = HashMap::new();
    let mut today_weights: HashMap<&str, f64> = HashMap::new();

    // Process each row
    for i in 0..n_rows {
        let date = input.dates.value(i);
        let symbol = input.symbols.value(i);
        let price = input.prices.value(i);
        let weight = input.weights.value(i);

        // Check if we've moved to a new date
        let date_changed = current_date.map_or(true, |d| d != date);

        if date_changed && current_date.is_some() {
            let prev_date = current_date.unwrap();

            // STEP 1: Update existing positions with today's prices
            update_positions(&mut portfolio, &today_prices);

            // STEP 2: Execute pending stop exits
            execute_pending_stops(
                &mut portfolio,
                &mut pending_stop_exits,
                &mut stopped_stocks,
                config,
            );

            // STEP 3: Execute pending rebalance (T+1 execution)
            if let Some(target_weights) = pending_weights.take() {
                execute_rebalance(
                    &mut portfolio,
                    &target_weights,
                    &today_prices,
                    &stopped_stocks,
                    config,
                );
                stopped_stocks.clear();
            }

            // STEP 4: Check if prev_date is a rebalance day
            let should_rebalance = match resample {
                ResampleFreq::Monthly => is_month_end(prev_date, date),
                ResampleFreq::Weekly => true, // TODO: proper weekly detection
                ResampleFreq::Daily => true,
            };

            if should_rebalance && !today_weights.is_empty() {
                // Normalize weights using Finlab's behavior
                let normalized = normalize_weights(
                    &today_weights,
                    &stopped_stocks,
                    config.position_limit,
                );
                if !normalized.is_empty() {
                    pending_weights = Some(normalized);
                }
            }

            // STEP 5: Record cumulative return
            creturn.push(portfolio.balance());

            // Clear today's data for new day
            today_prices.clear();
            today_weights.clear();
        }

        // Accumulate today's data
        current_date = Some(date);
        if price > 0.0 && !price.is_nan() {
            today_prices.insert(symbol, price);
        }
        // Handle null weights (from polars rolling operations) as 0.0
        if !weight.is_nan() && weight.abs() > 1e-10 {
            today_weights.insert(symbol, weight);
        }
    }

    // Process final day
    if current_date.is_some() && !today_prices.is_empty() {
        // STEP 1: Update positions
        update_positions(&mut portfolio, &today_prices);

        // STEP 2: Execute pending stops
        execute_pending_stops(
            &mut portfolio,
            &mut pending_stop_exits,
            &mut stopped_stocks,
            config,
        );

        // STEP 3: Execute pending rebalance
        if let Some(target_weights) = pending_weights.take() {
            execute_rebalance(
                &mut portfolio,
                &target_weights,
                &today_prices,
                &stopped_stocks,
                config,
            );
        }

        // STEP 4: Final balance
        creturn.push(portfolio.balance());
    }

    BacktestResult {
        creturn,
        trades: vec![],
    }
}

/// Check if prev_date is a month-end (next date is in a different month)
fn is_month_end(prev_date: &str, next_date: &str) -> bool {
    prev_date.get(..7) != next_date.get(..7)
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
fn update_positions(portfolio: &mut StringPortfolio, prices: &HashMap<&str, f64>) {
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

/// Execute pending stop exits
fn execute_pending_stops(
    portfolio: &mut StringPortfolio,
    pending_stops: &mut Vec<String>,
    stopped_stocks: &mut HashMap<String, bool>,
    config: &BacktestConfig,
) {
    for sym in pending_stops.drain(..) {
        if let Some(pos) = portfolio.positions.remove(&sym) {
            let sell_value =
                pos.last_market_value - pos.last_market_value.abs() * (config.fee_ratio + config.tax_ratio);
            portfolio.cash += sell_value;
            if config.stop_trading_next_period {
                stopped_stocks.insert(sym, true);
            }
        }
    }
}

/// Execute rebalance with string-keyed positions
fn execute_rebalance(
    portfolio: &mut StringPortfolio,
    target_weights: &HashMap<String, f64>,
    today_prices: &HashMap<&str, f64>,
    stopped_stocks: &HashMap<String, bool>,
    config: &BacktestConfig,
) {
    // Update existing positions to market value
    for (_sym, pos) in portfolio.positions.iter_mut() {
        pos.value = pos.last_market_value;
    }

    // Calculate total balance
    let balance = portfolio.balance();
    let total_target_weight: f64 = target_weights.values().map(|w| w.abs()).sum();

    if total_target_weight == 0.0 || balance <= 0.0 {
        // Exit all positions
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

    // Store old positions
    let old_positions: HashMap<String, f64> = portfolio
        .positions
        .iter()
        .map(|(k, v)| (k.clone(), v.value))
        .collect();

    // Clear and rebuild
    portfolio.positions.clear();
    let mut cash = portfolio.cash;

    for (sym, &target_weight) in target_weights {
        if target_weight.abs() < 1e-10 {
            // Exit position
            if let Some(&old_value) = old_positions.get(sym) {
                if old_value.abs() > 1e-10 {
                    let sell_fee = old_value.abs() * (config.fee_ratio + config.tax_ratio);
                    cash += old_value - sell_fee;
                }
            }
            continue;
        }

        // Skip stopped stocks
        if stopped_stocks.get(sym).copied().unwrap_or(false) {
            continue;
        }

        let price = match today_prices.get(sym.as_str()) {
            Some(&p) if p > 0.0 && !p.is_nan() => p,
            _ => continue,
        };

        let target_value = target_weight * ratio;
        let current_value = old_positions.get(sym).copied().unwrap_or(0.0);
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
            portfolio.positions.insert(
                sym.clone(),
                StringPosition {
                    value: new_position_value,
                    entry_price: price,
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
    for (sym, &old_value) in old_positions.iter() {
        if !target_weights.contains_key(sym) && old_value.abs() > 1e-10 {
            let sell_fee = old_value.abs() * (config.fee_ratio + config.tax_ratio);
            cash += old_value - sell_fee;
        }
    }

    portfolio.cash = cash;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input<'a>(
        dates: &'a StringArray,
        symbols: &'a StringArray,
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
        let dates = StringArray::from(Vec::<&str>::new());
        let symbols = StringArray::from(Vec::<&str>::new());
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
        let dates = StringArray::from(vec![
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
        ]);
        let symbols = StringArray::from(vec!["AAPL", "AAPL", "AAPL", "AAPL"]);
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
        let dates = StringArray::from(vec![
            "2024-01-01", "2024-01-01",
            "2024-01-02", "2024-01-02",
            "2024-01-03", "2024-01-03",
        ]);
        let symbols = StringArray::from(vec![
            "AAPL", "GOOG",
            "AAPL", "GOOG",
            "AAPL", "GOOG",
        ]);
        let prices = Float64Array::from(vec![
            100.0, 100.0,
            100.0, 100.0,
            110.0, 90.0,
        ]);
        let weights = Float64Array::from(vec![
            0.5, 0.5,
            0.0, 0.0,
            0.0, 0.0,
        ]);

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
        let dates = StringArray::from(vec!["2024-01-01", "2024-01-02", "2024-01-03"]);
        let symbols = StringArray::from(vec!["AAPL", "AAPL", "AAPL"]);
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
        let dates = StringArray::from(vec![
            "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02",
        ]);
        let symbols = StringArray::from(vec!["AAPL", "AAPL", "AAPL", "AAPL"]);
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
}
