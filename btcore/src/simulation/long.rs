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

use std::collections::{HashMap, VecDeque};

use arrow::array::{Float64Array, Int32Array, StringViewArray};

use crate::config::BacktestConfig;
use crate::position::{Position, PositionSnapshot};
use crate::tracker::{BacktestResult, StockOperations, TradeRecord, NoopSymbolTracker, SymbolTracker, TradeTracker};
use crate::{is_valid_price, FLOAT_EPSILON};

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
    /// Open prices for touched_exit (optional)
    pub open_prices: Option<&'a Float64Array>,
    /// High prices for touched_exit (optional)
    pub high_prices: Option<&'a Float64Array>,
    /// Low prices for touched_exit (optional)
    pub low_prices: Option<&'a Float64Array>,
    /// Factor for converting adj prices to raw prices (optional)
    /// raw_price = adj_price / factor
    /// Defaults to 1.0 if not provided
    pub factor: Option<&'a Float64Array>,
}

/// Resample frequency for rebalancing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleFreq {
    /// Daily rebalancing
    Daily,
    /// Weekly rebalancing (end of week, Friday)
    Weekly,
    /// Weekly on specific weekday (0=Mon, 1=Tue, ..., 6=Sun)
    WeeklyOn(u8),
    /// Monthly rebalancing (end of month)
    Monthly,
    /// Monthly rebalancing (start of month)
    MonthStart,
    /// Quarterly rebalancing (end of quarter)
    Quarterly,
    /// Quarterly rebalancing (start of quarter)
    QuarterStart,
    /// Yearly rebalancing (end of year)
    Yearly,
    /// Only rebalance when position changes (Finlab default, resample=None)
    PositionChange,
}

impl ResampleFreq {
    /// Parse from string (like Polars/Pandas resample)
    pub fn from_str(s: Option<&str>) -> Self {
        match s {
            Some("M") | Some("ME") => Self::Monthly,
            Some("MS") => Self::MonthStart,
            Some("W") | Some("W-FRI") => Self::Weekly,
            Some("W-MON") => Self::WeeklyOn(0),
            Some("W-TUE") => Self::WeeklyOn(1),
            Some("W-WED") => Self::WeeklyOn(2),
            Some("W-THU") => Self::WeeklyOn(3),
            Some("W-SAT") => Self::WeeklyOn(5),
            Some("W-SUN") => Self::WeeklyOn(6),
            Some("D") => Self::Daily,
            Some("Q") | Some("QE") => Self::Quarterly,
            Some("QS") => Self::QuarterStart,
            Some("Y") | Some("YE") | Some("A") => Self::Yearly,
            None => Self::PositionChange,
            _ => Self::Daily,
        }
    }
}

/// Resample offset for delaying rebalance execution
///
/// Positive values delay execution by N days after the boundary.
/// For example, with Monthly resample and offset of 1 day:
/// - Boundary detected: Jan 31 (month end)
/// - Normal execution: Feb 1 (T+1)
/// - With offset 1d: Feb 2 (T+1+offset)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResampleOffset {
    /// Number of days to delay (positive only for now)
    pub days: i32,
}

impl ResampleOffset {
    /// Create a new offset with the given number of days
    pub fn new(days: i32) -> Self {
        Self { days: days.max(0) } // Only positive offsets supported
    }

    /// Parse from string like "1d", "2d", "1D", "1W", etc.
    ///
    /// Supported formats:
    /// - "1d", "2d", "1D" - days
    /// - "1W", "2W" - weeks (converted to days)
    ///
    /// Returns None if the string is None or cannot be parsed.
    pub fn from_str(s: Option<&str>) -> Option<Self> {
        let s = s?.trim();
        if s.is_empty() {
            return None;
        }

        // Try to parse as days (e.g., "1d", "2d", "1D")
        if s.ends_with('d') || s.ends_with('D') {
            let num_str = &s[..s.len() - 1];
            if let Ok(days) = num_str.parse::<i32>() {
                if days >= 0 {
                    return Some(Self { days });
                }
            }
        }

        // Try to parse as weeks (e.g., "1W", "2W")
        if s.ends_with('W') || s.ends_with('w') {
            let num_str = &s[..s.len() - 1];
            if let Ok(weeks) = num_str.parse::<i32>() {
                if weeks >= 0 {
                    return Some(Self { days: weeks * 7 });
                }
            }
        }

        None
    }
}

/// Delayed rebalance entry for offset support
///
/// Only stores the target date. When triggered (date > target_date),
/// we use prev_date as the signal date and today_weights for rebalancing.
/// This matches Wide format's _resample_position behavior where
/// the signal date is the LAST trading day <= target_date.
#[derive(Debug, Clone, Copy)]
struct DelayedRebalance {
    /// The target date to trigger rebalance (boundary_date + offset.days)
    /// When date > target_date, the rebalance triggers with prev_date as signal date
    target_date: i32,
}

/// Optional OHLC getters for touched_exit mode
pub struct OhlcGetters<FO, FH, FL>
where
    FO: Fn(usize) -> f64,
    FH: Fn(usize) -> f64,
    FL: Fn(usize) -> f64,
{
    pub get_open: FO,
    pub get_high: FH,
    pub get_low: FL,
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
/// * `get_factor` - Closure to get factor at index (for raw price conversion, 1.0 if no factor)
/// * `resample` - Rebalancing frequency
/// * `offset` - Optional resample offset for delaying execution
/// * `config` - Backtest configuration
/// * `tracker` - Trade tracker (use NoopSymbolTracker::default() for no tracking)
/// * `ohlc` - Optional OHLC getters for touched_exit mode
fn backtest_impl<'a, FD, FS, FP, FW, FF, T, FO, FH, FL>(
    n_rows: usize,
    get_date: FD,
    get_symbol: FS,
    get_price: FP,
    get_weight: FW,
    get_factor: FF,
    resample: ResampleFreq,
    offset: Option<ResampleOffset>,
    config: &BacktestConfig,
    tracker: &mut T,
    ohlc: Option<OhlcGetters<FO, FH, FL>>,
) -> (Vec<i32>, Vec<f64>)
where
    FD: Fn(usize) -> i32,
    FS: Fn(usize) -> &'a str,
    FP: Fn(usize) -> f64,
    FW: Fn(usize) -> f64,
    FF: Fn(usize) -> f64,
    FO: Fn(usize) -> f64,
    FH: Fn(usize) -> f64,
    FL: Fn(usize) -> f64,
    T: TradeTracker<Key = String, Date = i32, Record = TradeRecord>,
{
    if n_rows == 0 {
        return (vec![], vec![]);
    }

    let mut portfolio = Portfolio::new();
    let mut dates: Vec<i32> = Vec::new();
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
    let mut today_factor: HashMap<&str, f64> = HashMap::new();

    // Delayed rebalance queue for offset support
    let mut delayed_rebalances: VecDeque<DelayedRebalance> = VecDeque::new();
    let offset_days = offset.map(|o| o.days).unwrap_or(0);

    // OHLC data for touched_exit mode
    let mut today_open: HashMap<&str, f64> = HashMap::new();
    let mut today_high: HashMap<&str, f64> = HashMap::new();
    let mut today_low: HashMap<&str, f64> = HashMap::new();
    let touched_exit_enabled = config.touched_exit && ohlc.is_some();

    for i in 0..n_rows {
        let date = get_date(i);
        let symbol = get_symbol(i);
        let price = get_price(i);
        let weight = get_weight(i);

        let date_changed = current_date.map_or(true, |d| d != date);

        if date_changed && current_date.is_some() {
            let prev_date = current_date.unwrap();

            // STEP 1: Update positions (cr *= r, maxcr update)
            update_positions(&mut portfolio, &today_prices);

            // Record prices for MAE/MFE calculation
            for (sym, _pos) in portfolio.positions.iter() {
                if let Some(&price) = today_prices.get(sym.as_str()) {
                    tracker.record_price(sym, price, price);
                }
            }

            // STEP 2: Stop detection and execution
            if touched_exit_enabled {
                // touched_exit mode: detect and execute IMMEDIATELY (T+0)
                let ohlc_refs = Some((&today_open, &today_high, &today_low));
                let stops = detect_stops_unified(&portfolio, &today_prices, ohlc_refs, config);
                execute_stops_impl(
                    &mut portfolio,
                    stops,
                    &mut stopped_stocks,
                    &today_prices,
                    &today_factor,
                    config,
                    prev_date,
                    Some(prev_date), // exit_sig_date = current date (same day)
                    tracker,
                );
                // Clear pending stops (not used in touched_exit mode)
                pending_stop_exits.clear();
            } else {
                // Regular mode: execute pending stops (T+1), then detect new stops
                // Execute pending stops from yesterday's detection
                let pending_results: Vec<StopResult> = pending_stop_exits
                    .drain(..)
                    .map(|sym| StopResult { symbol: sym, exit_ratio: 1.0 })
                    .collect();
                execute_stops_impl(
                    &mut portfolio,
                    pending_results,
                    &mut stopped_stocks,
                    &today_prices,
                    &today_factor,
                    config,
                    prev_date,
                    None, // No exit_sig_date for regular stops
                    tracker,
                );

                // Detect new stops for tomorrow's execution
                let new_stops = detect_stops_unified(&portfolio, &today_prices, None, config);
                for stop in new_stops {
                    pending_stop_exits.push(stop.symbol);
                }
            }

            // Update previous_price AFTER stop detection (matches wide format behavior)
            // This ensures touched_exit uses the correct previous_price for ratio calculations
            update_previous_prices(&mut portfolio, &today_prices);

            // Check if any delayed rebalances are due
            // Move them to pending_weights for execution
            //
            // Wide format behavior with offset:
            // - _resample_position shifts boundary dates forward by offset
            // - Signal date = LAST trading day <= target_date
            // - The weights used are from the signal date
            // - Signal date records creturn BEFORE rebalance
            // - Rebalance executes on T+1
            //
            // Key insight: When date > target_date, prev_date is the last trading day
            // before date. If there are no trading days between prev_date and date
            // that are <= target_date, then prev_date IS the last trading day <= target_date.
            // This is exactly the signal date that Wide format uses.
            //
            // Example with Q+1D (boundary = Dec 31, target_date = Jan 1):
            // Case 1: Jan 1 is NOT a trading day (holiday)
            // - Dec 31 → Jan 5: date=Jan5, prev_date=Dec31
            // - date (Jan5) > target_date (Jan1)? YES
            // - prev_date (Dec31) is the last trading day <= Jan1
            // - Use today_weights (Dec31's weights)
            //
            // Case 2: Apr 1 IS a trading day
            // - Mar 31 → Apr 1: Queue delayed with target_date = Apr 1
            // - Apr 1 → Apr 2: date=Apr2, prev_date=Apr1
            // - date (Apr2) > target_date (Apr1)? YES
            // - prev_date (Apr1) is the last trading day <= Apr1
            // - Use today_weights (Apr1's weights)
            //
            // Note: The delayed rebalance is checked AFTER queueing new ones (below).
            // This ensures that if a delayed rebalance is queued and triggers on
            // the same transition, it uses the correct weights (today_weights).
            let delayed_triggered = false;

            // STEP 3: Execute pending rebalance
            // Skip if delayed triggered this iteration - execute on T+1
            // This matches Wide format where signal_date records creturn BEFORE rebalance
            if delayed_triggered {
                // Keep pending_weights for next day (T+1 execution)
                // Don't take() so it stays in pending_weights
            } else if let Some(target_weights) = pending_weights.take() {
                let sig_date = pending_signal_date.take().unwrap_or(prev_date);
                execute_rebalance_impl(
                    &mut portfolio,
                    &target_weights,
                    &today_prices,
                    &today_factor,
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
            let is_quarter_end = is_quarter_end_i32(prev_date, date);
            let is_year_end = is_year_end_i32(prev_date, date);

            // Check if prev_date is a specific weekday (for WeeklyOn)
            let prev_weekday = weekday_of_i32(prev_date);

            // When offset > 0, has_first_signal is set when delayed rebalance triggers
            // (already handled above), not at boundary. This matches Wide format behavior.
            let set_first_signal_at_boundary = offset_days == 0;

            // Track if we have signals for later use
            let has_signals = !normalized.is_empty();

            match resample {
                ResampleFreq::Daily => {
                    if has_signals && set_first_signal_at_boundary {
                        has_first_signal = true;
                    }
                    if has_first_signal {
                        active_weights = normalized.clone();
                    }
                }
                ResampleFreq::Weekly => {
                    if is_week_end {
                        active_weights = normalized.clone();
                        if has_signals && set_first_signal_at_boundary {
                            has_first_signal = true;
                        }
                    }
                }
                ResampleFreq::WeeklyOn(weekday) => {
                    // Rebalance on specific weekday (0=Mon, ..., 6=Sun)
                    if prev_weekday == weekday {
                        active_weights = normalized.clone();
                        if has_signals && set_first_signal_at_boundary {
                            has_first_signal = true;
                        }
                    }
                }
                ResampleFreq::Monthly | ResampleFreq::MonthStart => {
                    // Both trigger at month boundary (end of one month / start of next)
                    if is_month_end {
                        active_weights = normalized.clone();
                        if has_signals && set_first_signal_at_boundary {
                            has_first_signal = true;
                        }
                    }
                }
                ResampleFreq::Quarterly | ResampleFreq::QuarterStart => {
                    // Both trigger at quarter boundary
                    if is_quarter_end {
                        active_weights = normalized.clone();
                        if has_signals && set_first_signal_at_boundary {
                            has_first_signal = true;
                        }
                    }
                }
                ResampleFreq::Yearly => {
                    if is_year_end {
                        active_weights = normalized.clone();
                        if has_signals && set_first_signal_at_boundary {
                            has_first_signal = true;
                        }
                    }
                }
                ResampleFreq::PositionChange => {
                    if has_signals && set_first_signal_at_boundary {
                        has_first_signal = true;
                    }
                    position_changed = weights_differ(&active_weights, &normalized);
                    active_weights = normalized;
                }
            }

            // STEP 5: Check rebalance
            // When offset > 0: push to delayed queue even before has_first_signal
            // (has_first_signal will be set when delayed rebalance triggers)
            // When offset = 0: only rebalance after first signal (normal behavior)
            let should_rebalance = match resample {
                ResampleFreq::Monthly | ResampleFreq::MonthStart => is_month_end,
                ResampleFreq::Weekly => is_week_end,
                ResampleFreq::WeeklyOn(weekday) => prev_weekday == weekday,
                ResampleFreq::Quarterly | ResampleFreq::QuarterStart => is_quarter_end,
                ResampleFreq::Yearly => is_year_end,
                ResampleFreq::Daily => true,
                ResampleFreq::PositionChange => position_changed,
            };

            if should_rebalance {
                if offset_days > 0 {
                    // Delay the rebalance by offset_days from the ACTUAL period boundary
                    // For weekly, boundary is Sunday (not last trading day)
                    // For monthly/quarterly, boundary is the last calendar day (= last trading day)
                    //
                    // IMPORTANT: Handle multi-period gaps (e.g., multiple weeks during holidays)
                    // We need to queue a delayed rebalance for EACH period boundary in the gap.
                    let boundaries = get_all_period_boundaries(prev_date, date, resample);
                    for boundary in boundaries {
                        delayed_rebalances.push_back(DelayedRebalance {
                            target_date: boundary + offset_days,
                        });
                    }
                } else if has_first_signal {
                    // No offset: set pending_weights for next day (T+1)
                    pending_weights = Some(active_weights.clone());
                    pending_signal_date = Some(prev_date);
                }
            }

            // STEP 5b: Check delayed rebalances AFTER queueing new ones
            // This is critical: if a delayed rebalance is queued and should trigger
            // on the same transition (e.g., Q+1D when target_date falls in a trading gap),
            // we need to check it NOW while today_weights has the correct weights.
            //
            // IMPORTANT: Wide format de-duplicates weeks with the same signal date.
            // When multiple weeks fall in a trading gap, they all use the same signal
            // (last trading day before the gap). We need to collapse all such delayed
            // rebalances into a single rebalance.
            //
            // Strategy: Pop ALL delayed rebalances with target_date < date, but only
            // set pending_weights once. This matches Wide format's de-duplication.
            let mut any_triggered = false;
            while let Some(delayed) = delayed_rebalances.front() {
                // Trigger when date crosses target_date
                if date > delayed.target_date {
                    let _delayed = delayed_rebalances.pop_front().unwrap();
                    any_triggered = true;
                    // Continue popping all triggered delayed rebalances
                } else {
                    break;
                }
            }

            // If any delayed rebalances triggered, set pending_weights once
            if any_triggered {
                // Use today_weights (prev_date's weights, which is the signal date)
                let signal_weights = normalize_weights(
                    &today_weights,
                    &stopped_stocks,
                    config.position_limit,
                );

                // Only set has_first_signal when we have actual non-empty weights
                // This matches Wide format behavior where creturn recording starts
                // from the first actual position, not from zero-weight signals.
                if !has_first_signal && !signal_weights.is_empty() {
                    has_first_signal = true;
                }
                pending_weights = Some(signal_weights);
                pending_signal_date = Some(prev_date);
                // Note: delayed_triggered is set for documentation purposes
                // The actual skip logic is in STEP 3 which runs before this
                let _ = delayed_triggered;  // suppress unused warning
            }

            // STEP 6: Record date and creturn (only after first signal)
            // This matches Wide format behavior where creturn starts from first signal date
            if has_first_signal {
                dates.push(prev_date);
                creturn.push(portfolio.balance());
            }

            today_prices.clear();
            today_weights.clear();
            today_factor.clear();
            if touched_exit_enabled {
                today_open.clear();
                today_high.clear();
                today_low.clear();
            }
        }

        current_date = Some(date);
        if is_valid_price(price) {
            today_prices.insert(symbol, price);
        }
        if !weight.is_nan() && weight.abs() > FLOAT_EPSILON {
            today_weights.insert(symbol, weight);
        }
        // Collect factor (for raw price calculation)
        let factor = get_factor(i);
        if factor.is_finite() && factor > 0.0 {
            today_factor.insert(symbol, factor);
        }
        // Collect OHLC data AFTER date_changed processing (for correct date)
        if let Some(ref ohlc_getters) = ohlc {
            today_open.insert(symbol, (ohlc_getters.get_open)(i));
            today_high.insert(symbol, (ohlc_getters.get_high)(i));
            today_low.insert(symbol, (ohlc_getters.get_low)(i));
        }
    }

    // Final day
    if let Some(last_date) = current_date {
        if !today_prices.is_empty() {
            update_positions(&mut portfolio, &today_prices);

            // Record prices for MAE/MFE calculation (final day)
            for (sym, _pos) in portfolio.positions.iter() {
                if let Some(&price) = today_prices.get(sym.as_str()) {
                    tracker.record_price(sym, price, price);
                }
            }

            // Stop detection and execution (same logic as main loop)
            if touched_exit_enabled {
                // touched_exit mode: detect and execute immediately (T+0)
                let ohlc_refs = Some((&today_open, &today_high, &today_low));
                let stops = detect_stops_unified(&portfolio, &today_prices, ohlc_refs, config);
                execute_stops_impl(
                    &mut portfolio,
                    stops,
                    &mut stopped_stocks,
                    &today_prices,
                    &today_factor,
                    config,
                    last_date,
                    Some(last_date),
                    tracker,
                );
            } else {
                // Regular mode: execute pending stops (T+1)
                execute_pending_stops_impl(
                    &mut portfolio,
                    &mut pending_stop_exits,
                    &mut stopped_stocks,
                    &today_prices,
                    &today_factor,
                    config,
                    last_date,
                    tracker,
                );
            }

            // Update previous_price AFTER stop detection
            update_previous_prices(&mut portfolio, &today_prices);

            if let Some(target_weights) = pending_weights.take() {
                let sig_date = pending_signal_date.take().unwrap_or(last_date);
                execute_rebalance_impl(
                    &mut portfolio,
                    &target_weights,
                    &today_prices,
                    &today_factor,
                    &stopped_stocks,
                    config,
                    last_date,
                    sig_date,
                    tracker,
                );
            }

            // Handle single-date case: if we have weights but no first signal yet,
            // record the signal day with creturn=1.0 (matching Wide format behavior)
            if !has_first_signal && !today_weights.is_empty() {
                has_first_signal = true;
            }

            // Only record if we have first signal (consistent with main loop)
            if has_first_signal {
                dates.push(last_date);
                creturn.push(portfolio.balance());
            }
        }
    }

    (dates, creturn)
}

/// Run backtest with closure-based data access (public API, no trade tracking)
///
/// # Arguments
/// * `n_rows` - Total number of rows
/// * `get_date` - Closure to get date (days since epoch) at index
/// * `get_symbol` - Closure to get symbol string at index
/// * `get_price` - Closure to get price at index
/// * `get_weight` - Closure to get weight at index (NaN = no signal)
/// * `get_factor` - Closure to get factor at index (for raw price conversion, 1.0 if no factor)
/// * `resample` - Rebalancing frequency
/// * `offset` - Optional resample offset for delaying execution
/// * `config` - Backtest configuration
pub fn backtest_with_accessor<'a, FD, FS, FP, FW, FF, FO, FH, FL>(
    n_rows: usize,
    get_date: FD,
    get_symbol: FS,
    get_price: FP,
    get_weight: FW,
    get_factor: FF,
    ohlc_accessors: Option<(FO, FH, FL)>,
    resample: ResampleFreq,
    offset: Option<ResampleOffset>,
    config: &BacktestConfig,
) -> BacktestResult
where
    FD: Fn(usize) -> i32,
    FS: Fn(usize) -> &'a str,
    FP: Fn(usize) -> f64,
    FW: Fn(usize) -> f64,
    FF: Fn(usize) -> f64,
    FO: Fn(usize) -> f64,
    FH: Fn(usize) -> f64,
    FL: Fn(usize) -> f64,
{
    let mut tracker = NoopSymbolTracker::default();

    let (dates, creturn) = if config.touched_exit && ohlc_accessors.is_some() {
        let (get_open, get_high, get_low) = ohlc_accessors.unwrap();
        let ohlc = Some(OhlcGetters {
            get_open,
            get_high,
            get_low,
        });
        backtest_impl(
            n_rows,
            get_date,
            get_symbol,
            get_price,
            get_weight,
            get_factor,
            resample,
            offset,
            config,
            &mut tracker,
            ohlc,
        )
    } else {
        let ohlc: Option<OhlcGetters<fn(usize) -> f64, fn(usize) -> f64, fn(usize) -> f64>> = None;
        backtest_impl(
            n_rows,
            get_date,
            get_symbol,
            get_price,
            get_weight,
            get_factor,
            resample,
            offset,
            config,
            &mut tracker,
            ohlc,
        )
    };

    BacktestResult {
        dates,
        creturn,
        trades: vec![],
        stock_operations: None,
    }
}

/// Run backtest on Arrow arrays with zero-copy access
///
/// Delegates to `backtest_with_accessor` using Arrow array closures.
/// Supports touched_exit mode when OHLC data is provided in input.
///
/// # Arguments
/// * `input` - Arrow arrays containing long format data (must be sorted by date)
/// * `resample` - Rebalancing frequency
/// * `offset` - Optional resample offset for delaying execution
/// * `config` - Backtest configuration
///
/// # Returns
/// BacktestResult containing dates, cumulative returns, and empty trades
pub fn backtest_long_arrow(
    input: &LongFormatArrowInput,
    resample: ResampleFreq,
    offset: Option<ResampleOffset>,
    config: &BacktestConfig,
) -> BacktestResult {
    // Build OHLC accessors if data is available
    let ohlc_accessors = if input.open_prices.is_some()
        && input.high_prices.is_some()
        && input.low_prices.is_some()
    {
        let open_arr = input.open_prices.unwrap();
        let high_arr = input.high_prices.unwrap();
        let low_arr = input.low_prices.unwrap();
        Some((
            move |i: usize| open_arr.value(i),
            move |i: usize| high_arr.value(i),
            move |i: usize| low_arr.value(i),
        ))
    } else {
        None
    };

    // Build factor accessor (default to 1.0 if not provided)
    let get_factor: Box<dyn Fn(usize) -> f64> = if let Some(factor_arr) = input.factor {
        Box::new(move |i: usize| factor_arr.value(i))
    } else {
        Box::new(|_: usize| 1.0)
    };

    backtest_with_accessor(
        input.dates.len(),
        |i| input.dates.value(i),
        |i| input.symbols.value(i),
        |i| input.prices.value(i),
        |i| input.weights.value(i),
        |i| get_factor(i),
        ohlc_accessors,
        resample,
        offset,
        config,
    )
}

/// Run backtest on native Rust slices
///
/// Supports touched_exit mode when OHLC data is provided.
pub fn backtest_long_slice(
    dates: &[i32],
    symbols: &[&str],
    prices: &[f64],
    weights: &[f64],
    factor: Option<&[f64]>,
    open_prices: Option<&[f64]>,
    high_prices: Option<&[f64]>,
    low_prices: Option<&[f64]>,
    resample: ResampleFreq,
    offset: Option<ResampleOffset>,
    config: &BacktestConfig,
) -> BacktestResult {
    // Build OHLC accessors if data is available
    let ohlc_accessors =
        if open_prices.is_some() && high_prices.is_some() && low_prices.is_some() {
            let open = open_prices.unwrap();
            let high = high_prices.unwrap();
            let low = low_prices.unwrap();
            Some((
                move |i: usize| open[i],
                move |i: usize| high[i],
                move |i: usize| low[i],
            ))
        } else {
            None
        };

    // Build factor accessor (default to 1.0 if not provided)
    let get_factor: Box<dyn Fn(usize) -> f64> = if let Some(f) = factor {
        Box::new(move |i: usize| f[i])
    } else {
        Box::new(|_: usize| 1.0)
    };

    backtest_with_accessor(
        dates.len(),
        |i| dates[i],
        |i| symbols[i],
        |i| prices[i],
        |i| weights[i],
        |i| get_factor(i),
        ohlc_accessors,
        resample,
        offset,
        config,
    )
}


// ============================================================================
// Backtest with trade tracking
// ============================================================================

/// Run backtest on Arrow arrays with full report (trades tracking)
///
/// Delegates to `backtest_impl` with `SymbolTracker` for trade tracking.
/// Same as `backtest_long_arrow` but returns trade records as well.
/// Supports touched_exit mode when OHLC data is provided in input.
pub fn backtest_with_report_long_arrow(
    input: &LongFormatArrowInput,
    resample: ResampleFreq,
    offset: Option<ResampleOffset>,
    config: &BacktestConfig,
) -> BacktestResult {
    let mut tracker = SymbolTracker::new();

    // Build factor accessor (default to 1.0 if not provided)
    let get_factor: Box<dyn Fn(usize) -> f64> = if let Some(factor_arr) = input.factor {
        Box::new(move |i: usize| factor_arr.value(i))
    } else {
        Box::new(|_: usize| 1.0)
    };

    // Build OHLC getters if data is available and touched_exit is enabled
    let (dates, creturn) = if config.touched_exit
        && input.open_prices.is_some()
        && input.high_prices.is_some()
        && input.low_prices.is_some()
    {
        let open_arr = input.open_prices.unwrap();
        let high_arr = input.high_prices.unwrap();
        let low_arr = input.low_prices.unwrap();

        let ohlc = Some(OhlcGetters {
            get_open: |i: usize| open_arr.value(i),
            get_high: |i: usize| high_arr.value(i),
            get_low: |i: usize| low_arr.value(i),
        });

        backtest_impl(
            input.dates.len(),
            |i| input.dates.value(i),
            |i| input.symbols.value(i),
            |i| input.prices.value(i),
            |i| input.weights.value(i),
            |i| get_factor(i),
            resample,
            offset,
            config,
            &mut tracker,
            ohlc,
        )
    } else {
        let ohlc: Option<OhlcGetters<fn(usize) -> f64, fn(usize) -> f64, fn(usize) -> f64>> = None;
        backtest_impl(
            input.dates.len(),
            |i| input.dates.value(i),
            |i| input.symbols.value(i),
            |i| input.prices.value(i),
            |i| input.weights.value(i),
            |i| get_factor(i),
            resample,
            offset,
            config,
            &mut tracker,
            ohlc,
        )
    };

    // Finalize trades (include open positions)
    let mut trades = tracker.finalize(config.fee_ratio, config.tax_ratio);

    // Calculate stock_operations (actions, weights, next_weights)
    // This matches Finlab's behavior where actions are computed at the end of backtest
    let stock_operations = calculate_stock_operations(input, &trades, &dates, resample, offset);

    // Post-process trades based on stock_operations (Finlab compatible)
    // 1. For "exit" actions: set exit_sig_date on open trades
    // 2. For "enter" actions: create pending entry trades
    if let Some(next_date) = stock_operations.next_weight_date {
        // Update exit_sig_date for "exit" actions
        for trade in trades.iter_mut() {
            if trade.entry_date.is_some() && trade.exit_date.is_none() {
                if let Some(action) = stock_operations.actions.get(&trade.symbol) {
                    if action == "exit" {
                        trade.exit_sig_date = Some(next_date);
                    }
                }
            }
        }

        // Create pending entry trades for "enter" actions
        for (symbol, action) in &stock_operations.actions {
            if action == "enter" {
                let weight = stock_operations.next_weights.get(symbol).copied().unwrap_or(0.0);
                let pending_trade = TradeRecord {
                    symbol: symbol.clone(),
                    entry_date: None,
                    exit_date: None,
                    entry_sig_date: next_date,
                    exit_sig_date: None,
                    position_weight: weight,
                    entry_price: f64::NAN,
                    exit_price: None,
                    entry_raw_price: f64::NAN,
                    exit_raw_price: None,
                    trade_return: None,
                    mae: None,
                    gmfe: None,
                    bmfe: None,
                    mdd: None,
                    pdays: None,
                    period: None,
                };
                trades.push(pending_trade);
            }
        }
    }

    BacktestResult {
        dates,
        creturn,
        trades,
        stock_operations: Some(stock_operations),
    }
}

/// Calculate the next signal date based on resample frequency and offset
///
/// Given a base date (typically the current signal date), calculate when the
/// next signal should occur based on the resample pattern.
///
/// For example:
/// - Monthly with 15D offset: if base = Jan 15, next = Feb 15
/// - Monthly (no offset): if base = Jan 31, next = Feb 28/29
fn calculate_next_signal_date(
    base_date: i32,
    resample: ResampleFreq,
    offset: Option<ResampleOffset>,
) -> i32 {
    let offset_days = offset.map(|o| o.days).unwrap_or(0);

    match resample {
        ResampleFreq::Monthly | ResampleFreq::MonthStart => {
            // Get current year/month from base_date
            let (year, month) = days_to_year_month(base_date);

            // Calculate next month
            let (next_year, next_month) = if month == 12 {
                (year + 1, 1)
            } else {
                (year, month + 1)
            };

            // If offset is provided, signal is on (1 + offset_days) day of month
            // Otherwise, signal is at month end
            if offset_days > 0 {
                // Signal on offset_days-th day of month (e.g., 15D means 15th)
                let signal_day = offset_days.min(days_in_month(next_year, next_month) as i32);
                ymd_to_days(next_year, next_month, signal_day as u32)
            } else {
                // Month end
                ymd_to_days(next_year, next_month, days_in_month(next_year, next_month))
            }
        }
        ResampleFreq::Weekly | ResampleFreq::WeeklyOn(_) => {
            // Add 7 days for next week
            base_date + 7 + offset_days
        }
        ResampleFreq::Quarterly => {
            // Get current quarter and move to next
            let (year, month) = days_to_year_month(base_date);
            let current_quarter = (month - 1) / 3;
            let (next_year, next_quarter) = if current_quarter == 3 {
                (year + 1, 0)
            } else {
                (year, current_quarter + 1)
            };

            // First month of next quarter
            let quarter_start_month = next_quarter * 3 + 1;

            if offset_days > 0 {
                ymd_to_days(next_year, quarter_start_month, offset_days.min(28) as u32)
            } else {
                // Last day of quarter (last day of 3rd month in quarter)
                let quarter_end_month = quarter_start_month + 2;
                ymd_to_days(
                    next_year,
                    quarter_end_month,
                    days_in_month(next_year, quarter_end_month),
                )
            }
        }
        ResampleFreq::Yearly => {
            let (year, _) = days_to_year_month(base_date);
            if offset_days > 0 {
                ymd_to_days(year + 1, 1, offset_days.min(28) as u32)
            } else {
                ymd_to_days(year + 1, 12, 31)
            }
        }
        _ => {
            // Daily or PositionChange - just add 1 day
            base_date + 1
        }
    }
}

/// Calculate stock operations (actions, weights, next_weights) at the end of backtest
///
/// This function computes the trading actions by comparing:
/// - Current positions (open trades with entry_date but no exit_date)
/// - Latest signal weights (last signal date's weights from input)
///
/// Actions:
/// - "enter": not in current positions, but in latest signal weights
/// - "exit": in current positions, but not in latest signal weights
/// - "hold": in both current positions and latest signal weights
fn calculate_stock_operations(
    input: &LongFormatArrowInput,
    trades: &[TradeRecord],
    dates: &[i32],
    resample: ResampleFreq,
    offset: Option<ResampleOffset>,
) -> StockOperations {
    use std::collections::HashSet;

    let mut ops = StockOperations::new();

    if input.dates.len() == 0 || dates.is_empty() {
        return ops;
    }

    // Find the last creturn date (last price date)
    let last_price_date = *dates.last().unwrap();

    // Find current positions: trades with entry_date but no exit_date
    let current_positions: HashSet<String> = trades
        .iter()
        .filter(|t| t.entry_date.is_some() && t.exit_date.is_none())
        .map(|t| t.symbol.clone())
        .collect();

    // Find signal dates from input
    // We need to find weights for action calculation and determine the next rebalance date
    let n = input.dates.len();
    let mut current_signal_date: Option<i32> = None;
    let mut future_signal_date: Option<i32> = None;
    let mut signal_weights: HashMap<String, f64> = HashMap::new();

    // First pass: find the NEXT signal date (first date > last_price_date with signals)
    for i in 0..n {
        let date = input.dates.value(i);
        let weight = input.weights.value(i);

        if date > last_price_date {
            if !weight.is_nan() && weight.abs() > FLOAT_EPSILON {
                if future_signal_date.is_none() {
                    future_signal_date = Some(date);
                }
                if future_signal_date == Some(date) {
                    let symbol = input.symbols.value(i).to_string();
                    signal_weights.insert(symbol, weight);
                } else {
                    // We've moved past the next signal date
                    break;
                }
            }
        }
    }

    // If no future signal found, fall back to latest signal <= last_price_date
    if future_signal_date.is_none() {
        // Find the latest date with any signal (non-zero weight)
        for i in (0..n).rev() {
            let date = input.dates.value(i);
            let weight = input.weights.value(i);

            // Skip if this date is after the last price date
            if date > last_price_date {
                continue;
            }

            // Check if this is a new date
            if current_signal_date.is_none() || date == current_signal_date.unwrap() {
                if !weight.is_nan() && weight.abs() > FLOAT_EPSILON {
                    current_signal_date = Some(date);
                    let symbol = input.symbols.value(i).to_string();
                    signal_weights.insert(symbol, weight);
                }
            } else if current_signal_date.is_some() && date < current_signal_date.unwrap() {
                // We've moved past the latest signal date
                break;
            }
        }
    }

    // Determine signal_date (for weights) and next_weight_date (next rebalance)
    let (_signal_date, calculated_next_weight_date) = if let Some(future_date) = future_signal_date {
        // Found future signal in input - use it for both
        (future_date, future_date)
    } else if let Some(current_date) = current_signal_date {
        // Fallback to current signal - calculate next rebalance date from resample pattern
        let next_date = calculate_next_signal_date(current_date, resample, offset);
        (current_date, next_date)
    } else {
        // No signal found
        return ops;
    };

    // Normalize signal weights
    let total_weight: f64 = signal_weights.values().map(|w| w.abs()).sum();
    if total_weight > 1.0 {
        for w in signal_weights.values_mut() {
            *w /= total_weight;
        }
    }

    // Find current weights (from current positions)
    // For simplicity, we use position_weight from trades
    let current_weights: HashMap<String, f64> = trades
        .iter()
        .filter(|t| t.entry_date.is_some() && t.exit_date.is_none())
        .map(|t| (t.symbol.clone(), t.position_weight))
        .collect();

    // Calculate total current weight for normalization
    let total_current: f64 = current_weights.values().map(|w| w.abs()).sum();
    let normalized_current: HashMap<String, f64> = if total_current > 1.0 {
        current_weights
            .iter()
            .map(|(k, v)| (k.clone(), v / total_current))
            .collect()
    } else {
        current_weights.clone()
    };

    // Calculate actions by comparing current positions and signal weights
    let signal_stocks: HashSet<String> = signal_weights
        .iter()
        .filter(|(_, &w)| w.abs() > FLOAT_EPSILON)
        .map(|(k, _)| k.clone())
        .collect();

    for symbol in current_positions.union(&signal_stocks) {
        let in_current = current_positions.contains(symbol);
        let in_signal = signal_stocks.contains(symbol);

        let action = match (in_current, in_signal) {
            (false, true) => "enter",   // new position
            (true, false) => "exit",    // close position
            (true, true) => "hold",     // keep position
            (false, false) => continue, // should not happen
        };
        ops.actions.insert(symbol.clone(), action.to_string());
    }

    // Set weight dates
    // weight_date: the date of current weights (entry_sig_date of most recent entry)
    let weight_date = trades
        .iter()
        .filter(|t| t.entry_date.is_some() && t.exit_date.is_none())
        .map(|t| t.entry_sig_date)
        .max();

    ops.weights = normalized_current;
    ops.next_weights = signal_weights;
    ops.weight_date = weight_date;
    ops.next_weight_date = Some(calculated_next_weight_date);

    ops
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

/// Check if prev_date is a quarter-end using i32 dates (days since 1970-01-01)
///
/// Returns true if prev_date and next_date are in different quarters.
fn is_quarter_end_i32(prev_days: i32, next_days: i32) -> bool {
    let (prev_year, prev_month) = days_to_year_month(prev_days);
    let (next_year, next_month) = days_to_year_month(next_days);
    let prev_quarter = (prev_year, (prev_month - 1) / 3);
    let next_quarter = (next_year, (next_month - 1) / 3);
    prev_quarter != next_quarter
}

/// Check if prev_date is a year-end using i32 dates (days since 1970-01-01)
///
/// Returns true if prev_date and next_date are in different years.
fn is_year_end_i32(prev_days: i32, next_days: i32) -> bool {
    let (prev_year, _) = days_to_year_month(prev_days);
    let (next_year, _) = days_to_year_month(next_days);
    prev_year != next_year
}

/// Get weekday of a date (days since 1970-01-01)
///
/// Returns 0=Monday, 1=Tuesday, ..., 6=Sunday (ISO 8601 weekday)
/// 1970-01-01 was a Thursday (weekday 3)
#[inline]
fn weekday_of_i32(days: i32) -> u8 {
    // rem_euclid handles negative days correctly
    ((days.rem_euclid(7) + 3) % 7) as u8
}

/// Get all period boundaries between prev_date (inclusive) and date (exclusive)
/// for a given resample frequency.
///
/// This handles multi-period gaps (e.g., multiple weeks during a long holiday).
/// Returns boundaries in chronological order.
///
/// The function includes the period containing prev_date if prev_date is before or on
/// the boundary. For example, if prev_date = Dec 31 (quarter end) for quarterly,
/// it includes Dec 31 as a boundary.
fn get_all_period_boundaries(
    prev_date: i32,
    date: i32,
    resample: ResampleFreq,
) -> Vec<i32> {
    let mut boundaries = Vec::new();

    match resample {
        ResampleFreq::Weekly | ResampleFreq::WeeklyOn(_) => {
            // Find all Sundays from prev_date's week to the week before date
            // First: compute the Sunday of prev_date's week (on or after prev_date)
            let weekday = weekday_of_i32(prev_date);
            let first_sunday = if weekday == 6 {
                prev_date // prev_date is already Sunday
            } else {
                prev_date + (6 - weekday as i32)
            };

            // Iterate through all Sundays before date
            let mut sunday = first_sunday;
            while sunday < date {
                boundaries.push(sunday);
                sunday += 7;
            }
        }
        ResampleFreq::Monthly | ResampleFreq::MonthStart => {
            // Find all month ends from prev_date's month to the month before date
            let (mut year, mut month) = days_to_year_month(prev_date);
            loop {
                let month_end = last_day_of_month_i32(ymd_to_days(year, month, 1));
                // Include if boundary >= prev_date (prev_date could be before or on boundary)
                // and boundary < date
                if month_end >= prev_date && month_end < date {
                    boundaries.push(month_end);
                }
                // Move to next month
                if month == 12 {
                    month = 1;
                    year += 1;
                } else {
                    month += 1;
                }
                // Check if we've passed date
                if ymd_to_days(year, month, 1) >= date {
                    break;
                }
            }
        }
        ResampleFreq::Quarterly | ResampleFreq::QuarterStart => {
            // Find all quarter ends from prev_date's quarter to the quarter before date
            let (mut year, month) = days_to_year_month(prev_date);
            let mut qtr = ((month - 1) / 3) + 1; // 1, 2, 3, or 4
            loop {
                let qtr_end_month = qtr * 3;
                let qtr_end = last_day_of_quarter_i32(ymd_to_days(year, qtr_end_month, 1));
                // Include if boundary >= prev_date and boundary < date
                if qtr_end >= prev_date && qtr_end < date {
                    boundaries.push(qtr_end);
                }
                // Move to next quarter
                if qtr == 4 {
                    qtr = 1;
                    year += 1;
                } else {
                    qtr += 1;
                }
                // Check if we've passed date
                if ymd_to_days(year, qtr * 3, 1) >= date {
                    break;
                }
            }
        }
        ResampleFreq::Yearly => {
            // Find all year ends from prev_date's year to the year before date
            let (mut year, _) = days_to_year_month(prev_date);
            loop {
                let year_end = last_day_of_year_i32(ymd_to_days(year, 12, 31));
                // Include if boundary >= prev_date and boundary < date
                if year_end >= prev_date && year_end < date {
                    boundaries.push(year_end);
                }
                year += 1;
                if ymd_to_days(year, 1, 1) >= date {
                    break;
                }
            }
        }
        _ => {
            // Daily and PositionChange don't have period boundaries
        }
    }

    boundaries
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

/// Convert (year, month, day) to days since 1970-01-01
#[inline]
fn ymd_to_days(year: i32, month: u32, day: u32) -> i32 {
    // Algorithm from Howard Hinnant's date library
    // https://howardhinnant.github.io/date_algorithms.html
    let y = if month <= 2 { year - 1 } else { year };
    let m = if month <= 2 { month + 12 } else { month };
    let era = if y >= 0 { y / 400 } else { (y - 399) / 400 };
    let yoe = (y - era * 400) as u32;
    let doy = (153 * (m - 3) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe as i32 - 719468
}

/// Get number of days in a month
#[inline]
fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0) {
                29
            } else {
                28
            }
        }
        _ => 30, // fallback
    }
}

/// Get the last day of the month for a given date (days since epoch).
/// Returns the days since epoch for the last day of that month.
#[inline]
fn last_day_of_month_i32(days: i32) -> i32 {
    let (year, month) = days_to_year_month(days);
    let last_day = days_in_month(year, month);
    ymd_to_days(year, month, last_day)
}

/// Get the last day of the quarter for a given date (days since epoch).
/// Returns the days since epoch for the last day of that quarter.
#[inline]
fn last_day_of_quarter_i32(days: i32) -> i32 {
    let (year, month) = days_to_year_month(days);
    // Quarter end months: 3 (Mar), 6 (Jun), 9 (Sep), 12 (Dec)
    let quarter_end_month = ((month - 1) / 3 + 1) * 3;
    let last_day = days_in_month(year, quarter_end_month);
    ymd_to_days(year, quarter_end_month, last_day)
}

/// Get the last day of the year for a given date (days since epoch).
/// Returns the days since epoch for Dec 31 of that year.
#[inline]
fn last_day_of_year_i32(days: i32) -> i32 {
    let (year, _) = days_to_year_month(days);
    ymd_to_days(year, 12, 31)
}

/// Update positions with daily returns (cr *= r, maxcr, last_market_value, max_price)
///
/// NOTE: Does NOT update previous_price - call update_previous_prices separately
/// after touched_exit detection. This matches wide format behavior.
fn update_positions(portfolio: &mut Portfolio, prices: &HashMap<&str, f64>) {
    for (sym, pos) in portfolio.positions.iter_mut() {
        if let Some(&curr_price) = prices.get(sym.as_str()) {
            pos.update_with_return(curr_price);
            // NOTE: previous_price is NOT updated here.
            // Call update_previous_prices after detect_touched_exit.
        }
    }
}

/// Update previous_price for all positions after touched_exit detection
///
/// This should be called AFTER update_positions and detect_stops_unified
/// to properly track price history for the next day's calculations.
fn update_previous_prices(portfolio: &mut Portfolio, prices: &HashMap<&str, f64>) {
    for (sym, pos) in portfolio.positions.iter_mut() {
        if let Some(&curr_price) = prices.get(sym.as_str()) {
            if is_valid_price(curr_price) {
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
    today_factor: &HashMap<&str, f64>,
    config: &BacktestConfig,
    current_date: i32,
    tracker: &mut T,
)
where
    T: TradeTracker<Key = String, Date = i32, Record = TradeRecord>,
{
    for sym in pending_stops.drain(..) {
        if let Some(pos) = portfolio.positions.remove(&sym) {
            let exit_price = today_prices.get(sym.as_str()).copied().unwrap_or(pos.previous_price);
            let exit_factor = today_factor.get(sym.as_str()).copied().unwrap_or(1.0);
            tracker.close_trade(
                &sym,
                current_date,
                None, // Stop exit, no signal date
                exit_price,
                exit_factor,
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

/// Stop detection result with optional exit ratio for touched_exit mode
#[derive(Debug, Clone)]
pub struct StopResult {
    pub symbol: String,
    /// Exit ratio for touched_exit (position value multiplier), 1.0 for regular stops
    pub exit_ratio: f64,
}

/// Unified stop detection that handles both regular stops and touched_exit mode
///
/// For regular stops (ohlc = None): Uses close price only, exit_ratio = 1.0
/// For touched_exit (ohlc = Some): Uses OHLC prices, exit_ratio adjusted to touched price
fn detect_stops_unified(
    portfolio: &Portfolio,
    close_prices: &HashMap<&str, f64>,
    ohlc: Option<(&HashMap<&str, f64>, &HashMap<&str, f64>, &HashMap<&str, f64>)>,
    config: &BacktestConfig,
) -> Vec<StopResult> {
    let mut results = Vec::new();

    for (sym, pos) in portfolio.positions.iter() {
        let sym_str = sym.as_str();

        // Get current close price
        let close_price = close_prices.get(sym_str).copied().unwrap_or(0.0);
        if !is_valid_price(close_price) {
            continue;
        }

        let is_long = pos.last_market_value >= 0.0;
        let cr = pos.cr;
        let maxcr = pos.maxcr;

        // Calculate thresholds
        let (min_r, max_r) = if is_long {
            let stop_threshold = 1.0 - config.stop_loss;
            let trail_threshold = if config.trail_stop < f64::INFINITY {
                maxcr - config.trail_stop
            } else {
                f64::NEG_INFINITY
            };
            (stop_threshold.max(trail_threshold), 1.0 + config.take_profit)
        } else {
            let stop_threshold = 1.0 + config.stop_loss;
            let trail_threshold = if config.trail_stop < f64::INFINITY {
                maxcr + config.trail_stop
            } else {
                f64::INFINITY
            };
            (1.0 - config.take_profit, stop_threshold.min(trail_threshold))
        };

        // Handle touched_exit mode with OHLC
        if let Some((open_prices, high_prices, low_prices)) = ohlc {
            let open_price = open_prices.get(sym_str).copied().unwrap_or(f64::NAN);
            let high_price = high_prices.get(sym_str).copied().unwrap_or(f64::NAN);
            let low_price = low_prices.get(sym_str).copied().unwrap_or(f64::NAN);
            let prev_price = pos.previous_price;

            // Skip if OHLC invalid
            if open_price.is_nan() || high_price.is_nan() || low_price.is_nan()
                || prev_price <= 0.0 || cr.is_nan() || cr <= 0.0
            {
                continue;
            }

            let r = close_price / prev_price;
            if r.is_nan() || r <= 0.0 {
                continue;
            }

            // Finlab formulas (lines 342-344 of backtest_core.pyx):
            // open_r = cr[sid] / r * (open / prev)  = cr_old * (open / prev)
            // high_r = cr[sid] / r * (high / prev)  = cr_old * (high / prev)
            // low_r  = cr[sid] / r * (low / prev)   = cr_old * (low / prev)
            //
            // We compute in the EXACT same order as Finlab to ensure
            // identical floating point behavior.
            let open_r = cr / r * (open_price / prev_price);
            let high_r = cr / r * (high_price / prev_price);
            let low_r = cr / r * (low_price / prev_price);

            // Check touch conditions (Finlab lines 348-350)
            // NOTE: Uses <= for min_r (not <) to match Finlab's behavior
            let touch_open = open_r >= max_r || open_r <= min_r;
            let touch_high = high_r >= max_r;
            let touch_low = low_r <= min_r;

            // Priority: open > high > low (Finlab lines 354-380)
            // Exit ratio adjusts pos[sid] to touched price level
            if touch_open {
                // Finlab: pos[sid] *= open_r / r
                results.push(StopResult { symbol: sym.clone(), exit_ratio: open_r / r });
            } else if touch_high {
                // Finlab: pos[sid] = entry_pos * max_r = pos / cr * max_r
                results.push(StopResult { symbol: sym.clone(), exit_ratio: max_r / cr });
            } else if touch_low {
                // Finlab: pos[sid] = entry_pos * min_r = pos / cr * min_r
                results.push(StopResult { symbol: sym.clone(), exit_ratio: min_r / cr });
            }
        } else {
            // Regular stop detection (close price only)
            let cr_at_close = cr * close_price / close_price; // Finlab floating point behavior

            if is_long {
                if config.take_profit < f64::INFINITY && cr_at_close >= max_r {
                    results.push(StopResult { symbol: sym.clone(), exit_ratio: 1.0 });
                    continue;
                }
                if cr_at_close < min_r {
                    results.push(StopResult { symbol: sym.clone(), exit_ratio: 1.0 });
                }
            } else {
                if cr_at_close >= max_r {
                    results.push(StopResult { symbol: sym.clone(), exit_ratio: 1.0 });
                    continue;
                }
                if config.take_profit < f64::INFINITY && cr_at_close < min_r {
                    results.push(StopResult { symbol: sym.clone(), exit_ratio: 1.0 });
                }
            }
        }
    }

    results
}

/// Execute stop exits with optional exit_ratio adjustment (for touched_exit)
///
/// This handles both:
/// - Regular stops: exit_ratio = 1.0, scheduled for T+1
/// - Touched exits: exit_ratio adjusted, executed immediately T+0
fn execute_stops_impl<T>(
    portfolio: &mut Portfolio,
    stops: Vec<StopResult>,
    stopped_stocks: &mut HashMap<String, bool>,
    today_prices: &HashMap<&str, f64>,
    today_factor: &HashMap<&str, f64>,
    config: &BacktestConfig,
    current_date: i32,
    exit_sig_date: Option<i32>,  // Some(date) for touched_exit, None for regular
    tracker: &mut T,
)
where
    T: TradeTracker<Key = String, Date = i32, Record = TradeRecord>,
{
    for stop in stops {
        if let Some(pos) = portfolio.positions.remove(&stop.symbol) {
            // Adjust position value by exit_ratio (1.0 for regular, adjusted for touched)
            let exit_value = pos.last_market_value * stop.exit_ratio;

            // For touched_exit, adjust exit_price by exit_ratio (touched_price / close_price)
            let close_price = today_prices.get(stop.symbol.as_str()).copied().unwrap_or(pos.previous_price);
            let exit_price = close_price * stop.exit_ratio;
            let exit_factor = today_factor.get(stop.symbol.as_str()).copied().unwrap_or(1.0);
            tracker.close_trade(
                &stop.symbol,
                current_date,
                exit_sig_date,
                exit_price,
                exit_factor,
                config.fee_ratio,
                config.tax_ratio,
            );

            let sell_value = exit_value - exit_value.abs() * (config.fee_ratio + config.tax_ratio);
            portfolio.cash += sell_value;

            if config.stop_trading_next_period {
                stopped_stocks.insert(stop.symbol, true);
            }
        }
    }
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
    today_factor: &HashMap<&str, f64>,
    stopped_stocks: &HashMap<String, bool>,
    config: &BacktestConfig,
    current_date: i32,
    signal_date: i32,
    tracker: &mut T,
)
where
    T: TradeTracker<Key = String, Date = i32, Record = TradeRecord>,
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
        let exit_factor = today_factor.get(sym.as_str()).copied().unwrap_or(1.0);
        tracker.close_trade(
            sym,
            current_date,
            Some(signal_date),
            exit_price,
            exit_factor,
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

    // Store old positions as snapshots (single iteration instead of 3)
    let old_snapshots: HashMap<String, PositionSnapshot> = portfolio
        .positions
        .iter()
        .map(|(k, v)| (k.clone(), PositionSnapshot::from(v)))
        .collect();

    // Clear and rebuild
    portfolio.positions.clear();
    let mut cash = portfolio.cash;

    for (sym, &target_weight) in &effective_weights {
        // Note: stopped stocks are already filtered out in effective_weights when
        // stop_trading_next_period is true

        // Get price and check validity
        let price_opt = today_prices.get(sym.as_str()).copied();
        let price_valid = price_opt.map_or(false, |p| is_valid_price(p));

        // Target position value (scaled by ratio)
        let target_value = target_weight * ratio;
        let snapshot = old_snapshots.get(sym);
        let current_value = snapshot.map(|s| s.cost_basis).unwrap_or(0.0);

        // Handle NaN price case (match Finlab behavior: enter even with NaN price)
        if !price_valid {
            // If target is 0 and we have an old position, sell it using old market value
            // Note: close_trade was already called at the start for ALL positions
            if target_weight.abs() < FLOAT_EPSILON {
                if let Some(snap) = snapshot {
                    if snap.market_value.abs() > FLOAT_EPSILON {
                        let sell_fee = snap.market_value.abs() * (config.fee_ratio + config.tax_ratio);
                        cash += snap.market_value - sell_fee;
                    }
                }
                continue;
            }

            // Finlab behavior: Enter/modify position even with NaN price
            if target_value.abs() > FLOAT_EPSILON {
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

                if new_value.abs() > FLOAT_EPSILON {
                    // Finlab behavior: Open trade for ALL positions after rebalance
                    let entry_factor = today_factor.get(sym.as_str()).copied().unwrap_or(1.0);
                    tracker.open_trade(sym.clone(), current_date, signal_date, f64::NAN, target_weight, entry_factor);
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
        if target_weight.abs() < FLOAT_EPSILON {
            if current_value.abs() > FLOAT_EPSILON {
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

        if new_position_value.abs() > FLOAT_EPSILON {
            // Finlab behavior: Open trade for ALL positions after rebalance
            // (All trades were closed at the start of rebalance)
            let entry_factor = today_factor.get(sym.as_str()).copied().unwrap_or(1.0);
            tracker.open_trade(sym.clone(), current_date, signal_date, price, target_weight, entry_factor);

            // Determine if this is a continuing same-direction position
            let is_continuing = current_value.abs() > FLOAT_EPSILON && current_value * target_weight > 0.0;

            let new_pos = if config.retain_cost_when_rebalance && is_continuing {
                // Preserve stop tracking for continuing same-direction positions
                let snap = snapshot.unwrap(); // Safe: is_continuing implies snapshot exists
                Position::new_from_snapshot(new_position_value, price, snap)
            } else {
                // New position or direction change or retain_cost=False: reset all
                Position::new(new_position_value, price)
            };

            portfolio.positions.insert(sym.clone(), new_pos);
        }
    }

    // Handle positions outside effective_weights (sell them)
    // Note: close_trade was already called at the start for ALL positions
    for (sym, snap) in old_snapshots.iter() {
        if !effective_weights.contains_key(sym) && snap.cost_basis.abs() > FLOAT_EPSILON {
            let sell_fee = snap.cost_basis.abs() * (config.fee_ratio + config.tax_ratio);
            cash += snap.cost_basis - sell_fee;
        }
    }

    portfolio.cash = cash;
}

/// Check if two weight maps differ (for PositionChange mode)
///
/// Two weight maps differ if:
/// 1. They have different keys (symbols)
/// 2. Any corresponding weights differ by more than FLOAT_EPSILON
fn weights_differ(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> bool {
    // Different number of symbols
    if a.len() != b.len() {
        return true;
    }

    // Check each symbol
    for (sym, &weight_a) in a.iter() {
        match b.get(sym) {
            Some(&weight_b) => {
                if (weight_a - weight_b).abs() > FLOAT_EPSILON {
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
            w.abs() > FLOAT_EPSILON && !is_stopped
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
            open_prices: None,
            high_prices: None,
            low_prices: None,
            factor: None,
        }
    }

    #[test]
    fn test_backtest_empty() {
        let dates = Int32Array::from(Vec::<i32>::new());
        let symbols = make_symbols(vec![]);
        let prices = Float64Array::from(Vec::<f64>::new());
        let weights = Float64Array::from(Vec::<f64>::new());

        let input = make_input(&dates, &symbols, &prices, &weights);
        let result = backtest_long_arrow(&input, ResampleFreq::Daily, None, &BacktestConfig::default());

        assert!(result.creturn.is_empty());
    }

    #[test]
    fn test_backtest_single_stock() {
        // 4 days, 1 stock
        // Day 0: signal weight=1.0
        // Day 1: entry at T+1
        // Day 2: +10%
        // Day 3: +10%
        // Note: With ResampleFreq::Daily, weights must stay at 1.0 to keep position open
        let dates = Int32Array::from(vec![
            date_to_days("2024-01-01"),
            date_to_days("2024-01-02"),
            date_to_days("2024-01-03"),
            date_to_days("2024-01-04"),
        ]);
        let symbols = make_symbols(vec!["AAPL", "AAPL", "AAPL", "AAPL"]);
        let prices = Float64Array::from(vec![100.0, 100.0, 110.0, 121.0]);
        let weights = Float64Array::from(vec![1.0, 1.0, 1.0, 1.0]);

        let input = make_input(&dates, &symbols, &prices, &weights);
        let config = BacktestConfig {
            fee_ratio: 0.0,
            tax_ratio: 0.0,
            finlab_mode: true,
            ..Default::default()
        };

        let result = backtest_long_arrow(&input, ResampleFreq::Daily, None, &config);

        assert_eq!(result.creturn.len(), 4);
        assert!((result.creturn[0] - 1.0).abs() < FLOAT_EPSILON, "Day 0: {}", result.creturn[0]);
        assert!((result.creturn[1] - 1.0).abs() < FLOAT_EPSILON, "Day 1: {}", result.creturn[1]);
        assert!((result.creturn[2] - 1.1).abs() < FLOAT_EPSILON, "Day 2: {}", result.creturn[2]);
        assert!((result.creturn[3] - 1.21).abs() < FLOAT_EPSILON, "Day 3: {}", result.creturn[3]);
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

        let result = backtest_long_arrow(&input, ResampleFreq::Daily, None, &config);

        assert_eq!(result.creturn.len(), 3);
        assert!((result.creturn[0] - 1.0).abs() < FLOAT_EPSILON);
        assert!((result.creturn[1] - 1.0).abs() < FLOAT_EPSILON);
        // Day 2: 0.5 * 1.1 + 0.5 * 0.9 = 1.0
        assert!((result.creturn[2] - 1.0).abs() < FLOAT_EPSILON);
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

        let result = backtest_long_arrow(&input, ResampleFreq::Daily, None, &config);

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

        let result = backtest_long_arrow(&input, ResampleFreq::Monthly, None, &config);

        // Jan 31 is month-end, so signal triggers entry on Feb 1
        // Recording starts from first signal (Jan 31), not from the beginning
        // creturn[0] = Jan 31 (signal day) = 1.0
        // creturn[1] = Feb 1 (entry day) = 1.0
        // creturn[2] = Feb 2 (+10%) = 1.1
        assert_eq!(result.creturn.len(), 3, "Expected 3 days from signal day onward");
        assert!((result.creturn[0] - 1.0).abs() < FLOAT_EPSILON, "Jan 31 (signal): {}", result.creturn[0]);
        assert!((result.creturn[1] - 1.0).abs() < FLOAT_EPSILON, "Feb 1 (entry): {}", result.creturn[1]);
        assert!((result.creturn[2] - 1.1).abs() < FLOAT_EPSILON, "Feb 2 (+10%): {}", result.creturn[2]);
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

        let result = backtest_long_slice(&dates, &symbols, &prices, &weights, None, None, None, None, ResampleFreq::Daily, None, &config);

        assert_eq!(result.creturn.len(), 3);
        assert!((result.creturn[2] - 1.1).abs() < FLOAT_EPSILON);
    }
}
