//! Polars extension for portfolio backtesting
//!
//! This crate provides Polars expression functions for backtesting
//! trading strategies. It wraps the btcore library for use in Python
//! via PyO3 and pyo3-polars.
//!
//! # API Overview
//!
//! - `backtest()` - Main API for long format data (zero-copy, fastest)
//! - `backtest_with_report()` - Long format with full report (trades tracking)
//! - `backtest_wide()` - Wide format API (for validation/compatibility)
//! - `backtest_with_report_wide_impl()` - Wide format with full report
//!
//! # Profiling
//!
//! Set environment variable `POLARS_BACKTEST_PROFILE=1` to enable profiling output.

mod expressions;
mod ffi_convert;
mod report;

pub use report::PyBacktestReport;

/// Check if profiling is enabled via environment variable
fn is_profile_enabled() -> bool {
    std::env::var("POLARS_BACKTEST_PROFILE").map(|v| v == "1" || v.to_lowercase() == "true").unwrap_or(false)
}

/// Profile macro - only prints when POLARS_BACKTEST_PROFILE=1
macro_rules! profile {
    ($($arg:tt)*) => {
        if is_profile_enabled() {
            eprintln!($($arg)*);
        }
    };
}

use arrow::array::{Float64Array, Int32Array, StringViewArray};
use polars_arrow::array::{Array, PrimitiveArray, Utf8ViewArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3_polars::PyDataFrame;

use polars::prelude::*;

use btcore::{
    config::ExecOptions,
    run_backtest, run_backtest_with_trades, BacktestConfig, WideBacktestResult,
    PriceData, TradeRecord, WideTradeRecord,
    simulation::{
        backtest_long_arrow, backtest_with_report_long_arrow,
        LongFormatArrowInput, ResampleFreq, ResampleOffset, StockOperations,
    },
};

// =============================================================================
// Python Wrapper Types
// =============================================================================

/// Python wrapper for BacktestConfig
#[pyclass(name = "BacktestConfig")]
#[derive(Clone)]
pub struct PyBacktestConfig {
    inner: BacktestConfig,
}

#[pymethods]
impl PyBacktestConfig {
    #[new]
    #[pyo3(signature = (
        fee_ratio=0.001425,
        tax_ratio=0.003,
        stop_loss=1.0,
        take_profit=f64::INFINITY,
        trail_stop=f64::INFINITY,
        position_limit=1.0,
        retain_cost_when_rebalance=false,
        stop_trading_next_period=true,
        finlab_mode=false,
        touched_exit=false,
    ))]
    fn new(
        fee_ratio: f64,
        tax_ratio: f64,
        stop_loss: f64,
        take_profit: f64,
        trail_stop: f64,
        position_limit: f64,
        retain_cost_when_rebalance: bool,
        stop_trading_next_period: bool,
        finlab_mode: bool,
        touched_exit: bool,
    ) -> Self {
        Self {
            inner: BacktestConfig {
                fee_ratio,
                tax_ratio,
                stop_loss,
                take_profit,
                trail_stop,
                position_limit,
                retain_cost_when_rebalance,
                stop_trading_next_period,
                finlab_mode,
                touched_exit,
                exec: ExecOptions::default(),
            },
        }
    }

    #[getter]
    fn fee_ratio(&self) -> f64 { self.inner.fee_ratio }
    #[getter]
    fn tax_ratio(&self) -> f64 { self.inner.tax_ratio }
    #[getter]
    fn stop_loss(&self) -> f64 { self.inner.stop_loss }
    #[getter]
    fn take_profit(&self) -> f64 { self.inner.take_profit }
    #[getter]
    fn trail_stop(&self) -> f64 { self.inner.trail_stop }
    #[getter]
    fn position_limit(&self) -> f64 { self.inner.position_limit }
    #[getter]
    fn retain_cost_when_rebalance(&self) -> bool { self.inner.retain_cost_when_rebalance }
    #[getter]
    fn stop_trading_next_period(&self) -> bool { self.inner.stop_trading_next_period }
    #[getter]
    fn finlab_mode(&self) -> bool { self.inner.finlab_mode }
    #[getter]
    fn touched_exit(&self) -> bool { self.inner.touched_exit }

    fn __repr__(&self) -> String {
        format!(
            "BacktestConfig(fee_ratio={}, tax_ratio={}, stop_loss={}, take_profit={}, \
             trail_stop={}, position_limit={}, retain_cost_when_rebalance={}, \
             stop_trading_next_period={}, finlab_mode={}, touched_exit={})",
            self.inner.fee_ratio, self.inner.tax_ratio, self.inner.stop_loss,
            self.inner.take_profit, self.inner.trail_stop, self.inner.position_limit,
            self.inner.retain_cost_when_rebalance, self.inner.stop_trading_next_period,
            self.inner.finlab_mode, self.inner.touched_exit,
        )
    }
}

/// Python wrapper for WideTradeRecord (wide format, usize indices)
#[pyclass(name = "WideTradeRecord")]
#[derive(Clone)]
pub struct PyWideTradeRecord {
    #[pyo3(get)]
    pub stock_id: usize,
    #[pyo3(get)]
    pub entry_index: Option<usize>,
    #[pyo3(get)]
    pub exit_index: Option<usize>,
    #[pyo3(get)]
    pub entry_sig_index: usize,
    #[pyo3(get)]
    pub exit_sig_index: Option<usize>,
    #[pyo3(get)]
    pub position_weight: f64,
    #[pyo3(get)]
    pub entry_price: f64,
    #[pyo3(get)]
    pub exit_price: Option<f64>,
    #[pyo3(get)]
    pub trade_return: Option<f64>,
    // MAE/MFE metrics
    #[pyo3(get)]
    pub mae: Option<f64>,
    #[pyo3(get)]
    pub gmfe: Option<f64>,
    #[pyo3(get)]
    pub bmfe: Option<f64>,
    #[pyo3(get)]
    pub mdd: Option<f64>,
    #[pyo3(get)]
    pub pdays: Option<u32>,
    #[pyo3(get)]
    pub period: Option<u32>,
}

#[pymethods]
impl PyWideTradeRecord {
    fn holding_period(&self) -> Option<usize> {
        match (self.entry_index, self.exit_index) {
            (Some(entry), Some(exit)) => Some(exit - entry),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "WideTradeRecord(stock_id={}, entry={:?}, exit={:?}, weight={:.4}, return={:?})",
            self.stock_id, self.entry_index, self.exit_index,
            self.position_weight, self.trade_return,
        )
    }
}

impl From<WideTradeRecord> for PyWideTradeRecord {
    fn from(r: WideTradeRecord) -> Self {
        Self {
            stock_id: r.stock_id,
            entry_index: r.entry_index,
            exit_index: r.exit_index,
            entry_sig_index: r.entry_sig_index,
            exit_sig_index: r.exit_sig_index,
            position_weight: r.position_weight,
            entry_price: r.entry_price,
            exit_price: r.exit_price,
            trade_return: r.trade_return,
            mae: r.mae,
            gmfe: r.gmfe,
            bmfe: r.bmfe,
            mdd: r.mdd,
            pdays: r.pdays,
            period: r.period,
        }
    }
}

/// Python wrapper for BacktestResult (long format - returns DataFrame creturn)
#[pyclass(name = "BacktestResult")]
#[derive(Clone)]
pub struct PyBacktestResult {
    creturn_df: DataFrame,
}

#[pymethods]
impl PyBacktestResult {
    /// Get cumulative returns as a Polars DataFrame with date column
    #[getter]
    fn creturn(&self) -> PyDataFrame {
        PyDataFrame(self.creturn_df.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "BacktestResult(creturn_len={})",
            self.creturn_df.height(),
        )
    }
}

/// Python wrapper for wide format BacktestResult (returns Vec<f64> creturn for Report compatibility)
#[pyclass(name = "WideBacktestResult")]
#[derive(Clone)]
pub struct PyWideBacktestResult {
    #[pyo3(get)]
    pub creturn: Vec<f64>,
    #[pyo3(get)]
    pub trades: Vec<PyWideTradeRecord>,
}

#[pymethods]
impl PyWideBacktestResult {
    fn __repr__(&self) -> String {
        format!(
            "WideBacktestResult(creturn_len={}, trades_count={})",
            self.creturn.len(), self.trades.len(),
        )
    }
}

impl From<WideBacktestResult> for PyWideBacktestResult {
    fn from(r: WideBacktestResult) -> Self {
        Self {
            creturn: r.creturn,
            trades: r.trades.into_iter().map(|t| t.into()).collect(),
        }
    }
}

/// Python wrapper for TradeRecord (default format - string symbols, i32 dates)
#[pyclass(name = "TradeRecord")]
#[derive(Clone)]
pub struct PyTradeRecord {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub entry_date: Option<i32>,
    #[pyo3(get)]
    pub exit_date: Option<i32>,
    #[pyo3(get)]
    pub entry_sig_date: i32,
    #[pyo3(get)]
    pub exit_sig_date: Option<i32>,
    #[pyo3(get)]
    pub position_weight: f64,
    #[pyo3(get)]
    pub entry_price: f64,
    #[pyo3(get)]
    pub exit_price: Option<f64>,
    #[pyo3(get)]
    pub trade_return: Option<f64>,
    // MAE/MFE metrics
    #[pyo3(get)]
    pub mae: Option<f64>,
    #[pyo3(get)]
    pub gmfe: Option<f64>,
    #[pyo3(get)]
    pub bmfe: Option<f64>,
    #[pyo3(get)]
    pub mdd: Option<f64>,
    #[pyo3(get)]
    pub pdays: Option<u32>,
    #[pyo3(get)]
    pub period: Option<i32>,
}

#[pymethods]
impl PyTradeRecord {
    fn holding_days(&self) -> Option<i32> {
        match (self.entry_date, self.exit_date) {
            (Some(entry), Some(exit)) => Some(exit - entry),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TradeRecord(symbol='{}', entry_date={:?}, exit_date={:?}, weight={:.4}, return={:?})",
            self.symbol, self.entry_date, self.exit_date,
            self.position_weight, self.trade_return,
        )
    }
}

impl From<TradeRecord> for PyTradeRecord {
    fn from(r: TradeRecord) -> Self {
        Self {
            symbol: r.symbol,
            entry_date: r.entry_date,
            exit_date: r.exit_date,
            entry_sig_date: r.entry_sig_date,
            exit_sig_date: r.exit_sig_date,
            position_weight: r.position_weight,
            entry_price: r.entry_price,
            exit_price: r.exit_price,
            trade_return: r.trade_return,
            mae: r.mae,
            gmfe: r.gmfe,
            bmfe: r.bmfe,
            mdd: r.mdd,
            pdays: r.pdays,
            period: r.period,
        }
    }
}

/// Convert Vec<TradeRecord> to a Polars DataFrame
///
/// Columns:
/// - stock_id: String (symbol)
/// - entry_date: Date (days since epoch -> Date)
/// - exit_date: Date (optional)
/// - entry_sig_date: Date
/// - exit_sig_date: Date (optional)
/// - position: Float64 (position weight)
/// - period: Int32 (holding period in days, optional)
/// - return: Float64 (trade return, optional)
/// - entry_price: Float64
/// - exit_price: Float64 (optional)
/// - mae: Float64 (Maximum Adverse Excursion, optional)
/// - gmfe: Float64 (Global Maximum Favorable Excursion, optional)
/// - bmfe: Float64 (Before-MAE MFE, optional)
/// - mdd: Float64 (Maximum Drawdown, optional)
/// - pdays: UInt32 (Profitable days, optional)
fn trades_to_dataframe(trades: &[TradeRecord]) -> PolarsResult<DataFrame> {
    // Build columns
    let stock_id: Vec<&str> = trades.iter().map(|t| t.symbol.as_str()).collect();
    let entry_date: Vec<Option<i32>> = trades.iter().map(|t| t.entry_date).collect();
    let exit_date: Vec<Option<i32>> = trades.iter().map(|t| t.exit_date).collect();
    let entry_sig_date: Vec<i32> = trades.iter().map(|t| t.entry_sig_date).collect();
    let exit_sig_date: Vec<Option<i32>> = trades.iter().map(|t| t.exit_sig_date).collect();
    let position: Vec<f64> = trades.iter().map(|t| t.position_weight).collect();
    let period: Vec<Option<i32>> = trades.iter().map(|t| t.period).collect();
    let trade_return: Vec<Option<f64>> = trades.iter().map(|t| t.trade_return).collect();
    let entry_price: Vec<f64> = trades.iter().map(|t| t.entry_price).collect();
    let exit_price: Vec<Option<f64>> = trades.iter().map(|t| t.exit_price).collect();
    // Raw prices (adj_price / factor, for liquidity calculations)
    let entry_raw_price: Vec<f64> = trades.iter().map(|t| t.entry_raw_price).collect();
    let exit_raw_price: Vec<Option<f64>> = trades.iter().map(|t| t.exit_raw_price).collect();
    // MAE/MFE columns
    let mae: Vec<Option<f64>> = trades.iter().map(|t| t.mae).collect();
    let gmfe: Vec<Option<f64>> = trades.iter().map(|t| t.gmfe).collect();
    let bmfe: Vec<Option<f64>> = trades.iter().map(|t| t.bmfe).collect();
    let mdd: Vec<Option<f64>> = trades.iter().map(|t| t.mdd).collect();
    let pdays: Vec<Option<u32>> = trades.iter().map(|t| t.pdays).collect();

    // Create Series
    let stock_id_series = Series::new("stock_id".into(), stock_id);
    let entry_date_series = Series::new("entry_date".into(), entry_date)
        .cast(&DataType::Date)?;
    let exit_date_series = Series::new("exit_date".into(), exit_date)
        .cast(&DataType::Date)?;
    let entry_sig_date_series = Series::new("entry_sig_date".into(), entry_sig_date)
        .cast(&DataType::Date)?;
    let exit_sig_date_series = Series::new("exit_sig_date".into(), exit_sig_date)
        .cast(&DataType::Date)?;
    let position_series = Series::new("position".into(), position);
    let period_series = Series::new("period".into(), period);
    let return_series = Series::new("return".into(), trade_return);
    let entry_price_series = Series::new("entry_price".into(), entry_price);
    let exit_price_series = Series::new("exit_price".into(), exit_price);
    let entry_raw_price_series = Series::new("entry_raw_price".into(), entry_raw_price);
    let exit_raw_price_series = Series::new("exit_raw_price".into(), exit_raw_price);
    // MAE/MFE series
    let mae_series = Series::new("mae".into(), mae);
    let gmfe_series = Series::new("gmfe".into(), gmfe);
    let bmfe_series = Series::new("bmfe".into(), bmfe);
    let mdd_series = Series::new("mdd".into(), mdd);
    let pdays_series = Series::new("pdays".into(), pdays);

    DataFrame::new(vec![
        stock_id_series.into_column(),
        entry_date_series.into_column(),
        exit_date_series.into_column(),
        entry_sig_date_series.into_column(),
        exit_sig_date_series.into_column(),
        position_series.into_column(),
        period_series.into_column(),
        return_series.into_column(),
        entry_price_series.into_column(),
        exit_price_series.into_column(),
        entry_raw_price_series.into_column(),
        exit_raw_price_series.into_column(),
        mae_series.into_column(),
        gmfe_series.into_column(),
        bmfe_series.into_column(),
        mdd_series.into_column(),
        pdays_series.into_column(),
    ])
}

/// Convert StockOperations to weights and next_weights DataFrames
///
/// Returns (weights_df, next_weights_df) where each DataFrame has columns:
/// - symbol: String
/// - weight: Float64
/// - date: Date (weight_date for weights, next_weight_date for next_weights)
fn stock_operations_to_dataframes(
    ops: &StockOperations,
) -> PolarsResult<(Option<DataFrame>, Option<DataFrame>)> {
    // Create weights DataFrame with date column
    let weights_df = if ops.weights.is_empty() {
        None
    } else {
        let n = ops.weights.len();
        let symbols: Vec<&str> = ops.weights.keys().map(|s| s.as_str()).collect();
        let weights: Vec<f64> = ops.weights.values().copied().collect();
        // Add date column (same date for all rows)
        let dates: Vec<Option<i32>> = vec![ops.weight_date; n];
        Some(DataFrame::new(vec![
            Series::new("symbol".into(), symbols).into_column(),
            Series::new("weight".into(), weights).into_column(),
            Series::new("date".into(), dates).cast(&DataType::Date)?.into_column(),
        ])?)
    };

    // Create next_weights DataFrame with date column
    let next_weights_df = if ops.next_weights.is_empty() {
        None
    } else {
        let n = ops.next_weights.len();
        let symbols: Vec<&str> = ops.next_weights.keys().map(|s| s.as_str()).collect();
        let weights: Vec<f64> = ops.next_weights.values().copied().collect();
        // Add date column (same date for all rows)
        let dates: Vec<Option<i32>> = vec![ops.next_weight_date; n];
        Some(DataFrame::new(vec![
            Series::new("symbol".into(), symbols).into_column(),
            Series::new("weight".into(), weights).into_column(),
            Series::new("date".into(), dates).cast(&DataType::Date)?.into_column(),
        ])?)
    };

    Ok((weights_df, next_weights_df))
}

// PyBacktestReport is now defined in report.rs

// =============================================================================
// Long Format Input Conversion
// =============================================================================

/// Column names of a long format DataFrame
struct LongColumnNames<'a> {
    date: &'a str,
    symbol: &'a str,
    trade_at_price: &'a str,
    position: &'a str,
    open: &'a str,
    high: &'a str,
    low: &'a str,
    factor: Option<&'a str>,
}

/// Long format columns as arrow-rs arrays (zero-copy), ready for btcore
struct LongArrays {
    dates: Int32Array,
    symbols: StringViewArray,
    prices: Float64Array,
    weights: Float64Array,
    open: Option<Float64Array>,
    high: Option<Float64Array>,
    low: Option<Float64Array>,
    factor: Option<Float64Array>,
}

impl LongArrays {
    fn input(&self) -> LongFormatArrowInput<'_> {
        LongFormatArrowInput {
            dates: &self.dates,
            symbols: &self.symbols,
            prices: &self.prices,
            weights: &self.weights,
            open_prices: self.open.as_ref(),
            high_prices: self.high.as_ref(),
            low_prices: self.low.as_ref(),
            factor: self.factor.as_ref(),
        }
    }
}

/// Config passed by the caller, or the long format default
fn long_config(config: Option<PyBacktestConfig>) -> BacktestConfig {
    config.map(|c| c.inner).unwrap_or_else(|| BacktestConfig {
        finlab_mode: true,
        ..Default::default()
    })
}

/// Check that the required columns exist, then sort by date unless told not to
fn validated_long_frame(
    df: DataFrame,
    names: &LongColumnNames,
    touched_exit: bool,
    skip_sort: bool,
) -> PyResult<DataFrame> {
    for col_name in [
        names.date,
        names.symbol,
        names.trade_at_price,
        names.position,
    ] {
        if df.column(col_name).is_err() {
            return Err(PyValueError::new_err(format!(
                "Missing required column: '{}'",
                col_name
            )));
        }
    }

    // Validate OHLC columns if touched_exit is enabled
    if touched_exit {
        for col_name in [names.open, names.high, names.low] {
            if df.column(col_name).is_err() {
                return Err(PyValueError::new_err(format!(
                    "touched_exit=True requires column '{}', but it is missing",
                    col_name
                )));
            }
        }
    }

    if skip_sort {
        return Ok(df);
    }
    df.sort([names.date], SortMultipleOptions::default())
        .map_err(|e| PyValueError::new_err(format!("Failed to sort: {}", e)))
}

/// Convert a column to an arrow-rs Float64Array, casting and rechunking only when needed
fn float_column_to_arrow(df: &DataFrame, name: &str) -> PyResult<Float64Array> {
    let column = df
        .column(name)
        .map_err(|e| PyValueError::new_err(format!("Failed to get {} column: {}", name, e)))?;
    let series = if column.dtype() == &DataType::Float64 {
        column.clone()
    } else {
        column
            .cast(&DataType::Float64)
            .map_err(|e| PyValueError::new_err(format!("Failed to cast {}: {}", name, e)))?
    };
    let values = series
        .f64()
        .map_err(|e| PyValueError::new_err(format!("{} must be f64: {}", name, e)))?;
    let rechunked;
    let single_chunk: &Float64Chunked = if values.chunks().len() > 1 {
        rechunked = values.rechunk();
        &rechunked
    } else {
        values
    };
    let array = single_chunk.chunks()[0]
        .as_any()
        .downcast_ref::<PrimitiveArray<f64>>()
        .ok_or_else(|| PyValueError::new_err(format!("Failed to downcast {} array", name)))?;
    ffi_convert::polars_f64_to_arrow(array)
        .map_err(|e| PyValueError::new_err(format!("FFI {} conversion failed: {}", name, e)))
}

/// Convert the columns btcore reads into arrow-rs arrays
fn long_arrays(
    df: &DataFrame,
    names: &LongColumnNames,
    touched_exit: bool,
    skip_sort: bool,
) -> PyResult<LongArrays> {
    let date_col_ref = df
        .column(names.date)
        .map_err(|e| PyValueError::new_err(format!("Failed to get date column: {}", e)))?;
    let date_series = if date_col_ref.dtype() == &DataType::Date {
        date_col_ref.clone()
    } else {
        date_col_ref
            .cast(&DataType::Date)
            .map_err(|e| PyValueError::new_err(format!("Failed to cast date: {}", e)))?
    };
    let date_phys = date_series
        .date()
        .map_err(|e| PyValueError::new_err(format!("Date column must be Date: {}", e)))?
        .physical();
    let date_ca_rechunked;
    let date_ca: &ChunkedArray<Int32Type> = if date_phys.chunks().len() > 1 {
        date_ca_rechunked = date_phys.rechunk();
        &date_ca_rechunked
    } else {
        date_phys
    };

    let symbol_ref = df
        .column(names.symbol)
        .map_err(|e| PyValueError::new_err(format!("Failed to get symbol column: {}", e)))?
        .str()
        .map_err(|e| PyValueError::new_err(format!("Symbol column must be string: {}", e)))?;
    let symbol_ca_rechunked;
    let symbol_ca: &StringChunked = if symbol_ref.chunks().len() > 1 {
        symbol_ca_rechunked = symbol_ref.rechunk();
        &symbol_ca_rechunked
    } else {
        symbol_ref
    };

    // Downcast to concrete polars-arrow types (single chunk guaranteed by rechunk)
    let dates_arrow = date_ca.chunks()[0]
        .as_any()
        .downcast_ref::<PrimitiveArray<i32>>()
        .ok_or_else(|| PyValueError::new_err("Failed to downcast date array"))?;

    let symbols_arrow = symbol_ca.chunks()[0]
        .as_any()
        .downcast_ref::<Utf8ViewArray>()
        .ok_or_else(|| PyValueError::new_err("Failed to downcast symbol array"))?;

    // Validity bitmaps are not consulted past this point, so null slots would be
    // read as undefined buffer bytes — reject them here.
    if dates_arrow.null_count() > 0 {
        return Err(PyValueError::new_err(
            "date column contains null values; drop or fill them before backtesting",
        ));
    }
    if symbols_arrow.null_count() > 0 {
        return Err(PyValueError::new_err(
            "symbol column contains null values; drop or fill them before backtesting",
        ));
    }
    // skip_sort callers promise date-sorted data; unsorted input silently corrupts
    // day grouping, so verify the promise.
    if skip_sort && dates_arrow.values().windows(2).any(|w| w[0] > w[1]) {
        return Err(PyValueError::new_err(
            "skip_sort=true but the date column is not sorted in ascending order",
        ));
    }

    // Convert polars-arrow arrays to arrow-rs arrays using FFI (zero-copy)
    let dates = ffi_convert::polars_i32_to_arrow(dates_arrow)
        .map_err(|e| PyValueError::new_err(format!("FFI date conversion failed: {}", e)))?;
    let symbols = ffi_convert::polars_utf8view_to_arrow(symbols_arrow)
        .map_err(|e| PyValueError::new_err(format!("FFI symbol conversion failed: {}", e)))?;
    let prices = float_column_to_arrow(df, names.trade_at_price)?;
    let weights = float_column_to_arrow(df, names.position)?;

    // OHLC columns are only read when touched_exit is enabled
    let (open, high, low) = if touched_exit {
        (
            Some(float_column_to_arrow(df, names.open)?),
            Some(float_column_to_arrow(df, names.high)?),
            Some(float_column_to_arrow(df, names.low)?),
        )
    } else {
        (None, None, None)
    };

    // A factor column that doesn't exist defaults to 1.0 in btcore
    let factor = match names.factor {
        Some(factor_col) if df.column(factor_col).is_ok() => {
            Some(float_column_to_arrow(df, factor_col)?)
        }
        _ => None,
    };

    Ok(LongArrays {
        dates,
        symbols,
        prices,
        weights,
        open,
        high,
        low,
        factor,
    })
}

/// Parse resample_offset, failing on anything the engine cannot honour
fn parse_resample_offset(resample_offset: Option<&str>) -> PyResult<Option<ResampleOffset>> {
    // from_str returns None for anything it cannot parse (e.g. negative offsets like
    // "-1D") — erroring beats silently rebalancing on unshifted boundaries.
    match resample_offset {
        Some(raw) => match ResampleOffset::from_str(resample_offset) {
            Some(parsed) => Ok(Some(parsed)),
            None => Err(PyValueError::new_err(format!(
                "Unsupported resample_offset '{}': only non-negative day/week \
offsets such as '5D' or '1W' are supported",
                raw
            ))),
        },
        None => Ok(None),
    }
}

/// Build the creturn DataFrame: drop the flat head and rebase to the first kept value
fn creturn_frame(date: &str, dates: &[i32], creturn: &[f64]) -> PyResult<DataFrame> {
    // Create unique_dates Series directly from btcore result (zero overhead)
    let unique_dates = Series::new(date.into(), dates)
        .cast(&DataType::Date)
        .map_err(|e| PyValueError::new_err(format!("Failed to cast to Date: {}", e)))?;

    // Handle empty result case (e.g., no valid signals)
    if creturn.is_empty() {
        let empty_creturn = Series::new("creturn".into(), Vec::<f64>::new());
        return DataFrame::new(vec![
            unique_dates.into_column(),
            empty_creturn.into_column(),
        ])
        .map_err(|e| PyValueError::new_err(format!("Failed to create empty DataFrame: {}", e)));
    }

    // Create creturn Series and find first index where != 1.0 using Polars boolean mask
    let creturn_series = Series::new("creturn".into(), creturn);
    let mask = creturn_series
        .f64()
        .map_err(|e| PyValueError::new_err(format!("creturn must be f64: {}", e)))?
        .not_equal(1.0);

    // Find first true index from boolean ChunkedArray
    let first_change_idx = mask
        .into_iter()
        .position(|opt| opt == Some(true))
        .unwrap_or(0);

    // Go back one day to include the signal/entry day
    let first_active_idx = if first_change_idx > 0 {
        first_change_idx - 1
    } else {
        0
    };
    let len = creturn.len() - first_active_idx;

    // Slice dates and creturn using Polars slice (zero-copy)
    let sliced_dates = unique_dates.slice(first_active_idx as i64, len);
    let sliced_creturn = creturn_series.slice(first_active_idx as i64, len);

    // Normalize: divide by first value using Polars ops
    let first_value = sliced_creturn.f64()
        .map_err(|e| PyValueError::new_err(format!("slice f64 failed: {}", e)))?
        .first()
        .unwrap_or(1.0);

    let creturn_col = if first_value != 0.0 && (first_value - 1.0).abs() > 1e-10 {
        (sliced_creturn / first_value).into_column()
    } else {
        sliced_creturn.into_column()
    };

    DataFrame::new(vec![sliced_dates.into_column(), creturn_col])
        .map_err(|e| PyValueError::new_err(format!("Failed to create creturn DataFrame: {}", e)))
}

// =============================================================================
// Main API: Long Format Backtest (zero-copy)
// =============================================================================

/// Run backtest on long format DataFrame (zero-copy, fastest)
///
/// This is the main API for backtesting. It processes long format data directly
/// using Arrow arrays with true zero-copy access via FFI.
///
/// Args:
///     df: DataFrame with columns [date, symbol, trade_at_price, position]
///     date: Name of date column (default: "date")
///     symbol: Name of symbol column (default: "symbol")
///     trade_at_price: Name of price column (default: "close")
///     position: Name of position/weight column (default: "weight")
///     open: Name of open price column (default: "open", for touched_exit)
///     high: Name of high price column (default: "high", for touched_exit)
///     low: Name of low price column (default: "low", for touched_exit)
///     factor: Name of factor column for raw price calculation (default: None)
///             If provided, raw_price = adj_price / factor
///             If column doesn't exist, factor defaults to 1.0
///     resample: Rebalancing frequency ("D", "W", "M", or None for daily)
///     config: BacktestConfig (optional)
///     skip_sort: Skip sorting if data is already sorted by date (default: false)
///
/// Returns:
///     PyBacktestResult with creturn
#[pyfunction]
#[pyo3(signature = (
    df,
    date="date",
    symbol="symbol",
    trade_at_price="close",
    position="weight",
    open="open",
    high="high",
    low="low",
    factor=None,
    resample=None,
    resample_offset=None,
    config=None,
    skip_sort=false
))]
fn backtest(
    df: PyDataFrame,
    date: &str,
    symbol: &str,
    trade_at_price: &str,
    position: &str,
    open: &str,
    high: &str,
    low: &str,
    factor: Option<&str>,
    resample: Option<&str>,
    resample_offset: Option<&str>,
    config: Option<PyBacktestConfig>,
    skip_sort: bool,
) -> PyResult<PyBacktestResult> {
    use fastant::Instant;

    let total_start = Instant::now();
    let mut step_start = Instant::now();

    let n_rows = df.0.height();
    let names = LongColumnNames {
        date,
        symbol,
        trade_at_price,
        position,
        open,
        high,
        low,
        factor,
    };
    let cfg = long_config(config);
    let df = validated_long_frame(df.0, &names, cfg.touched_exit, skip_sort)?;

    profile!(
        "[PROFILE] Sort: {:?} (rows={})",
        step_start.elapsed(),
        n_rows
    );
    step_start = Instant::now();

    let arrays = long_arrays(&df, &names, cfg.touched_exit, skip_sort)?;

    profile!("[PROFILE] Columns to arrow-rs: {:?}", step_start.elapsed());
    step_start = Instant::now();

    let resample_freq = ResampleFreq::from_str(resample);
    let offset = parse_resample_offset(resample_offset)?;

    // Run backtest using btcore with arrow-rs arrays
    let result = backtest_long_arrow(&arrays.input(), resample_freq, offset, &cfg);

    profile!("[PROFILE] Backtest (btcore): {:?}", step_start.elapsed());
    step_start = Instant::now();

    let creturn_df = creturn_frame(date, &result.dates, &result.creturn)?;

    profile!("[PROFILE] Build result: {:?}", step_start.elapsed());
    profile!("[PROFILE] TOTAL: {:?}", total_start.elapsed());

    Ok(PyBacktestResult { creturn_df })
}

/// Run backtest with report (trades as Polars DataFrame)
///
/// Args:
///     df: DataFrame with columns [date, symbol, trade_at_price, position]
///     date: Name of date column (default: "date")
///     symbol: Name of symbol column (default: "symbol")
///     trade_at_price: Name of price column (default: "close")
///     position: Name of position/weight column (default: "weight")
///     open: Name of open price column (default: "open", for touched_exit)
///     high: Name of high price column (default: "high", for touched_exit)
///     low: Name of low price column (default: "low", for touched_exit)
///     factor: Name of factor column for raw price calculation (default: None)
///             If provided, raw_price = adj_price / factor
///             If column doesn't exist, factor defaults to 1.0
///     resample: Rebalancing frequency ("D", "W", "M", or None for daily)
///     resample_offset: Optional offset for rebalance dates (e.g., "1d", "2d", "1W")
///     config: BacktestConfig (optional)
///     skip_sort: Skip sorting if data is already sorted by date (default: false)
///
/// Returns:
///     BacktestReport with creturn (Vec<f64>) and trades (DataFrame)
#[pyfunction]
#[pyo3(signature = (
    df,
    date="date",
    symbol="symbol",
    trade_at_price="close",
    position="weight",
    open="open",
    high="high",
    low="low",
    factor=None,
    resample=None,
    resample_offset=None,
    config=None,
    skip_sort=false,
    benchmark=None,
    limit_up="limit_up",
    limit_down="limit_down",
    trading_value="trading_value"
))]
fn backtest_with_report(
    df: PyDataFrame,
    date: &str,
    symbol: &str,
    trade_at_price: &str,
    position: &str,
    open: &str,
    high: &str,
    low: &str,
    factor: Option<&str>,
    resample: Option<&str>,
    resample_offset: Option<&str>,
    config: Option<PyBacktestConfig>,
    skip_sort: bool,
    benchmark: Option<Py<PyAny>>,
    limit_up: &str,
    limit_down: &str,
    trading_value: &str,
) -> PyResult<PyBacktestReport> {
    let n_rows = df.0.height();
    let names = LongColumnNames {
        date,
        symbol,
        trade_at_price,
        position,
        open,
        high,
        low,
        factor,
    };
    let cfg = long_config(config);
    let df = validated_long_frame(df.0, &names, cfg.touched_exit, skip_sort)?;
    let arrays = long_arrays(&df, &names, cfg.touched_exit, skip_sort)?;

    let resample_freq = ResampleFreq::from_str(resample);
    let offset = parse_resample_offset(resample_offset)?;

    // Run backtest with report using btcore
    let result = backtest_with_report_long_arrow(&arrays.input(), resample_freq, offset, &cfg);

    // Process benchmark parameter (can be str column name or PyDataFrame)
    let benchmark_df: Option<DataFrame> = if let Some(bm_obj) = benchmark {
        Python::attach(|py| -> PyResult<Option<DataFrame>> {
            // Try to extract as string (column name)
            if let Ok(bm_col) = bm_obj.extract::<&str>(py) {
                // Extract benchmark from df: group by date, take first value per date
                let bm_df = df
                    .clone()
                    .lazy()
                    .group_by([col(date)])
                    .agg([col(bm_col).first().alias("creturn")])
                    .sort([date], Default::default())
                    .with_column(col(date).alias("date"))
                    .select([col("date"), col("creturn")])
                    .collect()
                    .map_err(|e| PyValueError::new_err(format!("Failed to extract benchmark column '{}': {}", bm_col, e)))?;
                Ok(Some(bm_df))
            }
            // Try to extract as PyDataFrame
            else if let Ok(bm_pydf) = bm_obj.extract::<PyDataFrame>(py) {
                let bm_df = bm_pydf.0;
                // Validate benchmark has required columns
                if bm_df.column("date").is_err() || bm_df.column("creturn").is_err() {
                    return Err(PyValueError::new_err(
                        "Benchmark DataFrame must have 'date' and 'creturn' columns"
                    ));
                }
                Ok(Some(bm_df))
            }
            else {
                Err(PyValueError::new_err(
                    "benchmark must be a column name (str) or DataFrame with 'date' and 'creturn' columns"
                ))
            }
        })?
    } else {
        None
    };

    // Extract limit_prices_df if limit_up or limit_down columns exist
    let limit_prices_df: Option<DataFrame> = {
        let has_limit_up = df.column(limit_up).is_ok();
        let has_limit_down = df.column(limit_down).is_ok();

        if has_limit_up || has_limit_down {
            let mut select_cols = vec![col(date).alias("date"), col(symbol).alias("symbol")];
            if has_limit_up {
                select_cols.push(col(limit_up).alias("limit_up"));
            }
            if has_limit_down {
                select_cols.push(col(limit_down).alias("limit_down"));
            }
            // Dedup: duplicate (date, symbol) rows would multiply trades in the
            // liquidity-metric joins and silently skew buyHigh/sellLow/capacity
            Some(df.clone().lazy().select(select_cols)
                .unique(Some(cols(["date", "symbol"])), UniqueKeepStrategy::First)
                .collect()
                .map_err(|e| PyValueError::new_err(format!("Failed to extract limit prices: {}", e)))?)
        } else {
            None
        }
    };

    // Extract trading_value_df if trading_value column exists (for capacity metric)
    let trading_value_df: Option<DataFrame> = if df.column(trading_value).is_ok() {
        Some(df.clone().lazy().select([
            col(date).alias("date"),
            col(symbol).alias("symbol"),
            col(trading_value).alias("trading_value"),
        ])
            .unique(Some(cols(["date", "symbol"])), UniqueKeepStrategy::First)
            .collect()
            .map_err(|e| PyValueError::new_err(format!("Failed to extract trading value: {}", e)))?)
    } else {
        None
    };

    let creturn_df = creturn_frame(date, &result.dates, &result.creturn)?;

    // Convert trades to DataFrame
    let trades_df = trades_to_dataframe(&result.trades)
        .map_err(|e| PyValueError::new_err(format!("Failed to create trades DataFrame: {}", e)))?;

    profile!("[PROFILE] backtest_with_report: {} rows, {} trades", n_rows, result.trades.len());

    // Extract weights and next_weights from stock_operations
    let (weights_df, next_weights_df) = if let Some(ref ops) = result.stock_operations {
        stock_operations_to_dataframes(ops)
            .map_err(|e| PyValueError::new_err(format!("Failed to create weights DataFrames: {}", e)))?
    } else {
        (None, None)
    };

    Ok(PyBacktestReport::new(
        creturn_df,
        trades_df,
        cfg,
        resample.map(|s| s.to_string()),
        benchmark_df,
        limit_prices_df,
        trading_value_df,
        weights_df,
        next_weights_df,
    ))
}

// =============================================================================
// Wide Format API (for validation/compatibility)
// =============================================================================

/// Run backtest with wide format data (for validation)
///
/// Args:
///     prices: DataFrame with dates as rows, stocks as columns (Float64)
///     weights: DataFrame with rebalance dates as rows, stocks as columns (Float64)
///     rebalance_indices: List of row indices where rebalancing occurs
///     config: BacktestConfig (optional)
///
/// Returns:
///     List[float]: Cumulative returns at each time step
#[pyfunction]
#[pyo3(signature = (prices, weights, rebalance_indices, config=None))]
fn backtest_wide(
    prices: PyDataFrame,
    weights: PyDataFrame,
    rebalance_indices: Vec<usize>,
    config: Option<PyBacktestConfig>,
) -> PyResult<Vec<f64>> {
    let prices_df = prices.0;
    let weights_df = weights.0;

    let prices_2d = df_to_f64_2d(&prices_df)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert prices: {}", e)))?;

    let weights_2d = df_to_f64_2d(&weights_df)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert weights: {}", e)))?;

    let cfg = config.map(|c| c.inner).unwrap_or_default();

    Ok(run_backtest(&prices_2d, &weights_2d, &rebalance_indices, &cfg))
}

/// Run backtest with trades tracking (wide format)
///
/// Args:
///     adj_prices: DataFrame with adjusted prices (for creturn)
///     original_prices: DataFrame with original prices (for trades)
///     weights: DataFrame with weights
///     rebalance_indices: List of row indices where rebalancing occurs
///     config: BacktestConfig (optional)
///     open_prices/high_prices/low_prices: Optional OHLC for touched_exit
///
/// Returns:
///     WideBacktestResult with creturn (Vec<f64>) and trades
#[pyfunction]
#[pyo3(signature = (adj_prices, original_prices, weights, rebalance_indices, config=None, open_prices=None, high_prices=None, low_prices=None))]
fn backtest_with_report_wide_impl(
    adj_prices: PyDataFrame,
    original_prices: PyDataFrame,
    weights: PyDataFrame,
    rebalance_indices: Vec<usize>,
    config: Option<PyBacktestConfig>,
    open_prices: Option<PyDataFrame>,
    high_prices: Option<PyDataFrame>,
    low_prices: Option<PyDataFrame>,
) -> PyResult<PyWideBacktestResult> {
    let adj_prices_df = adj_prices.0;
    let original_prices_df = original_prices.0;
    let weights_df = weights.0;

    let adj_prices_2d = df_to_f64_2d(&adj_prices_df)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert adj_prices: {}", e)))?;

    let original_prices_2d = df_to_f64_2d(&original_prices_df)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert original_prices: {}", e)))?;

    let weights_2d = df_to_f64_2d(&weights_df)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert weights: {}", e)))?;

    let open_prices_2d = open_prices
        .map(|df| df_to_f64_2d(&df.0))
        .transpose()
        .map_err(|e| PyValueError::new_err(format!("Failed to convert open_prices: {}", e)))?;

    let high_prices_2d = high_prices
        .map(|df| df_to_f64_2d(&df.0))
        .transpose()
        .map_err(|e| PyValueError::new_err(format!("Failed to convert high_prices: {}", e)))?;

    let low_prices_2d = low_prices
        .map(|df| df_to_f64_2d(&df.0))
        .transpose()
        .map_err(|e| PyValueError::new_err(format!("Failed to convert low_prices: {}", e)))?;

    let cfg = config.map(|c| c.inner).unwrap_or_default();

    let prices = match (&open_prices_2d, &high_prices_2d, &low_prices_2d) {
        (Some(open), Some(high), Some(low)) => {
            PriceData::with_ohlc(&adj_prices_2d, &original_prices_2d, open, high, low)
        }
        _ => PriceData::new(&adj_prices_2d, &original_prices_2d),
    };

    let result = run_backtest_with_trades(&prices, &weights_2d, &rebalance_indices, &cfg);

    // Wide format returns raw creturn Vec<f64> for Report compatibility
    Ok(result.into())
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Convert DataFrame to Vec<Vec<f64>> (row-major)
fn df_to_f64_2d(df: &DataFrame) -> Result<Vec<Vec<f64>>, String> {
    let n_rows = df.height();
    let n_cols = df.width();

    if n_cols == 0 {
        return Ok(vec![]);
    }

    // Get column slices
    let col_slices: Vec<Vec<f64>> = df
        .get_columns()
        .iter()
        .map(|col| {
            let f64_col = col.cast(&DataType::Float64)
                .map_err(|e| format!("Failed to cast column: {}", e))?;
            let ca = f64_col.f64()
                .map_err(|e| format!("Failed to get f64 chunked array: {}", e))?;

            match ca.cont_slice() {
                Ok(slice) => Ok(slice.to_vec()),
                Err(_) => Ok(ca.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect()),
            }
        })
        .collect::<Result<Vec<_>, String>>()?;

    // Build row-major result
    let mut result = Vec::with_capacity(n_rows);
    for row_idx in 0..n_rows {
        let mut row = Vec::with_capacity(n_cols);
        for col_data in &col_slices {
            row.push(col_data[row_idx]);
        }
        result.push(row);
    }

    Ok(result)
}

// =============================================================================
// Module Initialization
// =============================================================================

#[pymodule]
fn _polars_backtest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // The zero-copy FFI path transmutes between polars-arrow and arrow-rs C-ABI
    // structs; verify layout compatibility once instead of assuming it forever.
    ffi_convert::verify_ffi_compatibility()
        .map_err(|e| PyValueError::new_err(format!("Arrow FFI incompatibility: {}", e)))?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // Config
    m.add_class::<PyBacktestConfig>()?;
    // Default types (long format)
    m.add_class::<PyTradeRecord>()?;
    m.add_class::<PyBacktestResult>()?;
    m.add_class::<PyBacktestReport>()?;
    // Wide format types
    m.add_class::<PyWideTradeRecord>()?;
    m.add_class::<PyWideBacktestResult>()?;
    // Main API (long format)
    m.add_function(wrap_pyfunction!(backtest, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_with_report, m)?)?;
    // Wide format API (for validation)
    m.add_function(wrap_pyfunction!(backtest_wide, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_with_report_wide_impl, m)?)?;
    Ok(())
}
