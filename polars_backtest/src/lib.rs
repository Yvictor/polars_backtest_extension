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

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3_polars::PyDataFrame;

use polars::prelude::*;

use btcore::{
    run_backtest, run_backtest_with_trades, BacktestConfig, WideBacktestResult,
    PriceData, TradeRecord, WideTradeRecord,
    simulation::{
        backtest_long_arrow, backtest_with_report_long_arrow,
        LongFormatArrowInput, ResampleFreq, ResampleOffset,
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
fn trades_to_dataframe(trades: &[TradeRecord]) -> PolarsResult<DataFrame> {
    // Build columns
    let stock_id: Vec<&str> = trades.iter().map(|t| t.symbol.as_str()).collect();
    let entry_date: Vec<Option<i32>> = trades.iter().map(|t| t.entry_date).collect();
    let exit_date: Vec<Option<i32>> = trades.iter().map(|t| t.exit_date).collect();
    let entry_sig_date: Vec<i32> = trades.iter().map(|t| t.entry_sig_date).collect();
    let exit_sig_date: Vec<Option<i32>> = trades.iter().map(|t| t.exit_sig_date).collect();
    let position: Vec<f64> = trades.iter().map(|t| t.position_weight).collect();
    let period: Vec<Option<i32>> = trades.iter().map(|t| {
        match (t.entry_date, t.exit_date) {
            (Some(e), Some(x)) => Some(x - e),
            _ => None,
        }
    }).collect();
    let trade_return: Vec<Option<f64>> = trades.iter().map(|t| t.trade_return).collect();
    let entry_price: Vec<f64> = trades.iter().map(|t| t.entry_price).collect();
    let exit_price: Vec<Option<f64>> = trades.iter().map(|t| t.exit_price).collect();

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
    ])
}

/// Python wrapper for backtest report with trades as DataFrame
#[pyclass(name = "BacktestReport")]
#[derive(Clone)]
pub struct PyBacktestReport {
    creturn_df: DataFrame,
    trades_df: DataFrame,
}

#[pymethods]
impl PyBacktestReport {
    /// Get cumulative returns as a Polars DataFrame with date column
    #[getter]
    fn creturn(&self) -> PyDataFrame {
        PyDataFrame(self.creturn_df.clone())
    }

    /// Get trades as a Polars DataFrame
    #[getter]
    fn trades(&self) -> PyDataFrame {
        PyDataFrame(self.trades_df.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "BacktestReport(creturn_len={}, trades_count={})",
            self.creturn_df.height(), self.trades_df.height(),
        )
    }
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
    resample: Option<&str>,
    resample_offset: Option<&str>,
    config: Option<PyBacktestConfig>,
    skip_sort: bool,
) -> PyResult<PyBacktestResult> {
    use fastant::Instant;
    use polars_arrow::array::{PrimitiveArray, Utf8ViewArray};

    let total_start = Instant::now();
    let mut step_start = Instant::now();

    let df = df.0;
    let n_rows = df.height();

    // Get config first to check if touched_exit is enabled
    let cfg = config.map(|c| c.inner).unwrap_or_else(|| {
        BacktestConfig {
            finlab_mode: true,
            ..Default::default()
        }
    });
    let touched_exit = cfg.touched_exit;

    // Validate required columns exist
    for col_name in [date, symbol, trade_at_price, position] {
        if df.column(col_name).is_err() {
            return Err(PyValueError::new_err(format!(
                "Missing required column: '{}'", col_name
            )));
        }
    }

    // Validate OHLC columns if touched_exit is enabled
    if touched_exit {
        for col_name in [open, high, low] {
            if df.column(col_name).is_err() {
                return Err(PyValueError::new_err(format!(
                    "touched_exit=True requires column '{}', but it is missing", col_name
                )));
            }
        }
    }

    // Sort by date if needed
    let df = if skip_sort {
        profile!("[PROFILE] Sort: SKIPPED (skip_sort=true, rows={})", n_rows);
        df
    } else {
        let sorted = df
            .sort([date], SortMultipleOptions::default())
            .map_err(|e| PyValueError::new_err(format!("Failed to sort: {}", e)))?;
        profile!("[PROFILE] Sort: {:?} (rows={})", step_start.elapsed(), n_rows);
        sorted
    };
    step_start = Instant::now();

    // Get ChunkedArrays - only cast/rechunk when necessary
    let date_col_ref = df.column(date)
        .map_err(|e| PyValueError::new_err(format!("Failed to get date column: {}", e)))?;
    let date_series = if date_col_ref.dtype() == &DataType::Date {
        date_col_ref.clone()
    } else {
        date_col_ref.cast(&DataType::Date)
            .map_err(|e| PyValueError::new_err(format!("Failed to cast date: {}", e)))?
    };
    let date_phys = date_series.date()
        .map_err(|e| PyValueError::new_err(format!("Date column must be Date: {}", e)))?
        .physical();
    let date_nc = date_phys.chunks().len();
    let date_ca_rechunked;
    let date_ca: &ChunkedArray<Int32Type> = if date_nc > 1 {
        date_ca_rechunked = date_phys.rechunk();
        &date_ca_rechunked
    } else {
        date_phys
    };

    let symbol_ref = df.column(symbol)
        .map_err(|e| PyValueError::new_err(format!("Failed to get symbol column: {}", e)))?
        .str()
        .map_err(|e| PyValueError::new_err(format!("Symbol column must be string: {}", e)))?;
    let sym_nc = symbol_ref.chunks().len();
    let symbol_ca_rechunked;
    let symbol_ca: &StringChunked = if sym_nc > 1 {
        symbol_ca_rechunked = symbol_ref.rechunk();
        &symbol_ca_rechunked
    } else {
        symbol_ref
    };

    let price_col_ref = df.column(trade_at_price)
        .map_err(|e| PyValueError::new_err(format!("Failed to get price column: {}", e)))?;
    let price_series = if price_col_ref.dtype() == &DataType::Float64 {
        price_col_ref.clone()
    } else {
        price_col_ref.cast(&DataType::Float64)
            .map_err(|e| PyValueError::new_err(format!("Failed to cast price: {}", e)))?
    };
    let price_f64 = price_series.f64()
        .map_err(|e| PyValueError::new_err(format!("Price must be f64: {}", e)))?;
    let price_nc = price_f64.chunks().len();
    let price_ca_rechunked;
    let price_ca: &Float64Chunked = if price_nc > 1 {
        price_ca_rechunked = price_f64.rechunk();
        &price_ca_rechunked
    } else {
        price_f64
    };

    let position_col_ref = df.column(position)
        .map_err(|e| PyValueError::new_err(format!("Failed to get position column: {}", e)))?;
    let position_series = if position_col_ref.dtype() == &DataType::Float64 {
        position_col_ref.clone()
    } else {
        position_col_ref.cast(&DataType::Float64)
            .map_err(|e| PyValueError::new_err(format!("Failed to cast position: {}", e)))?
    };
    let position_f64 = position_series.f64()
        .map_err(|e| PyValueError::new_err(format!("Position must be f64: {}", e)))?;
    let position_nc = position_f64.chunks().len();
    let position_ca_rechunked;
    let position_ca: &Float64Chunked = if position_nc > 1 {
        position_ca_rechunked = position_f64.rechunk();
        &position_ca_rechunked
    } else {
        position_f64
    };

    profile!("[PROFILE] Get ChunkedArrays (chunks: d={}, s={}, p={}, pos={}): {:?}",
              date_nc, sym_nc, price_nc, position_nc, step_start.elapsed());
    step_start = Instant::now();

    // Get underlying polars-arrow arrays (single chunk guaranteed by rechunk)
    let date_chunks = date_ca.chunks();
    let symbol_chunks = symbol_ca.chunks();
    let price_chunks = price_ca.chunks();
    let position_chunks = position_ca.chunks();

    // Downcast to concrete polars-arrow types
    let dates_arrow = date_chunks[0]
        .as_any()
        .downcast_ref::<PrimitiveArray<i32>>()
        .ok_or_else(|| PyValueError::new_err("Failed to downcast date array"))?;

    let symbols_arrow = symbol_chunks[0]
        .as_any()
        .downcast_ref::<Utf8ViewArray>()
        .ok_or_else(|| PyValueError::new_err("Failed to downcast symbol array"))?;

    let prices_arrow = price_chunks[0]
        .as_any()
        .downcast_ref::<PrimitiveArray<f64>>()
        .ok_or_else(|| PyValueError::new_err("Failed to downcast price array"))?;

    let positions_arrow = position_chunks[0]
        .as_any()
        .downcast_ref::<PrimitiveArray<f64>>()
        .ok_or_else(|| PyValueError::new_err("Failed to downcast position array"))?;

    profile!("[PROFILE] Get polars-arrow arrays: {:?}", step_start.elapsed());
    step_start = Instant::now();

    // Convert polars-arrow arrays to arrow-rs arrays using FFI (zero-copy)
    let dates_rs = ffi_convert::polars_i32_to_arrow(dates_arrow)
        .map_err(|e| PyValueError::new_err(format!("FFI date conversion failed: {}", e)))?;
    let symbols_rs = ffi_convert::polars_utf8view_to_arrow(symbols_arrow)
        .map_err(|e| PyValueError::new_err(format!("FFI symbol conversion failed: {}", e)))?;
    let prices_rs = ffi_convert::polars_f64_to_arrow(prices_arrow)
        .map_err(|e| PyValueError::new_err(format!("FFI price conversion failed: {}", e)))?;
    let positions_rs = ffi_convert::polars_f64_to_arrow(positions_arrow)
        .map_err(|e| PyValueError::new_err(format!("FFI position conversion failed: {}", e)))?;

    profile!("[PROFILE] FFI conversion (polars-arrow -> arrow-rs): {:?}", step_start.elapsed());
    step_start = Instant::now();

    // Process OHLC columns if touched_exit is enabled
    let (open_rs, high_rs, low_rs) = if touched_exit {
        // Helper to process a float column
        let process_float_col = |col_name: &str| -> PyResult<arrow::array::Float64Array> {
            let col_ref = df.column(col_name)
                .map_err(|e| PyValueError::new_err(format!("Failed to get {} column: {}", col_name, e)))?;
            let col_series = if col_ref.dtype() == &DataType::Float64 {
                col_ref.clone()
            } else {
                col_ref.cast(&DataType::Float64)
                    .map_err(|e| PyValueError::new_err(format!("Failed to cast {}: {}", col_name, e)))?
            };
            let col_f64 = col_series.f64()
                .map_err(|e| PyValueError::new_err(format!("{} must be f64: {}", col_name, e)))?;
            let col_rechunked;
            let col_ca: &Float64Chunked = if col_f64.chunks().len() > 1 {
                col_rechunked = col_f64.rechunk();
                &col_rechunked
            } else {
                col_f64
            };
            let col_chunks = col_ca.chunks();
            let col_arrow = col_chunks[0]
                .as_any()
                .downcast_ref::<PrimitiveArray<f64>>()
                .ok_or_else(|| PyValueError::new_err(format!("Failed to downcast {} array", col_name)))?;
            ffi_convert::polars_f64_to_arrow(col_arrow)
                .map_err(|e| PyValueError::new_err(format!("FFI {} conversion failed: {}", col_name, e)))
        };

        (
            Some(process_float_col(open)?),
            Some(process_float_col(high)?),
            Some(process_float_col(low)?),
        )
    } else {
        (None, None, None)
    };

    profile!("[PROFILE] OHLC processing: {:?}", step_start.elapsed());
    step_start = Instant::now();

    // Parse resample frequency and offset
    let resample_freq = ResampleFreq::from_str(resample);
    let offset = ResampleOffset::from_str(resample_offset);

    // Build arrow input for btcore
    let input = LongFormatArrowInput {
        dates: &dates_rs,
        symbols: &symbols_rs,
        prices: &prices_rs,
        weights: &positions_rs,
        open_prices: open_rs.as_ref(),
        high_prices: high_rs.as_ref(),
        low_prices: low_rs.as_ref(),
    };

    // Run backtest using btcore with arrow-rs arrays
    let result = backtest_long_arrow(&input, resample_freq, offset, &cfg);

    profile!("[PROFILE] Backtest (btcore): {:?}", step_start.elapsed());
    step_start = Instant::now();

    // Handle empty result case (e.g., no valid signals)
    if result.dates.is_empty() {
        let empty_dates = Series::new(date.into(), Vec::<i32>::new())
            .cast(&DataType::Date)
            .map_err(|e| PyValueError::new_err(format!("Failed to cast to Date: {}", e)))?;
        let empty_creturn = Series::new("creturn".into(), Vec::<f64>::new());
        let creturn_df = DataFrame::new(vec![
            empty_dates.into_column(),
            empty_creturn.into_column(),
        ]).map_err(|e| PyValueError::new_err(format!("Failed to create empty DataFrame: {}", e)))?;
        return Ok(PyBacktestResult { creturn_df });
    }

    // Create unique_dates Series directly from btcore result (zero overhead)
    let unique_dates = Series::new(date.into(), &result.dates)
        .cast(&DataType::Date)
        .map_err(|e| PyValueError::new_err(format!("Failed to cast to Date: {}", e)))?;

    // Create creturn Series and find first index where != 1.0 using Polars boolean mask
    let creturn_series = Series::new("creturn".into(), &result.creturn);
    let mask = creturn_series
        .f64()
        .map_err(|e| PyValueError::new_err(format!("creturn must be f64: {}", e)))?
        .not_equal(1.0);

    // Find first true index from boolean ChunkedArray
    let first_change_idx = mask.into_iter()
        .position(|opt| opt == Some(true))
        .unwrap_or(0);

    // Go back one day to include the signal/entry day
    let first_active_idx = if first_change_idx > 0 { first_change_idx - 1 } else { 0 };
    let len = result.creturn.len() - first_active_idx;

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

    // Build creturn DataFrame
    let creturn_df = DataFrame::new(vec![
        sliced_dates.into_column(),
        creturn_col,
    ]).map_err(|e| PyValueError::new_err(format!("Failed to create creturn DataFrame: {}", e)))?;

    profile!("[PROFILE] Build result: {:?}", step_start.elapsed());
    profile!("[PROFILE] TOTAL: {:?}", total_start.elapsed());

    Ok(PyBacktestResult {
        creturn_df,
    })
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
    resample=None,
    resample_offset=None,
    config=None,
    skip_sort=false
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
    resample: Option<&str>,
    resample_offset: Option<&str>,
    config: Option<PyBacktestConfig>,
    skip_sort: bool,
) -> PyResult<PyBacktestReport> {
    use polars_arrow::array::{PrimitiveArray, Utf8ViewArray};

    let df = df.0;
    let n_rows = df.height();

    // Get config first to check if touched_exit is enabled
    let cfg = config.map(|c| c.inner).unwrap_or_else(|| {
        BacktestConfig {
            finlab_mode: true,
            ..Default::default()
        }
    });
    let touched_exit = cfg.touched_exit;

    // Validate required columns exist
    for col_name in [date, symbol, trade_at_price, position] {
        if df.column(col_name).is_err() {
            return Err(PyValueError::new_err(format!(
                "Missing required column: '{}'", col_name
            )));
        }
    }

    // Validate OHLC columns if touched_exit is enabled
    if touched_exit {
        for col_name in [open, high, low] {
            if df.column(col_name).is_err() {
                return Err(PyValueError::new_err(format!(
                    "touched_exit=True requires column '{}', but it is missing", col_name
                )));
            }
        }
    }

    // Sort by date if needed
    let df = if skip_sort {
        df
    } else {
        df.sort([date], SortMultipleOptions::default())
            .map_err(|e| PyValueError::new_err(format!("Failed to sort: {}", e)))?
    };

    // Get ChunkedArrays - only cast/rechunk when necessary
    let date_col_ref = df.column(date)
        .map_err(|e| PyValueError::new_err(format!("Failed to get date column: {}", e)))?;
    let date_series = if date_col_ref.dtype() == &DataType::Date {
        date_col_ref.clone()
    } else {
        date_col_ref.cast(&DataType::Date)
            .map_err(|e| PyValueError::new_err(format!("Failed to cast date: {}", e)))?
    };
    let date_phys = date_series.date()
        .map_err(|e| PyValueError::new_err(format!("Date column must be Date: {}", e)))?
        .physical();
    let date_nc = date_phys.chunks().len();
    let date_ca_rechunked;
    let date_ca: &ChunkedArray<Int32Type> = if date_nc > 1 {
        date_ca_rechunked = date_phys.rechunk();
        &date_ca_rechunked
    } else {
        date_phys
    };

    let symbol_ref = df.column(symbol)
        .map_err(|e| PyValueError::new_err(format!("Failed to get symbol column: {}", e)))?
        .str()
        .map_err(|e| PyValueError::new_err(format!("Symbol column must be string: {}", e)))?;
    let sym_nc = symbol_ref.chunks().len();
    let symbol_ca_rechunked;
    let symbol_ca: &StringChunked = if sym_nc > 1 {
        symbol_ca_rechunked = symbol_ref.rechunk();
        &symbol_ca_rechunked
    } else {
        symbol_ref
    };

    let price_col_ref = df.column(trade_at_price)
        .map_err(|e| PyValueError::new_err(format!("Failed to get price column: {}", e)))?;
    let price_series = if price_col_ref.dtype() == &DataType::Float64 {
        price_col_ref.clone()
    } else {
        price_col_ref.cast(&DataType::Float64)
            .map_err(|e| PyValueError::new_err(format!("Failed to cast price: {}", e)))?
    };
    let price_f64 = price_series.f64()
        .map_err(|e| PyValueError::new_err(format!("Price must be f64: {}", e)))?;
    let price_nc = price_f64.chunks().len();
    let price_ca_rechunked;
    let price_ca: &Float64Chunked = if price_nc > 1 {
        price_ca_rechunked = price_f64.rechunk();
        &price_ca_rechunked
    } else {
        price_f64
    };

    let position_col_ref = df.column(position)
        .map_err(|e| PyValueError::new_err(format!("Failed to get position column: {}", e)))?;
    let position_series = if position_col_ref.dtype() == &DataType::Float64 {
        position_col_ref.clone()
    } else {
        position_col_ref.cast(&DataType::Float64)
            .map_err(|e| PyValueError::new_err(format!("Failed to cast position: {}", e)))?
    };
    let position_f64 = position_series.f64()
        .map_err(|e| PyValueError::new_err(format!("Position must be f64: {}", e)))?;
    let position_nc = position_f64.chunks().len();
    let position_ca_rechunked;
    let position_ca: &Float64Chunked = if position_nc > 1 {
        position_ca_rechunked = position_f64.rechunk();
        &position_ca_rechunked
    } else {
        position_f64
    };

    // Get underlying polars-arrow arrays (single chunk guaranteed by rechunk)
    let date_chunks = date_ca.chunks();
    let symbol_chunks = symbol_ca.chunks();
    let price_chunks = price_ca.chunks();
    let position_chunks = position_ca.chunks();

    // Downcast to concrete polars-arrow types
    let dates_arrow = date_chunks[0]
        .as_any()
        .downcast_ref::<PrimitiveArray<i32>>()
        .ok_or_else(|| PyValueError::new_err("Failed to downcast date array"))?;

    let symbols_arrow = symbol_chunks[0]
        .as_any()
        .downcast_ref::<Utf8ViewArray>()
        .ok_or_else(|| PyValueError::new_err("Failed to downcast symbol array"))?;

    let prices_arrow = price_chunks[0]
        .as_any()
        .downcast_ref::<PrimitiveArray<f64>>()
        .ok_or_else(|| PyValueError::new_err("Failed to downcast price array"))?;

    let positions_arrow = position_chunks[0]
        .as_any()
        .downcast_ref::<PrimitiveArray<f64>>()
        .ok_or_else(|| PyValueError::new_err("Failed to downcast position array"))?;

    // Convert polars-arrow arrays to arrow-rs arrays using FFI (zero-copy)
    let dates_rs = ffi_convert::polars_i32_to_arrow(dates_arrow)
        .map_err(|e| PyValueError::new_err(format!("FFI date conversion failed: {}", e)))?;
    let symbols_rs = ffi_convert::polars_utf8view_to_arrow(symbols_arrow)
        .map_err(|e| PyValueError::new_err(format!("FFI symbol conversion failed: {}", e)))?;
    let prices_rs = ffi_convert::polars_f64_to_arrow(prices_arrow)
        .map_err(|e| PyValueError::new_err(format!("FFI price conversion failed: {}", e)))?;
    let positions_rs = ffi_convert::polars_f64_to_arrow(positions_arrow)
        .map_err(|e| PyValueError::new_err(format!("FFI position conversion failed: {}", e)))?;

    // Process OHLC columns if touched_exit is enabled
    let (open_rs, high_rs, low_rs) = if touched_exit {
        // Helper to process a float column
        let process_float_col = |col_name: &str| -> PyResult<arrow::array::Float64Array> {
            let col_ref = df.column(col_name)
                .map_err(|e| PyValueError::new_err(format!("Failed to get {} column: {}", col_name, e)))?;
            let col_series = if col_ref.dtype() == &DataType::Float64 {
                col_ref.clone()
            } else {
                col_ref.cast(&DataType::Float64)
                    .map_err(|e| PyValueError::new_err(format!("Failed to cast {}: {}", col_name, e)))?
            };
            let col_f64 = col_series.f64()
                .map_err(|e| PyValueError::new_err(format!("{} must be f64: {}", col_name, e)))?;
            let col_rechunked;
            let col_ca: &Float64Chunked = if col_f64.chunks().len() > 1 {
                col_rechunked = col_f64.rechunk();
                &col_rechunked
            } else {
                col_f64
            };
            let col_chunks = col_ca.chunks();
            let col_arrow = col_chunks[0]
                .as_any()
                .downcast_ref::<PrimitiveArray<f64>>()
                .ok_or_else(|| PyValueError::new_err(format!("Failed to downcast {} array", col_name)))?;
            ffi_convert::polars_f64_to_arrow(col_arrow)
                .map_err(|e| PyValueError::new_err(format!("FFI {} conversion failed: {}", col_name, e)))
        };

        (
            Some(process_float_col(open)?),
            Some(process_float_col(high)?),
            Some(process_float_col(low)?),
        )
    } else {
        (None, None, None)
    };

    // Parse resample frequency and offset
    let resample_freq = ResampleFreq::from_str(resample);
    let offset = ResampleOffset::from_str(resample_offset);

    // Build arrow input for btcore
    let input = LongFormatArrowInput {
        dates: &dates_rs,
        symbols: &symbols_rs,
        prices: &prices_rs,
        weights: &positions_rs,
        open_prices: open_rs.as_ref(),
        high_prices: high_rs.as_ref(),
        low_prices: low_rs.as_ref(),
    };

    // Run backtest with report using btcore
    let result = backtest_with_report_long_arrow(&input, resample_freq, offset, &cfg);

    // Convert trades to DataFrame
    let trades_df = trades_to_dataframe(&result.trades)
        .map_err(|e| PyValueError::new_err(format!("Failed to create trades DataFrame: {}", e)))?;

    // Create unique_dates Series directly from btcore result (zero overhead)
    let unique_dates = Series::new(date.into(), &result.dates)
        .cast(&DataType::Date)
        .map_err(|e| PyValueError::new_err(format!("Failed to cast to Date: {}", e)))?;

    // Find first index where creturn != 1.0 using Polars boolean mask
    let creturn_series = Series::new("creturn".into(), &result.creturn);
    let mask = creturn_series
        .f64()
        .map_err(|e| PyValueError::new_err(format!("creturn must be f64: {}", e)))?
        .not_equal(1.0);

    // Find first true index from boolean ChunkedArray
    let first_change_idx = mask.into_iter()
        .position(|opt| opt == Some(true))
        .unwrap_or(0);

    // Go back one day to include the signal/entry day
    let first_active_idx = if first_change_idx > 0 { first_change_idx - 1 } else { 0 };
    let len = result.creturn.len() - first_active_idx;

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

    // Build creturn DataFrame
    let creturn_df = DataFrame::new(vec![
        sliced_dates.into_column(),
        creturn_col,
    ]).map_err(|e| PyValueError::new_err(format!("Failed to create creturn DataFrame: {}", e)))?;

    profile!("[PROFILE] backtest_with_report: {} rows, {} trades", n_rows, result.trades.len());

    Ok(PyBacktestReport {
        creturn_df,
        trades_df,
    })
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
