//! Polars extension for portfolio backtesting
//!
//! This crate provides Polars expression functions for backtesting
//! trading strategies. It wraps the btcore library for use in Python
//! via PyO3 and pyo3-polars.
//!
//! # API Overview
//!
//! - `backtest()` - Main API for long format data (zero-copy, fastest)
//! - `backtest_wide()` - Wide format API (for validation/compatibility)
//! - `backtest_with_trades_wide()` - Wide format with trade tracking

mod expressions;
mod ffi_convert;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3_polars::PyDataFrame;

use polars::prelude::*;

use btcore::{
    run_backtest, run_backtest_with_trades, BacktestConfig, BacktestResult, PriceData,
    SimTradeRecord,
    simulation::{backtest_long_arrow, LongFormatArrowInput, ResampleFreq},
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

/// Python wrapper for TradeRecord
#[pyclass(name = "TradeRecord")]
#[derive(Clone)]
pub struct PyTradeRecord {
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
impl PyTradeRecord {
    fn holding_period(&self) -> Option<usize> {
        match (self.entry_index, self.exit_index) {
            (Some(entry), Some(exit)) => Some(exit - entry),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TradeRecord(stock_id={}, entry={:?}, exit={:?}, weight={:.4}, return={:?})",
            self.stock_id, self.entry_index, self.exit_index,
            self.position_weight, self.trade_return,
        )
    }
}

impl From<SimTradeRecord> for PyTradeRecord {
    fn from(r: SimTradeRecord) -> Self {
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

/// Python wrapper for BacktestResult
#[pyclass(name = "BacktestResult")]
#[derive(Clone)]
pub struct PyBacktestResult {
    #[pyo3(get)]
    pub creturn: Vec<f64>,
    #[pyo3(get)]
    pub trades: Vec<PyTradeRecord>,
}

#[pymethods]
impl PyBacktestResult {
    fn __repr__(&self) -> String {
        format!(
            "BacktestResult(creturn_len={}, trades_count={})",
            self.creturn.len(), self.trades.len(),
        )
    }
}

impl From<BacktestResult> for PyBacktestResult {
    fn from(r: BacktestResult) -> Self {
        Self {
            creturn: r.creturn,
            trades: r.trades.into_iter().map(|t| t.into()).collect(),
        }
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
///     df: DataFrame with columns [date, symbol, price, weight]
///     date_col: Name of date column (default: "date")
///     symbol_col: Name of symbol column (default: "symbol")
///     price_col: Name of price column (default: "close")
///     weight_col: Name of weight column (default: "weight")
///     resample: Rebalancing frequency ("D", "W", "M", or None for daily)
///     config: BacktestConfig (optional)
///     skip_sort: Skip sorting if data is already sorted by date (default: false)
///
/// Returns:
///     PyBacktestResult with creturn
#[pyfunction]
#[pyo3(signature = (
    df,
    date_col="date",
    symbol_col="symbol",
    price_col="close",
    weight_col="weight",
    resample=None,
    config=None,
    skip_sort=false
))]
fn backtest(
    df: PyDataFrame,
    date_col: &str,
    symbol_col: &str,
    price_col: &str,
    weight_col: &str,
    resample: Option<&str>,
    config: Option<PyBacktestConfig>,
    skip_sort: bool,
) -> PyResult<PyBacktestResult> {
    use std::time::Instant;
    use polars_arrow::array::{PrimitiveArray, Utf8ViewArray};

    let total_start = Instant::now();
    let mut step_start = Instant::now();

    let df = df.0;
    let n_rows = df.height();

    // Validate required columns exist
    for col_name in [date_col, symbol_col, price_col, weight_col] {
        if df.column(col_name).is_err() {
            return Err(PyValueError::new_err(format!(
                "Missing required column: '{}'", col_name
            )));
        }
    }

    // Sort by date if needed
    let df = if skip_sort {
        eprintln!("[PROFILE] Sort: SKIPPED (skip_sort=true, rows={})", n_rows);
        df
    } else {
        let sorted = df
            .sort([date_col], SortMultipleOptions::default())
            .map_err(|e| PyValueError::new_err(format!("Failed to sort: {}", e)))?;
        eprintln!("[PROFILE] Sort: {:?} (rows={})", step_start.elapsed(), n_rows);
        sorted
    };
    step_start = Instant::now();

    // Get ChunkedArrays - only cast/rechunk when necessary
    let date_col_ref = df.column(date_col)
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

    let symbol_ref = df.column(symbol_col)
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

    let price_col_ref = df.column(price_col)
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

    let weight_col_ref = df.column(weight_col)
        .map_err(|e| PyValueError::new_err(format!("Failed to get weight column: {}", e)))?;
    let weight_series = if weight_col_ref.dtype() == &DataType::Float64 {
        weight_col_ref.clone()
    } else {
        weight_col_ref.cast(&DataType::Float64)
            .map_err(|e| PyValueError::new_err(format!("Failed to cast weight: {}", e)))?
    };
    let weight_f64 = weight_series.f64()
        .map_err(|e| PyValueError::new_err(format!("Weight must be f64: {}", e)))?;
    let weight_nc = weight_f64.chunks().len();
    let weight_ca_rechunked;
    let weight_ca: &Float64Chunked = if weight_nc > 1 {
        weight_ca_rechunked = weight_f64.rechunk();
        &weight_ca_rechunked
    } else {
        weight_f64
    };

    eprintln!("[PROFILE] Get ChunkedArrays (chunks: d={}, s={}, p={}, w={}): {:?}",
              date_nc, sym_nc, price_nc, weight_nc, step_start.elapsed());
    step_start = Instant::now();

    // Get underlying polars-arrow arrays (single chunk guaranteed by rechunk)
    let date_chunks = date_ca.chunks();
    let symbol_chunks = symbol_ca.chunks();
    let price_chunks = price_ca.chunks();
    let weight_chunks = weight_ca.chunks();

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

    let weights_arrow = weight_chunks[0]
        .as_any()
        .downcast_ref::<PrimitiveArray<f64>>()
        .ok_or_else(|| PyValueError::new_err("Failed to downcast weight array"))?;

    eprintln!("[PROFILE] Get polars-arrow arrays: {:?}", step_start.elapsed());
    step_start = Instant::now();

    // Convert polars-arrow arrays to arrow-rs arrays using FFI (zero-copy)
    let dates_rs = ffi_convert::polars_i32_to_arrow(dates_arrow)
        .map_err(|e| PyValueError::new_err(format!("FFI date conversion failed: {}", e)))?;
    let symbols_rs = ffi_convert::polars_utf8view_to_arrow(symbols_arrow)
        .map_err(|e| PyValueError::new_err(format!("FFI symbol conversion failed: {}", e)))?;
    let prices_rs = ffi_convert::polars_f64_to_arrow(prices_arrow)
        .map_err(|e| PyValueError::new_err(format!("FFI price conversion failed: {}", e)))?;
    let weights_rs = ffi_convert::polars_f64_to_arrow(weights_arrow)
        .map_err(|e| PyValueError::new_err(format!("FFI weight conversion failed: {}", e)))?;

    eprintln!("[PROFILE] FFI conversion (polars-arrow -> arrow-rs): {:?}", step_start.elapsed());
    step_start = Instant::now();

    // Get config (default to finlab_mode=true for long format)
    let cfg = config.map(|c| c.inner).unwrap_or_else(|| {
        BacktestConfig {
            finlab_mode: true,
            ..Default::default()
        }
    });

    // Parse resample frequency
    let resample_freq = ResampleFreq::from_str(resample);

    // Build arrow input for btcore
    let input = LongFormatArrowInput {
        dates: &dates_rs,
        symbols: &symbols_rs,
        prices: &prices_rs,
        weights: &weights_rs,
    };

    // Run backtest using btcore with arrow-rs arrays
    let result = backtest_long_arrow(&input, resample_freq, &cfg);

    eprintln!("[PROFILE] Backtest (btcore arrow): {:?}", step_start.elapsed());
    eprintln!("[PROFILE] TOTAL: {:?}", total_start.elapsed());

    Ok(PyBacktestResult {
        creturn: result.creturn,
        trades: vec![],
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
///     BacktestResult with creturn and trades
#[pyfunction]
#[pyo3(signature = (adj_prices, original_prices, weights, rebalance_indices, config=None, open_prices=None, high_prices=None, low_prices=None))]
fn backtest_with_trades_wide(
    adj_prices: PyDataFrame,
    original_prices: PyDataFrame,
    weights: PyDataFrame,
    rebalance_indices: Vec<usize>,
    config: Option<PyBacktestConfig>,
    open_prices: Option<PyDataFrame>,
    high_prices: Option<PyDataFrame>,
    low_prices: Option<PyDataFrame>,
) -> PyResult<PyBacktestResult> {
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
    m.add_class::<PyBacktestConfig>()?;
    m.add_class::<PyTradeRecord>()?;
    m.add_class::<PyBacktestResult>()?;
    // Main API (long format)
    m.add_function(wrap_pyfunction!(backtest, m)?)?;
    // Wide format API (for validation)
    m.add_function(wrap_pyfunction!(backtest_wide, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_with_trades_wide, m)?)?;
    Ok(())
}
