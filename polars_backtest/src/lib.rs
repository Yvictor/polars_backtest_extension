//! Polars extension for portfolio backtesting
//!
//! This crate provides Polars expression functions for backtesting
//! trading strategies. It wraps the btcore library for use in Python
//! via PyO3 and pyo3-polars.

mod expressions;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3_polars::PyDataFrame;

use btcore::{
    run_backtest, run_backtest_with_trades,
    BacktestConfig, BacktestResult, SimTradeRecord,
};

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
            },
        }
    }

    #[getter]
    fn fee_ratio(&self) -> f64 {
        self.inner.fee_ratio
    }

    #[getter]
    fn tax_ratio(&self) -> f64 {
        self.inner.tax_ratio
    }

    #[getter]
    fn stop_loss(&self) -> f64 {
        self.inner.stop_loss
    }

    #[getter]
    fn take_profit(&self) -> f64 {
        self.inner.take_profit
    }

    #[getter]
    fn trail_stop(&self) -> f64 {
        self.inner.trail_stop
    }

    #[getter]
    fn position_limit(&self) -> f64 {
        self.inner.position_limit
    }

    #[getter]
    fn retain_cost_when_rebalance(&self) -> bool {
        self.inner.retain_cost_when_rebalance
    }

    #[getter]
    fn stop_trading_next_period(&self) -> bool {
        self.inner.stop_trading_next_period
    }

    #[getter]
    fn finlab_mode(&self) -> bool {
        self.inner.finlab_mode
    }

    fn __repr__(&self) -> String {
        format!(
            "BacktestConfig(fee_ratio={}, tax_ratio={}, stop_loss={}, take_profit={}, trail_stop={}, position_limit={}, retain_cost_when_rebalance={}, stop_trading_next_period={}, finlab_mode={})",
            self.inner.fee_ratio,
            self.inner.tax_ratio,
            self.inner.stop_loss,
            self.inner.take_profit,
            self.inner.trail_stop,
            self.inner.position_limit,
            self.inner.retain_cost_when_rebalance,
            self.inner.stop_trading_next_period,
            self.inner.finlab_mode,
        )
    }
}

/// Run backtest simulation with boolean signals (equal weight)
///
/// Args:
///     prices: DataFrame with dates as rows, stocks as columns (Float64)
///     signals: DataFrame with rebalance dates as rows, stocks as columns (Bool)
///     rebalance_indices: List of row indices in prices where rebalancing occurs
///     config: BacktestConfig (optional)
///
/// Returns:
///     List[float]: Cumulative returns at each time step
#[pyfunction]
#[pyo3(signature = (prices, signals, rebalance_indices, config=None))]
fn backtest_signals(
    prices: PyDataFrame,
    signals: PyDataFrame,
    rebalance_indices: Vec<usize>,
    config: Option<PyBacktestConfig>,
) -> PyResult<Vec<f64>> {
    let prices_df = prices.0;
    let signals_df = signals.0;

    // Convert prices DataFrame to Vec<Vec<f64>>
    let prices_2d = df_to_f64_2d(&prices_df)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert prices: {}", e)))?;

    // Convert signals DataFrame to Vec<Vec<bool>>
    let signals_2d = df_to_bool_2d(&signals_df)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert signals: {}", e)))?;

    let cfg = config.map(|c| c.inner).unwrap_or_default();

    // Run backtest (pure Rust, releases GIL implicitly)
    Ok(run_backtest(&prices_2d, &signals_2d, &rebalance_indices, &cfg))
}

/// Run backtest simulation with custom float weights
///
/// Args:
///     prices: DataFrame with dates as rows, stocks as columns (Float64)
///     weights: DataFrame with rebalance dates as rows, stocks as columns (Float64)
///     rebalance_indices: List of row indices in prices where rebalancing occurs
///     config: BacktestConfig (optional)
///
/// Returns:
///     List[float]: Cumulative returns at each time step
#[pyfunction]
#[pyo3(signature = (prices, weights, rebalance_indices, config=None))]
fn backtest_weights(
    prices: PyDataFrame,
    weights: PyDataFrame,
    rebalance_indices: Vec<usize>,
    config: Option<PyBacktestConfig>,
) -> PyResult<Vec<f64>> {
    let prices_df = prices.0;
    let weights_df = weights.0;

    // Convert prices DataFrame to Vec<Vec<f64>>
    let prices_2d = df_to_f64_2d(&prices_df)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert prices: {}", e)))?;

    // Convert weights DataFrame to Vec<Vec<f64>>
    let weights_2d = df_to_f64_2d(&weights_df)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert weights: {}", e)))?;

    let cfg = config.map(|c| c.inner).unwrap_or_default();

    // Run backtest (pure Rust, releases GIL implicitly)
    Ok(run_backtest(&prices_2d, &weights_2d, &rebalance_indices, &cfg))
}

/// Helper to convert DataFrame to Vec<Vec<f64>> (row-major)
fn df_to_f64_2d(df: &polars::prelude::DataFrame) -> Result<Vec<Vec<f64>>, String> {
    use polars::prelude::*;

    let n_rows = df.height();
    let n_cols = df.width();

    if n_cols == 0 {
        return Ok(vec![]);
    }

    let mut result = Vec::with_capacity(n_rows);

    for row_idx in 0..n_rows {
        let mut row = Vec::with_capacity(n_cols);
        for col in df.get_columns() {
            let val = col
                .get(row_idx)
                .map_err(|e| format!("Failed to get value: {}", e))?;

            let f64_val = match val {
                AnyValue::Float64(v) => v,
                AnyValue::Float32(v) => v as f64,
                AnyValue::Int64(v) => v as f64,
                AnyValue::Int32(v) => v as f64,
                AnyValue::UInt64(v) => v as f64,
                AnyValue::UInt32(v) => v as f64,
                AnyValue::Null => f64::NAN,
                _ => return Err(format!("Unsupported type: {:?}", val)),
            };
            row.push(f64_val);
        }
        result.push(row);
    }

    Ok(result)
}

/// Helper to convert DataFrame to Vec<Vec<bool>> (row-major)
fn df_to_bool_2d(df: &polars::prelude::DataFrame) -> Result<Vec<Vec<bool>>, String> {
    use polars::prelude::*;

    let n_rows = df.height();
    let n_cols = df.width();

    if n_cols == 0 {
        return Ok(vec![]);
    }

    let mut result = Vec::with_capacity(n_rows);

    for row_idx in 0..n_rows {
        let mut row = Vec::with_capacity(n_cols);
        for col in df.get_columns() {
            let val = col
                .get(row_idx)
                .map_err(|e| format!("Failed to get value: {}", e))?;

            let bool_val = match val {
                AnyValue::Boolean(v) => v,
                AnyValue::Int64(v) => v != 0,
                AnyValue::Int32(v) => v != 0,
                AnyValue::Float64(v) => v != 0.0 && !v.is_nan(),
                AnyValue::Null => false,
                _ => return Err(format!("Unsupported type for bool: {:?}", val)),
            };
            row.push(bool_val);
        }
        result.push(row);
    }

    Ok(result)
}

/// Python wrapper for TradeRecord
#[pyclass(name = "TradeRecord")]
#[derive(Clone)]
pub struct PyTradeRecord {
    /// Stock ID (column index in price matrix)
    #[pyo3(get)]
    pub stock_id: usize,
    /// Actual entry date (row index in price matrix, T+1 after signal)
    #[pyo3(get)]
    pub entry_index: usize,
    /// Actual exit date (row index in price matrix)
    #[pyo3(get)]
    pub exit_index: Option<usize>,
    /// Signal date for entry (row index in price matrix)
    #[pyo3(get)]
    pub entry_sig_index: usize,
    /// Signal date for exit (row index in price matrix)
    #[pyo3(get)]
    pub exit_sig_index: Option<usize>,
    /// Position weight at entry
    #[pyo3(get)]
    pub position_weight: f64,
    /// Entry price (ORIGINAL price, not adjusted)
    #[pyo3(get)]
    pub entry_price: f64,
    /// Exit price (ORIGINAL price, not adjusted)
    #[pyo3(get)]
    pub exit_price: Option<f64>,
    /// Trade return (calculated using original prices with fees)
    #[pyo3(get)]
    pub trade_return: Option<f64>,
}

#[pymethods]
impl PyTradeRecord {
    /// Calculate holding period in days
    fn holding_period(&self) -> Option<usize> {
        self.exit_index.map(|exit| exit - self.entry_index)
    }

    fn __repr__(&self) -> String {
        format!(
            "TradeRecord(stock_id={}, entry_index={}, exit_index={:?}, position={:.4}, entry_price={:.2}, exit_price={:?}, return={:?})",
            self.stock_id,
            self.entry_index,
            self.exit_index,
            self.position_weight,
            self.entry_price,
            self.exit_price,
            self.trade_return,
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
    /// Cumulative returns at each time step
    #[pyo3(get)]
    pub creturn: Vec<f64>,
    /// List of completed trades
    #[pyo3(get)]
    pub trades: Vec<PyTradeRecord>,
}

#[pymethods]
impl PyBacktestResult {
    fn __repr__(&self) -> String {
        format!(
            "BacktestResult(creturn_len={}, trades_count={})",
            self.creturn.len(),
            self.trades.len(),
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

/// Run backtest with trades tracking and dual price support
///
/// This function returns both cumulative returns and trade records.
/// It uses:
/// - adj_prices: Adjusted prices for return calculation (creturn)
/// - original_prices: Original prices for trade records (entry/exit prices)
///
/// The trade records match Finlab's trades DataFrame format, using
/// original prices for entry/exit to match real trading execution.
///
/// Args:
///     adj_prices: DataFrame with adjusted prices [n_times x n_assets] (for creturn)
///     original_prices: DataFrame with original prices [n_times x n_assets] (for trades)
///     weights: DataFrame with rebalance dates as rows, stocks as columns (Float64)
///     rebalance_indices: List of row indices in prices where rebalancing occurs
///     config: BacktestConfig (optional)
///
/// Returns:
///     BacktestResult containing creturn and trades list
#[pyfunction]
#[pyo3(signature = (adj_prices, original_prices, weights, rebalance_indices, config=None))]
fn backtest_with_trades(
    adj_prices: PyDataFrame,
    original_prices: PyDataFrame,
    weights: PyDataFrame,
    rebalance_indices: Vec<usize>,
    config: Option<PyBacktestConfig>,
) -> PyResult<PyBacktestResult> {
    let adj_prices_df = adj_prices.0;
    let original_prices_df = original_prices.0;
    let weights_df = weights.0;

    // Convert DataFrames to Vec<Vec<f64>>
    let adj_prices_2d = df_to_f64_2d(&adj_prices_df)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert adj_prices: {}", e)))?;

    let original_prices_2d = df_to_f64_2d(&original_prices_df)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert original_prices: {}", e)))?;

    let weights_2d = df_to_f64_2d(&weights_df)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert weights: {}", e)))?;

    let cfg = config.map(|c| c.inner).unwrap_or_default();

    // Run backtest with trades tracking
    let result = run_backtest_with_trades(
        &adj_prices_2d,
        &original_prices_2d,
        &weights_2d,
        &rebalance_indices,
        &cfg,
    );

    Ok(result.into())
}

/// Initialize the Python module
#[pymodule]
fn _polars_backtest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyBacktestConfig>()?;
    m.add_class::<PyTradeRecord>()?;
    m.add_class::<PyBacktestResult>()?;
    m.add_function(wrap_pyfunction!(backtest_signals, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_weights, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_with_trades, m)?)?;
    Ok(())
}
