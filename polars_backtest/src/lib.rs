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

use polars::prelude::*;
use polars_ops::pivot;

use btcore::{
    run_backtest, run_backtest_with_trades, BacktestConfig, BacktestResult, PriceData,
    SimTradeRecord,
    simulation::{backtest_long, backtest_long_with_accessor, LongFormatInput, ResampleFreq},
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

    #[getter]
    fn touched_exit(&self) -> bool {
        self.inner.touched_exit
    }

    fn __repr__(&self) -> String {
        format!(
            "BacktestConfig(fee_ratio={}, tax_ratio={}, stop_loss={}, take_profit={}, trail_stop={}, position_limit={}, retain_cost_when_rebalance={}, stop_trading_next_period={}, finlab_mode={}, touched_exit={})",
            self.inner.fee_ratio,
            self.inner.tax_ratio,
            self.inner.stop_loss,
            self.inner.take_profit,
            self.inner.trail_stop,
            self.inner.position_limit,
            self.inner.retain_cost_when_rebalance,
            self.inner.stop_trading_next_period,
            self.inner.finlab_mode,
            self.inner.touched_exit,
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
/// Optimized version using columnar access with cont_slice()
fn df_to_f64_2d(df: &polars::prelude::DataFrame) -> Result<Vec<Vec<f64>>, String> {
    use polars::prelude::*;

    let n_rows = df.height();
    let n_cols = df.width();

    if n_cols == 0 {
        return Ok(vec![]);
    }

    // Get column slices (zero-copy when possible)
    let col_slices: Vec<Vec<f64>> = df
        .get_columns()
        .iter()
        .map(|col| {
            // Cast to f64 if needed
            let f64_col = col.cast(&DataType::Float64)
                .map_err(|e| format!("Failed to cast column: {}", e))?;
            let ca = f64_col.f64()
                .map_err(|e| format!("Failed to get f64 chunked array: {}", e))?;

            // Try zero-copy slice, fallback to collect if chunked or has nulls
            match ca.cont_slice() {
                Ok(slice) => Ok(slice.to_vec()),
                Err(_) => {
                    // Fallback: collect with NaN for nulls
                    Ok(ca.into_iter()
                        .map(|v| v.unwrap_or(f64::NAN))
                        .collect())
                }
            }
        })
        .collect::<Result<Vec<_>, String>>()?;

    // Build row-major result from column slices
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
    /// None for pending entries that have signal but not yet executed
    #[pyo3(get)]
    pub entry_index: Option<usize>,
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
        match (self.entry_index, self.exit_index) {
            (Some(entry), Some(exit)) => Some(exit - entry),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TradeRecord(stock_id={}, entry_index={:?}, exit_index={:?}, position={:.4}, entry_price={:.2}, exit_price={:?}, return={:?})",
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
/// - open_prices/high_prices/low_prices: Optional, for touched_exit support
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
///     open_prices: DataFrame with open prices (optional, for touched_exit)
///     high_prices: DataFrame with high prices (optional, for touched_exit)
///     low_prices: DataFrame with low prices (optional, for touched_exit)
///
/// Returns:
///     BacktestResult containing creturn and trades list
#[pyfunction]
#[pyo3(signature = (adj_prices, original_prices, weights, rebalance_indices, config=None, open_prices=None, high_prices=None, low_prices=None))]
fn backtest_with_trades(
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

    // Convert DataFrames to Vec<Vec<f64>>
    let adj_prices_2d = df_to_f64_2d(&adj_prices_df)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert adj_prices: {}", e)))?;

    let original_prices_2d = df_to_f64_2d(&original_prices_df)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert original_prices: {}", e)))?;

    let weights_2d = df_to_f64_2d(&weights_df)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert weights: {}", e)))?;

    // Convert optional OHLC prices
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

    // Build PriceData with optional OHLC
    let prices = match (&open_prices_2d, &high_prices_2d, &low_prices_2d) {
        (Some(open), Some(high), Some(low)) => {
            PriceData::with_ohlc(&adj_prices_2d, &original_prices_2d, open, high, low)
        }
        _ => PriceData::new(&adj_prices_2d, &original_prices_2d),
    };

    // Run backtest with trades tracking
    let result = run_backtest_with_trades(&prices, &weights_2d, &rebalance_indices, &cfg);

    Ok(result.into())
}

// =============================================================================
// Long Format Backtest API (Stage 4)
// =============================================================================

/// Convert long format DataFrame to wide format
///
/// Input: DataFrame with columns [date, symbol, value_col]
/// Output: DataFrame with date as rows, symbols as columns
fn long_to_wide(
    df: &DataFrame,
    date_col: &str,
    symbol_col: &str,
    value_col: &str,
) -> PolarsResult<DataFrame> {
    // Select only needed columns first
    let selected = df.select([date_col, symbol_col, value_col])?;

    // Pivot: on=symbol (becomes columns), index=date (row index), values=value
    // None for agg_fn defaults to first() - expects one value per (date, symbol) pair
    pivot::pivot(
        &selected,
        [PlSmallStr::from(symbol_col)],           // on - column values become new columns
        Some([PlSmallStr::from(date_col)]),       // index - row index (group by)
        Some([PlSmallStr::from(value_col)]),      // values to aggregate
        false,                                    // sort columns
        None,                                     // no aggregation (defaults to first)
        None,                                     // separator
    )
}

/// Compute rebalance indices based on resample rule
///
/// Returns indices where rebalancing should occur
fn compute_rebalance_indices(
    dates: &[String],
    resample: Option<&str>,
) -> Vec<usize> {
    match resample {
        None | Some("D") => {
            // Daily rebalancing: every day is a rebalance day
            (0..dates.len()).collect()
        }
        Some("W") => {
            // Weekly: first day of each week
            let mut indices = vec![0];
            for i in 1..dates.len() {
                // Simple heuristic: new week if date string differs in week portion
                // For proper implementation, parse dates
                if i > 0 {
                    indices.push(i);
                }
            }
            // For now, just return all indices (will be refined)
            (0..dates.len()).collect()
        }
        Some("M") => {
            // Monthly: last day of each month (like Finlab's resample)
            // Rebalance happens on the last trading day before month changes
            let mut indices = Vec::new();
            for i in 0..dates.len() {
                if i == dates.len() - 1 {
                    // Last day is always a rebalance day
                    indices.push(i);
                } else {
                    let curr = &dates[i];
                    let next = &dates[i + 1];
                    // Compare YYYY-MM prefix: if next month differs, current is month-end
                    if curr.len() >= 7 && next.len() >= 7 && curr[..7] != next[..7] {
                        indices.push(i);
                    }
                }
            }
            indices
        }
        _ => (0..dates.len()).collect(),
    }
}

/// Build wide format matrices from partitions (avoids pivot)
///
/// Uses partition_by to split data by date, then builds matrices directly.
/// This can be faster than pivot for certain data patterns.
fn build_wide_from_partitions(
    df: &DataFrame,
    date_col: &str,
    symbol_col: &str,
    value_col: &str,
) -> PolarsResult<(Vec<String>, Vec<String>, Vec<Vec<f64>>)> {
    use std::collections::HashMap;

    // Get unique symbols for column ordering (MUST be sorted for consistency!)
    let symbols_series = df.column(symbol_col)?.str()?;
    let mut unique_symbols: Vec<String> = symbols_series
        .unique()?
        .into_iter()
        .filter_map(|s| s.map(|s| s.to_string()))
        .collect();
    unique_symbols.sort();  // Critical: ensure consistent order across prices and weights

    // Build symbol → index mapping
    let symbol_to_idx: HashMap<&str, usize> = unique_symbols
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    let n_symbols = unique_symbols.len();

    // Sort by date and use partition_by_stable to maintain order
    let df_sorted = df.sort([date_col], SortMultipleOptions::default())?;
    let partitions = df_sorted.partition_by_stable([date_col], true)?;

    let mut dates: Vec<String> = Vec::with_capacity(partitions.len());
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(partitions.len());

    for partition in &partitions {
        // Get date from first row
        let date_val = partition
            .column(date_col)?
            .get(0)
            .map_err(|e| PolarsError::ComputeError(format!("Failed to get date: {}", e).into()))?;
        let date_str = match date_val {
            AnyValue::String(s) => s.to_string(),
            AnyValue::StringOwned(s) => s.to_string(),
            _ => format!("{:?}", date_val),
        };
        dates.push(date_str);

        // Build row with NaN for missing symbols
        let mut row = vec![f64::NAN; n_symbols];

        let symbols = partition.column(symbol_col)?.str()?;
        let values = partition.column(value_col)?.cast(&DataType::Float64)?;
        let values_f64 = values.f64()?;

        for i in 0..partition.height() {
            if let Some(symbol) = symbols.get(i) {
                if let Some(&idx) = symbol_to_idx.get(symbol) {
                    row[idx] = values_f64.get(i).unwrap_or(f64::NAN);
                }
            }
        }

        rows.push(row);
    }

    Ok((dates, unique_symbols, rows))
}

/// Build TWO wide format matrices from partitions in a single pass
///
/// This is more efficient than calling build_wide_from_partitions twice
/// because we only sort and partition once.
fn build_wide_from_partitions_dual(
    df: &DataFrame,
    date_col: &str,
    symbol_col: &str,
    value_col1: &str,
    value_col2: &str,
) -> PolarsResult<(Vec<String>, Vec<String>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    use std::collections::HashMap;

    // Get unique symbols for column ordering (MUST be sorted for consistency!)
    let symbols_series = df.column(symbol_col)?.str()?;
    let mut unique_symbols: Vec<String> = symbols_series
        .unique()?
        .into_iter()
        .filter_map(|s| s.map(|s| s.to_string()))
        .collect();
    unique_symbols.sort();

    // Build symbol → index mapping
    let symbol_to_idx: HashMap<&str, usize> = unique_symbols
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    let n_symbols = unique_symbols.len();

    // Sort by date and partition ONCE
    let df_sorted = df.sort([date_col], SortMultipleOptions::default())?;
    let partitions = df_sorted.partition_by_stable([date_col], true)?;

    let mut dates: Vec<String> = Vec::with_capacity(partitions.len());
    let mut rows1: Vec<Vec<f64>> = Vec::with_capacity(partitions.len());
    let mut rows2: Vec<Vec<f64>> = Vec::with_capacity(partitions.len());

    for partition in &partitions {
        // Get date from first row
        let date_val = partition
            .column(date_col)?
            .get(0)
            .map_err(|e| PolarsError::ComputeError(format!("Failed to get date: {}", e).into()))?;
        let date_str = match date_val {
            AnyValue::String(s) => s.to_string(),
            AnyValue::StringOwned(s) => s.to_string(),
            _ => format!("{:?}", date_val),
        };
        dates.push(date_str);

        // Build rows with NaN for missing symbols
        let mut row1 = vec![f64::NAN; n_symbols];
        let mut row2 = vec![f64::NAN; n_symbols];

        let symbols = partition.column(symbol_col)?.str()?;
        let values1 = partition.column(value_col1)?.cast(&DataType::Float64)?;
        let values1_f64 = values1.f64()?;
        let values2 = partition.column(value_col2)?.cast(&DataType::Float64)?;
        let values2_f64 = values2.f64()?;

        for i in 0..partition.height() {
            if let Some(symbol) = symbols.get(i) {
                if let Some(&idx) = symbol_to_idx.get(symbol) {
                    row1[idx] = values1_f64.get(i).unwrap_or(f64::NAN);
                    row2[idx] = values2_f64.get(i).unwrap_or(f64::NAN);
                }
            }
        }

        rows1.push(row1);
        rows2.push(row2);
    }

    Ok((dates, unique_symbols, rows1, rows2))
}

/// Run backtest on long format DataFrame using partition_by
///
/// This is an alternative to backtest() that avoids pivot by using partition_by.
/// May be faster for certain data patterns.
#[pyfunction]
#[pyo3(signature = (
    df,
    date_col="date",
    symbol_col="symbol",
    price_col="close",
    weight_col="weight",
    resample=None,
    config=None
))]
fn backtest_partitioned(
    df: PyDataFrame,
    date_col: &str,
    symbol_col: &str,
    price_col: &str,
    weight_col: &str,
    resample: Option<&str>,
    config: Option<PyBacktestConfig>,
) -> PyResult<PyBacktestResult> {
    let df = df.0;

    // Validate required columns exist
    for col_name in [date_col, symbol_col, price_col, weight_col] {
        if df.column(col_name).is_err() {
            return Err(PyValueError::new_err(format!(
                "Missing required column: '{}'", col_name
            )));
        }
    }

    // Build wide format from partitions (prices and weights in one pass)
    let (dates, _symbols, prices_2d, weights_2d) = build_wide_from_partitions_dual(
        &df, date_col, symbol_col, price_col, weight_col
    ).map_err(|e| PyValueError::new_err(format!("Failed to build wide format: {}", e)))?;

    // Compute rebalance indices
    let rebalance_indices = compute_rebalance_indices(&dates, resample);

    // Extract only rebalance-day weights (run_backtest expects weights.len() == rebalance_indices.len())
    let rebalance_weights: Vec<Vec<f64>> = rebalance_indices
        .iter()
        .map(|&idx| weights_2d[idx].clone())
        .collect();

    // Get config
    let cfg = config.map(|c| c.inner).unwrap_or_default();

    // Run backtest
    let creturn = run_backtest(&prices_2d, &rebalance_weights, &rebalance_indices, &cfg);

    Ok(PyBacktestResult {
        creturn,
        trades: vec![],
    })
}

/// Run backtest with trades on long format DataFrame using partition_by
///
/// Like backtest_partitioned but also tracks trades for Report generation.
#[pyfunction]
#[pyo3(signature = (
    df,
    date_col="date",
    symbol_col="symbol",
    price_col="close",
    weight_col="weight",
    resample=None,
    config=None
))]
fn backtest_with_trades_partitioned(
    df: PyDataFrame,
    date_col: &str,
    symbol_col: &str,
    price_col: &str,
    weight_col: &str,
    resample: Option<&str>,
    config: Option<PyBacktestConfig>,
) -> PyResult<PyBacktestResult> {
    let df = df.0;

    // Validate required columns exist
    for col_name in [date_col, symbol_col, price_col, weight_col] {
        if df.column(col_name).is_err() {
            return Err(PyValueError::new_err(format!(
                "Missing required column: '{}'", col_name
            )));
        }
    }

    // Build wide format from partitions (prices and weights in one pass)
    let (dates, _symbols, prices_2d, weights_2d) = build_wide_from_partitions_dual(
        &df, date_col, symbol_col, price_col, weight_col
    ).map_err(|e| PyValueError::new_err(format!("Failed to build wide format: {}", e)))?;

    // Compute rebalance indices
    let rebalance_indices = compute_rebalance_indices(&dates, resample);

    // Extract only rebalance-day weights (run_backtest expects weights.len() == rebalance_indices.len())
    let rebalance_weights: Vec<Vec<f64>> = rebalance_indices
        .iter()
        .map(|&idx| weights_2d[idx].clone())
        .collect();

    // Get config
    let cfg = config.map(|c| c.inner).unwrap_or_default();

    // Build PriceData (no OHLC for this version)
    let prices = PriceData::new(&prices_2d, &prices_2d);

    // Run backtest with trades tracking
    let result = run_backtest_with_trades(&prices, &rebalance_weights, &rebalance_indices, &cfg);

    Ok(result.into())
}

// =============================================================================
// Zero-copy backtest using btcore's arrow-based engine
// =============================================================================

/// Run backtest using long format engine (zero-copy, no pivot/partition_by)
///
/// This function processes long format data directly using Polars ChunkedArray
/// with zero-copy access via btcore's accessor-based API.
///
/// Data flow: DataFrame → sort → ChunkedArray accessors → btcore → creturn
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
fn backtest_long_from_df(
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

    // Note: is_sorted_flag is lost when DataFrame is passed from Python to Rust via pyo3-polars.
    // See: https://github.com/pola-rs/pyo3-polars/issues/51
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

    // Get ChunkedArrays (zero-copy access to underlying data)
    let date_ca = df.column(date_col)
        .map_err(|e| PyValueError::new_err(format!("Failed to get date column: {}", e)))?
        .str()
        .map_err(|e| PyValueError::new_err(format!("Date column must be string: {}", e)))?
        .clone();

    let symbol_ca = df.column(symbol_col)
        .map_err(|e| PyValueError::new_err(format!("Failed to get symbol column: {}", e)))?
        .str()
        .map_err(|e| PyValueError::new_err(format!("Symbol column must be string: {}", e)))?
        .clone();

    let price_series = df.column(price_col)
        .map_err(|e| PyValueError::new_err(format!("Failed to get price column: {}", e)))?
        .cast(&DataType::Float64)
        .map_err(|e| PyValueError::new_err(format!("Failed to cast price: {}", e)))?;
    let price_ca = price_series.f64()
        .map_err(|e| PyValueError::new_err(format!("Price must be f64: {}", e)))?
        .clone();

    let weight_series = df.column(weight_col)
        .map_err(|e| PyValueError::new_err(format!("Failed to get weight column: {}", e)))?
        .cast(&DataType::Float64)
        .map_err(|e| PyValueError::new_err(format!("Failed to cast weight: {}", e)))?;
    let weight_ca = weight_series.f64()
        .map_err(|e| PyValueError::new_err(format!("Weight must be f64: {}", e)))?
        .clone();

    eprintln!("[PROFILE] Get ChunkedArrays: {:?}", step_start.elapsed());
    step_start = Instant::now();

    // Get config
    let cfg = config.map(|c| c.inner).unwrap_or_else(|| {
        BacktestConfig {
            finlab_mode: true,
            ..Default::default()
        }
    });

    // Parse resample frequency
    let resample_freq = ResampleFreq::from_str(resample);

    // Run backtest using btcore with closure-based accessors (zero-copy)
    let result = backtest_long_with_accessor(
        n_rows,
        |i| date_ca.get(i).unwrap_or(""),
        |i| symbol_ca.get(i).unwrap_or(""),
        |i| price_ca.get(i).unwrap_or(f64::NAN),
        |i| weight_ca.get(i).unwrap_or(f64::NAN),
        resample_freq,
        &cfg,
    );

    eprintln!("[PROFILE] Backtest (btcore): {:?}", step_start.elapsed());
    eprintln!("[PROFILE] TOTAL: {:?}", total_start.elapsed());

    Ok(PyBacktestResult {
        creturn: result.creturn,
        trades: vec![],
    })
}

/// Run backtest with long format input (Stage 4 API)
///
/// This is the main API for backtesting with long format data.
/// Internally converts to wide format and calls existing backtest logic.
///
/// Args:
///     df: DataFrame in long format with columns [date, symbol, price, weight]
///     date_col: Name of date column (default: "date")
///     symbol_col: Name of symbol column (default: "symbol")
///     price_col: Name of price column (default: "close")
///     weight_col: Name of weight column (default: "weight")
///     resample: Rebalancing frequency ("D", "W", "M", or None for daily)
///     config: BacktestConfig (optional)
///
/// Returns:
///     PyBacktestResult with creturn and trades
#[pyfunction]
#[pyo3(signature = (
    df,
    date_col="date",
    symbol_col="symbol",
    price_col="close",
    weight_col="weight",
    resample=None,
    config=None
))]
fn backtest(
    df: PyDataFrame,
    date_col: &str,
    symbol_col: &str,
    price_col: &str,
    weight_col: &str,
    resample: Option<&str>,
    config: Option<PyBacktestConfig>,
) -> PyResult<PyBacktestResult> {
    let df = df.0;

    // Validate required columns exist
    for col_name in [date_col, symbol_col, price_col, weight_col] {
        if df.column(col_name).is_err() {
            return Err(PyValueError::new_err(format!(
                "Missing required column: '{}'", col_name
            )));
        }
    }

    // Sort by date to ensure correct ordering
    let df = df
        .sort([date_col], SortMultipleOptions::default())
        .map_err(|e| PyValueError::new_err(format!("Failed to sort: {}", e)))?;

    // Convert to wide format
    let wide_prices = long_to_wide(&df, date_col, symbol_col, price_col)
        .map_err(|e| PyValueError::new_err(format!("Failed to pivot prices: {}", e)))?;

    let wide_weights = long_to_wide(&df, date_col, symbol_col, weight_col)
        .map_err(|e| PyValueError::new_err(format!("Failed to pivot weights: {}", e)))?;

    // Extract dates for rebalance computation
    let dates: Vec<String> = wide_prices
        .column(date_col)
        .map_err(|e| PyValueError::new_err(format!("Failed to get date column: {}", e)))?
        .str()
        .map_err(|e| PyValueError::new_err(format!("Date column is not string: {}", e)))?
        .into_iter()
        .filter_map(|s| s.map(|s| s.to_string()))
        .collect();

    // Compute rebalance indices
    let rebalance_indices = compute_rebalance_indices(&dates, resample);

    // Drop date column for price/weight matrices
    let price_cols: Vec<_> = wide_prices
        .get_column_names()
        .into_iter()
        .filter(|&c| c != date_col)
        .map(|c| c.to_string())
        .collect();

    let prices_only = wide_prices
        .select(price_cols.iter().map(|s| s.as_str()))
        .map_err(|e| PyValueError::new_err(format!("Failed to select price columns: {}", e)))?;

    let weights_only = wide_weights
        .select(price_cols.iter().map(|s| s.as_str()))
        .map_err(|e| PyValueError::new_err(format!("Failed to select weight columns: {}", e)))?;

    // Convert to 2D arrays
    let prices_2d = df_to_f64_2d(&prices_only)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert prices: {}", e)))?;

    let weights_2d = df_to_f64_2d(&weights_only)
        .map_err(|e| PyValueError::new_err(format!("Failed to convert weights: {}", e)))?;

    // Get config
    let cfg = config.map(|c| c.inner).unwrap_or_default();

    // Run backtest
    let creturn = run_backtest(&prices_2d, &weights_2d, &rebalance_indices, &cfg);

    Ok(PyBacktestResult {
        creturn,
        trades: vec![],
    })
}

/// Backtest using long format data with zero-copy slice access
///
/// This function processes long format data directly without pivot/partition_by,
/// using zero-copy slice access from Polars arrays.
///
/// # Arguments
/// * `df` - Long format DataFrame with columns: date_idx, symbol_id, price, weight
/// * `n_dates` - Total number of unique dates
/// * `n_symbols` - Total number of unique symbols
/// * `rebalance_mask` - Boolean Series indicating rebalance days
/// * `config` - Backtest configuration
///
/// # Returns
/// List of cumulative returns (one per date)
#[pyfunction]
#[pyo3(signature = (
    date_indices,
    symbol_ids,
    prices,
    weights,
    n_dates,
    n_symbols,
    rebalance_mask,
    config=None
))]
fn backtest_long_format(
    date_indices: PyDataFrame,
    symbol_ids: PyDataFrame,
    prices: PyDataFrame,
    weights: PyDataFrame,
    n_dates: usize,
    n_symbols: usize,
    rebalance_mask: PyDataFrame,
    config: Option<PyBacktestConfig>,
) -> PyResult<Vec<f64>> {
    // Extract Series from single-column DataFrames
    let date_idx_series = date_indices.0.column("date_idx")
        .map_err(|e| PyValueError::new_err(format!("Missing date_idx column: {}", e)))?;
    let symbol_id_series = symbol_ids.0.column("symbol_id")
        .map_err(|e| PyValueError::new_err(format!("Missing symbol_id column: {}", e)))?;
    let price_series = prices.0.column("price")
        .map_err(|e| PyValueError::new_err(format!("Missing price column: {}", e)))?;
    let weight_series = weights.0.column("weight")
        .map_err(|e| PyValueError::new_err(format!("Missing weight column: {}", e)))?;
    let mask_series = rebalance_mask.0.column("rebalance")
        .map_err(|e| PyValueError::new_err(format!("Missing rebalance column: {}", e)))?;

    // Get zero-copy slices from ChunkedArrays
    let date_idx_ca = date_idx_series.u32()
        .map_err(|e| PyValueError::new_err(format!("date_idx must be u32: {}", e)))?;
    let symbol_id_ca = symbol_id_series.u32()
        .map_err(|e| PyValueError::new_err(format!("symbol_id must be u32: {}", e)))?;
    let price_ca = price_series.f64()
        .map_err(|e| PyValueError::new_err(format!("price must be f64: {}", e)))?;
    let weight_ca = weight_series.f64()
        .map_err(|e| PyValueError::new_err(format!("weight must be f64: {}", e)))?;
    let mask_ca = mask_series.bool()
        .map_err(|e| PyValueError::new_err(format!("rebalance must be bool: {}", e)))?;

    // Zero-copy slice access (cont_slice returns &[T] directly from Arrow buffer)
    let date_idx_slice = date_idx_ca.cont_slice()
        .map_err(|e| PyValueError::new_err(format!("date_idx not contiguous: {}", e)))?;
    let symbol_id_slice = symbol_id_ca.cont_slice()
        .map_err(|e| PyValueError::new_err(format!("symbol_id not contiguous: {}", e)))?;
    let price_slice = price_ca.cont_slice()
        .map_err(|e| PyValueError::new_err(format!("price not contiguous: {}", e)))?;
    let weight_slice = weight_ca.cont_slice()
        .map_err(|e| PyValueError::new_err(format!("weight not contiguous: {}", e)))?;

    // Boolean mask needs to be collected (no cont_slice for bool)
    let rebalance_mask_vec: Vec<bool> = mask_ca.into_iter()
        .map(|v| v.unwrap_or(false))
        .collect();

    let cfg = config.map(|c| c.inner).unwrap_or_else(|| {
        BacktestConfig {
            finlab_mode: true,
            ..Default::default()
        }
    });

    // Create input struct with zero-copy slices
    let input = LongFormatInput {
        date_indices: date_idx_slice,
        symbol_ids: symbol_id_slice,
        prices: price_slice,
        weights: weight_slice,
    };

    // Run backtest
    let result = backtest_long(&input, n_dates, n_symbols, &rebalance_mask_vec, &cfg);

    Ok(result.creturn)
}

/// Initialize the Python module
#[pymodule]
fn _polars_backtest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyBacktestConfig>()?;
    m.add_class::<PyTradeRecord>()?;
    m.add_class::<PyBacktestResult>()?;
    m.add_function(wrap_pyfunction!(backtest, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_partitioned, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_with_trades_partitioned, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_signals, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_weights, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_with_trades, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_long_format, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_long_from_df, m)?)?;
    Ok(())
}
