//! Backtest Report with statistics and metrics
//!
//! This module provides the PyBacktestReport struct with methods for
//! calculating statistics, metrics, and position information.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3_polars::PyDataFrame;
use polars::prelude::*;
use polars_ops::pivot::pivot;

use btcore::BacktestConfig;

// Helper to convert PolarsError to PyErr
fn to_py_err(e: PolarsError) -> PyErr {
    PyValueError::new_err(format!("{}", e))
}

/// Python wrapper for backtest report with trades as DataFrame
#[pyclass(name = "BacktestReport")]
#[derive(Clone)]
pub struct PyBacktestReport {
    pub(crate) creturn_df: DataFrame,
    pub(crate) trades_df: DataFrame,
    pub(crate) config: BacktestConfig,
    pub(crate) resample: Option<String>,
}

impl PyBacktestReport {
    /// Create a new PyBacktestReport
    pub fn new(
        creturn_df: DataFrame,
        trades_df: DataFrame,
        config: BacktestConfig,
        resample: Option<String>,
    ) -> Self {
        Self {
            creturn_df,
            trades_df,
            config,
            resample,
        }
    }
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

    /// Get fee ratio
    #[getter]
    fn fee_ratio(&self) -> f64 {
        self.config.fee_ratio
    }

    /// Get tax ratio
    #[getter]
    fn tax_ratio(&self) -> f64 {
        self.config.tax_ratio
    }

    /// Get stop loss threshold
    #[getter]
    fn stop_loss(&self) -> Option<f64> {
        if self.config.stop_loss >= 1.0 {
            None
        } else {
            Some(self.config.stop_loss)
        }
    }

    /// Get take profit threshold
    #[getter]
    fn take_profit(&self) -> Option<f64> {
        if self.config.take_profit.is_infinite() {
            None
        } else {
            Some(self.config.take_profit)
        }
    }

    /// Get trail stop threshold
    #[getter]
    fn trail_stop(&self) -> Option<f64> {
        if self.config.trail_stop.is_infinite() {
            None
        } else {
            Some(self.config.trail_stop)
        }
    }

    /// Get trade_at setting (always "close" for now)
    #[getter]
    fn trade_at(&self) -> &str {
        "close"
    }

    /// Get resample setting
    #[getter]
    fn get_resample(&self) -> Option<&str> {
        self.resample.as_deref()
    }

    /// Get daily resampled cumulative return DataFrame
    fn daily_creturn(&self) -> PyResult<PyDataFrame> {
        let df = self.compute_daily_creturn().map_err(to_py_err)?;
        Ok(PyDataFrame(df))
    }

    /// Get backtest statistics as a single-row DataFrame
    #[pyo3(signature = (riskfree_rate=0.02))]
    fn get_stats(&self, riskfree_rate: f64) -> PyResult<PyDataFrame> {
        let daily = self.compute_daily_creturn().map_err(to_py_err)?;

        if daily.height() < 2 {
            return Err(PyValueError::new_err("Insufficient data for statistics"));
        }

        let nperiods = 252.0_f64;
        let rf_periodic = (1.0 + riskfree_rate).powf(1.0 / nperiods) - 1.0;

        // Calculate avg_drawdown separately (need period logic)
        let avg_dd = self.calc_avg_drawdown(&daily).map_err(to_py_err)?;
        let win_ratio = self.calc_win_ratio()?;

        // Use expressions for stats calculation
        let result = daily
            .lazy()
            .with_columns([
                // Daily return
                (col("creturn") / col("creturn").shift(lit(1)) - lit(1.0))
                    .fill_null(lit(0.0))
                    .alias("return"),
                // Drawdown
                (col("creturn") / col("creturn").cum_max(false) - lit(1.0))
                    .alias("drawdown"),
            ])
            .select([
                // Start/end dates and riskfree rate
                col("date").first().alias("start"),
                col("date").last().alias("end"),
                lit(riskfree_rate).alias("rf"),
                // Total return
                (col("creturn").last() / col("creturn").first() - lit(1.0))
                    .alias("total_return"),
                // CAGR - use dt().total_days(false) to get duration in days
                ((col("creturn").last() / col("creturn").first())
                    .pow(lit(1.0) / ((col("date").last() - col("date").first())
                        .dt().total_days(false).cast(DataType::Float64) / lit(365.25)))
                    - lit(1.0))
                    .alias("cagr"),
                // Max drawdown
                col("drawdown").min().alias("max_drawdown"),
                // Avg drawdown (pre-calculated)
                lit(avg_dd).alias("avg_drawdown"),
                // Daily mean (annualized)
                (col("return").mean() * lit(nperiods)).alias("daily_mean"),
                // Daily volatility (annualized)
                (col("return").std(1) * lit(nperiods.sqrt())).alias("daily_vol"),
                // Sharpe ratio
                (((col("return") - lit(rf_periodic)).mean())
                    / (col("return") - lit(rf_periodic)).std(1)
                    * lit(nperiods.sqrt()))
                    .alias("daily_sharpe"),
                // Sortino ratio
                (((col("return") - lit(rf_periodic)).mean())
                    / when(col("return").lt(lit(rf_periodic)))
                        .then(col("return") - lit(rf_periodic))
                        .otherwise(lit(0.0))
                        .std(1)
                    * lit(nperiods.sqrt()))
                    .alias("daily_sortino"),
                // Best/worst day
                col("return").max().alias("best_day"),
                col("return").min().alias("worst_day"),
                // Calmar ratio - also use dt().total_days(false)
                (((col("creturn").last() / col("creturn").first())
                    .pow(lit(1.0) / ((col("date").last() - col("date").first())
                        .dt().total_days(false).cast(DataType::Float64) / lit(365.25)))
                    - lit(1.0))
                    / (lit(0.0) - col("drawdown").min()))
                    .alias("calmar"),
                // Win ratio (pre-calculated)
                lit(win_ratio).alias("win_ratio"),
            ])
            .collect()
            .map_err(to_py_err)?;

        Ok(PyDataFrame(result))
    }

    /// Get monthly statistics as a single-row DataFrame
    #[pyo3(signature = (riskfree_rate=0.02))]
    fn get_monthly_stats(&self, riskfree_rate: f64) -> PyResult<PyDataFrame> {
        let daily = self.compute_daily_creturn().map_err(to_py_err)?;
        let nperiods = 12.0_f64;
        let rf_periodic = (1.0 + riskfree_rate).powf(1.0 / nperiods) - 1.0;

        let result = daily
            .lazy()
            .with_column(col("date").dt().truncate(lit("1mo")).alias("month"))
            .group_by([col("month")])
            .agg([col("creturn").last()])
            .sort(["month"], Default::default())
            .with_column(
                (col("creturn") / col("creturn").shift(lit(1)) - lit(1.0))
                    .fill_null(lit(0.0))
                    .alias("return")
            )
            .select([
                (col("return").mean() * lit(nperiods)).alias("monthly_mean"),
                (col("return").std(1) * lit(nperiods.sqrt())).alias("monthly_vol"),
                (((col("return") - lit(rf_periodic)).mean())
                    / (col("return") - lit(rf_periodic)).std(1)
                    * lit(nperiods.sqrt()))
                    .alias("monthly_sharpe"),
                (((col("return") - lit(rf_periodic)).mean())
                    / when(col("return").lt(lit(rf_periodic)))
                        .then(col("return") - lit(rf_periodic))
                        .otherwise(lit(0.0))
                        .std(1)
                    * lit(nperiods.sqrt()))
                    .alias("monthly_sortino"),
                col("return").max().alias("best_month"),
                col("return").min().alias("worst_month"),
            ])
            .collect()
            .map_err(to_py_err)?;

        Ok(PyDataFrame(result))
    }

    /// Get monthly return table (year x month pivot)
    fn get_return_table(&self) -> PyResult<PyDataFrame> {
        let daily = self.compute_daily_creturn().map_err(to_py_err)?;

        let monthly = daily
            .lazy()
            .with_columns([
                col("date").dt().year().alias("year"),
                col("date").dt().month().alias("month"),
            ])
            .group_by([col("year"), col("month")])
            .agg([col("creturn").last().alias("month_end")])
            .sort(["year", "month"], Default::default())
            .with_column(
                (col("month_end") / col("month_end").shift(lit(1)) - lit(1.0))
                    .fill_null(lit(0.0))
                    .alias("monthly_return")
            )
            .collect()
            .map_err(to_py_err)?;

        // Pivot to year x month format
        let pivoted = pivot(
            &monthly,
            [PlSmallStr::from_static("month")],
            Some([PlSmallStr::from_static("year")]),
            Some([PlSmallStr::from_static("monthly_return")]),
            false,
            None,
            None,
        )
        .map_err(to_py_err)?;

        Ok(PyDataFrame(pivoted))
    }

    /// Get current trades (active positions)
    fn current_trades(&self) -> PyResult<PyDataFrame> {
        let trades = &self.trades_df;
        if trades.height() == 0 {
            return Ok(PyDataFrame(trades.clone()));
        }

        // Get last date from creturn
        let last_date = self.get_last_date_expr()?;

        let current = trades
            .clone()
            .lazy()
            .filter(
                col("exit_date").is_null()
                    .or(col("exit_date").eq(lit(last_date)))
            )
            .collect()
            .map_err(to_py_err)?;

        Ok(PyDataFrame(current))
    }

    /// Get trade actions (enter/exit/hold)
    fn actions(&self) -> PyResult<PyDataFrame> {
        let trades = &self.trades_df;
        if trades.height() == 0 {
            let empty = DataFrame::new(vec![
                Series::new_empty("stock_id".into(), &DataType::String).into_column(),
                Series::new_empty("action".into(), &DataType::String).into_column(),
            ]).map_err(to_py_err)?;
            return Ok(PyDataFrame(empty));
        }

        let last_date = self.get_last_date_expr()?;

        let result = trades
            .clone()
            .lazy()
            .select([
                col("stock_id"),
                when(col("entry_date").eq(lit(last_date)))
                    .then(lit("enter"))
                    .when(col("exit_date").eq(lit(last_date)))
                    .then(lit("exit"))
                    .when(col("exit_date").is_null())
                    .then(lit("hold"))
                    .otherwise(lit("closed"))
                    .alias("action"),
            ])
            .filter(col("action").neq(lit("closed")))
            .collect()
            .map_err(to_py_err)?;

        Ok(PyDataFrame(result))
    }

    /// Check if any trade was triggered by stop loss or take profit
    fn is_stop_triggered(&self) -> PyResult<bool> {
        let current = self.current_trades()?;
        let current_df = &current.0;

        if current_df.height() == 0 {
            return Ok(false);
        }

        // Check stop loss
        if self.config.stop_loss < 1.0 {
            let sl_count = current_df
                .clone()
                .lazy()
                .filter(
                    col("return").is_not_null()
                        .and(col("return").lt_eq(lit(-self.config.stop_loss)))
                )
                .collect()
                .map_err(to_py_err)?
                .height();
            if sl_count > 0 {
                return Ok(true);
            }
        }

        // Check take profit
        if !self.config.take_profit.is_infinite() {
            let tp_count = current_df
                .clone()
                .lazy()
                .filter(
                    col("return").is_not_null()
                        .and(col("return").gt_eq(lit(self.config.take_profit)))
                )
                .collect()
                .map_err(to_py_err)?
                .height();
            if tp_count > 0 {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn __repr__(&self) -> String {
        format!(
            "BacktestReport(creturn_len={}, trades_count={})",
            self.creturn_df.height(), self.trades_df.height(),
        )
    }
}

// Helper methods (not exposed to Python)
impl PyBacktestReport {
    /// Compute daily creturn DataFrame
    fn compute_daily_creturn(&self) -> PolarsResult<DataFrame> {
        self.creturn_df
            .clone()
            .lazy()
            .with_column(col("date").cast(DataType::Date))
            .group_by([col("date")])
            .agg([col("creturn").last()])
            .sort(["date"], Default::default())
            .collect()
    }

    /// Get last date as a scalar for filtering
    fn get_last_date_expr(&self) -> PyResult<i32> {
        let date_col = self.creturn_df.column("date")
            .map_err(to_py_err)?
            .date()
            .map_err(to_py_err)?;

        // Access physical representation
        let phys = &date_col.phys;
        phys.get(phys.len() - 1)
            .ok_or_else(|| PyValueError::new_err("No dates in creturn"))
    }

    /// Calculate average drawdown (mean of per-period minimum drawdowns)
    fn calc_avg_drawdown(&self, daily: &DataFrame) -> PolarsResult<f64> {
        // Add drawdown column and period detection
        let dd_df = daily
            .clone()
            .lazy()
            .with_column(
                (col("creturn") / col("creturn").cum_max(false) - lit(1.0))
                    .alias("drawdown")
            )
            .with_column(
                when(
                    col("drawdown").lt(lit(0.0))
                        .and(col("drawdown").shift(lit(1)).fill_null(lit(0.0)).gt_eq(lit(0.0)))
                )
                .then(lit(1i32))
                .otherwise(lit(0i32))
                .cum_sum(false)
                .alias("dd_period")
            )
            .filter(col("drawdown").lt(lit(0.0)))
            .collect()?;

        if dd_df.height() == 0 {
            return Ok(0.0);
        }

        // Get min drawdown per period and average
        let result = dd_df
            .lazy()
            .group_by([col("dd_period")])
            .agg([col("drawdown").min()])
            .select([col("drawdown").mean()])
            .collect()?;

        Ok(result
            .column("drawdown")
            .ok()
            .and_then(|c| c.f64().ok())
            .and_then(|c| c.get(0))
            .unwrap_or(0.0))
    }

    /// Calculate win ratio from trades
    fn calc_win_ratio(&self) -> PyResult<f64> {
        let trades = &self.trades_df;

        let stats = trades
            .clone()
            .lazy()
            .filter(col("return").is_not_null().and(col("return").is_not_nan()))
            .select([
                col("return").count().alias("total"),
                col("return").filter(col("return").gt(lit(0.0))).count().alias("winners"),
            ])
            .collect()
            .map_err(to_py_err)?;

        let total = stats.column("total")
            .ok()
            .and_then(|c| c.u32().ok())
            .and_then(|c| c.get(0))
            .unwrap_or(0) as f64;

        let winners = stats.column("winners")
            .ok()
            .and_then(|c| c.u32().ok())
            .and_then(|c| c.get(0))
            .unwrap_or(0) as f64;

        if total == 0.0 {
            Ok(0.0)
        } else {
            Ok(winners / total)
        }
    }
}
