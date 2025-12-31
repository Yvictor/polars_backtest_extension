//! Backtest Report with statistics and metrics
//!
//! This module provides the PyBacktestReport struct with methods for
//! calculating statistics, metrics, and position information.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3_polars::PyDataFrame;
use polars::prelude::*;
use polars::prelude::QuantileMethod;
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

    /// Get structured metrics as single-row DataFrame
    ///
    /// Args:
    ///     sections: List of sections to include. Options: "backtest", "profitability",
    ///              "risk", "ratio", "winrate". Defaults to all sections.
    ///     riskfree_rate: Annual risk-free rate for Sharpe/Sortino calculations.
    ///
    /// Returns:
    ///     Single-row DataFrame with each metric as a column.
    #[pyo3(signature = (sections=None, riskfree_rate=0.02))]
    fn get_metrics(
        &self,
        sections: Option<Vec<String>>,
        riskfree_rate: f64,
    ) -> PyResult<PyDataFrame> {
        let all_sections = vec!["backtest", "profitability", "risk", "ratio", "winrate"];
        let sections_list: Vec<&str> = match &sections {
            Some(s) => {
                // Validate sections
                for sec in s {
                    if !all_sections.contains(&sec.as_str()) {
                        return Err(PyValueError::new_err(format!(
                            "Invalid section: '{}'. Valid: {:?}",
                            sec, all_sections
                        )));
                    }
                }
                s.iter().map(|s| s.as_str()).collect()
            }
            None => all_sections.clone(),
        };

        let daily = self.compute_daily_creturn().map_err(to_py_err)?;

        if daily.height() < 2 {
            return Err(PyValueError::new_err("Insufficient data for metrics"));
        }

        let nperiods = 252.0_f64;
        let rf_periodic = (1.0 + riskfree_rate).powf(1.0 / nperiods) - 1.0;

        // Prepare daily with returns and drawdown
        let daily_with_return = daily
            .clone()
            .lazy()
            .with_columns([
                (col("creturn") / col("creturn").shift(lit(1)) - lit(1.0))
                    .fill_null(lit(0.0))
                    .alias("return"),
                (col("creturn") / col("creturn").cum_max(false) - lit(1.0))
                    .alias("drawdown"),
            ])
            .collect()
            .map_err(to_py_err)?;

        // Prepare monthly returns for VaR/CVaR
        let monthly_with_return = daily
            .clone()
            .lazy()
            .with_column(col("date").dt().truncate(lit("1mo")).alias("month"))
            .group_by([col("month")])
            .agg([col("creturn").last()])
            .sort(["month"], Default::default())
            .with_column(
                (col("creturn") / col("creturn").shift(lit(1)) - lit(1.0))
                    .fill_null(lit(0.0))
                    .alias("return"),
            )
            .collect()
            .map_err(to_py_err)?;

        // Build expressions based on sections
        let mut exprs: Vec<Expr> = Vec::new();

        // === BACKTEST SECTION ===
        if sections_list.contains(&"backtest") {
            exprs.push(col("date").first().cast(DataType::String).alias("startDate"));
            exprs.push(col("date").last().cast(DataType::String).alias("endDate"));
            exprs.push(lit(self.config.fee_ratio).alias("feeRatio"));
            exprs.push(lit(self.config.tax_ratio).alias("taxRatio"));
            exprs.push(lit("daily").alias("freq"));
            exprs.push(lit("close").alias("tradeAt"));
            exprs.push(if self.config.stop_loss >= 1.0 {
                lit(NULL).alias("stopLoss")
            } else {
                lit(self.config.stop_loss).alias("stopLoss")
            });
            exprs.push(if self.config.take_profit.is_infinite() {
                lit(NULL).alias("takeProfit")
            } else {
                lit(self.config.take_profit).alias("takeProfit")
            });
            exprs.push(if self.config.trail_stop.is_infinite() {
                lit(NULL).alias("trailStop")
            } else {
                lit(self.config.trail_stop).alias("trailStop")
            });
        }

        // === PROFITABILITY SECTION ===
        if sections_list.contains(&"profitability") {
            // Annual return (CAGR)
            exprs.push(
                ((col("creturn").last() / col("creturn").first())
                    .pow(lit(1.0) / ((col("date").last() - col("date").first())
                        .dt().total_days(false).cast(DataType::Float64) / lit(365.25)))
                    - lit(1.0))
                    .alias("annualReturn"),
            );

            // Calculate avg/max number of concurrent positions from trades
            let (avg_n_stock, max_n_stock) =
                self.calc_position_stats().map_err(to_py_err)?;
            exprs.push(lit(avg_n_stock).alias("avgNStock"));
            exprs.push(lit(max_n_stock).alias("maxNStock"));
        }

        // === RISK SECTION ===
        if sections_list.contains(&"risk") {
            exprs.push(col("drawdown").min().alias("maxDrawdown"));

            let avg_dd = self.calc_avg_drawdown(&daily).map_err(to_py_err)?;
            exprs.push(lit(avg_dd).alias("avgDrawdown"));

            // Calculate avgDrawdownDays
            let avg_dd_days = self.calc_avg_drawdown_days(&daily).map_err(to_py_err)?;
            exprs.push(lit(avg_dd_days).alias("avgDrawdownDays"));

            // VaR and CVaR (5% percentile of monthly returns)
            let (var_5, cvar_5) =
                self.calc_var_cvar(&monthly_with_return).map_err(to_py_err)?;
            exprs.push(lit(var_5).alias("valueAtRisk"));
            exprs.push(lit(cvar_5).alias("cvalueAtRisk"));
        }

        // === RATIO SECTION ===
        if sections_list.contains(&"ratio") {
            // Sharpe ratio
            exprs.push(
                (((col("return") - lit(rf_periodic)).mean())
                    / (col("return") - lit(rf_periodic)).std(1)
                    * lit(nperiods.sqrt()))
                    .alias("sharpeRatio"),
            );

            // Sortino ratio
            exprs.push(
                (((col("return") - lit(rf_periodic)).mean())
                    / when(col("return").lt(lit(rf_periodic)))
                        .then(col("return") - lit(rf_periodic))
                        .otherwise(lit(0.0))
                        .std(1)
                    * lit(nperiods.sqrt()))
                    .alias("sortinoRatio"),
            );

            // Calmar ratio
            exprs.push(
                (((col("creturn").last() / col("creturn").first())
                    .pow(lit(1.0) / ((col("date").last() - col("date").first())
                        .dt().total_days(false).cast(DataType::Float64) / lit(365.25)))
                    - lit(1.0))
                    / (lit(0.0) - col("drawdown").min()))
                    .alias("calmarRatio"),
            );

            // Volatility (annualized daily vol)
            exprs.push(
                (col("return").std(1) * lit(nperiods.sqrt())).alias("volatility"),
            );

            // Profit factor and tail ratio (pre-computed)
            let profit_factor = self.calc_profit_factor()?;
            let tail_ratio = self.calc_tail_ratio(&daily_with_return).map_err(to_py_err)?;
            exprs.push(lit(profit_factor).alias("profitFactor"));
            exprs.push(lit(tail_ratio).alias("tailRatio"));
        }

        // === WINRATE SECTION ===
        if sections_list.contains(&"winrate") {
            let win_ratio = self.calc_win_ratio()?;
            let expectancy = self.calc_expectancy()?;
            let (mae, mfe) = self.calc_mae_mfe()?;

            exprs.push(lit(win_ratio).alias("winRate"));
            exprs.push(lit(expectancy).alias("expectancy"));
            exprs.push(lit(mae).alias("mae"));
            exprs.push(lit(mfe).alias("mfe"));
        }

        if exprs.is_empty() {
            return Err(PyValueError::new_err("No sections specified"));
        }

        let result = daily_with_return
            .lazy()
            .select(exprs)
            .collect()
            .map_err(to_py_err)?;

        Ok(PyDataFrame(result))
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

    /// Calculate average drawdown days
    fn calc_avg_drawdown_days(&self, daily: &DataFrame) -> PolarsResult<f64> {
        let dd_df = daily
            .clone()
            .lazy()
            .with_column(
                (col("creturn") / col("creturn").cum_max(false) - lit(1.0))
                    .alias("drawdown"),
            )
            .with_columns([
                // Mark start of new drawdown period
                when(
                    col("drawdown").lt(lit(0.0))
                        .and(col("drawdown").shift(lit(1)).fill_null(lit(0.0)).gt_eq(lit(0.0))),
                )
                .then(lit(1i32))
                .otherwise(lit(0i32))
                .cum_sum(false)
                .alias("dd_period"),
            ])
            .filter(col("drawdown").lt(lit(0.0)))
            .collect()?;

        if dd_df.height() == 0 {
            return Ok(0.0);
        }

        // Count days per drawdown period and average
        let result = dd_df
            .lazy()
            .group_by([col("dd_period")])
            .agg([col("drawdown").len().alias("length")])
            .select([col("length").mean()])
            .collect()?;

        Ok(result
            .column("length")
            .ok()
            .and_then(|c| c.f64().ok())
            .and_then(|c| c.get(0))
            .unwrap_or(0.0))
    }

    /// Calculate VaR and CVaR (5% percentile of monthly returns)
    fn calc_var_cvar(&self, monthly: &DataFrame) -> PolarsResult<(f64, f64)> {
        let return_col = monthly.column("return")?.f64()?;

        // Calculate 5% quantile (VaR)
        let var_5 = return_col.quantile(0.05, QuantileMethod::Linear)?
            .unwrap_or(f64::NAN);

        if var_5.is_nan() {
            return Ok((f64::NAN, f64::NAN));
        }

        // CVaR = mean of returns below VaR
        let cvar_df = monthly
            .clone()
            .lazy()
            .filter(col("return").lt_eq(lit(var_5)))
            .select([col("return").mean()])
            .collect()?;

        let cvar_5 = cvar_df
            .column("return")
            .ok()
            .and_then(|c| c.f64().ok())
            .and_then(|c| c.get(0))
            .unwrap_or(f64::NAN);

        Ok((var_5, cvar_5))
    }

    /// Calculate position statistics (avg and max concurrent positions) from trades
    ///
    /// For each date in creturn, count how many trades are active (entry <= date <= exit).
    fn calc_position_stats(&self) -> PolarsResult<(f64, i64)> {
        let trades = &self.trades_df;

        if trades.height() == 0 {
            return Ok((0.0, 0));
        }

        // Check if we have the required columns
        if trades.column("entry_date").is_err() || trades.column("exit_date").is_err() {
            return Ok((0.0, 0));
        }

        // Get creturn dates as i32 (days since epoch)
        let creturn_dates = self.creturn_df.column("date")?.date()?;
        let last_date = creturn_dates.physical().get(creturn_dates.len() - 1).unwrap_or(0);

        // Get entry/exit dates as Date type
        let entry_col = trades.column("entry_date")?.date()?;
        let exit_col = trades.column("exit_date")?.date()?;

        // Build trade ranges: (entry, exit) as days since epoch
        // For null exit (open positions), use last_date + 1 so they're counted on last_date
        let n_trades = trades.height();
        let mut trade_ranges: Vec<(i32, i32)> = Vec::with_capacity(n_trades);

        for i in 0..n_trades {
            if let Some(entry) = entry_col.physical().get(i) {
                let exit = exit_col.physical().get(i).unwrap_or(last_date + 1);
                trade_ranges.push((entry, exit));
            }
        }

        if trade_ranges.is_empty() {
            return Ok((0.0, 0));
        }

        // Count active positions for each date
        // Active means: entry_date <= date < exit_date
        // (exit_date is the day position is closed at close, so not counted)
        let mut sum = 0i64;
        let mut max = 0i64;
        let n_dates = creturn_dates.len();

        for i in 0..n_dates {
            if let Some(date) = creturn_dates.physical().get(i) {
                let count = trade_ranges
                    .iter()
                    .filter(|(entry, exit)| *entry <= date && date < *exit)
                    .count() as i64;
                sum += count;
                if count > max {
                    max = count;
                }
            }
        }

        let avg = if n_dates > 0 { sum as f64 / n_dates as f64 } else { 0.0 };
        Ok((avg, max))
    }

    /// Calculate profit factor (sum of positive returns / abs(sum of negative returns))
    fn calc_profit_factor(&self) -> PyResult<f64> {
        let trades = &self.trades_df;

        let sums = trades
            .clone()
            .lazy()
            .filter(col("return").is_not_null().and(col("return").is_not_nan()))
            .select([
                col("return")
                    .filter(col("return").gt(lit(0.0)))
                    .sum()
                    .alias("pos_sum"),
                col("return")
                    .filter(col("return").lt(lit(0.0)))
                    .sum()
                    .alias("neg_sum"),
            ])
            .collect()
            .map_err(to_py_err)?;

        let pos_sum = sums
            .column("pos_sum")
            .ok()
            .and_then(|c| c.f64().ok())
            .and_then(|c| c.get(0))
            .unwrap_or(0.0);

        let neg_sum = sums
            .column("neg_sum")
            .ok()
            .and_then(|c| c.f64().ok())
            .and_then(|c| c.get(0))
            .unwrap_or(0.0);

        if neg_sum == 0.0 || neg_sum.is_nan() {
            Ok(f64::INFINITY)
        } else {
            Ok((pos_sum / neg_sum).abs())
        }
    }

    /// Calculate tail ratio (95th percentile / 5th percentile of daily returns)
    fn calc_tail_ratio(&self, daily_with_return: &DataFrame) -> PolarsResult<f64> {
        let return_col = daily_with_return.column("return")?.f64()?;

        let p95 = return_col.quantile(0.95, QuantileMethod::Linear)?
            .unwrap_or(f64::NAN);
        let p05 = return_col.quantile(0.05, QuantileMethod::Linear)?
            .unwrap_or(f64::NAN);

        if p05 == 0.0 || p05.is_nan() || p95.is_nan() {
            Ok(f64::INFINITY)
        } else {
            Ok((p95 / p05).abs())
        }
    }

    /// Calculate expectancy (mean of trade returns)
    fn calc_expectancy(&self) -> PyResult<f64> {
        let trades = &self.trades_df;

        let result = trades
            .clone()
            .lazy()
            .filter(col("return").is_not_null().and(col("return").is_not_nan()))
            .select([col("return").mean()])
            .collect()
            .map_err(to_py_err)?;

        Ok(result
            .column("return")
            .ok()
            .and_then(|c| c.f64().ok())
            .and_then(|c| c.get(0))
            .unwrap_or(f64::NAN))
    }

    /// Calculate MAE and MFE means
    fn calc_mae_mfe(&self) -> PyResult<(f64, f64)> {
        let trades = &self.trades_df;

        // Check if columns exist
        let has_mae = trades.column("mae").is_ok();
        let has_gmfe = trades.column("gmfe").is_ok();

        let mae = if has_mae {
            let result = trades
                .clone()
                .lazy()
                .select([col("mae").mean()])
                .collect()
                .map_err(to_py_err)?;

            result
                .column("mae")
                .ok()
                .and_then(|c| c.f64().ok())
                .and_then(|c| c.get(0))
                .unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };

        let mfe = if has_gmfe {
            let result = trades
                .clone()
                .lazy()
                .select([col("gmfe").mean()])
                .collect()
                .map_err(to_py_err)?;

            result
                .column("gmfe")
                .ok()
                .and_then(|c| c.f64().ok())
                .and_then(|c| c.get(0))
                .unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };

        Ok((mae, mfe))
    }
}
