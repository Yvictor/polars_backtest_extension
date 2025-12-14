//! Polars expression implementations for backtesting

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use btcore::{daily_returns, cumulative_returns, sharpe_ratio, sortino_ratio, max_drawdown};

/// Calculate daily returns from a price series
///
/// Returns a Float64 series with the daily percentage returns.
/// First value is null (no previous price).
#[polars_expr(output_type=Float64)]
fn pl_daily_returns(inputs: &[Series]) -> PolarsResult<Series> {
    let prices = inputs[0].f64()?;
    let prices_vec: Vec<f64> = prices
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();

    let returns = daily_returns(&prices_vec);

    let result: Float64Chunked = returns
        .into_iter()
        .map(|r| r.filter(|v| v.is_finite()))
        .collect();

    Ok(result.into_series().with_name(prices.name().clone()))
}

/// Calculate cumulative returns from a daily returns series
///
/// Returns a Float64 series with cumulative returns starting at 1.0.
#[polars_expr(output_type=Float64)]
fn pl_cumulative_returns(inputs: &[Series]) -> PolarsResult<Series> {
    let returns = inputs[0].f64()?;
    let returns_vec: Vec<Option<f64>> = returns.into_iter().collect();

    let creturn = cumulative_returns(&returns_vec);

    let result: Float64Chunked = creturn.into_iter().map(Some).collect();

    Ok(result.into_series().with_name(returns.name().clone()))
}

/// Calculate Sharpe ratio from a returns series
///
/// Arguments:
/// - inputs[0]: Returns series (Float64)
/// - kwargs: rf (risk-free rate), annualize (trading days per year)
#[polars_expr(output_type=Float64)]
fn pl_sharpe_ratio(inputs: &[Series]) -> PolarsResult<Series> {
    let returns = inputs[0].f64()?;
    let returns_vec: Vec<f64> = returns
        .into_iter()
        .filter_map(|v| v.filter(|x| x.is_finite()))
        .collect();

    // Default: rf=0, annualize=252
    let sharpe = sharpe_ratio(&returns_vec, 0.0, 252.0);

    let result = Float64Chunked::from_slice(returns.name().clone(), &[sharpe]);
    Ok(result.into_series())
}

/// Calculate Sortino ratio from a returns series
#[polars_expr(output_type=Float64)]
fn pl_sortino_ratio(inputs: &[Series]) -> PolarsResult<Series> {
    let returns = inputs[0].f64()?;
    let returns_vec: Vec<f64> = returns
        .into_iter()
        .filter_map(|v| v.filter(|x| x.is_finite()))
        .collect();

    let sortino = sortino_ratio(&returns_vec, 0.0, 252.0);

    let result = Float64Chunked::from_slice(returns.name().clone(), &[sortino]);
    Ok(result.into_series())
}

/// Calculate maximum drawdown from cumulative returns
#[polars_expr(output_type=Float64)]
fn pl_max_drawdown(inputs: &[Series]) -> PolarsResult<Series> {
    let creturn = inputs[0].f64()?;
    let creturn_vec: Vec<f64> = creturn
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .filter(|v| v.is_finite())
        .collect();

    let mdd = max_drawdown(&creturn_vec);

    let result = Float64Chunked::from_slice(creturn.name().clone(), &[mdd]);
    Ok(result.into_series())
}

/// Calculate drawdown series from cumulative returns
#[polars_expr(output_type=Float64)]
fn pl_drawdown_series(inputs: &[Series]) -> PolarsResult<Series> {
    let creturn = inputs[0].f64()?;
    let creturn_vec: Vec<f64> = creturn
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();

    let dd = btcore::stats::drawdown_series(&creturn_vec);

    let result: Float64Chunked = dd.into_iter().map(Some).collect();
    Ok(result.into_series().with_name(creturn.name().clone()))
}

/// Calculate portfolio return for a single period given weights and returns
#[polars_expr(output_type=Float64)]
fn pl_portfolio_return(inputs: &[Series]) -> PolarsResult<Series> {
    let weights = inputs[0].f64()?;
    let returns = inputs[1].f64()?;

    let weights_vec: Vec<f64> = weights
        .into_iter()
        .map(|v| v.unwrap_or(0.0))
        .collect();

    let returns_vec: Vec<f64> = returns
        .into_iter()
        .map(|v| v.unwrap_or(0.0))
        .collect();

    let port_ret = btcore::portfolio_return(&weights_vec, &returns_vec);

    let result = Float64Chunked::from_slice(PlSmallStr::from_static("portfolio_return"), &[port_ret]);
    Ok(result.into_series())
}

/// Calculate equal weights from boolean signals
#[polars_expr(output_type=Float64)]
fn pl_equal_weights(inputs: &[Series]) -> PolarsResult<Series> {
    let signals = inputs[0].bool()?;
    let signals_vec: Vec<bool> = signals
        .into_iter()
        .map(|v| v.unwrap_or(false))
        .collect();

    let weights = btcore::returns::equal_weights(&signals_vec);

    let result: Float64Chunked = weights.into_iter().map(Some).collect();
    Ok(result.into_series().with_name(signals.name().clone()))
}
