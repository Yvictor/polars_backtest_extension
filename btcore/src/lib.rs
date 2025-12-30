//! btcore: High-performance portfolio backtesting engine
//!
//! Pure Rust implementation without Python dependencies.
//! This crate can be used standalone or as the backend for polars_backtest.
//!
//! # Overview
//!
//! This crate provides a complete portfolio backtesting engine that matches
//! the behavior of Finlab's backtest.sim() function.
//!
//! # Key Features
//!
//! - Equal-weight portfolio simulation
//! - Configurable transaction costs (fee + tax)
//! - Stop loss, take profit, and trailing stop
//! - Position limit per stock
//! - Comprehensive statistics calculation

/// Threshold for floating-point comparisons.
///
/// Used throughout the codebase for:
/// - Checking if values are effectively zero
/// - Comparing floating-point numbers for equality
/// - Finlab compatibility (matches their precision handling)
pub const FLOAT_EPSILON: f64 = 1e-10;

/// Check if a price is valid for calculations.
///
/// A price is valid if it is positive and not NaN.
/// Used for filtering out missing data (NaN) and invalid prices (<=0).
#[inline]
pub fn is_valid_price(price: f64) -> bool {
    price > 0.0 && !price.is_nan()
}

pub mod config;
pub mod portfolio;
pub mod position;
pub mod rebalance;
pub mod returns;
pub mod simulation;
pub mod stats;
pub mod stops;
pub mod tracker;
pub mod trades;
pub mod weights;

// Re-export commonly used items
pub use returns::{cumulative_returns, daily_returns, portfolio_return};
pub use simulation::{
    run_backtest, run_backtest_with_trades, BacktestConfig, BacktestResult, WideBacktestResult,
    PriceData, TradeRecord, WideTradeRecord,
};
pub use stats::{calc_cagr, max_drawdown, sharpe_ratio, sortino_ratio, BacktestStats};
pub use trades::TradeRecord as TradeBookRecord;
pub use weights::IntoWeights;
