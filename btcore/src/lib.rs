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

pub mod portfolio;
pub mod rebalance;
pub mod returns;
pub mod simulation;
pub mod stats;
pub mod trades;
pub mod weights;

// Re-export commonly used items
pub use returns::{cumulative_returns, daily_returns, portfolio_return};
pub use simulation::{
    run_backtest, run_backtest_finlab_full, run_backtest_with_trades,
    BacktestConfig, BacktestResult, FinlabPriceData, OhlcPrices,
    TradeRecord as SimTradeRecord,
};
pub use stats::{calc_cagr, max_drawdown, sharpe_ratio, sortino_ratio, BacktestStats};
pub use trades::TradeRecord;
pub use weights::IntoWeights;
