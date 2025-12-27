//! Backtest simulation engine
//!
//! This module implements the main backtest simulation loop that matches
//! Finlab's backtest_core Cython implementation.
//!
//! # Module Structure
//!
//! - `wide`: Wide format simulation (current implementation)
//! - Future: `long`: Long format simulation (Stage 6)
//!
//! # Weight Modes
//!
//! Supports two input modes:
//! 1. **Boolean signals** - Converted to equal weights (like Finlab with bool positions)
//! 2. **Float weights** - Custom weights, normalized to sum=1 (like Finlab with float positions)

mod wide;

// Re-export public API from wide module
pub use wide::{run_backtest, run_backtest_with_trades, PriceData};

// Re-export from other modules for convenience
pub use crate::config::BacktestConfig;
pub use crate::tracker::{BacktestResult, TradeRecord};
pub use crate::weights::IntoWeights;
