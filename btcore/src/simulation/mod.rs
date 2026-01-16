//! Backtest simulation engine
//!
//! This module implements the main backtest simulation loop that matches
//! Finlab's backtest_core Cython implementation.
//!
//! # Module Structure
//!
//! - `wide`: Wide format simulation (2D array input)
//! - `long`: Long format simulation (sorted 1D arrays, multiple interfaces)
//!
//! # Weight Modes
//!
//! Supports two input modes:
//! 1. **Boolean signals** - Converted to equal weights (like Finlab with bool positions)
//! 2. **Float weights** - Custom weights, normalized to sum=1 (like Finlab with float positions)

mod long;
mod wide;

// Re-export public API from wide module
pub use wide::{run_backtest, run_backtest_with_trades, PriceData};

// Re-export public API from long module
pub use long::{
    // Interfaces
    backtest_long_arrow,
    backtest_long_slice,
    backtest_with_accessor,
    backtest_with_report_long_arrow,
    // Types
    LongFormatArrowInput,
    Portfolio,
    ResampleFreq,
    ResampleOffset,
};

// Re-export from other modules for convenience
pub use crate::config::BacktestConfig;
pub use crate::tracker::{BacktestResult, WideBacktestResult, TradeRecord, WideTradeRecord, StockOperations};
pub use crate::weights::IntoWeights;
