//! Trade tracking for backtest simulation
//!
//! This module provides the TradeTracker trait and implementations for
//! tracking trades during simulation. Uses zero-cost abstraction to
//! avoid overhead when trade tracking is not needed.
//!
//! ## Tracker Types
//!
//! - `NoopTracker` - Zero overhead, for backtest without trade records
//! - `IndexTracker` - Wide format (usize keys, usize dates)
//! - `SymbolTracker` - Long format (String keys, i32 dates)

use std::collections::HashMap;
use std::hash::Hash;

use crate::mae_mfe::calculate_mae_mfe_at_exit;

/// Round raw price to avoid floating point precision issues
/// Uses 6 decimal places which is sufficient for price calculations
#[inline]
fn round_raw_price(price: f64) -> f64 {
    (price * 1_000_000.0).round() / 1_000_000.0
}

// ============================================================================
// Trade Records
// ============================================================================

/// A single trade record for wide format (usize indices)
///
/// Fields use original prices (not adjusted) for entry/exit prices,
/// matching Finlab's actual trading record format.
#[derive(Debug, Clone)]
pub struct WideTradeRecord {
    /// Stock ID (column index in price matrix)
    pub stock_id: usize,
    /// Actual entry date (row index in price matrix, T+1 after signal)
    /// None for pending entries that have signal but not yet executed
    pub entry_index: Option<usize>,
    /// Actual exit date (row index in price matrix)
    pub exit_index: Option<usize>,
    /// Signal date for entry (row index in price matrix)
    pub entry_sig_index: usize,
    /// Signal date for exit (row index in price matrix)
    pub exit_sig_index: Option<usize>,
    /// Position weight at entry
    pub position_weight: f64,
    /// Entry price (ORIGINAL price, not adjusted)
    pub entry_price: f64,
    /// Exit price (ORIGINAL price, not adjusted)
    pub exit_price: Option<f64>,
    /// Trade return (calculated using original prices with fees)
    pub trade_return: Option<f64>,

    // MAE/MFE metrics (calculated at exit)
    /// Maximum Adverse Excursion (max loss during trade, negative)
    pub mae: Option<f64>,
    /// Global Maximum Favorable Excursion (max profit during trade)
    pub gmfe: Option<f64>,
    /// Before-MAE MFE (MFE at the time when MAE occurred)
    pub bmfe: Option<f64>,
    /// Maximum drawdown during the trade
    pub mdd: Option<f64>,
    /// Number of profitable days
    pub pdays: Option<u32>,
    /// Holding period in days
    pub period: Option<u32>,
}

impl WideTradeRecord {
    /// Calculate holding period in days
    pub fn holding_period(&self) -> Option<usize> {
        match (self.entry_index, self.exit_index) {
            (Some(entry), Some(exit)) => Some(exit - entry),
            _ => None,
        }
    }

    /// Calculate trade return with fees
    ///
    /// Finlab formula:
    /// trade_return = (1 - fee_ratio) * (exit_price / entry_price) * (1 - tax_ratio - fee_ratio) - 1
    pub fn calculate_return(&self, fee_ratio: f64, tax_ratio: f64) -> Option<f64> {
        self.exit_price.map(|exit_price| {
            (1.0 - fee_ratio) * (exit_price / self.entry_price) * (1.0 - tax_ratio - fee_ratio) - 1.0
        })
    }
}

/// Trade record (string symbols, i32 dates) - default format
///
/// Uses string symbols and i32 dates (days since epoch) for direct use with Polars DataFrames.
/// This is the default/recommended format.
#[derive(Debug, Clone)]
pub struct TradeRecord {
    /// Stock symbol (string key)
    pub symbol: String,
    /// Actual entry date (days since epoch, T+1 after signal)
    /// None for pending entries that have signal but not yet executed
    pub entry_date: Option<i32>,
    /// Actual exit date (days since epoch)
    pub exit_date: Option<i32>,
    /// Signal date for entry (days since epoch)
    pub entry_sig_date: i32,
    /// Signal date for exit (days since epoch)
    pub exit_sig_date: Option<i32>,
    /// Position weight at entry
    pub position_weight: f64,
    /// Entry price (adjusted price, for return calculation)
    pub entry_price: f64,
    /// Exit price (adjusted price, for return calculation)
    pub exit_price: Option<f64>,
    /// Entry raw price (unadjusted = entry_price / factor, for liquidity metrics)
    pub entry_raw_price: f64,
    /// Exit raw price (unadjusted = exit_price / factor, for liquidity metrics)
    pub exit_raw_price: Option<f64>,
    /// Trade return (calculated using adjusted prices with fees)
    pub trade_return: Option<f64>,

    // MAE/MFE metrics (calculated at exit)
    /// Maximum Adverse Excursion (max loss during trade, negative)
    pub mae: Option<f64>,
    /// Global Maximum Favorable Excursion (max profit during trade)
    pub gmfe: Option<f64>,
    /// Before-MAE MFE (MFE at the time when MAE occurred)
    pub bmfe: Option<f64>,
    /// Maximum drawdown during the trade
    pub mdd: Option<f64>,
    /// Number of profitable days
    pub pdays: Option<u32>,
    /// Holding period in days
    pub period: Option<i32>,
}

impl TradeRecord {
    /// Calculate holding period in days
    pub fn holding_days(&self) -> Option<i32> {
        match (self.entry_date, self.exit_date) {
            (Some(entry), Some(exit)) => Some(exit - entry),
            _ => None,
        }
    }

    /// Calculate trade return with fees
    pub fn calculate_return(&self, fee_ratio: f64, tax_ratio: f64) -> Option<f64> {
        self.exit_price.map(|exit_price| {
            (1.0 - fee_ratio) * (exit_price / self.entry_price) * (1.0 - tax_ratio - fee_ratio) - 1.0
        })
    }
}

// ============================================================================
// Backtest Results
// ============================================================================

/// Stock operations calculated at the end of backtest
///
/// This structure captures the actions (enter/exit/hold) for each stock
/// based on comparing current positions vs the latest signal weights.
/// Matches Finlab's `stock_operations` dictionary.
#[derive(Debug, Clone, Default)]
pub struct StockOperations {
    /// Actions for each stock: "enter", "exit", "hold", "sl", "tp", etc.
    pub actions: HashMap<String, String>,
    /// Current normalized weights (at weight_date)
    pub weights: HashMap<String, f64>,
    /// Next signal weights (at next_weight_date)
    pub next_weights: HashMap<String, f64>,
    /// Date of current weights (days since epoch)
    pub weight_date: Option<i32>,
    /// Date of next weights/signal (days since epoch)
    pub next_weight_date: Option<i32>,
}

impl StockOperations {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Result of a backtest simulation including trades (wide format)
#[derive(Debug, Clone)]
pub struct WideBacktestResult {
    /// Cumulative returns at each time step
    pub creturn: Vec<f64>,
    /// List of completed trades
    pub trades: Vec<WideTradeRecord>,
}

/// Result of a backtest with trades - default format
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Unique dates (i32 days since epoch) - same length as creturn
    pub dates: Vec<i32>,
    /// Cumulative returns at each time step (one per unique date)
    pub creturn: Vec<f64>,
    /// List of completed trades
    pub trades: Vec<TradeRecord>,
    /// Stock operations (actions, weights, next_weights) - Finlab compatible
    pub stock_operations: Option<StockOperations>,
}

// ============================================================================
// TradeTracker Trait - Unified with Associated Types
// ============================================================================

/// Trait for tracking trades during simulation
///
/// Uses associated types to support both:
/// - Wide format: `Key=usize`, `Date=usize`, `Record=WideTradeRecord`
/// - Long format: `Key=String`, `Date=i32`, `Record=TradeRecord`
///
/// This allows zero-cost abstraction when trade tracking is not needed.
pub trait TradeTracker {
    /// Key type for identifying positions (usize for wide, String for long)
    type Key: Clone + Eq + Hash;
    /// Date type (usize index for wide, i32 days since epoch for long)
    type Date: Copy;
    /// Output record type
    type Record;

    /// Create a new tracker
    fn new() -> Self
    where
        Self: Sized;

    /// Record opening a new trade
    /// - `entry_factor`: Factor at entry for raw price calculation (raw = adj / factor)
    fn open_trade(
        &mut self,
        key: Self::Key,
        entry_date: Self::Date,
        signal_date: Self::Date,
        entry_price: f64,
        weight: f64,
        entry_factor: f64,
    );

    /// Record closing a trade (rebalance or stop)
    /// - `exit_factor`: Factor at exit for raw price calculation (raw = adj / factor)
    fn close_trade(
        &mut self,
        key: &Self::Key,
        exit_date: Self::Date,
        exit_sig_date: Option<Self::Date>,
        exit_price: f64,
        exit_factor: f64,
        fee_ratio: f64,
        tax_ratio: f64,
    );

    /// Check if a trade is open for a key
    fn has_open_trade(&self, key: &Self::Key) -> bool;

    /// Add a pending entry (signal given but not yet executed)
    fn add_pending_entry(&mut self, key: Self::Key, signal_date: Self::Date, weight: f64);

    /// Record a price for MAE/MFE calculation
    /// Call this each day while a position is open
    fn record_price(&mut self, key: &Self::Key, close_price: f64, trade_price: f64);

    /// Finalize all open trades at the end of simulation
    fn finalize(self, fee_ratio: f64, tax_ratio: f64) -> Vec<Self::Record>;
}

// ============================================================================
// NoopTracker - Zero overhead implementation
// ============================================================================

/// No-op trade tracker - zero overhead for simple backtest
///
/// Generic over Key, Date, and Record types to work with both wide and long formats.
pub struct NoopTracker<K, D, R> {
    _phantom: std::marker::PhantomData<(K, D, R)>,
}

impl<K, D, R> Default for NoopTracker<K, D, R> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<K, D, R> TradeTracker for NoopTracker<K, D, R>
where
    K: Clone + Eq + Hash,
    D: Copy,
{
    type Key = K;
    type Date = D;
    type Record = R;

    #[inline]
    fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn open_trade(&mut self, _: Self::Key, _: Self::Date, _: Self::Date, _: f64, _: f64, _: f64) {}

    #[inline]
    fn close_trade(
        &mut self,
        _: &Self::Key,
        _: Self::Date,
        _: Option<Self::Date>,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
    ) {
    }

    #[inline]
    fn has_open_trade(&self, _: &Self::Key) -> bool {
        false
    }

    #[inline]
    fn add_pending_entry(&mut self, _: Self::Key, _: Self::Date, _: f64) {}

    #[inline]
    fn record_price(&mut self, _: &Self::Key, _: f64, _: f64) {}

    #[inline]
    fn finalize(self, _: f64, _: f64) -> Vec<Self::Record> {
        vec![]
    }
}

// Type aliases for convenience
pub type NoopIndexTracker = NoopTracker<usize, usize, WideTradeRecord>;
pub type NoopSymbolTracker = NoopTracker<String, i32, TradeRecord>;

// ============================================================================
// Generic TradeTracker Implementation
// ============================================================================

/// Generic open trade info - shared by IndexTracker and SymbolTracker
#[derive(Debug, Clone)]
struct OpenTradeInfo<K: Clone, D: Copy> {
    key: K,
    entry_date: D,
    signal_date: D,
    weight: f64,
    entry_price: f64,
    /// Factor at entry (for raw price calculation: raw = adj / factor)
    entry_factor: f64,
    /// Close prices during the trade (for MAE/MFE calculation)
    close_prices: Vec<f64>,
    /// Trade prices during the trade (for MAE/MFE calculation)
    trade_prices: Vec<f64>,
}

/// Trait for building trade records from generic open trade info
pub trait RecordBuilder: Sized {
    type Key: Clone + Eq + Hash;
    type Date: Copy;

    /// Build a completed trade record (without MAE/MFE calculation)
    fn build_completed(
        key: Self::Key,
        entry_date: Self::Date,
        exit_date: Self::Date,
        signal_date: Self::Date,
        exit_sig_date: Option<Self::Date>,
        weight: f64,
        entry_price: f64,
        exit_price: f64,
        entry_raw_price: f64,
        exit_raw_price: f64,
        fee_ratio: f64,
        tax_ratio: f64,
    ) -> Self;

    /// Build a completed trade record with MAE/MFE calculation
    fn build_completed_with_mae_mfe(
        key: Self::Key,
        entry_date: Self::Date,
        exit_date: Self::Date,
        signal_date: Self::Date,
        exit_sig_date: Option<Self::Date>,
        weight: f64,
        entry_price: f64,
        exit_price: f64,
        entry_raw_price: f64,
        exit_raw_price: f64,
        fee_ratio: f64,
        tax_ratio: f64,
        close_prices: &[f64],
        trade_prices: &[f64],
    ) -> Self;

    /// Build a pending entry record (signal given but not executed)
    fn build_pending(key: Self::Key, signal_date: Self::Date, weight: f64) -> Self;

    /// Build an open position record (entry executed but not exited)
    fn build_open(
        key: Self::Key,
        entry_date: Self::Date,
        signal_date: Self::Date,
        weight: f64,
        entry_price: f64,
        entry_raw_price: f64,
    ) -> Self;

    /// Build an open position record with calculated metrics
    ///
    /// This calculates unrealized return, MAE/MFE using the last price
    /// as a "virtual exit" without exit transaction costs.
    fn build_open_with_metrics(
        key: Self::Key,
        entry_date: Self::Date,
        signal_date: Self::Date,
        weight: f64,
        entry_price: f64,
        entry_raw_price: f64,
        close_prices: &[f64],
        trade_prices: &[f64],
        fee_ratio: f64,
        tax_ratio: f64,
    ) -> Self;
}

impl RecordBuilder for WideTradeRecord {
    type Key = usize;
    type Date = usize;

    fn build_completed(
        key: usize,
        entry_date: usize,
        exit_date: usize,
        signal_date: usize,
        exit_sig_date: Option<usize>,
        weight: f64,
        entry_price: f64,
        exit_price: f64,
        _entry_raw_price: f64,
        _exit_raw_price: f64,
        fee_ratio: f64,
        tax_ratio: f64,
    ) -> Self {
        let trade = Self {
            stock_id: key,
            entry_index: Some(entry_date),
            exit_index: Some(exit_date),
            entry_sig_index: signal_date,
            exit_sig_index: exit_sig_date,
            position_weight: weight,
            entry_price,
            exit_price: Some(exit_price),
            trade_return: None,
            mae: None,
            gmfe: None,
            bmfe: None,
            mdd: None,
            pdays: None,
            period: Some((exit_date - entry_date) as u32),
        };
        Self {
            trade_return: trade.calculate_return(fee_ratio, tax_ratio),
            ..trade
        }
    }

    fn build_completed_with_mae_mfe(
        key: usize,
        entry_date: usize,
        exit_date: usize,
        signal_date: usize,
        exit_sig_date: Option<usize>,
        weight: f64,
        entry_price: f64,
        exit_price: f64,
        _entry_raw_price: f64,
        _exit_raw_price: f64,
        fee_ratio: f64,
        tax_ratio: f64,
        close_prices: &[f64],
        trade_prices: &[f64],
    ) -> Self {
        // Calculate MAE/MFE metrics
        let is_long = weight >= 0.0;
        let exit_idx = close_prices.len().saturating_sub(1);
        let metrics = calculate_mae_mfe_at_exit(
            close_prices,
            trade_prices,
            0,         // entry is at index 0 in our collected prices
            exit_idx,  // exit is at the last index
            is_long,
            true,  // has_entry_transaction
            true,  // has_exit_transaction
            fee_ratio,
            tax_ratio,
        );

        let trade = Self {
            stock_id: key,
            entry_index: Some(entry_date),
            exit_index: Some(exit_date),
            entry_sig_index: signal_date,
            exit_sig_index: exit_sig_date,
            position_weight: weight,
            entry_price,
            exit_price: Some(exit_price),
            trade_return: None,
            mae: Some(metrics.mae),
            gmfe: Some(metrics.gmfe),
            bmfe: Some(metrics.bmfe),
            mdd: Some(metrics.mdd),
            pdays: Some(metrics.pdays),
            period: Some((exit_date - entry_date) as u32),
        };
        Self {
            trade_return: trade.calculate_return(fee_ratio, tax_ratio),
            ..trade
        }
    }

    fn build_pending(key: usize, signal_date: usize, weight: f64) -> Self {
        Self {
            stock_id: key,
            entry_index: None,
            exit_index: None,
            entry_sig_index: signal_date,
            exit_sig_index: None,
            position_weight: weight,
            entry_price: f64::NAN,
            exit_price: None,
            trade_return: None,
            mae: None,
            gmfe: None,
            bmfe: None,
            mdd: None,
            pdays: None,
            period: None,
        }
    }

    fn build_open(
        key: usize,
        entry_date: usize,
        signal_date: usize,
        weight: f64,
        entry_price: f64,
        _entry_raw_price: f64,
    ) -> Self {
        Self {
            stock_id: key,
            entry_index: Some(entry_date),
            exit_index: None,
            entry_sig_index: signal_date,
            exit_sig_index: None,
            position_weight: weight,
            entry_price,
            exit_price: None,
            trade_return: None,
            mae: None,
            gmfe: None,
            bmfe: None,
            mdd: None,
            pdays: None,
            period: None,
        }
    }

    fn build_open_with_metrics(
        key: usize,
        entry_date: usize,
        signal_date: usize,
        weight: f64,
        entry_price: f64,
        _entry_raw_price: f64,
        close_prices: &[f64],
        trade_prices: &[f64],
        fee_ratio: f64,
        _tax_ratio: f64,
    ) -> Self {
        // Calculate MAE/MFE metrics for open position
        // Use last price as "virtual exit" WITHOUT exit transaction costs
        let is_long = weight >= 0.0;
        let exit_idx = close_prices.len().saturating_sub(1);
        let metrics = calculate_mae_mfe_at_exit(
            close_prices,
            trade_prices,
            0,         // entry is at index 0 in our collected prices
            exit_idx,  // exit is at the last index
            is_long,
            true,   // has_entry_transaction
            false,  // NO exit transaction (still open)
            fee_ratio,
            0.0,    // No tax for unrealized return
        );

        let period = close_prices.len().saturating_sub(1) as u32;

        Self {
            stock_id: key,
            entry_index: Some(entry_date),
            exit_index: None,
            entry_sig_index: signal_date,
            exit_sig_index: None,
            position_weight: weight,
            entry_price,
            exit_price: None,
            trade_return: Some(metrics.ret),
            mae: Some(metrics.mae),
            gmfe: Some(metrics.gmfe),
            bmfe: Some(metrics.bmfe),
            mdd: Some(metrics.mdd),
            pdays: Some(metrics.pdays),
            period: Some(period),
        }
    }
}

impl RecordBuilder for TradeRecord {
    type Key = String;
    type Date = i32;

    fn build_completed(
        key: String,
        entry_date: i32,
        exit_date: i32,
        signal_date: i32,
        exit_sig_date: Option<i32>,
        weight: f64,
        entry_price: f64,
        exit_price: f64,
        entry_raw_price: f64,
        exit_raw_price: f64,
        fee_ratio: f64,
        tax_ratio: f64,
    ) -> Self {
        let trade = Self {
            symbol: key,
            entry_date: Some(entry_date),
            exit_date: Some(exit_date),
            entry_sig_date: signal_date,
            exit_sig_date,
            position_weight: weight,
            entry_price,
            exit_price: Some(exit_price),
            entry_raw_price,
            exit_raw_price: Some(exit_raw_price),
            trade_return: None,
            mae: None,
            gmfe: None,
            bmfe: None,
            mdd: None,
            pdays: None,
            period: Some(exit_date - entry_date),
        };
        Self {
            trade_return: trade.calculate_return(fee_ratio, tax_ratio),
            ..trade
        }
    }

    fn build_completed_with_mae_mfe(
        key: String,
        entry_date: i32,
        exit_date: i32,
        signal_date: i32,
        exit_sig_date: Option<i32>,
        weight: f64,
        entry_price: f64,
        exit_price: f64,
        entry_raw_price: f64,
        exit_raw_price: f64,
        fee_ratio: f64,
        tax_ratio: f64,
        close_prices: &[f64],
        trade_prices: &[f64],
    ) -> Self {
        // Calculate MAE/MFE metrics
        let is_long = weight >= 0.0;
        let exit_idx = close_prices.len().saturating_sub(1);
        let metrics = calculate_mae_mfe_at_exit(
            close_prices,
            trade_prices,
            0,         // entry is at index 0 in our collected prices
            exit_idx,  // exit is at the last index
            is_long,
            true,  // has_entry_transaction
            true,  // has_exit_transaction
            fee_ratio,
            tax_ratio,
        );

        let trade = Self {
            symbol: key,
            entry_date: Some(entry_date),
            exit_date: Some(exit_date),
            entry_sig_date: signal_date,
            exit_sig_date,
            position_weight: weight,
            entry_price,
            exit_price: Some(exit_price),
            entry_raw_price,
            exit_raw_price: Some(exit_raw_price),
            trade_return: None,
            mae: Some(metrics.mae),
            gmfe: Some(metrics.gmfe),
            bmfe: Some(metrics.bmfe),
            mdd: Some(metrics.mdd),
            pdays: Some(metrics.pdays),
            period: Some(exit_date - entry_date),
        };
        Self {
            trade_return: trade.calculate_return(fee_ratio, tax_ratio),
            ..trade
        }
    }

    fn build_pending(key: String, signal_date: i32, weight: f64) -> Self {
        Self {
            symbol: key,
            entry_date: None,
            exit_date: None,
            entry_sig_date: signal_date,
            exit_sig_date: None,
            position_weight: weight,
            entry_price: f64::NAN,
            exit_price: None,
            entry_raw_price: f64::NAN,
            exit_raw_price: None,
            trade_return: None,
            mae: None,
            gmfe: None,
            bmfe: None,
            mdd: None,
            pdays: None,
            period: None,
        }
    }

    fn build_open(
        key: String,
        entry_date: i32,
        signal_date: i32,
        weight: f64,
        entry_price: f64,
        entry_raw_price: f64,
    ) -> Self {
        Self {
            symbol: key,
            entry_date: Some(entry_date),
            exit_date: None,
            entry_sig_date: signal_date,
            exit_sig_date: None,
            position_weight: weight,
            entry_price,
            exit_price: None,
            entry_raw_price,
            exit_raw_price: None,
            trade_return: None,
            mae: None,
            gmfe: None,
            bmfe: None,
            mdd: None,
            pdays: None,
            period: None,
        }
    }

    fn build_open_with_metrics(
        key: String,
        entry_date: i32,
        signal_date: i32,
        weight: f64,
        entry_price: f64,
        entry_raw_price: f64,
        close_prices: &[f64],
        trade_prices: &[f64],
        fee_ratio: f64,
        _tax_ratio: f64,
    ) -> Self {
        // Calculate MAE/MFE metrics for open position
        // Use last price as "virtual exit" WITHOUT exit transaction costs
        let is_long = weight >= 0.0;
        let exit_idx = close_prices.len().saturating_sub(1);
        let metrics = calculate_mae_mfe_at_exit(
            close_prices,
            trade_prices,
            0,         // entry is at index 0 in our collected prices
            exit_idx,  // exit is at the last index
            is_long,
            true,   // has_entry_transaction
            false,  // NO exit transaction (still open)
            fee_ratio,
            0.0,    // No tax for unrealized return
        );

        let period = close_prices.len().saturating_sub(1) as i32;

        Self {
            symbol: key,
            entry_date: Some(entry_date),
            exit_date: None,
            entry_sig_date: signal_date,
            exit_sig_date: None,
            position_weight: weight,
            entry_price,
            exit_price: None,
            entry_raw_price,
            exit_raw_price: None,
            trade_return: Some(metrics.ret),
            mae: Some(metrics.mae),
            gmfe: Some(metrics.gmfe),
            bmfe: Some(metrics.bmfe),
            mdd: Some(metrics.mdd),
            pdays: Some(metrics.pdays),
            period: Some(period),
        }
    }
}

/// Generic trade tracker - unified implementation for both formats
pub struct GenericTracker<R: RecordBuilder> {
    open_trades: HashMap<R::Key, OpenTradeInfo<R::Key, R::Date>>,
    completed_trades: Vec<R>,
}

impl<R: RecordBuilder> TradeTracker for GenericTracker<R> {
    type Key = R::Key;
    type Date = R::Date;
    type Record = R;

    fn new() -> Self {
        Self {
            open_trades: HashMap::new(),
            completed_trades: Vec::new(),
        }
    }

    fn open_trade(
        &mut self,
        key: Self::Key,
        entry_date: Self::Date,
        signal_date: Self::Date,
        entry_price: f64,
        weight: f64,
        entry_factor: f64,
    ) {
        // Initialize with the entry price as first price point
        self.open_trades.insert(
            key.clone(),
            OpenTradeInfo {
                key,
                entry_date,
                signal_date,
                weight,
                entry_price,
                entry_factor,
                close_prices: vec![entry_price],
                trade_prices: vec![entry_price],
            },
        );
    }

    fn close_trade(
        &mut self,
        key: &Self::Key,
        exit_date: Self::Date,
        exit_sig_date: Option<Self::Date>,
        exit_price: f64,
        exit_factor: f64,
        fee_ratio: f64,
        tax_ratio: f64,
    ) {
        if let Some(open_trade) = self.open_trades.remove(key) {
            // Calculate raw prices with rounding to avoid floating point precision issues
            let entry_raw_price = round_raw_price(open_trade.entry_price / open_trade.entry_factor);
            let exit_raw_price = round_raw_price(exit_price / exit_factor);

            // Use MAE/MFE calculation if we have price history (more than just entry price)
            if open_trade.close_prices.len() > 1 {
                self.completed_trades.push(R::build_completed_with_mae_mfe(
                    open_trade.key,
                    open_trade.entry_date,
                    exit_date,
                    open_trade.signal_date,
                    exit_sig_date,
                    open_trade.weight,
                    open_trade.entry_price,
                    exit_price,
                    entry_raw_price,
                    exit_raw_price,
                    fee_ratio,
                    tax_ratio,
                    &open_trade.close_prices,
                    &open_trade.trade_prices,
                ));
            } else {
                // No price history - skip MAE/MFE calculation
                self.completed_trades.push(R::build_completed(
                    open_trade.key,
                    open_trade.entry_date,
                    exit_date,
                    open_trade.signal_date,
                    exit_sig_date,
                    open_trade.weight,
                    open_trade.entry_price,
                    exit_price,
                    entry_raw_price,
                    exit_raw_price,
                    fee_ratio,
                    tax_ratio,
                ));
            }
        }
    }

    fn has_open_trade(&self, key: &Self::Key) -> bool {
        self.open_trades.contains_key(key)
    }

    fn add_pending_entry(&mut self, key: Self::Key, signal_date: Self::Date, weight: f64) {
        self.completed_trades
            .push(R::build_pending(key, signal_date, weight));
    }

    fn record_price(&mut self, key: &Self::Key, close_price: f64, trade_price: f64) {
        if let Some(trade) = self.open_trades.get_mut(key) {
            trade.close_prices.push(close_price);
            trade.trade_prices.push(trade_price);
        }
    }

    fn finalize(mut self, fee_ratio: f64, tax_ratio: f64) -> Vec<R> {
        for (_, open_trade) in self.open_trades.drain() {
            // Calculate raw price for open positions with rounding
            let entry_raw_price = round_raw_price(open_trade.entry_price / open_trade.entry_factor);

            // Always calculate metrics for open positions
            // Even if close_prices.len() == 1 (entry day only), we can still calculate
            // basic metrics (period=0, return=-fee, mae=-fee, gmfe=0)
            self.completed_trades.push(R::build_open_with_metrics(
                open_trade.key,
                open_trade.entry_date,
                open_trade.signal_date,
                open_trade.weight,
                open_trade.entry_price,
                entry_raw_price,
                &open_trade.close_prices,
                &open_trade.trade_prices,
                fee_ratio,
                tax_ratio,
            ));
        }
        self.completed_trades
    }
}

// Type aliases for backward compatibility
pub type IndexTracker = GenericTracker<WideTradeRecord>;
pub type SymbolTracker = GenericTracker<TradeRecord>;

// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wide_trade_record_holding_period() {
        let trade = WideTradeRecord {
            stock_id: 0,
            entry_index: Some(5),
            exit_index: Some(15),
            entry_sig_index: 4,
            exit_sig_index: Some(14),
            position_weight: 0.5,
            entry_price: 100.0,
            exit_price: Some(110.0),
            trade_return: None,
            mae: None,
            gmfe: None,
            bmfe: None,
            mdd: None,
            pdays: None,
            period: Some(10),
        };
        assert_eq!(trade.holding_period(), Some(10));
    }

    #[test]
    fn test_wide_trade_record_calculate_return() {
        let trade = WideTradeRecord {
            stock_id: 0,
            entry_index: Some(0),
            exit_index: Some(10),
            entry_sig_index: 0,
            exit_sig_index: Some(9),
            position_weight: 1.0,
            entry_price: 100.0,
            exit_price: Some(110.0),
            trade_return: None,
            mae: None,
            gmfe: None,
            bmfe: None,
            mdd: None,
            pdays: None,
            period: Some(10),
        };

        let ret = trade.calculate_return(0.001425, 0.003).unwrap();
        let expected = (1.0 - 0.001425) * 1.1 * (1.0 - 0.003 - 0.001425) - 1.0;
        assert!((ret - expected).abs() < 1e-10);
    }

    #[test]
    fn test_noop_tracker() {
        let mut tracker: NoopIndexTracker = NoopTracker::new();
        tracker.open_trade(0, 1, 0, 100.0, 0.5, 1.0);
        assert!(!tracker.has_open_trade(&0));

        let trades = tracker.finalize(0.001, 0.003);
        assert!(trades.is_empty());
    }

    #[test]
    fn test_index_tracker_open_close() {
        let mut tracker = IndexTracker::new();

        tracker.open_trade(0, 1, 0, 100.0, 0.5, 1.0);
        assert!(tracker.has_open_trade(&0));
        assert!(!tracker.has_open_trade(&1));

        tracker.close_trade(&0, 10, Some(9), 110.0, 1.0, 0.001425, 0.003);
        assert!(!tracker.has_open_trade(&0));

        let trades = tracker.finalize(0.001425, 0.003);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].stock_id, 0);
        assert_eq!(trades[0].entry_index, Some(1));
        assert_eq!(trades[0].exit_index, Some(10));
    }

    #[test]
    fn test_index_tracker_pending_entry() {
        let mut tracker = IndexTracker::new();
        tracker.add_pending_entry(0, 9, 0.5);

        let trades = tracker.finalize(0.001425, 0.003);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].entry_index, None);
        assert!(trades[0].entry_price.is_nan());
    }

    #[test]
    fn test_symbol_tracker_open_close() {
        let mut tracker = SymbolTracker::new();

        // factor = 1.0 means raw price equals adj price
        tracker.open_trade("2330".to_string(), 19000, 18999, 100.0, 0.5, 1.0);
        assert!(tracker.has_open_trade(&"2330".to_string()));
        assert!(!tracker.has_open_trade(&"2317".to_string()));

        tracker.close_trade(&"2330".to_string(), 19010, Some(19009), 110.0, 1.0, 0.001425, 0.003);
        assert!(!tracker.has_open_trade(&"2330".to_string()));

        let trades = tracker.finalize(0.001425, 0.003);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].symbol, "2330");
        assert_eq!(trades[0].entry_date, Some(19000));
        assert_eq!(trades[0].exit_date, Some(19010));
        // With factor = 1.0, raw_price should equal adj price
        assert_eq!(trades[0].entry_raw_price, 100.0);
        assert_eq!(trades[0].exit_raw_price, Some(110.0));
    }

    #[test]
    fn test_symbol_tracker_with_factor() {
        let mut tracker = SymbolTracker::new();

        // adj_price = 100.0, factor = 2.0 => raw_price = 50.0
        tracker.open_trade("2330".to_string(), 19000, 18999, 100.0, 0.5, 2.0);

        // adj_price = 110.0, factor = 2.2 => raw_price = 50.0
        tracker.close_trade(&"2330".to_string(), 19010, Some(19009), 110.0, 2.2, 0.001425, 0.003);

        let trades = tracker.finalize(0.001425, 0.003);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].entry_price, 100.0);        // adj price
        assert_eq!(trades[0].exit_price, Some(110.0));   // adj price
        assert_eq!(trades[0].entry_raw_price, 50.0);     // 100.0 / 2.0
        assert_eq!(trades[0].exit_raw_price, Some(50.0)); // 110.0 / 2.2 = 50.0 (rounded)
    }

    #[test]
    fn test_symbol_tracker_pending_entry() {
        let mut tracker = SymbolTracker::new();
        tracker.add_pending_entry("2330".to_string(), 19009, 0.5);

        let trades = tracker.finalize(0.001425, 0.003);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].entry_date, None);
        assert!(trades[0].entry_price.is_nan());
        assert!(trades[0].entry_raw_price.is_nan());
    }

    #[test]
    fn test_trade_record_holding_days() {
        let trade = TradeRecord {
            symbol: "2330".to_string(),
            entry_date: Some(19000),
            exit_date: Some(19010),
            entry_sig_date: 18999,
            exit_sig_date: Some(19009),
            position_weight: 0.5,
            entry_price: 100.0,
            exit_price: Some(110.0),
            entry_raw_price: 100.0,
            exit_raw_price: Some(110.0),
            trade_return: None,
            mae: None,
            gmfe: None,
            bmfe: None,
            mdd: None,
            pdays: None,
            period: Some(10),
        };
        assert_eq!(trade.holding_days(), Some(10));
    }
}
