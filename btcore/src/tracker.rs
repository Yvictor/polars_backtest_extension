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

// ============================================================================
// Trade Records
// ============================================================================

/// A single trade record for wide format (usize indices)
///
/// Fields use original prices (not adjusted) for entry/exit prices,
/// matching Finlab's actual trading record format.
#[derive(Debug, Clone)]
pub struct TradeRecord {
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
}

impl TradeRecord {
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

/// Trade record for long format (string symbols, i32 dates)
///
/// Similar to TradeRecord but uses string symbols and i32 dates (days since epoch)
/// instead of usize indices for direct use with Polars DataFrames.
#[derive(Debug, Clone)]
pub struct LongTradeRecord {
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
    /// Entry price (original price, not adjusted)
    pub entry_price: f64,
    /// Exit price (original price, not adjusted)
    pub exit_price: Option<f64>,
    /// Trade return (calculated using original prices with fees)
    pub trade_return: Option<f64>,
}

impl LongTradeRecord {
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

/// Result of a backtest simulation including trades (wide format)
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Cumulative returns at each time step
    pub creturn: Vec<f64>,
    /// List of completed trades
    pub trades: Vec<TradeRecord>,
}

/// Result of a long format backtest with trades
#[derive(Debug, Clone)]
pub struct LongBacktestResult {
    /// Unique dates (i32 days since epoch) - same length as creturn
    pub dates: Vec<i32>,
    /// Cumulative returns at each time step (one per unique date)
    pub creturn: Vec<f64>,
    /// List of completed trades with string symbols
    pub trades: Vec<LongTradeRecord>,
}

// ============================================================================
// TradeTracker Trait - Unified with Associated Types
// ============================================================================

/// Trait for tracking trades during simulation
///
/// Uses associated types to support both:
/// - Wide format: `Key=usize`, `Date=usize`, `Record=TradeRecord`
/// - Long format: `Key=String`, `Date=i32`, `Record=LongTradeRecord`
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
    fn open_trade(
        &mut self,
        key: Self::Key,
        entry_date: Self::Date,
        signal_date: Self::Date,
        entry_price: f64,
        weight: f64,
    );

    /// Record closing a trade (rebalance or stop)
    fn close_trade(
        &mut self,
        key: &Self::Key,
        exit_date: Self::Date,
        exit_sig_date: Option<Self::Date>,
        exit_price: f64,
        fee_ratio: f64,
        tax_ratio: f64,
    );

    /// Check if a trade is open for a key
    fn has_open_trade(&self, key: &Self::Key) -> bool;

    /// Add a pending entry (signal given but not yet executed)
    fn add_pending_entry(&mut self, key: Self::Key, signal_date: Self::Date, weight: f64);

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
    fn open_trade(&mut self, _: Self::Key, _: Self::Date, _: Self::Date, _: f64, _: f64) {}

    #[inline]
    fn close_trade(
        &mut self,
        _: &Self::Key,
        _: Self::Date,
        _: Option<Self::Date>,
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
    fn finalize(self, _: f64, _: f64) -> Vec<Self::Record> {
        vec![]
    }
}

// Type aliases for convenience
pub type NoopIndexTracker = NoopTracker<usize, usize, TradeRecord>;
pub type NoopSymbolTracker = NoopTracker<String, i32, LongTradeRecord>;

// ============================================================================
// IndexTracker - Wide format (usize keys)
// ============================================================================

/// Open position tracking for wide format
#[derive(Debug, Clone)]
struct IndexOpenTrade {
    stock_id: usize,
    entry_index: usize,
    entry_sig_index: usize,
    weight: f64,
    entry_price: f64,
}

/// Trade tracker for wide format - uses usize indices
pub struct IndexTracker {
    open_trades: HashMap<usize, IndexOpenTrade>,
    completed_trades: Vec<TradeRecord>,
}

impl TradeTracker for IndexTracker {
    type Key = usize;
    type Date = usize;
    type Record = TradeRecord;

    fn new() -> Self {
        Self {
            open_trades: HashMap::new(),
            completed_trades: Vec::new(),
        }
    }

    fn open_trade(
        &mut self,
        key: usize,
        entry_date: usize,
        signal_date: usize,
        entry_price: f64,
        weight: f64,
    ) {
        self.open_trades.insert(
            key,
            IndexOpenTrade {
                stock_id: key,
                entry_index: entry_date,
                entry_sig_index: signal_date,
                weight,
                entry_price,
            },
        );
    }

    fn close_trade(
        &mut self,
        key: &usize,
        exit_date: usize,
        exit_sig_date: Option<usize>,
        exit_price: f64,
        fee_ratio: f64,
        tax_ratio: f64,
    ) {
        if let Some(open_trade) = self.open_trades.remove(key) {
            let trade = TradeRecord {
                stock_id: *key,
                entry_index: Some(open_trade.entry_index),
                exit_index: Some(exit_date),
                entry_sig_index: open_trade.entry_sig_index,
                exit_sig_index: exit_sig_date,
                position_weight: open_trade.weight,
                entry_price: open_trade.entry_price,
                exit_price: Some(exit_price),
                trade_return: None,
            };
            let trade_return = trade.calculate_return(fee_ratio, tax_ratio);

            self.completed_trades.push(TradeRecord {
                trade_return,
                ..trade
            });
        }
    }

    fn has_open_trade(&self, key: &usize) -> bool {
        self.open_trades.contains_key(key)
    }

    fn add_pending_entry(&mut self, key: usize, signal_date: usize, weight: f64) {
        self.completed_trades.push(TradeRecord {
            stock_id: key,
            entry_index: None,
            exit_index: None,
            entry_sig_index: signal_date,
            exit_sig_index: None,
            position_weight: weight,
            entry_price: f64::NAN,
            exit_price: None,
            trade_return: None,
        });
    }

    fn finalize(mut self, _fee_ratio: f64, _tax_ratio: f64) -> Vec<TradeRecord> {
        // Report open positions as still open (like Finlab: exit_date=NaT)
        for (stock_id, open_trade) in self.open_trades.drain() {
            self.completed_trades.push(TradeRecord {
                stock_id,
                entry_index: Some(open_trade.entry_index),
                exit_index: None,
                entry_sig_index: open_trade.entry_sig_index,
                exit_sig_index: None,
                position_weight: open_trade.weight,
                entry_price: open_trade.entry_price,
                exit_price: None,
                trade_return: None,
            });
        }
        self.completed_trades
    }
}

// ============================================================================
// SymbolTracker - Long format (String keys)
// ============================================================================

/// Open position tracking for long format
#[derive(Debug, Clone)]
struct SymbolOpenTrade {
    symbol: String,
    entry_date: i32,
    entry_sig_date: i32,
    weight: f64,
    entry_price: f64,
}

/// Trade tracker for long format - uses String symbols and i32 dates
pub struct SymbolTracker {
    open_trades: HashMap<String, SymbolOpenTrade>,
    completed_trades: Vec<LongTradeRecord>,
}

impl TradeTracker for SymbolTracker {
    type Key = String;
    type Date = i32;
    type Record = LongTradeRecord;

    fn new() -> Self {
        Self {
            open_trades: HashMap::new(),
            completed_trades: Vec::new(),
        }
    }

    fn open_trade(
        &mut self,
        key: String,
        entry_date: i32,
        signal_date: i32,
        entry_price: f64,
        weight: f64,
    ) {
        self.open_trades.insert(
            key.clone(),
            SymbolOpenTrade {
                symbol: key,
                entry_date,
                entry_sig_date: signal_date,
                weight,
                entry_price,
            },
        );
    }

    fn close_trade(
        &mut self,
        key: &String,
        exit_date: i32,
        exit_sig_date: Option<i32>,
        exit_price: f64,
        fee_ratio: f64,
        tax_ratio: f64,
    ) {
        if let Some(open_trade) = self.open_trades.remove(key) {
            let trade = LongTradeRecord {
                symbol: open_trade.symbol,
                entry_date: Some(open_trade.entry_date),
                exit_date: Some(exit_date),
                entry_sig_date: open_trade.entry_sig_date,
                exit_sig_date,
                position_weight: open_trade.weight,
                entry_price: open_trade.entry_price,
                exit_price: Some(exit_price),
                trade_return: None,
            };
            let trade_return = trade.calculate_return(fee_ratio, tax_ratio);

            self.completed_trades.push(LongTradeRecord {
                trade_return,
                ..trade
            });
        }
    }

    fn has_open_trade(&self, key: &String) -> bool {
        self.open_trades.contains_key(key)
    }

    fn add_pending_entry(&mut self, key: String, signal_date: i32, weight: f64) {
        self.completed_trades.push(LongTradeRecord {
            symbol: key,
            entry_date: None,
            exit_date: None,
            entry_sig_date: signal_date,
            exit_sig_date: None,
            position_weight: weight,
            entry_price: f64::NAN,
            exit_price: None,
            trade_return: None,
        });
    }

    fn finalize(mut self, _fee_ratio: f64, _tax_ratio: f64) -> Vec<LongTradeRecord> {
        // Report open positions as still open
        for (_symbol, open_trade) in self.open_trades.drain() {
            self.completed_trades.push(LongTradeRecord {
                symbol: open_trade.symbol,
                entry_date: Some(open_trade.entry_date),
                exit_date: None,
                entry_sig_date: open_trade.entry_sig_date,
                exit_sig_date: None,
                position_weight: open_trade.weight,
                entry_price: open_trade.entry_price,
                exit_price: None,
                trade_return: None,
            });
        }
        self.completed_trades
    }
}

// ============================================================================
// Legacy aliases for backward compatibility
// ============================================================================

// Keep old names as aliases during transition
pub type RealTracker = IndexTracker;
pub type OpenTrade = IndexOpenTrade;

// Old trait - will be removed after wide.rs is updated
pub(crate) trait LegacyTradeTracker {
    fn new() -> Self;
    fn open_trade(&mut self, stock_id: usize, entry_index: usize, signal_index: usize, entry_price: f64, weight: f64);
    fn close_trade(&mut self, stock_id: usize, exit_index: usize, exit_sig_index: Option<usize>, exit_price: f64, fee_ratio: f64, tax_ratio: f64);
    fn has_open_trade(&self, stock_id: usize) -> bool;
    #[allow(dead_code)]
    fn get_entry_price(&self, stock_id: usize) -> Option<f64>;
    fn add_pending_entry(&mut self, stock_id: usize, signal_index: usize, weight: f64);
    fn finalize(self, last_index: usize, trade_prices: &[f64], fee_ratio: f64, tax_ratio: f64) -> Vec<TradeRecord>;
}

/// Legacy NoopTracker for wide.rs compatibility
pub(crate) struct LegacyNoopTracker;

impl LegacyTradeTracker for LegacyNoopTracker {
    #[inline]
    fn new() -> Self { Self }
    #[inline]
    fn open_trade(&mut self, _: usize, _: usize, _: usize, _: f64, _: f64) {}
    #[inline]
    fn close_trade(&mut self, _: usize, _: usize, _: Option<usize>, _: f64, _: f64, _: f64) {}
    #[inline]
    fn has_open_trade(&self, _: usize) -> bool { false }
    #[inline]
    fn get_entry_price(&self, _: usize) -> Option<f64> { None }
    #[inline]
    fn add_pending_entry(&mut self, _: usize, _: usize, _: f64) {}
    #[inline]
    fn finalize(self, _: usize, _: &[f64], _: f64, _: f64) -> Vec<TradeRecord> { vec![] }
}

/// Legacy RealTracker for wide.rs compatibility
pub(crate) struct LegacyRealTracker {
    open_trades: HashMap<usize, IndexOpenTrade>,
    completed_trades: Vec<TradeRecord>,
}

impl LegacyTradeTracker for LegacyRealTracker {
    fn new() -> Self {
        Self {
            open_trades: HashMap::new(),
            completed_trades: Vec::new(),
        }
    }

    fn open_trade(&mut self, stock_id: usize, entry_index: usize, signal_index: usize, entry_price: f64, weight: f64) {
        self.open_trades.insert(
            stock_id,
            IndexOpenTrade {
                stock_id,
                entry_index,
                entry_sig_index: signal_index,
                weight,
                entry_price,
            },
        );
    }

    fn close_trade(&mut self, stock_id: usize, exit_index: usize, exit_sig_index: Option<usize>, exit_price: f64, fee_ratio: f64, tax_ratio: f64) {
        if let Some(open_trade) = self.open_trades.remove(&stock_id) {
            let trade = TradeRecord {
                stock_id,
                entry_index: Some(open_trade.entry_index),
                exit_index: Some(exit_index),
                entry_sig_index: open_trade.entry_sig_index,
                exit_sig_index,
                position_weight: open_trade.weight,
                entry_price: open_trade.entry_price,
                exit_price: Some(exit_price),
                trade_return: None,
            };
            let trade_return = trade.calculate_return(fee_ratio, tax_ratio);
            self.completed_trades.push(TradeRecord { trade_return, ..trade });
        }
    }

    fn has_open_trade(&self, stock_id: usize) -> bool {
        self.open_trades.contains_key(&stock_id)
    }

    fn get_entry_price(&self, stock_id: usize) -> Option<f64> {
        self.open_trades.get(&stock_id).map(|t| t.entry_price)
    }

    fn add_pending_entry(&mut self, stock_id: usize, signal_index: usize, weight: f64) {
        self.completed_trades.push(TradeRecord {
            stock_id,
            entry_index: None,
            exit_index: None,
            entry_sig_index: signal_index,
            exit_sig_index: None,
            position_weight: weight,
            entry_price: f64::NAN,
            exit_price: None,
            trade_return: None,
        });
    }

    fn finalize(mut self, _last_index: usize, _trade_prices: &[f64], _fee_ratio: f64, _tax_ratio: f64) -> Vec<TradeRecord> {
        for (stock_id, open_trade) in self.open_trades.drain() {
            self.completed_trades.push(TradeRecord {
                stock_id,
                entry_index: Some(open_trade.entry_index),
                exit_index: None,
                entry_sig_index: open_trade.entry_sig_index,
                exit_sig_index: None,
                position_weight: open_trade.weight,
                entry_price: open_trade.entry_price,
                exit_price: None,
                trade_return: None,
            });
        }
        self.completed_trades
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_record_holding_period() {
        let trade = TradeRecord {
            stock_id: 0,
            entry_index: Some(5),
            exit_index: Some(15),
            entry_sig_index: 4,
            exit_sig_index: Some(14),
            position_weight: 0.5,
            entry_price: 100.0,
            exit_price: Some(110.0),
            trade_return: None,
        };
        assert_eq!(trade.holding_period(), Some(10));
    }

    #[test]
    fn test_trade_record_calculate_return() {
        let trade = TradeRecord {
            stock_id: 0,
            entry_index: Some(0),
            exit_index: Some(10),
            entry_sig_index: 0,
            exit_sig_index: Some(9),
            position_weight: 1.0,
            entry_price: 100.0,
            exit_price: Some(110.0),
            trade_return: None,
        };

        let ret = trade.calculate_return(0.001425, 0.003).unwrap();
        let expected = (1.0 - 0.001425) * 1.1 * (1.0 - 0.003 - 0.001425) - 1.0;
        assert!((ret - expected).abs() < 1e-10);
    }

    #[test]
    fn test_noop_tracker() {
        let mut tracker: NoopIndexTracker = NoopTracker::new();
        tracker.open_trade(0, 1, 0, 100.0, 0.5);
        assert!(!tracker.has_open_trade(&0));

        let trades = tracker.finalize(0.001, 0.003);
        assert!(trades.is_empty());
    }

    #[test]
    fn test_index_tracker_open_close() {
        let mut tracker = IndexTracker::new();

        tracker.open_trade(0, 1, 0, 100.0, 0.5);
        assert!(tracker.has_open_trade(&0));
        assert!(!tracker.has_open_trade(&1));

        tracker.close_trade(&0, 10, Some(9), 110.0, 0.001425, 0.003);
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

        tracker.open_trade("2330".to_string(), 19000, 18999, 100.0, 0.5);
        assert!(tracker.has_open_trade(&"2330".to_string()));
        assert!(!tracker.has_open_trade(&"2317".to_string()));

        tracker.close_trade(&"2330".to_string(), 19010, Some(19009), 110.0, 0.001425, 0.003);
        assert!(!tracker.has_open_trade(&"2330".to_string()));

        let trades = tracker.finalize(0.001425, 0.003);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].symbol, "2330");
        assert_eq!(trades[0].entry_date, Some(19000));
        assert_eq!(trades[0].exit_date, Some(19010));
    }

    #[test]
    fn test_symbol_tracker_pending_entry() {
        let mut tracker = SymbolTracker::new();
        tracker.add_pending_entry("2330".to_string(), 19009, 0.5);

        let trades = tracker.finalize(0.001425, 0.003);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].entry_date, None);
        assert!(trades[0].entry_price.is_nan());
    }

    #[test]
    fn test_long_trade_record_holding_days() {
        let trade = LongTradeRecord {
            symbol: "2330".to_string(),
            entry_date: Some(19000),
            exit_date: Some(19010),
            entry_sig_date: 18999,
            exit_sig_date: Some(19009),
            position_weight: 0.5,
            entry_price: 100.0,
            exit_price: Some(110.0),
            trade_return: None,
        };
        assert_eq!(trade.holding_days(), Some(10));
    }
}
