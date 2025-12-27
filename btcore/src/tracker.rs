//! Trade tracking for backtest simulation
//!
//! This module provides the TradeTracker trait and implementations for
//! tracking trades during simulation. Uses zero-cost abstraction to
//! avoid overhead when trade tracking is not needed.

use std::collections::HashMap;

/// A single trade record matching Finlab's trades DataFrame structure
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

/// Open position tracking for trades generation
#[derive(Debug, Clone)]
pub(crate) struct OpenTrade {
    /// Stock ID
    #[allow(dead_code)]
    pub stock_id: usize,
    /// Entry index (actual entry date, T+1)
    pub entry_index: usize,
    /// Signal index (entry signal date)
    pub entry_sig_index: usize,
    /// Position weight
    pub weight: f64,
    /// Entry price (original, not adjusted)
    pub entry_price: f64,
}

/// Result of a backtest simulation including trades
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Cumulative returns at each time step
    pub creturn: Vec<f64>,
    /// List of completed trades
    pub trades: Vec<TradeRecord>,
}

// ============================================================================
// Trade Tracker Trait - abstracts trade tracking for zero-cost abstraction
// ============================================================================

/// Trait for tracking trades during simulation
///
/// This allows `run_backtest` to use `NoopTracker` (zero overhead)
/// while `run_backtest_with_trades` uses `RealTracker` (full tracking).
pub(crate) trait TradeTracker {
    /// Create a new tracker
    fn new() -> Self;

    /// Record opening a new trade
    fn open_trade(
        &mut self,
        stock_id: usize,
        entry_index: usize,
        signal_index: usize,
        entry_price: f64,
        weight: f64,
    );

    /// Record closing a trade (rebalance or stop)
    fn close_trade(
        &mut self,
        stock_id: usize,
        exit_index: usize,
        exit_sig_index: Option<usize>,
        exit_price: f64,
        fee_ratio: f64,
        tax_ratio: f64,
    );

    /// Check if a trade is open for a stock
    fn has_open_trade(&self, stock_id: usize) -> bool;

    /// Get entry price for an open trade (needed for return calculation)
    #[allow(dead_code)]
    fn get_entry_price(&self, stock_id: usize) -> Option<f64>;

    /// Add a pending entry (signal given but not yet executed)
    /// Used for trades at the end of simulation where entry would happen on the next day
    fn add_pending_entry(&mut self, stock_id: usize, signal_index: usize, weight: f64);

    /// Finalize all open trades at the end of simulation
    fn finalize(
        self,
        last_index: usize,
        trade_prices: &[f64],
        fee_ratio: f64,
        tax_ratio: f64,
    ) -> Vec<TradeRecord>;
}

/// No-op trade tracker - zero overhead for simple backtest
pub(crate) struct NoopTracker;

impl TradeTracker for NoopTracker {
    #[inline]
    fn new() -> Self {
        Self
    }

    #[inline]
    fn open_trade(&mut self, _: usize, _: usize, _: usize, _: f64, _: f64) {}

    #[inline]
    fn close_trade(&mut self, _: usize, _: usize, _: Option<usize>, _: f64, _: f64, _: f64) {}

    #[inline]
    fn has_open_trade(&self, _: usize) -> bool {
        false
    }

    #[inline]
    fn get_entry_price(&self, _: usize) -> Option<f64> {
        None
    }

    #[inline]
    fn add_pending_entry(&mut self, _: usize, _: usize, _: f64) {}

    #[inline]
    fn finalize(self, _: usize, _: &[f64], _: f64, _: f64) -> Vec<TradeRecord> {
        vec![]
    }
}

/// Real trade tracker - tracks all trades for full reporting
pub(crate) struct RealTracker {
    open_trades: HashMap<usize, OpenTrade>,
    completed_trades: Vec<TradeRecord>,
}

impl TradeTracker for RealTracker {
    fn new() -> Self {
        Self {
            open_trades: HashMap::new(),
            completed_trades: Vec::new(),
        }
    }

    fn open_trade(
        &mut self,
        stock_id: usize,
        entry_index: usize,
        signal_index: usize,
        entry_price: f64,
        weight: f64,
    ) {
        self.open_trades.insert(
            stock_id,
            OpenTrade {
                stock_id,
                entry_index,
                entry_sig_index: signal_index,
                weight,
                entry_price,
            },
        );
    }

    fn close_trade(
        &mut self,
        stock_id: usize,
        exit_index: usize,
        exit_sig_index: Option<usize>,
        exit_price: f64,
        fee_ratio: f64,
        tax_ratio: f64,
    ) {
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

            self.completed_trades.push(TradeRecord {
                trade_return,
                ..trade
            });
        }
    }

    fn has_open_trade(&self, stock_id: usize) -> bool {
        self.open_trades.contains_key(&stock_id)
    }

    fn get_entry_price(&self, stock_id: usize) -> Option<f64> {
        self.open_trades.get(&stock_id).map(|t| t.entry_price)
    }

    fn add_pending_entry(&mut self, stock_id: usize, signal_index: usize, weight: f64) {
        // Add a pending entry record (Finlab: entry_date=NaT for signals not yet executed)
        // This is for stocks that have a buy signal on the last day but would execute on the next day
        self.completed_trades.push(TradeRecord {
            stock_id,
            entry_index: None,  // Not yet executed
            exit_index: None,
            entry_sig_index: signal_index,
            exit_sig_index: None,
            position_weight: weight,
            entry_price: f64::NAN,  // No entry price yet
            exit_price: None,
            trade_return: None,
        });
    }

    fn finalize(
        mut self,
        _last_index: usize,
        _trade_prices: &[f64],
        _fee_ratio: f64,
        _tax_ratio: f64,
    ) -> Vec<TradeRecord> {
        // Report open positions as still open (like Finlab: exit_date=NaT)
        // Don't close them at the last index
        for (stock_id, open_trade) in self.open_trades.drain() {
            self.completed_trades.push(TradeRecord {
                stock_id,
                entry_index: Some(open_trade.entry_index),
                exit_index: None, // Open position - no exit
                entry_sig_index: open_trade.entry_sig_index,
                exit_sig_index: None,
                position_weight: open_trade.weight,
                entry_price: open_trade.entry_price,
                exit_price: None, // Open position - no exit price
                trade_return: None, // Open position - no return calculated
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
        // (1 - 0.001425) * (110/100) * (1 - 0.003 - 0.001425) - 1
        let expected = (1.0 - 0.001425) * 1.1 * (1.0 - 0.003 - 0.001425) - 1.0;
        assert!((ret - expected).abs() < 1e-10);
    }

    #[test]
    fn test_noop_tracker() {
        let mut tracker = NoopTracker::new();
        tracker.open_trade(0, 1, 0, 100.0, 0.5);
        assert!(!tracker.has_open_trade(0));

        let trades = tracker.finalize(10, &[], 0.001, 0.003);
        assert!(trades.is_empty());
    }

    #[test]
    fn test_real_tracker_open_close() {
        let mut tracker = RealTracker::new();

        tracker.open_trade(0, 1, 0, 100.0, 0.5);
        assert!(tracker.has_open_trade(0));
        assert!(!tracker.has_open_trade(1));

        tracker.close_trade(0, 10, Some(9), 110.0, 0.001425, 0.003);
        assert!(!tracker.has_open_trade(0));

        let trades = tracker.finalize(10, &[], 0.001425, 0.003);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].stock_id, 0);
        assert_eq!(trades[0].entry_index, Some(1));
        assert_eq!(trades[0].exit_index, Some(10));
    }

    #[test]
    fn test_real_tracker_pending_entry() {
        let mut tracker = RealTracker::new();

        tracker.add_pending_entry(0, 9, 0.5);

        let trades = tracker.finalize(10, &[], 0.001425, 0.003);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].entry_index, None);
        assert!(trades[0].entry_price.is_nan());
    }

    #[test]
    fn test_real_tracker_open_position_at_end() {
        let mut tracker = RealTracker::new();

        tracker.open_trade(0, 1, 0, 100.0, 0.5);
        // Don't close - position remains open

        let trades = tracker.finalize(10, &[], 0.001425, 0.003);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].entry_index, Some(1));
        assert_eq!(trades[0].exit_index, None); // Still open
        assert_eq!(trades[0].exit_price, None);
    }
}
