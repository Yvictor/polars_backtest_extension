//! Position tracking for backtest simulation
//!
//! This module contains the Position struct that tracks the state of a single stock position.

/// Position in a single stock
///
/// Tracks all state needed for:
/// - Value calculation (standard and Finlab mode)
/// - Stop loss / take profit / trailing stop detection
/// - Cumulative return tracking (cr/maxcr for Finlab mode)
#[derive(Debug, Clone)]
pub struct Position {
    /// Current value of the position (as fraction of portfolio)
    /// In standard mode: updated daily with returns
    /// In Finlab mode: this is the cost_basis (constant after entry)
    pub value: f64,

    /// Entry price for value calculation in Finlab mode
    /// In Finlab mode: used to calculate market value = cost_basis * close / entry_price
    /// This is reset on rebalance when retain_cost_when_rebalance=false
    pub entry_price: f64,

    /// Original entry price for stop loss/take profit calculation
    /// This is NEVER reset during rebalance - always the original entry price
    pub stop_entry_price: f64,

    /// Maximum price since entry (for trailing stop)
    pub max_price: f64,

    /// Last valid market value (Finlab: pos[sid] updated daily)
    /// Used when current price is NaN
    pub last_market_value: f64,

    /// Cumulative return ratio for stop detection (Finlab: cr[sid] *= r)
    /// This uses cumulative multiplication to match Finlab's floating point behavior
    pub cr: f64,

    /// Maximum cumulative return for trail stop (Finlab: maxcr[sid] = max(maxcr[sid], cr[sid]))
    /// This uses cumulative max to match Finlab's floating point behavior
    pub maxcr: f64,

    /// Previous price for daily return calculation (r = current / previous)
    pub previous_price: f64,
}

impl Position {
    /// Create a new position at the given price
    pub fn new(value: f64, price: f64) -> Self {
        Self {
            value,
            entry_price: price,
            stop_entry_price: price,
            max_price: price,
            last_market_value: value,
            cr: 1.0,
            maxcr: 1.0,
            previous_price: price,
        }
    }

    /// Create a new position with NaN price (price not available at entry)
    pub fn new_with_nan_price(value: f64) -> Self {
        Self {
            value,
            entry_price: 0.0,       // Signal that price was NaN at entry
            stop_entry_price: 0.0,
            max_price: 0.0,
            last_market_value: value, // Use position value as market value
            cr: 1.0,
            maxcr: 1.0,
            previous_price: 0.0,
        }
    }

    /// Update position with daily return
    ///
    /// Updates: cr, last_market_value, maxcr
    /// Does NOT update previous_price (call update_previous_price separately)
    pub fn update_with_return(&mut self, current_price: f64) {
        if current_price.is_nan() || current_price <= 0.0 {
            return;
        }

        // Update max_price for trailing stop
        if current_price > self.max_price {
            self.max_price = current_price;
        }

        // Update cr, last_market_value using Finlab's cumulative multiplication
        // Finlab line 304-320: r = price / previous_price; pos *= r; cr *= r; maxcr = max(maxcr, cr)
        if self.previous_price > 0.0 {
            let r = current_price / self.previous_price;
            self.cr *= r;
            self.last_market_value *= r;  // Finlab: pos[sid] *= r
        }

        // Update maxcr using Finlab's cumulative max
        self.maxcr = self.maxcr.max(self.cr);
    }

    /// Update previous_price after return calculation
    pub fn update_previous_price(&mut self, price: f64) {
        if !price.is_nan() && price > 0.0 {
            self.previous_price = price;
        }
    }

    /// Check if this is a long position
    pub fn is_long(&self) -> bool {
        self.last_market_value >= 0.0
    }

    /// Reset stop tracking (used after rebalance when retain_cost=false)
    pub fn reset_stop_tracking(&mut self, price: f64) {
        self.stop_entry_price = price;
        self.max_price = price;
        self.cr = 1.0;
        self.maxcr = 1.0;
        self.previous_price = price;
    }
}

impl Default for Position {
    fn default() -> Self {
        Self {
            value: 0.0,
            entry_price: 0.0,
            stop_entry_price: 0.0,
            max_price: 0.0,
            last_market_value: 0.0,
            cr: 1.0,
            maxcr: 1.0,
            previous_price: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_position() {
        let pos = Position::new(1000.0, 100.0);
        assert!((pos.value - 1000.0).abs() < 1e-10);
        assert!((pos.entry_price - 100.0).abs() < 1e-10);
        assert!((pos.stop_entry_price - 100.0).abs() < 1e-10);
        assert!((pos.max_price - 100.0).abs() < 1e-10);
        assert!((pos.cr - 1.0).abs() < 1e-10);
        assert!((pos.maxcr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_update_with_return() {
        let mut pos = Position::new(1000.0, 100.0);

        // Price goes up 10%
        pos.update_with_return(110.0);

        assert!((pos.cr - 1.1).abs() < 1e-10);
        assert!((pos.maxcr - 1.1).abs() < 1e-10);
        assert!((pos.last_market_value - 1100.0).abs() < 1e-10);
        assert!((pos.max_price - 110.0).abs() < 1e-10);
    }

    #[test]
    fn test_update_with_return_down() {
        let mut pos = Position::new(1000.0, 100.0);

        // Price goes up then down
        pos.update_with_return(110.0);
        pos.update_previous_price(110.0);
        pos.update_with_return(99.0);

        // cr = 1.1 * (99/110) = 0.99
        assert!((pos.cr - 0.99).abs() < 1e-10);
        // maxcr stays at 1.1
        assert!((pos.maxcr - 1.1).abs() < 1e-10);
        // max_price stays at 110
        assert!((pos.max_price - 110.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_long() {
        let pos = Position::new(1000.0, 100.0);
        assert!(pos.is_long());

        let short_pos = Position {
            last_market_value: -1000.0,
            ..Default::default()
        };
        assert!(!short_pos.is_long());
    }

    #[test]
    fn test_reset_stop_tracking() {
        let mut pos = Position::new(1000.0, 100.0);
        pos.update_with_return(120.0);
        pos.update_previous_price(120.0);

        // Reset at new price
        pos.reset_stop_tracking(115.0);

        assert!((pos.stop_entry_price - 115.0).abs() < 1e-10);
        assert!((pos.max_price - 115.0).abs() < 1e-10);
        assert!((pos.cr - 1.0).abs() < 1e-10);
        assert!((pos.maxcr - 1.0).abs() < 1e-10);
        assert!((pos.previous_price - 115.0).abs() < 1e-10);
    }
}
