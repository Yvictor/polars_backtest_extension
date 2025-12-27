//! Backtest configuration
//!
//! This module contains the configuration struct for backtest simulation.

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Transaction fee ratio (default: 0.001425 for Taiwan stocks)
    pub fee_ratio: f64,
    /// Transaction tax ratio (default: 0.003 for Taiwan stocks)
    pub tax_ratio: f64,
    /// Stop loss threshold (1.0 = disabled)
    pub stop_loss: f64,
    /// Take profit threshold (f64::INFINITY = disabled)
    pub take_profit: f64,
    /// Trailing stop threshold (f64::INFINITY = disabled)
    pub trail_stop: f64,
    /// Maximum weight per stock (default: 1.0)
    pub position_limit: f64,
    /// Retain cost when rebalancing (default: false)
    pub retain_cost_when_rebalance: bool,
    /// Stop trading next period after stop loss/take profit (default: true)
    pub stop_trading_next_period: bool,
    /// Use Finlab-compatible calculation mode (default: false)
    ///
    /// When enabled:
    /// - Positions track cost_basis + entry_price (not current_value)
    /// - Balance = cash + Σ(cost_basis * close_price / entry_price)
    /// - Rebalance uses Σ(cost_basis) as base (not market value)
    ///
    /// This mode exactly replicates Finlab's backtest_core.pyx calculation.
    pub finlab_mode: bool,
    /// Use touched exit mode (default: false)
    ///
    /// When enabled, uses OHLC prices for intraday stop detection.
    /// Exits happen on the same day when high/low prices touch thresholds,
    /// rather than T+1 execution.
    ///
    /// Finlab behavior (lines 339-393 of backtest_core.pyx):
    /// - Check open price first (touch_open)
    /// - Then check high/low for take_profit/stop_loss
    /// - Exit immediately at touched price (not close price)
    pub touched_exit: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            fee_ratio: 0.001425,
            tax_ratio: 0.003,
            stop_loss: 1.0,              // disabled
            take_profit: f64::INFINITY,  // disabled
            trail_stop: f64::INFINITY,   // disabled
            position_limit: 1.0,
            retain_cost_when_rebalance: false,
            stop_trading_next_period: true,
            finlab_mode: false,          // Use standard calculation by default
            touched_exit: false,         // Use close-based stop detection by default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BacktestConfig::default();
        assert!((config.fee_ratio - 0.001425).abs() < 1e-10);
        assert!((config.tax_ratio - 0.003).abs() < 1e-10);
        assert!((config.stop_loss - 1.0).abs() < 1e-10);
        assert!(config.take_profit.is_infinite());
        assert!(config.trail_stop.is_infinite());
        assert!((config.position_limit - 1.0).abs() < 1e-10);
        assert!(!config.retain_cost_when_rebalance);
        assert!(config.stop_trading_next_period);
        assert!(!config.finlab_mode);
        assert!(!config.touched_exit);
    }

    #[test]
    fn test_config_clone() {
        let config = BacktestConfig {
            fee_ratio: 0.002,
            stop_loss: 0.1,
            ..Default::default()
        };
        let cloned = config.clone();
        assert!((cloned.fee_ratio - 0.002).abs() < 1e-10);
        assert!((cloned.stop_loss - 0.1).abs() < 1e-10);
    }
}
