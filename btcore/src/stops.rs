//! Stop loss, take profit, and trailing stop detection
//!
//! This module provides functions to detect when positions should be closed
//! due to stop conditions. Supports both standard mode and Finlab mode
//! (using cr/maxcr cumulative tracking).

use crate::config::BacktestConfig;
use crate::is_valid_price;
use crate::position::Position;
use std::collections::HashMap;

/// Result of touched exit detection
///
/// Contains information needed to execute the exit at the touched price.
#[derive(Debug, Clone)]
pub struct TouchedExitResult {
    /// Stock ID
    pub stock_id: usize,
    /// Ratio to multiply position by to get exit value
    /// This adjusts position to the touched price level
    pub exit_ratio: f64,
    /// Whether this is a take profit (true) or stop loss (false)
    pub is_take_profit: bool,
}

/// Detect positions that should be closed due to stop conditions (standard mode)
///
/// Uses price-based stop detection:
/// - Stop loss: return_since_entry <= -stop_loss
/// - Take profit: return_since_entry >= take_profit
/// - Trailing stop: (max_price - current_price) / entry_price >= trail_stop
pub fn detect_stops(
    positions: &HashMap<usize, Position>,
    prices: &[f64],
    config: &BacktestConfig,
) -> Vec<usize> {
    positions
        .iter()
        .filter_map(|(&stock_id, pos)| {
            if stock_id >= prices.len() {
                return None;
            }

            let current_price = prices[stock_id];
            // Use stop_entry_price for stop loss calculation (original entry, never reset)
            let stop_entry = pos.stop_entry_price;

            if stop_entry <= 0.0 {
                return None;
            }

            let return_since_entry = (current_price - stop_entry) / stop_entry;

            // Check stop loss
            if config.stop_loss < 1.0 && return_since_entry <= -config.stop_loss {
                return Some(stock_id);
            }

            // Check take profit
            if config.take_profit < f64::INFINITY && return_since_entry >= config.take_profit {
                return Some(stock_id);
            }

            // Check trailing stop
            // Finlab formula: triggers when (max_price - current_price) / entry_price >= trail_stop
            if config.trail_stop < f64::INFINITY {
                let drawdown_from_entry = (pos.max_price - current_price) / stop_entry;
                if drawdown_from_entry >= config.trail_stop {
                    return Some(stock_id);
                }
            }

            None
        })
        .collect()
}

/// Detect positions that should be closed due to stop conditions in Finlab mode (T+1 preparation)
///
/// Uses Finlab's cr/maxcr formula for stop detection:
/// - cr is updated daily via cumulative multiplication: cr *= r (where r = today/yesterday)
///   This matches Finlab's floating point behavior exactly (line 319 of restored_backtest_core.pyx)
/// - maxcr = max cumulative return ratio
///
/// For long positions (entry_pos > 0):
/// - max_r = 1 + take_profit
/// - min_r = max(1 - stop_loss, maxcr - trail_stop)
/// - Trigger: cr >= max_r (take profit) or cr < min_r (stop loss/trail)
///
/// For short positions (entry_pos < 0):
/// - max_r = min(1 + stop_loss, maxcr + trail_stop)
/// - min_r = 1 - take_profit
/// - Trigger: cr >= max_r (stop loss/trail) or cr < min_r (take profit)
pub fn detect_stops_finlab(
    positions: &HashMap<usize, Position>,
    prices: &[f64],
    config: &BacktestConfig,
) -> Vec<usize> {
    positions
        .iter()
        .filter_map(|(&stock_id, pos)| {
            if stock_id >= prices.len() {
                return None;
            }

            let current_price = prices[stock_id];
            let stop_entry = pos.stop_entry_price;

            if stop_entry <= 0.0 || !is_valid_price(current_price) {
                return None;
            }

            // Use cumulative cr from Position (Finlab: cr *= r)
            // Note: cr is updated by update_max_prices() before this function is called
            let cr = pos.cr;

            // Finlab uses cr_at_close = cr * close / price for stop detection (line 387)
            // Even when close == price (both adj_close), the multiply-divide operation
            // affects floating point precision, which matters at exact threshold boundaries.
            // Example: cr = 0.9499999999999998 < 0.95, but cr * p / p = 0.95 exactly!
            let cr_at_close = cr * current_price / current_price;

            // Use cumulative maxcr from Position (Finlab: maxcr[sid] = max(maxcr[sid], cr[sid]))
            // Note: maxcr is updated by update_max_prices() before this function is called
            let maxcr = pos.maxcr;

            // Determine if position is long or short
            // Finlab: entry_pos = pos[sid] / cr[sid]
            // Since cr is always positive, the sign depends on pos[sid] (last_market_value)
            let is_long = pos.last_market_value >= 0.0;

            // Finlab stop conditions (lines 326-393 of restored_backtest_core.pyx):
            if is_long {
                // Long positions:
                //   max_r = 1 + take_profit
                //   min_r = max(1 - stop_loss, maxcr - trail_stop)
                // Trigger: cr_at_close >= max_r (take profit) or cr_at_close < min_r (stop loss/trail)

                // Check take profit: cr_at_close >= 1 + take_profit
                if config.take_profit < f64::INFINITY && cr_at_close >= 1.0 + config.take_profit {
                    return Some(stock_id);
                }

                // Calculate min_r using Finlab formula
                let stop_threshold = 1.0 - config.stop_loss;
                let trail_threshold = if config.trail_stop < f64::INFINITY {
                    maxcr - config.trail_stop
                } else {
                    f64::NEG_INFINITY
                };
                let min_r = stop_threshold.max(trail_threshold);

                // Check stop loss / trail stop: cr_at_close < min_r (Finlab uses < not <=)
                if cr_at_close < min_r {
                    return Some(stock_id);
                }
            } else {
                // Short positions:
                //   max_r = min(1 + stop_loss, maxcr + trail_stop)
                //   min_r = 1 - take_profit
                // Trigger: cr_at_close >= max_r (stop loss/trail) or cr_at_close < min_r (take profit)

                // Calculate max_r for short positions
                let stop_threshold = 1.0 + config.stop_loss;
                let trail_threshold = if config.trail_stop < f64::INFINITY {
                    maxcr + config.trail_stop
                } else {
                    f64::INFINITY
                };
                let max_r = stop_threshold.min(trail_threshold);

                // Check stop loss / trail stop: cr_at_close >= max_r
                if cr_at_close >= max_r {
                    return Some(stock_id);
                }

                // Check take profit: cr_at_close < 1 - take_profit
                let min_r = 1.0 - config.take_profit;
                if config.take_profit < f64::INFINITY && cr_at_close < min_r {
                    return Some(stock_id);
                }
            }

            None
        })
        .collect()
}

/// Detect intraday stop exits using OHLC prices (touched_exit mode)
///
/// Finlab's touched_exit logic (lines 339-393 of backtest_core.pyx):
/// 1. Calculate open_r, high_r, low_r (cumulative return ratios at each price)
/// 2. Check if any price touches the stop/take profit thresholds
/// 3. Priority: open > high > low
/// 4. Adjust position to touched price and exit immediately
///
/// IMPORTANT: This function should be called AFTER update_max_prices
/// (which updates cr *= r). We need to pass close_prices to calculate r
/// and then derive cr_old = cr / r.
///
/// This differs from `detect_stops_finlab` in that:
/// - Uses OHLC prices instead of just close price
/// - Exits happen on the same day (T+0), not T+1
/// - Position value is adjusted to the touched price
pub fn detect_touched_exit(
    positions: &HashMap<usize, Position>,
    open_prices: &[f64],
    high_prices: &[f64],
    low_prices: &[f64],
    close_prices: &[f64],
    _prev_prices: &[f64], // Kept for API compatibility but we use pos.previous_price
    config: &BacktestConfig,
) -> Vec<TouchedExitResult> {
    positions
        .iter()
        .filter_map(|(&stock_id, pos)| {
            if stock_id >= open_prices.len()
                || stock_id >= high_prices.len()
                || stock_id >= low_prices.len()
                || stock_id >= close_prices.len()
            {
                return None;
            }

            let open_price = open_prices[stock_id];
            let high_price = high_prices[stock_id];
            let low_price = low_prices[stock_id];
            let close_price = close_prices[stock_id];

            // Use pos.previous_price which tracks the last valid price for this position.
            // This handles NaN days correctly (previous_price only updates on valid days).
            let prev_price = pos.previous_price;

            // Skip if any OHLC price is invalid or prev_price is not set
            if open_price.is_nan()
                || high_price.is_nan()
                || low_price.is_nan()
                || close_price.is_nan()
                || close_price <= 0.0
                || prev_price <= 0.0
            {
                return None;
            }

            // Get cr from position (already updated by update_max_prices: cr *= r)
            let cr_new = pos.cr;
            let maxcr = pos.maxcr;

            // Skip if cr is invalid (shouldn't happen but safety check)
            if cr_new.is_nan() || cr_new <= 0.0 {
                return None;
            }

            // Calculate r = close / prev_price (same as Finlab line 305)
            let r = close_price / prev_price;

            // Finlab calculates these AFTER cr *= r (lines 342-344):
            // high_r = cr[sid] / r * (high / prev)
            // low_r = cr[sid] / r * (low / prev)
            // open_r = cr[sid] / r * (open / prev)
            //
            // We compute in the EXACT same order as Finlab (single expression)
            // to ensure identical floating point behavior.
            let open_r = cr_new / r * (open_price / prev_price);
            let high_r = cr_new / r * (high_price / prev_price);
            let low_r = cr_new / r * (low_price / prev_price);

            // Determine position direction
            // Finlab line 326: entry_pos = pos[sid] / cr[sid], if entry_pos > 0: long
            // Since cr is always positive, we just check pos sign
            let is_long = pos.last_market_value > 0.0;

            // Calculate thresholds (same as detect_stops_finlab)
            // Note: Use maxcr which was updated in update_max_prices
            let (max_r, min_r) = if is_long {
                // Long: max_r = 1 + take_profit, min_r = max(1 - stop_loss, maxcr - trail_stop)
                let max_r = 1.0 + config.take_profit;
                let stop_threshold = 1.0 - config.stop_loss;
                let trail_threshold = if config.trail_stop < f64::INFINITY {
                    maxcr - config.trail_stop
                } else {
                    f64::NEG_INFINITY
                };
                let min_r = stop_threshold.max(trail_threshold);
                (max_r, min_r)
            } else {
                // Short: max_r = min(1 + stop_loss, maxcr + trail_stop), min_r = 1 - take_profit
                let stop_threshold = 1.0 + config.stop_loss;
                let trail_threshold = if config.trail_stop < f64::INFINITY {
                    maxcr + config.trail_stop
                } else {
                    f64::INFINITY
                };
                let max_r = stop_threshold.min(trail_threshold);
                let min_r = 1.0 - config.take_profit;
                (max_r, min_r)
            };

            // Check touch conditions (Finlab lines 348-350)
            let touch_open = open_r >= max_r || open_r <= min_r;
            let touch_high = high_r >= max_r;
            let touch_low = low_r <= min_r;

            // Determine exit ratio
            // Finlab adjusts pos to touched price:
            // - touch_open: pos *= open_r / r  =>  pos_final = pos_before_r_update * open_r
            // - touch_high: pos = entry_pos * max_r  =>  entry_pos = pos / cr, so pos_final = pos * max_r / cr
            // - touch_low: pos = entry_pos * min_r  =>  pos_final = pos * min_r / cr
            //
            // After pos *= r, pos = pos_old * r = entry_pos * cr_old * r = entry_pos * cr_new
            // So exit_ratio relative to current pos (after r update):
            // - touch_open: exit_ratio = open_r / r
            // - touch_high: exit_ratio = max_r / cr_new
            // - touch_low: exit_ratio = min_r / cr_new
            //
            // Priority: open > high > low (Finlab lines 354-359)
            if touch_open {
                // Finlab: pos[sid] *= open_r / r
                let exit_ratio = open_r / r;
                let is_take_profit = if is_long {
                    open_r >= max_r
                } else {
                    open_r <= min_r
                };
                Some(TouchedExitResult {
                    stock_id,
                    exit_ratio,
                    is_take_profit,
                })
            } else if touch_high {
                // Finlab: pos[sid] = entry_pos * max_r = pos / cr * max_r
                let exit_ratio = max_r / cr_new;
                let is_take_profit = is_long; // high touch is TP for long, SL for short
                Some(TouchedExitResult {
                    stock_id,
                    exit_ratio,
                    is_take_profit,
                })
            } else if touch_low {
                // Finlab: pos[sid] = entry_pos * min_r = pos / cr * min_r
                let exit_ratio = min_r / cr_new;
                let is_take_profit = !is_long; // low touch is SL for long, TP for short
                Some(TouchedExitResult {
                    stock_id,
                    exit_ratio,
                    is_take_profit,
                })
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(stop_loss: f64, take_profit: f64, trail_stop: f64) -> BacktestConfig {
        BacktestConfig {
            fee_ratio: 0.001425,
            tax_ratio: 0.003,
            stop_loss,
            take_profit,
            trail_stop,
            position_limit: 1.0,
            retain_cost_when_rebalance: true,
            stop_trading_next_period: false,
            finlab_mode: false,
            touched_exit: false,
        }
    }

    #[test]
    fn test_detect_stops_stop_loss() {
        let mut positions = HashMap::new();
        let mut pos = Position::new(1000.0, 100.0);
        pos.stop_entry_price = 100.0;
        positions.insert(0, pos);

        let prices = vec![89.0]; // -11% return
        let config = make_config(0.10, f64::INFINITY, f64::INFINITY);

        let stops = detect_stops(&positions, &prices, &config);
        assert_eq!(stops, vec![0]);
    }

    #[test]
    fn test_detect_stops_take_profit() {
        let mut positions = HashMap::new();
        let mut pos = Position::new(1000.0, 100.0);
        pos.stop_entry_price = 100.0;
        positions.insert(0, pos);

        let prices = vec![121.0]; // +21% return
        let config = make_config(1.0, 0.20, f64::INFINITY);

        let stops = detect_stops(&positions, &prices, &config);
        assert_eq!(stops, vec![0]);
    }

    #[test]
    fn test_detect_stops_trailing() {
        let mut positions = HashMap::new();
        let mut pos = Position::new(1000.0, 100.0);
        pos.stop_entry_price = 100.0;
        pos.max_price = 120.0; // Max was 120
        positions.insert(0, pos);

        let prices = vec![105.0]; // Drawdown from max: (120-105)/100 = 15%
        let config = make_config(1.0, f64::INFINITY, 0.10);

        let stops = detect_stops(&positions, &prices, &config);
        assert_eq!(stops, vec![0]);
    }

    #[test]
    fn test_detect_stops_no_trigger() {
        let mut positions = HashMap::new();
        let mut pos = Position::new(1000.0, 100.0);
        pos.stop_entry_price = 100.0;
        positions.insert(0, pos);

        let prices = vec![105.0]; // +5%, no stop triggered
        let config = make_config(0.10, 0.20, f64::INFINITY);

        let stops = detect_stops(&positions, &prices, &config);
        assert!(stops.is_empty());
    }

    #[test]
    fn test_detect_stops_finlab_take_profit() {
        let mut positions = HashMap::new();
        let mut pos = Position::new(1000.0, 100.0);
        pos.stop_entry_price = 100.0;
        pos.cr = 1.21; // cr >= 1 + take_profit (0.20)
        pos.maxcr = 1.21;
        pos.last_market_value = 1210.0;
        positions.insert(0, pos);

        let prices = vec![121.0];
        let config = make_config(1.0, 0.20, f64::INFINITY);

        let stops = detect_stops_finlab(&positions, &prices, &config);
        assert_eq!(stops, vec![0]);
    }

    #[test]
    fn test_detect_stops_finlab_stop_loss() {
        let mut positions = HashMap::new();
        let mut pos = Position::new(1000.0, 100.0);
        pos.stop_entry_price = 100.0;
        pos.cr = 0.89; // cr < 1 - stop_loss (0.10)
        pos.maxcr = 1.0;
        pos.last_market_value = 890.0;
        positions.insert(0, pos);

        let prices = vec![89.0];
        let config = make_config(0.10, f64::INFINITY, f64::INFINITY);

        let stops = detect_stops_finlab(&positions, &prices, &config);
        assert_eq!(stops, vec![0]);
    }

    #[test]
    fn test_detect_stops_finlab_trail_stop() {
        let mut positions = HashMap::new();
        let mut pos = Position::new(1000.0, 100.0);
        pos.stop_entry_price = 100.0;
        pos.cr = 1.05; // Current cr
        pos.maxcr = 1.20; // Max was 1.20, trail_stop = 0.10, so min_r = max(0.9, 1.20 - 0.10) = 1.10
        pos.last_market_value = 1050.0;
        positions.insert(0, pos);

        let prices = vec![105.0];
        let config = make_config(1.0, f64::INFINITY, 0.10);

        let stops = detect_stops_finlab(&positions, &prices, &config);
        // cr (1.05) < min_r (1.10), should trigger
        assert_eq!(stops, vec![0]);
    }

    #[test]
    fn test_touched_exit_result_fields() {
        let result = TouchedExitResult {
            stock_id: 5,
            exit_ratio: 0.95,
            is_take_profit: false,
        };
        assert_eq!(result.stock_id, 5);
        assert!((result.exit_ratio - 0.95).abs() < 1e-10);
        assert!(!result.is_take_profit);
    }

    #[test]
    fn test_detect_touched_exit_open_touch() {
        let mut positions = HashMap::new();
        let mut pos = Position::new(1000.0, 100.0);
        pos.stop_entry_price = 100.0;
        pos.cr = 0.85; // Will be updated with r
        pos.maxcr = 1.0;
        pos.last_market_value = 850.0;
        pos.previous_price = 100.0;
        positions.insert(0, pos);

        // Scenario: Open at 88 (touches -10% stop loss), close at 95
        let open_prices = vec![88.0];
        let high_prices = vec![96.0];
        let low_prices = vec![87.0];
        let close_prices = vec![95.0];
        let prev_prices = vec![100.0];

        let config = make_config(0.10, f64::INFINITY, f64::INFINITY);

        let results = detect_touched_exit(
            &positions,
            &open_prices,
            &high_prices,
            &low_prices,
            &close_prices,
            &prev_prices,
            &config,
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].stock_id, 0);
        assert!(!results[0].is_take_profit); // Stop loss, not take profit
    }

    #[test]
    fn test_detect_touched_exit_no_touch() {
        let mut positions = HashMap::new();
        let mut pos = Position::new(1000.0, 100.0);
        pos.stop_entry_price = 100.0;
        pos.cr = 1.05;
        pos.maxcr = 1.05;
        pos.last_market_value = 1050.0;
        pos.previous_price = 100.0;
        positions.insert(0, pos);

        // All prices within range
        let open_prices = vec![102.0];
        let high_prices = vec![108.0];
        let low_prices = vec![98.0];
        let close_prices = vec![105.0];
        let prev_prices = vec![100.0];

        let config = make_config(0.10, 0.20, f64::INFINITY);

        let results = detect_touched_exit(
            &positions,
            &open_prices,
            &high_prices,
            &low_prices,
            &close_prices,
            &prev_prices,
            &config,
        );

        assert!(results.is_empty());
    }
}
