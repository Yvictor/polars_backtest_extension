//! MAE/MFE (Maximum Adverse/Favorable Excursion) calculation
//!
//! This module provides trade analysis metrics that track the best and worst
//! points during each trade's lifetime.
//!
//! # Metrics
//!
//! - **MAE**: Maximum Adverse Excursion - the maximum loss experienced during the trade
//! - **GMFE**: Global Maximum Favorable Excursion - the maximum profit during the trade
//! - **BMFE**: Before-MAE MFE - the MFE at the point when MAE occurred
//! - **MDD**: Maximum Drawdown during the trade
//! - **pdays**: Number of profitable days (days where cumulative return > 1)

use crate::is_valid_price;

/// MAE/MFE metrics at a specific point in time
#[derive(Debug, Clone, Copy, Default)]
pub struct MaeMfeMetrics {
    /// Maximum Adverse Excursion (negative value, e.g., -0.05 = -5%)
    pub mae: f64,
    /// Global Maximum Favorable Excursion (positive value, e.g., 0.10 = 10%)
    pub gmfe: f64,
    /// Before-MAE MFE - MFE at the time when MAE occurred
    pub bmfe: f64,
    /// Maximum Drawdown during the trade (negative value)
    pub mdd: f64,
    /// Number of profitable days
    pub pdays: u32,
    /// Cumulative return at this point (e.g., 0.05 = 5% profit)
    pub ret: f64,
}

/// Configuration for MAE/MFE calculation
#[derive(Debug, Clone, Copy)]
pub struct MaeMfeConfig {
    /// Maximum days to track (0 = only calculate at exit)
    pub window: usize,
    /// Step size for window sampling (default: 1)
    pub window_step: usize,
}

impl Default for MaeMfeConfig {
    fn default() -> Self {
        Self {
            window: 0,
            window_step: 1,
        }
    }
}

/// Calculate MAE/MFE metrics for a single trade
///
/// # Arguments
///
/// * `close_prices` - Close prices for the stock (full price series)
/// * `trade_prices` - Trade execution prices for the stock (full price series)
/// * `entry_index` - Index of trade entry
/// * `exit_index` - Index of trade exit
/// * `is_long` - Whether this is a long position
/// * `has_entry_transaction` - Whether entry fee should be applied
/// * `has_exit_transaction` - Whether exit fee/tax should be applied
/// * `fee_ratio` - Transaction fee ratio
/// * `tax_ratio` - Transaction tax ratio
/// * `config` - MAE/MFE calculation configuration
///
/// # Returns
///
/// Vector of MaeMfeMetrics at each window step, plus final metrics at exit
pub fn calculate_mae_mfe(
    close_prices: &[f64],
    trade_prices: &[f64],
    entry_index: usize,
    exit_index: usize,
    is_long: bool,
    has_entry_transaction: bool,
    has_exit_transaction: bool,
    fee_ratio: f64,
    tax_ratio: f64,
    config: &MaeMfeConfig,
) -> Vec<MaeMfeMetrics> {
    // Validate indices
    if entry_index >= close_prices.len() || exit_index >= close_prices.len() {
        return vec![MaeMfeMetrics::default()];
    }

    // Calculate the maximum index we need to track
    let mut exit_max = exit_index;
    if config.window > 0 && config.window + entry_index > exit_max {
        exit_max = config.window + entry_index;
    }
    exit_max = exit_max.min(close_prices.len() - 1);

    let capacity = exit_max - entry_index + 1;

    // Initialize tracking vectors
    let mut cummax = Vec::with_capacity(capacity);
    let mut cummin = Vec::with_capacity(capacity);
    let mut cummin_i = Vec::with_capacity(capacity); // Index where cummin occurred
    let mut mdd = Vec::with_capacity(capacity);
    let mut profit_period = Vec::with_capacity(capacity);
    let mut returns = Vec::with_capacity(capacity);

    // Get entry prices
    let entry_price = trade_prices[entry_index];
    let entry_close = close_prices[entry_index];

    if !is_valid_price(entry_price) || !is_valid_price(entry_close) {
        return vec![MaeMfeMetrics::default()];
    }

    // Calculate initial price ratio
    let mut price_ratio = if is_long {
        entry_close / entry_price
    } else {
        2.0 - entry_close / entry_price
    };

    // Apply entry fee
    if has_entry_transaction {
        price_ratio *= 1.0 - fee_ratio;
    }

    // Initialize first values
    returns.push(price_ratio);
    cummax.push(price_ratio.max(1.0));
    cummin.push(price_ratio.min(1.0));
    cummin_i.push(0);
    mdd.push((price_ratio - 1.0).min(0.0));
    profit_period.push(if price_ratio > 1.0 { 1 } else { 0 });

    // Track through the trade
    let mut pv = entry_close; // Previous valid close price

    for (i, ith) in (entry_index + 1..=exit_max).enumerate() {
        let p = close_prices[ith];

        if is_valid_price(p) {
            let v = p / pv;
            pv = p;

            if is_long {
                price_ratio *= v;
            } else {
                price_ratio = 2.0 - (2.0 - price_ratio) * v;
            }
        }

        let prev_idx = i; // Index in our vectors (0-based from entry)
        let cmax = cummax[prev_idx];
        let cmin = cummin[prev_idx];

        // Update cummax
        if price_ratio > cmax {
            cummax.push(price_ratio);
        } else {
            cummax.push(cmax);
        }

        // Update cummin and track when it occurred
        if price_ratio < cmin {
            cummin.push(price_ratio);
            cummin_i.push(i + 1);
        } else {
            cummin.push(cmin);
            cummin_i.push(cummin_i[prev_idx]);
        }

        // Update MDD
        let new_mdd = price_ratio / cummax[i + 1] - 1.0;
        if new_mdd < mdd[prev_idx] {
            mdd.push(new_mdd);
        } else {
            mdd.push(mdd[prev_idx]);
        }

        // Update profit period
        profit_period.push(profit_period[prev_idx] + if price_ratio > 1.0 { 1 } else { 0 });

        returns.push(price_ratio);
    }

    // Apply exit transaction costs to final return
    if has_exit_transaction && entry_index != exit_max {
        let last_idx = returns.len() - 1;

        // Adjust for trade price vs close price at exit
        let exit_trade_price = trade_prices[exit_max];
        let exit_close = close_prices[exit_max];

        if is_valid_price(exit_trade_price) && is_valid_price(exit_close) {
            if is_long {
                returns[last_idx] *= exit_trade_price / exit_close;
            } else {
                returns[last_idx] = 2.0 - (2.0 - returns[last_idx]) * exit_trade_price / exit_close;
            }
        }

        // Apply fee and tax
        returns[last_idx] *= 1.0 - fee_ratio - tax_ratio;
    }

    // Build output metrics
    let mut result = Vec::new();

    // Add metrics at each window step
    if config.window > 0 {
        let window = config.window.min(cummax.len());
        for w in (0..window).step_by(config.window_step) {
            if w < cummax.len() {
                let mae_i = cummin_i[w];
                result.push(MaeMfeMetrics {
                    mae: cummin[w] - 1.0,
                    gmfe: cummax[w] - 1.0,
                    bmfe: cummax[mae_i] - 1.0,
                    mdd: mdd[w],
                    pdays: profit_period[w],
                    ret: returns[w] - 1.0,
                });
            }
        }
    }

    // Add final metrics at exit
    let exit_w = (exit_index - entry_index).min(cummax.len() - 1);
    let mae_i = cummin_i[exit_w];
    result.push(MaeMfeMetrics {
        mae: cummin[exit_w] - 1.0,
        gmfe: cummax[exit_w] - 1.0,
        bmfe: cummax[mae_i] - 1.0,
        mdd: mdd[exit_w],
        pdays: profit_period[exit_w],
        ret: returns[exit_w] - 1.0,
    });

    result
}

/// Calculate MAE/MFE for a trade using only exit metrics (no windowing)
///
/// Simplified version when only final metrics are needed.
pub fn calculate_mae_mfe_at_exit(
    close_prices: &[f64],
    trade_prices: &[f64],
    entry_index: usize,
    exit_index: usize,
    is_long: bool,
    has_entry_transaction: bool,
    has_exit_transaction: bool,
    fee_ratio: f64,
    tax_ratio: f64,
) -> MaeMfeMetrics {
    let config = MaeMfeConfig::default();
    let metrics = calculate_mae_mfe(
        close_prices,
        trade_prices,
        entry_index,
        exit_index,
        is_long,
        has_entry_transaction,
        has_exit_transaction,
        fee_ratio,
        tax_ratio,
        &config,
    );
    metrics.into_iter().last().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_uptrend() {
        // Simple uptrend: 100 -> 105 -> 110 -> 108 -> 115
        let close = vec![100.0, 105.0, 110.0, 108.0, 115.0];
        let trade = close.clone(); // Trade at close

        let metrics = calculate_mae_mfe_at_exit(
            &close,
            &trade,
            0,     // entry at index 0
            4,     // exit at index 4
            true,  // is_long
            false, // no entry transaction
            false, // no exit transaction
            0.0,
            0.0,
        );

        // Final return: 115/100 - 1 = 0.15
        assert!((metrics.ret - 0.15).abs() < 1e-10);

        // GMFE: max was at 115 -> 0.15
        assert!((metrics.gmfe - 0.15).abs() < 1e-10);

        // MAE: min was at entry (1.0) -> 0.0
        // Actually, we need to recalculate...
        // At entry: close/trade = 100/100 = 1.0
        // After day 1: 105/100 = 1.05
        // After day 2: 110/100 = 1.10
        // After day 3: 108/100 = 1.08
        // After day 4: 115/100 = 1.15
        // cummin = min(1.0, 1.05, 1.10, 1.08, 1.15) = 1.0
        // mae = 1.0 - 1 = 0.0
        assert!((metrics.mae - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_long_with_drawdown() {
        // Price goes up then down: 100 -> 120 -> 110 -> 90 -> 105
        let close = vec![100.0, 120.0, 110.0, 90.0, 105.0];
        let trade = close.clone();

        let metrics = calculate_mae_mfe_at_exit(
            &close,
            &trade,
            0,
            4,
            true,
            false,
            false,
            0.0,
            0.0,
        );

        // Final return: 105/100 - 1 = 0.05
        assert!((metrics.ret - 0.05).abs() < 1e-10);

        // GMFE: max was at 120 -> 0.20
        assert!((metrics.gmfe - 0.20).abs() < 1e-10);

        // MAE: min was at 90 -> -0.10
        assert!((metrics.mae - (-0.10)).abs() < 1e-10);

        // MDD: 90/120 - 1 = -0.25
        assert!((metrics.mdd - (-0.25)).abs() < 1e-10);
    }

    #[test]
    fn test_short_position() {
        // Short: profit when price goes down
        // Price: 100 -> 95 -> 90 -> 92 -> 85
        let close = vec![100.0, 95.0, 90.0, 92.0, 85.0];
        let trade = close.clone();

        let metrics = calculate_mae_mfe_at_exit(
            &close,
            &trade,
            0,
            4,
            false, // is_short
            false,
            false,
            0.0,
            0.0,
        );

        // For short: return = 2 - price_ratio
        // price_ratio at exit = 85/100 = 0.85
        // short return = 2 - 0.85 = 1.15 -> return = 0.15
        assert!((metrics.ret - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_with_fees() {
        let close = vec![100.0, 110.0];
        let trade = close.clone();

        let fee_ratio = 0.001425;
        let tax_ratio = 0.003;

        let metrics = calculate_mae_mfe_at_exit(
            &close,
            &trade,
            0,
            1,
            true,
            true, // has entry transaction
            true, // has exit transaction
            fee_ratio,
            tax_ratio,
        );

        // Entry: 100/100 * (1 - 0.001425) = 0.998575
        // Exit: 0.998575 * 1.1 * (1 - 0.001425 - 0.003)
        //     = 0.998575 * 1.1 * 0.995575
        //     = 1.0937...
        // return = 1.0937 - 1 = 0.0937...
        let expected = (1.0 - fee_ratio) * 1.1 * (1.0 - fee_ratio - tax_ratio) - 1.0;
        assert!((metrics.ret - expected).abs() < 1e-6);
    }

    #[test]
    fn test_window_metrics() {
        // 10-day trade
        let close: Vec<f64> = (0..10).map(|i| 100.0 + i as f64 * 2.0).collect();
        let trade = close.clone();

        let config = MaeMfeConfig {
            window: 10,
            window_step: 2,
        };

        let metrics = calculate_mae_mfe(
            &close,
            &trade,
            0,
            9,
            true,
            false,
            false,
            0.0,
            0.0,
            &config,
        );

        // Should have metrics at [0, 2, 4, 6, 8] + exit
        // window_step=2, window=10 -> 5 window points + 1 exit = 6 total
        assert_eq!(metrics.len(), 6);

        // First metric at w=0
        assert!((metrics[0].ret - 0.0).abs() < 1e-10);

        // Last metric at exit (index 9)
        // 118/100 - 1 = 0.18
        assert!((metrics[5].ret - 0.18).abs() < 1e-10);
    }

    #[test]
    fn test_pdays_counting() {
        // Alternating profit/loss days
        let close = vec![100.0, 101.0, 99.0, 102.0, 98.0, 103.0];
        let trade = close.clone();

        let metrics = calculate_mae_mfe_at_exit(&close, &trade, 0, 5, true, false, false, 0.0, 0.0);

        // At each point:
        // 0: 100/100 = 1.0 (not > 1, pdays=0)
        // 1: 101/100 = 1.01 (> 1, pdays=1)
        // 2: 99/100 = 0.99 (not > 1, pdays=1)
        // 3: 102/100 = 1.02 (> 1, pdays=2)
        // 4: 98/100 = 0.98 (not > 1, pdays=2)
        // 5: 103/100 = 1.03 (> 1, pdays=3)
        assert_eq!(metrics.pdays, 3);
    }

    #[test]
    fn test_bmfe_calculation() {
        // Price goes up, then crashes, then recovers partially
        // We want to track BMFE: the MFE at the point when MAE occurred
        let close = vec![100.0, 110.0, 120.0, 80.0, 90.0];
        let trade = close.clone();

        let metrics = calculate_mae_mfe_at_exit(&close, &trade, 0, 4, true, false, false, 0.0, 0.0);

        // cummax = [1.0, 1.1, 1.2, 1.2, 1.2]
        // cummin = [1.0, 1.0, 1.0, 0.8, 0.8]
        // cummin_i = [0, 0, 0, 3, 3]
        // At exit, cummin_i = 3
        // BMFE = cummax[3] - 1 = 1.2 - 1 = 0.2
        assert!((metrics.bmfe - 0.2).abs() < 1e-10);

        // MAE = 0.8 - 1 = -0.2
        assert!((metrics.mae - (-0.2)).abs() < 1e-10);
    }
}
