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
    /// Long-format execution options, all off by default
    pub exec: ExecOptions,
}

/// Largest `rebalance_every`, about a century of weeks
pub const MAX_REBALANCE_EVERY: i32 = 5200;

/// Long-format execution options
///
/// Every field defaults to today's behaviour. The wide engine ignores this struct.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecOptions {
    /// Rebalance every k calendar weeks. Weekly resample only.
    pub rebalance_every: i32,
    /// Which of the k weeks rebalances, in `0..rebalance_every`
    pub rebalance_phase: i32,
    /// Any date (days since epoch) in the week counted as week 0
    pub rebalance_anchor: Option<i32>,
}

impl Default for ExecOptions {
    fn default() -> Self {
        Self {
            rebalance_every: 1,
            rebalance_phase: 0,
            rebalance_anchor: None,
        }
    }
}

impl ExecOptions {
    /// Reject schedule settings the loop cannot honour
    pub fn validate_schedule(&self, weekly_resample: bool) -> Result<(), String> {
        let every = self.rebalance_every;
        let phase = self.rebalance_phase;
        if !(1..=MAX_REBALANCE_EVERY).contains(&every) || phase < 0 || phase >= every {
            return Err(format!(
                "rebalance_every must be positive, at most {MAX_REBALANCE_EVERY}, and \
rebalance_phase in [0, every): every={every}, phase={phase}"
            ));
        }
        let scheduled = every != 1 || phase != 0;
        if (scheduled || self.rebalance_anchor.is_some()) && !weekly_resample {
            return Err("rebalance_every requires a weekly resample".into());
        }
        if scheduled && self.rebalance_anchor.is_none() {
            return Err("rebalance_every requires rebalance_anchor".into());
        }
        Ok(())
    }

    /// Whether `date` falls in a week selected by the schedule
    ///
    /// Weeks are counted on the calendar from Monday of the anchor's week, so
    /// a week without trading days still counts and slicing the input does not
    /// shift the phase.
    pub fn week_selected(&self, date: i32) -> bool {
        if self.rebalance_every == 1 {
            return true;
        }
        let anchor_monday = monday_of(self.rebalance_anchor.unwrap_or(0));
        let week = (date - anchor_monday).div_euclid(7);
        week.rem_euclid(self.rebalance_every) == self.rebalance_phase
    }

    /// First date on or after `date` that lies `days_after_monday` past the
    /// Monday of a selected week
    pub fn next_selected_date(&self, date: i32, days_after_monday: i32) -> i32 {
        let mut monday = monday_of(date - days_after_monday);
        if monday + days_after_monday < date {
            monday += 7;
        }
        (0..self.rebalance_every)
            .map(|weeks| monday + 7 * weeks)
            .find(|&candidate| self.week_selected(candidate))
            .unwrap_or(monday)
            + days_after_monday
    }
}

/// Monday of the week holding `date` (days since epoch)
fn monday_of(date: i32) -> i32 {
    // 1970-01-01 (day 0) was a Thursday
    date - (date + 3).rem_euclid(7)
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
            exec: ExecOptions::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule(every: i32, phase: i32, anchor: Option<i32>) -> ExecOptions {
        ExecOptions {
            rebalance_every: every,
            rebalance_phase: phase,
            rebalance_anchor: anchor,
        }
    }

    #[test]
    fn test_default_schedule_selects_every_week() {
        let exec = ExecOptions::default();
        assert!(exec.validate_schedule(false).is_ok());
        assert!((-20..20).all(|day| exec.week_selected(day)));
    }

    #[test]
    fn test_week_selected_counts_calendar_weeks_from_the_anchor_monday() {
        // Day 4 (1970-01-05) is a Monday; a Saturday anchor names the same week
        let exec = schedule(2, 0, Some(9));
        assert!((4..11).all(|day| exec.week_selected(day)));
        assert!((11..18).all(|day| !exec.week_selected(day)));
        assert!((18..25).all(|day| exec.week_selected(day)));
    }

    #[test]
    fn test_week_selected_handles_dates_before_the_anchor() {
        let exec = schedule(3, 1, Some(4));
        assert!(exec.week_selected(4 - 14));
        assert!(!exec.week_selected(4 - 7));
    }

    #[test]
    fn test_next_selected_date_stays_on_selected_weeks() {
        // Weeks start on days 4, 11, 18, 25; every other week is selected
        let exec = schedule(2, 0, Some(4));
        assert_eq!(exec.next_selected_date(4, 4), 8);
        assert_eq!(exec.next_selected_date(8, 4), 8);
        assert_eq!(exec.next_selected_date(9, 4), 22);
        assert_eq!(exec.next_selected_date(15, 4), 22);
    }

    #[test]
    fn test_validate_schedule() {
        assert!(schedule(2, 1, Some(0)).validate_schedule(true).is_ok());
        assert!(schedule(0, 0, Some(0)).validate_schedule(true).is_err());
        assert!(schedule(MAX_REBALANCE_EVERY, 0, Some(0))
            .validate_schedule(true)
            .is_ok());
        assert!(schedule(MAX_REBALANCE_EVERY + 1, 0, Some(0))
            .validate_schedule(true)
            .is_err());
        assert!(schedule(2, 2, Some(0)).validate_schedule(true).is_err());
        assert!(schedule(2, -1, Some(0)).validate_schedule(true).is_err());
        assert!(schedule(2, 0, None).validate_schedule(true).is_err());
        assert!(schedule(2, 0, Some(0)).validate_schedule(false).is_err());
        assert!(schedule(1, 0, Some(0)).validate_schedule(false).is_err());
    }

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
