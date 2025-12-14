//! Statistical metrics for backtest analysis

/// Backtest statistics result
#[derive(Debug, Clone, Default)]
pub struct BacktestStats {
    pub total_return: f64,
    pub cagr: f64,
    pub max_drawdown: f64,
    pub daily_sharpe: f64,
    pub daily_sortino: f64,
    pub calmar: f64,
    pub win_ratio: f64,
}

/// Calculate Sharpe ratio
///
/// # Arguments
/// * `returns` - Daily returns
/// * `rf` - Risk-free rate (annualized)
/// * `annualize` - Trading days per year (typically 252)
///
/// # Returns
/// Annualized Sharpe ratio
pub fn sharpe_ratio(returns: &[f64], rf: f64, annualize: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;

    if n < 2.0 {
        return 0.0;
    }

    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = variance.sqrt();

    if std > 0.0 {
        (mean - rf / annualize) / std * annualize.sqrt()
    } else {
        0.0
    }
}

/// Calculate Sortino ratio (downside risk only)
///
/// # Arguments
/// * `returns` - Daily returns
/// * `rf` - Risk-free rate (annualized)
/// * `annualize` - Trading days per year (typically 252)
///
/// # Returns
/// Annualized Sortino ratio
pub fn sortino_ratio(returns: &[f64], rf: f64, annualize: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;
    let daily_rf = rf / annualize;

    // Only consider returns below the risk-free rate
    let downside_returns: Vec<f64> = returns
        .iter()
        .filter(|&&r| r < daily_rf)
        .map(|&r| (r - daily_rf).powi(2))
        .collect();

    if downside_returns.is_empty() {
        return f64::INFINITY; // No downside risk
    }

    let downside_variance: f64 = downside_returns.iter().sum::<f64>() / n;
    let downside_std = downside_variance.sqrt();

    if downside_std > 0.0 {
        (mean - daily_rf) / downside_std * annualize.sqrt()
    } else {
        0.0
    }
}

/// Calculate maximum drawdown from cumulative returns
///
/// # Arguments
/// * `creturn` - Cumulative return values
///
/// # Returns
/// Maximum drawdown (negative value, e.g., -0.20 for 20% drawdown)
pub fn max_drawdown(creturn: &[f64]) -> f64 {
    if creturn.is_empty() {
        return 0.0;
    }

    let mut peak = creturn[0];
    let mut max_dd = 0.0f64;

    for &val in creturn {
        if val > peak {
            peak = val;
        }
        if peak > 0.0 {
            let dd = (val - peak) / peak;
            if dd < max_dd {
                max_dd = dd;
            }
        }
    }

    max_dd
}

/// Calculate drawdown series from cumulative returns
///
/// # Arguments
/// * `creturn` - Cumulative return values
///
/// # Returns
/// Drawdown at each point (negative values)
pub fn drawdown_series(creturn: &[f64]) -> Vec<f64> {
    if creturn.is_empty() {
        return vec![];
    }

    let mut peak = creturn[0];
    let mut drawdowns = Vec::with_capacity(creturn.len());

    for &val in creturn {
        if val > peak {
            peak = val;
        }
        let dd = if peak > 0.0 {
            (val - peak) / peak
        } else {
            0.0
        };
        drawdowns.push(dd);
    }

    drawdowns
}

/// Calculate CAGR (Compound Annual Growth Rate)
///
/// # Arguments
/// * `start` - Starting value
/// * `end` - Ending value
/// * `years` - Number of years
///
/// # Returns
/// CAGR as decimal (e.g., 0.10 for 10%)
pub fn calc_cagr(start: f64, end: f64, years: f64) -> f64 {
    if start <= 0.0 || end <= 0.0 || years <= 0.0 {
        return 0.0;
    }
    (end / start).powf(1.0 / years) - 1.0
}

/// Calculate Calmar ratio (CAGR / |Max Drawdown|)
///
/// # Arguments
/// * `cagr` - Compound annual growth rate
/// * `max_dd` - Maximum drawdown (negative value)
///
/// # Returns
/// Calmar ratio
pub fn calmar_ratio(cagr: f64, max_dd: f64) -> f64 {
    let abs_dd = max_dd.abs();
    if abs_dd > 0.0 {
        cagr / abs_dd
    } else {
        0.0
    }
}

/// Calculate win ratio from trade returns
///
/// # Arguments
/// * `trade_returns` - Returns for each trade
///
/// # Returns
/// Proportion of winning trades (0.0 to 1.0)
pub fn win_ratio(trade_returns: &[f64]) -> f64 {
    if trade_returns.is_empty() {
        return 0.0;
    }

    let wins = trade_returns.iter().filter(|&&r| r > 0.0).count();
    wins as f64 / trade_returns.len() as f64
}

impl BacktestStats {
    /// Calculate all statistics from returns and cumulative returns
    pub fn from_returns(
        daily_returns: &[f64],
        creturn: &[f64],
        years: f64,
        rf: f64,
    ) -> Self {
        let total_return = if !creturn.is_empty() {
            creturn.last().copied().unwrap_or(1.0) - 1.0
        } else {
            0.0
        };

        let start = creturn.first().copied().unwrap_or(1.0);
        let end = creturn.last().copied().unwrap_or(1.0);
        let cagr = calc_cagr(start, end, years);

        let max_dd = max_drawdown(creturn);
        let daily_sharpe = sharpe_ratio(daily_returns, rf, 252.0);
        let daily_sortino = sortino_ratio(daily_returns, rf, 252.0);
        let calmar = calmar_ratio(cagr, max_dd);

        Self {
            total_return,
            cagr,
            max_drawdown: max_dd,
            daily_sharpe,
            daily_sortino,
            calmar,
            win_ratio: 0.0, // Calculated from trades separately
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_ratio_positive() {
        // Consistent positive returns should give positive Sharpe
        let returns = vec![0.01, 0.02, 0.01, 0.015, 0.01];
        let sharpe = sharpe_ratio(&returns, 0.0, 252.0);
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_sharpe_ratio_negative() {
        // Consistent negative returns should give negative Sharpe
        let returns = vec![-0.01, -0.02, -0.01, -0.015, -0.01];
        let sharpe = sharpe_ratio(&returns, 0.0, 252.0);
        assert!(sharpe < 0.0);
    }

    #[test]
    fn test_sharpe_ratio_empty() {
        let sharpe = sharpe_ratio(&[], 0.0, 252.0);
        assert_eq!(sharpe, 0.0);
    }

    #[test]
    fn test_max_drawdown_basic() {
        // Price goes 100 -> 120 -> 90 -> 100
        // Max DD = (90 - 120) / 120 = -0.25
        let creturn = vec![1.0, 1.2, 0.9, 1.0];
        let max_dd = max_drawdown(&creturn);
        assert!((max_dd - (-0.25)).abs() < 1e-10);
    }

    #[test]
    fn test_max_drawdown_no_drawdown() {
        // Always increasing
        let creturn = vec![1.0, 1.1, 1.2, 1.3];
        let max_dd = max_drawdown(&creturn);
        assert_eq!(max_dd, 0.0);
    }

    #[test]
    fn test_calc_cagr() {
        // 100 -> 200 in 5 years
        let cagr = calc_cagr(1.0, 2.0, 5.0);
        let expected = 2.0_f64.powf(1.0 / 5.0) - 1.0;
        assert!((cagr - expected).abs() < 1e-10);
    }

    #[test]
    fn test_win_ratio() {
        let returns = vec![0.1, -0.05, 0.02, -0.01, 0.03];
        let ratio = win_ratio(&returns);
        assert!((ratio - 0.6).abs() < 1e-10); // 3 wins out of 5
    }

    #[test]
    fn test_drawdown_series() {
        let creturn = vec![1.0, 1.2, 1.1, 1.3, 1.2];
        let dd = drawdown_series(&creturn);

        assert_eq!(dd.len(), 5);
        assert!((dd[0] - 0.0).abs() < 1e-10);
        assert!((dd[1] - 0.0).abs() < 1e-10);
        assert!((dd[2] - (-1.0 / 12.0)).abs() < 1e-10); // (1.1 - 1.2) / 1.2
        assert!((dd[3] - 0.0).abs() < 1e-10);
        assert!((dd[4] - (-1.0 / 13.0)).abs() < 1e-10); // (1.2 - 1.3) / 1.3
    }
}
