//! Trade record tracking and analysis

use chrono::NaiveDate;

/// A single trade record
#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub symbol: String,
    pub entry_date: NaiveDate,
    pub exit_date: Option<NaiveDate>,
    pub entry_price: f64,
    pub exit_price: Option<f64>,
    pub shares: f64,
    pub side: TradeSide,
    pub pnl: Option<f64>,
    pub return_pct: Option<f64>,
}

/// Trade direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeSide {
    Long,
    Short,
}

impl TradeRecord {
    /// Create a new open trade
    pub fn open(
        symbol: String,
        entry_date: NaiveDate,
        entry_price: f64,
        shares: f64,
        side: TradeSide,
    ) -> Self {
        Self {
            symbol,
            entry_date,
            exit_date: None,
            entry_price,
            exit_price: None,
            shares,
            side,
            pnl: None,
            return_pct: None,
        }
    }

    /// Close the trade
    pub fn close(&mut self, exit_date: NaiveDate, exit_price: f64, fee_ratio: f64, tax_ratio: f64) {
        self.exit_date = Some(exit_date);
        self.exit_price = Some(exit_price);

        let gross_pnl = match self.side {
            TradeSide::Long => (exit_price - self.entry_price) * self.shares,
            TradeSide::Short => (self.entry_price - exit_price) * self.shares,
        };

        // Calculate transaction costs
        let entry_cost = self.entry_price * self.shares * fee_ratio;
        let exit_cost = exit_price * self.shares * (fee_ratio + tax_ratio);
        let total_cost = entry_cost + exit_cost;

        self.pnl = Some(gross_pnl - total_cost);

        if self.entry_price > 0.0 {
            self.return_pct = Some(self.pnl.unwrap() / (self.entry_price * self.shares));
        }
    }

    /// Check if trade is still open
    pub fn is_open(&self) -> bool {
        self.exit_date.is_none()
    }

    /// Get holding period in days (None if still open)
    pub fn holding_days(&self) -> Option<i64> {
        self.exit_date.map(|exit| (exit - self.entry_date).num_days())
    }
}

/// Trade book for tracking all trades
#[derive(Debug, Default)]
pub struct TradeBook {
    pub trades: Vec<TradeRecord>,
    pub fee_ratio: f64,
    pub tax_ratio: f64,
}

impl TradeBook {
    pub fn new(fee_ratio: f64, tax_ratio: f64) -> Self {
        Self {
            trades: Vec::new(),
            fee_ratio,
            tax_ratio,
        }
    }

    /// Open a new trade
    pub fn open_trade(
        &mut self,
        symbol: String,
        date: NaiveDate,
        price: f64,
        shares: f64,
        side: TradeSide,
    ) {
        self.trades.push(TradeRecord::open(symbol, date, price, shares, side));
    }

    /// Close a trade by symbol
    pub fn close_trade(&mut self, symbol: &str, date: NaiveDate, price: f64) -> Option<f64> {
        for trade in self.trades.iter_mut().rev() {
            if trade.symbol == symbol && trade.is_open() {
                trade.close(date, price, self.fee_ratio, self.tax_ratio);
                return trade.pnl;
            }
        }
        None
    }

    /// Get all closed trades
    pub fn closed_trades(&self) -> Vec<&TradeRecord> {
        self.trades.iter().filter(|t| !t.is_open()).collect()
    }

    /// Get all open trades
    pub fn open_trades(&self) -> Vec<&TradeRecord> {
        self.trades.iter().filter(|t| t.is_open()).collect()
    }

    /// Calculate trade statistics
    pub fn stats(&self) -> TradeStats {
        let closed: Vec<_> = self.closed_trades();

        if closed.is_empty() {
            return TradeStats::default();
        }

        let returns: Vec<f64> = closed
            .iter()
            .filter_map(|t| t.return_pct)
            .collect();

        let wins: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).copied().collect();
        let losses: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

        let total_trades = closed.len();
        let winning_trades = wins.len();
        let losing_trades = losses.len();

        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let avg_win = if !wins.is_empty() {
            wins.iter().sum::<f64>() / wins.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losses.is_empty() {
            losses.iter().sum::<f64>() / losses.len() as f64
        } else {
            0.0
        };

        let profit_factor = if avg_loss.abs() > 0.0 {
            avg_win / avg_loss.abs()
        } else {
            f64::INFINITY
        };

        let holding_periods: Vec<i64> = closed
            .iter()
            .filter_map(|t| t.holding_days())
            .collect();

        let avg_holding_days = if !holding_periods.is_empty() {
            holding_periods.iter().sum::<i64>() as f64 / holding_periods.len() as f64
        } else {
            0.0
        };

        TradeStats {
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            avg_win,
            avg_loss,
            profit_factor,
            avg_holding_days,
        }
    }
}

/// Summary statistics for trades
#[derive(Debug, Clone, Default)]
pub struct TradeStats {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    pub avg_holding_days: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap()
    }

    #[test]
    fn test_trade_open_close() {
        let mut trade = TradeRecord::open(
            "AAPL".to_string(),
            date(2024, 1, 1),
            100.0,
            10.0,
            TradeSide::Long,
        );

        assert!(trade.is_open());
        assert!(trade.pnl.is_none());

        trade.close(date(2024, 1, 10), 110.0, 0.001, 0.003);

        assert!(!trade.is_open());
        assert!(trade.pnl.is_some());

        // Gross PnL: (110 - 100) * 10 = 100
        // Entry cost: 100 * 10 * 0.001 = 1
        // Exit cost: 110 * 10 * 0.004 = 4.4
        // Net PnL: 100 - 1 - 4.4 = 94.6
        assert!((trade.pnl.unwrap() - 94.6).abs() < 1e-10);
    }

    #[test]
    fn test_trade_holding_days() {
        let mut trade = TradeRecord::open(
            "TSLA".to_string(),
            date(2024, 1, 1),
            200.0,
            5.0,
            TradeSide::Long,
        );

        assert!(trade.holding_days().is_none());

        trade.close(date(2024, 1, 15), 220.0, 0.001, 0.003);

        assert_eq!(trade.holding_days(), Some(14));
    }

    #[test]
    fn test_trade_book() {
        let mut book = TradeBook::new(0.001, 0.003);

        book.open_trade("AAPL".to_string(), date(2024, 1, 1), 100.0, 10.0, TradeSide::Long);
        book.open_trade("GOOG".to_string(), date(2024, 1, 2), 150.0, 5.0, TradeSide::Long);

        assert_eq!(book.open_trades().len(), 2);
        assert_eq!(book.closed_trades().len(), 0);

        book.close_trade("AAPL", date(2024, 1, 10), 110.0);

        assert_eq!(book.open_trades().len(), 1);
        assert_eq!(book.closed_trades().len(), 1);

        let stats = book.stats();
        assert_eq!(stats.total_trades, 1);
    }

    #[test]
    fn test_short_trade() {
        let mut trade = TradeRecord::open(
            "SPY".to_string(),
            date(2024, 1, 1),
            450.0,
            10.0,
            TradeSide::Short,
        );

        trade.close(date(2024, 1, 10), 440.0, 0.001, 0.003);

        // Gross PnL for short: (450 - 440) * 10 = 100
        // Entry cost: 450 * 10 * 0.001 = 4.5
        // Exit cost: 440 * 10 * 0.004 = 17.6
        // Net PnL: 100 - 4.5 - 17.6 = 77.9
        assert!((trade.pnl.unwrap() - 77.9).abs() < 1e-10);
    }
}
