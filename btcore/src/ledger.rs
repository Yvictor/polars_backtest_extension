//! Daily and per-leg ledger of a backtest
//!
//! Filled by the tracker while the simulation books each leg, so it reports
//! the amounts that actually moved through cash. `trades` keeps one row per
//! rebalance; the ledger is where the net legs are visible.

/// Why a leg was executed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillReason {
    Rebalance,
    Stop,
}

impl FillReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rebalance => "rebalance",
            Self::Stop => "stop",
        }
    }
}

/// One executed leg, as booked against cash
#[derive(Debug, Clone, Copy)]
pub struct LegFill {
    /// Signed traded value: positive for a buy, negative for a sell
    pub notional: f64,
    /// Execution price; NaN for a leg booked without a valid price
    pub price: f64,
    pub commission: f64,
    /// Tax charged on the configured side of this leg
    pub tax: f64,
    pub slippage: f64,
    pub surcharge: f64,
    pub reason: FillReason,
}

/// Portfolio state after the close of one day
#[derive(Debug, Clone, Copy)]
pub struct DayClose {
    pub equity: f64,
    pub cash: f64,
    pub gross_long: f64,
    pub gross_short: f64,
}

/// One ledger row per executed leg
#[derive(Debug, Clone)]
pub struct FillRecord<K, D> {
    pub date: D,
    pub key: K,
    pub fill: LegFill,
}

/// One ledger row per recorded day
#[derive(Debug, Clone)]
pub struct DailyRecord<D> {
    pub date: D,
    pub close: DayClose,
    pub commission: f64,
    pub tax: f64,
    pub slippage: f64,
    pub surcharge: f64,
    pub borrow_cost: f64,
    pub traded_notional: f64,
    /// `traded_notional` over the previous day's equity; None if that was not positive
    pub turnover: Option<f64>,
}

/// Cash paid for borrowing a position over the previous calendar interval
#[derive(Debug, Clone)]
pub struct BorrowRecord<K, D> {
    pub date: D,
    pub key: K,
    pub amount: f64,
}

/// Ledger collected during a simulation
#[derive(Debug, Clone)]
pub struct Ledger<K, D> {
    pub fills: Vec<FillRecord<K, D>>,
    pub daily: Vec<DailyRecord<D>>,
    pub cashflows: Vec<BorrowRecord<K, D>>,
    commission: f64,
    tax: f64,
    slippage: f64,
    surcharge: f64,
    borrow_cost: f64,
    traded_notional: f64,
    previous_equity: f64,
}

impl<K, D> Ledger<K, D> {
    /// Create an empty ledger for a portfolio that starts at `initial_equity`
    pub fn new(initial_equity: f64) -> Self {
        Self {
            fills: Vec::new(),
            daily: Vec::new(),
            cashflows: Vec::new(),
            commission: 0.0,
            tax: 0.0,
            slippage: 0.0,
            surcharge: 0.0,
            borrow_cost: 0.0,
            traded_notional: 0.0,
            previous_equity: initial_equity,
        }
    }

    /// Add a leg to the current day
    pub fn add_fill(&mut self, key: K, date: D, fill: LegFill) {
        self.commission += fill.commission;
        self.tax += fill.tax;
        self.slippage += fill.slippage;
        self.surcharge += fill.surcharge;
        self.traded_notional += fill.notional.abs();
        self.fills.push(FillRecord { date, key, fill });
    }

    pub fn add_borrow(&mut self, key: K, date: D, cost: f64) {
        self.borrow_cost += cost;
        self.cashflows.push(BorrowRecord {
            date,
            key,
            amount: -cost,
        });
    }

    /// Write the current day's row and start the next day from zero
    pub fn close_day(&mut self, date: D, close: DayClose) {
        let turnover =
            (self.previous_equity > 0.0).then(|| self.traded_notional / self.previous_equity);
        self.daily.push(DailyRecord {
            date,
            close,
            commission: std::mem::take(&mut self.commission),
            tax: std::mem::take(&mut self.tax),
            slippage: std::mem::take(&mut self.slippage),
            surcharge: std::mem::take(&mut self.surcharge),
            borrow_cost: std::mem::take(&mut self.borrow_cost),
            traded_notional: std::mem::take(&mut self.traded_notional),
            turnover,
        });
        self.previous_equity = close.equity;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leg(notional: f64) -> LegFill {
        LegFill {
            notional,
            price: 10.0,
            commission: 0.01,
            tax: 0.02,
            slippage: 0.0,
            surcharge: 0.0,
            reason: FillReason::Rebalance,
        }
    }

    fn close(equity: f64) -> DayClose {
        DayClose {
            equity,
            cash: 0.0,
            gross_long: equity,
            gross_short: 0.0,
        }
    }

    #[test]
    fn test_day_sums_its_legs_and_the_next_day_starts_empty() {
        let mut ledger: Ledger<&str, i32> = Ledger::new(2.0);
        ledger.add_fill("A", 1, leg(1.0));
        ledger.add_fill("B", 1, leg(-3.0));
        ledger.close_day(1, close(4.0));
        ledger.close_day(2, close(4.0));

        let first = &ledger.daily[0];
        assert_eq!(first.traded_notional, 4.0);
        assert_eq!(first.commission, 0.02);
        assert_eq!(first.tax, 0.04);
        assert_eq!(first.turnover, Some(2.0));
        assert_eq!(ledger.daily[1].traded_notional, 0.0);
        assert_eq!(ledger.daily[1].turnover, Some(0.0));
    }

    #[test]
    fn test_turnover_is_undefined_after_nonpositive_equity() {
        let mut ledger: Ledger<&str, i32> = Ledger::new(1.0);
        ledger.close_day(1, close(0.0));
        ledger.close_day(2, close(1.0));
        assert_eq!(ledger.daily[1].turnover, None);
    }
}
