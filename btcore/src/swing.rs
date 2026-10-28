//! Opt-in whole-share cash accounting. Independent of the legacy weight engine.
//! Dates are epoch days; symbols are caller-assigned IDs in stable lexical order.
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
pub struct Bar {
    pub date: i32,
    pub symbol: usize,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub limit_up: f64,
    pub limit_down: f64,
    pub dividend: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Target {
    pub date: i32,
    pub symbol: usize,
    pub weight: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Borrow {
    pub date: i32,
    pub symbol: usize,
    pub can_borrow: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Hedge {
    pub date: i32,
    pub ret: f64,
    pub roll: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct SwingCosts {
    pub fee_rate: f64,
    pub fee_discount: f64,
    pub sell_tax: f64,
    pub short_sell_fee: f64,
    pub borrow_rate: f64,
    pub slippage: f64,
    pub hedge_cost: f64,
}

#[cfg(test)]
impl Default for SwingCosts {
    fn default() -> Self {
        Self {
            fee_rate: 0.001425,
            fee_discount: 0.2,
            sell_tax: 0.003,
            short_sell_fee: 0.0008,
            borrow_rate: 0.0,
            slippage: 0.0,
            hedge_cost: 0.0002,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SwingRules {
    pub take_profit: Option<f64>,
    pub stop_loss: Option<f64>,
    pub exit_slippage: f64,
    pub hedge_ratio: f64,
    pub execution_close: bool,
}

#[cfg(test)]
impl Default for SwingRules {
    fn default() -> Self {
        Self {
            take_profit: None,
            stop_loss: None,
            exit_slippage: 0.001,
            hedge_ratio: 0.0,
            execution_close: false,
        }
    }
}

#[derive(Debug)]
pub struct Daily {
    pub date: i32,
    pub pnl: f64,
    pub costs: f64,
    pub equity: f64,
    pub long: f64,
    pub short: f64,
    pub hedge_pnl: f64,
    pub cash: f64,
    pub hedge_notional: f64,
    pub dividends: f64,
    pub borrow_cost: f64,
    pub hedge_cost: f64,
}

#[derive(Debug)]
pub struct Fill {
    pub date: i32,
    pub symbol: usize,
    pub shares: i64,
    pub price: f64,
    pub cost: f64,
    pub reason: &'static str,
}

#[derive(Debug)]
pub struct PositionSnapshot {
    pub date: i32,
    pub symbol: usize,
    pub shares: i64,
    pub entry: f64,
    pub mark: f64,
}

#[derive(Debug, Default)]
pub struct SwingResult {
    pub daily: Vec<Daily>,
    pub fills: Vec<Fill>,
    pub positions: Vec<PositionSnapshot>,
}

#[derive(Clone, Copy)]
struct Position {
    shares: i64,
    entry: f64,
    mark: f64,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            shares: 0,
            entry: f64::NAN,
            mark: f64::NAN,
        }
    }
}

fn exit_price(p: Position, b: &Bar, r: &SwingRules) -> Option<f64> {
    let long = p.shares > 0;
    let locked = b.high == b.low && b.low == if long { b.limit_down } else { b.limit_up };
    let tp = r
        .take_profit
        .map(|v| p.entry * if long { 1.0 + v } else { 1.0 - v });
    let sl = r
        .stop_loss
        .map(|v| p.entry * if long { 1.0 - v } else { 1.0 + v });
    if long {
        if sl.is_some_and(|v| b.open <= v) {
            return if locked { None } else { Some(b.open) };
        }
        if tp.is_some_and(|v| b.open >= v) {
            return Some(b.open);
        }
        if let Some(v) = sl.filter(|v| b.low < *v) {
            return if locked {
                None
            } else {
                Some(v * (1.0 - r.exit_slippage))
            };
        }
        tp.filter(|v| b.high > *v)
            .map(|v| v * (1.0 - r.exit_slippage))
    } else {
        if sl.is_some_and(|v| b.open >= v) {
            return if locked { None } else { Some(b.open) };
        }
        if tp.is_some_and(|v| b.open <= v) {
            return Some(b.open);
        }
        if let Some(v) = sl.filter(|v| b.high > *v) {
            return if locked {
                None
            } else {
                Some(v * (1.0 + r.exit_slippage))
            };
        }
        tp.filter(|v| b.low < *v)
            .map(|v| v * (1.0 + r.exit_slippage))
    }
}

struct Book<'a> {
    cash: f64,
    positions: BTreeMap<usize, Position>,
    pending: BTreeMap<usize, i64>,
    costs: &'a SwingCosts,
    fills: Vec<Fill>,
}

impl Book<'_> {
    fn trade(
        &mut self,
        date: i32,
        symbol: usize,
        delta: i64,
        price: f64,
        reason: &'static str,
        short_open: i64,
    ) -> f64 {
        let value = (delta as f64).abs() * price;
        let mut cost = value * self.costs.fee_rate * self.costs.fee_discount;
        if delta < 0 {
            cost +=
                value * self.costs.sell_tax + short_open as f64 * price * self.costs.short_sell_fee;
        }
        self.cash -= delta as f64 * price + cost;
        let p = self.positions.entry(symbol).or_insert(Position {
            mark: price,
            ..Position::default()
        });
        let new = p.shares + delta;
        if new == 0 {
            p.entry = f64::NAN;
        } else if p.shares == 0 || (p.shares > 0) != (new > 0) {
            p.entry = price;
        } else if new.abs() > p.shares.abs() {
            p.entry = (p.entry * (p.shares as f64).abs() + price * (delta as f64).abs())
                / (new as f64).abs();
        }
        p.shares = new;
        self.fills.push(Fill {
            date,
            symbol,
            shares: delta,
            price,
            cost,
            reason,
        });
        cost
    }

    fn execute(
        &mut self,
        date: i32,
        bars: &BTreeMap<usize, &Bar>,
        close: bool,
        borrow: Option<&BTreeMap<(i32, usize), bool>>,
    ) -> f64 {
        let mut total = 0.0;
        let orders: Vec<_> = self.pending.iter().map(|(&s, &q)| (s, q)).collect();
        for (symbol, target) in orders {
            let Some(b) = bars.get(&symbol) else { continue };
            let px = if close { b.close } else { b.open };
            if !px.is_finite() {
                continue;
            }
            let held = self.positions.get(&symbol).map_or(0, |p| p.shares);
            let mut delta = target - held;
            if delta == 0 {
                self.pending.remove(&symbol);
                continue;
            }
            if (delta > 0 && px >= b.limit_up) || (delta < 0 && px <= b.limit_down) {
                continue;
            }
            let mut short_open = (-(held + delta) - (-held).max(0)).max(0);
            if short_open > 0
                && borrow.is_some_and(|v| !v.get(&(date, symbol)).copied().unwrap_or(false))
            {
                delta += short_open;
                short_open = 0;
                if delta == 0 {
                    continue;
                }
            }
            let price = px
                * if delta > 0 {
                    1.0 + self.costs.slippage
                } else {
                    1.0 - self.costs.slippage
                };
            total += self.trade(date, symbol, delta, price, "rebalance", short_open);
            if self.positions[&symbol].shares == target {
                self.pending.remove(&symbol);
            }
        }
        total
    }
}

/// Run daily events. Missing bars retain the previous mark and pending order.
/// Missing borrow keys deny new short shares when a borrow table is supplied.
pub fn run_swing(
    bars: &[Bar],
    targets: &[Target],
    capital: f64,
    costs: &SwingCosts,
    rules: &SwingRules,
    borrow: Option<&[Borrow]>,
    hedge: Option<&[Hedge]>,
) -> Result<SwingResult, String> {
    if !capital.is_finite() || capital <= 0.0 {
        return Err("capital must be finite and positive".into());
    }
    let rates = [
        costs.fee_rate,
        costs.fee_discount,
        costs.sell_tax,
        costs.short_sell_fee,
        costs.borrow_rate,
        costs.slippage,
        costs.hedge_cost,
        rules.exit_slippage,
        rules.hedge_ratio,
    ];
    if rates.iter().any(|x| !x.is_finite() || *x < 0.0)
        || costs.slippage >= 1.0
        || rules.exit_slippage >= 1.0
        || [rules.take_profit, rules.stop_loss]
            .into_iter()
            .flatten()
            .any(|v| !v.is_finite() || v < 0.0)
    {
        return Err(
            "rates and thresholds must be finite and nonnegative; slippage must be < 1".into(),
        );
    }
    if rules.hedge_ratio != 0.0 && hedge.is_none() {
        return Err("hedge data required for hedge_ratio".into());
    }
    let mut by_day: BTreeMap<i32, BTreeMap<usize, &Bar>> = BTreeMap::new();
    for b in bars {
        if by_day
            .entry(b.date)
            .or_default()
            .insert(b.symbol, b)
            .is_some()
        {
            return Err("duplicate bar (date, symbol)".into());
        }
        if [b.open, b.high, b.low, b.close, b.limit_up, b.limit_down]
            .iter()
            .any(|v| v.is_infinite() || (v.is_finite() && *v <= 0.0))
            || !b.dividend.is_finite()
        {
            return Err("prices must be positive or missing; dividends must be finite".into());
        }
    }
    let mut target_days: BTreeMap<i32, BTreeMap<usize, f64>> = BTreeMap::new();
    for t in targets {
        if !t.weight.is_finite() || !by_day.contains_key(&t.date) {
            return Err("targets need finite weights and session dates".into());
        }
        if target_days
            .entry(t.date)
            .or_default()
            .insert(t.symbol, t.weight)
            .is_some()
        {
            return Err("duplicate target (date, symbol)".into());
        }
    }
    let mut can_borrow = BTreeMap::new();
    for b in borrow.unwrap_or(&[]) {
        if can_borrow
            .insert((b.date, b.symbol), b.can_borrow)
            .is_some()
        {
            return Err("duplicate borrow key".into());
        }
    }
    let mut hedge_days = BTreeMap::new();
    for h in hedge.unwrap_or(&[]) {
        if !h.ret.is_finite() || hedge_days.insert(h.date, h).is_some() {
            return Err("hedge needs finite returns and unique dates".into());
        }
    }
    let mut book = Book {
        cash: capital,
        positions: BTreeMap::new(),
        pending: BTreeMap::new(),
        costs,
        fills: Vec::new(),
    };
    let mut result = SwingResult::default();
    let mut prev_day = None;
    let mut prev_equity = capital;
    let mut hedge_notional = 0.0;
    for (&date, bar) in &by_day {
        let cal_days = prev_day.map_or(0, |p| i64::from(date) - i64::from(p));
        let mut borrow_cost = 0.0;
        let mut dividends = 0.0;
        for (&symbol, p) in &book.positions {
            if p.shares == 0 {
                continue;
            }
            if let Some(b) = bar.get(&symbol) {
                let income = p.shares as f64 * b.dividend;
                book.cash += income;
                dividends += income;
            }
            if p.shares < 0 {
                let fee = -p.shares as f64 * p.mark * costs.borrow_rate * cal_days as f64 / 365.0;
                book.cash -= fee;
                borrow_cost += fee;
            }
        }
        let mut cost_today = borrow_cost;
        let borrow_map = borrow.map(|_| &can_borrow);
        if !rules.execution_close {
            cost_today += book.execute(date, bar, false, borrow_map);
        }
        let exits: Vec<_> = book
            .positions
            .iter()
            .filter_map(|(&symbol, &p)| {
                if p.shares == 0 {
                    return None;
                }
                bar.get(&symbol)
                    .and_then(|b| exit_price(p, b, rules))
                    .map(|px| (symbol, p.shares, px))
            })
            .collect();
        for (symbol, shares, price) in exits {
            cost_today += book.trade(date, symbol, -shares, price, "exit", 0);
            book.positions.get_mut(&symbol).unwrap().mark = price;
            if book
                .pending
                .get(&symbol)
                .is_some_and(|q| (*q > 0 && shares > 0) || (*q < 0 && shares < 0))
            {
                book.pending.remove(&symbol);
            }
        }
        if rules.execution_close {
            cost_today += book.execute(date, bar, true, borrow_map);
        }
        let (mut long, mut short) = (0.0, 0.0);
        for (&symbol, p) in &mut book.positions {
            if let Some(b) = bar.get(&symbol) {
                if b.close.is_finite() {
                    p.mark = b.close;
                }
            }
            let value = if p.shares == 0 {
                0.0
            } else {
                p.shares as f64 * p.mark
            };
            long += value.max(0.0);
            short += value.min(0.0);
            if p.shares != 0 {
                result.positions.push(PositionSnapshot {
                    date,
                    symbol,
                    shares: p.shares,
                    entry: p.entry,
                    mark: p.mark,
                });
            }
        }
        let mut hedge_pnl = 0.0;
        let mut hedge_cost = 0.0;
        if hedge.is_some() {
            if let Some(h) = hedge_days.get(&date) {
                hedge_pnl = hedge_notional * h.ret;
                hedge_notional += hedge_pnl;
                book.cash += hedge_pnl;
                if h.roll {
                    let c = 2.0 * hedge_notional.abs() * costs.hedge_cost;
                    book.cash -= c;
                    cost_today += c;
                    hedge_cost += c;
                }
            }
            let new_h = -rules.hedge_ratio * (long + short);
            let c = (new_h - hedge_notional).abs() * costs.hedge_cost;
            book.cash -= c;
            cost_today += c;
            hedge_cost += c;
            hedge_notional = new_h;
        }
        let equity = book.cash + long + short;
        if let Some(w) = target_days.get(&date) {
            book.pending.clear();
            let mut symbols: BTreeMap<usize, f64> = w.clone();
            for (&symbol, p) in &book.positions {
                if p.shares != 0 {
                    symbols.entry(symbol).or_insert(0.0);
                }
            }
            for (symbol, weight) in symbols {
                let reference = bar
                    .get(&symbol)
                    .map(|b| b.close)
                    .unwrap_or_else(|| book.positions.get(&symbol).map_or(f64::NAN, |p| p.mark));
                if !reference.is_finite() || reference <= 0.0 {
                    continue;
                }
                let lots = (weight * equity / reference / 1000.0).trunc();
                // Bound arithmetic so target-held and share reversals cannot overflow.
                if !lots.is_finite() || lots.abs() > (i64::MAX / 4000) as f64 {
                    return Err("target share count overflow".into());
                }
                book.pending.insert(symbol, lots as i64 * 1000);
            }
        }
        result.daily.push(Daily {
            date,
            pnl: equity - prev_equity,
            costs: cost_today,
            equity,
            long,
            short,
            hedge_pnl,
            cash: book.cash,
            hedge_notional,
            dividends,
            borrow_cost,
            hedge_cost,
        });
        prev_day = Some(date);
        prev_equity = equity;
    }
    result.fills = book.fills;
    Ok(result)
}

#[cfg(test)]
#[path = "swing_tests.rs"]
mod tests;
