//! Typed Python transport; accounting is exclusively in btcore::swing.
use btcore::swing::{self, Bar, Borrow, Hedge, SwingCosts, SwingRules, Target};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type BarRow = (i32, usize, f64, f64, f64, f64, f64, f64, f64);
type CostRow = (f64, f64, f64, f64, f64, f64, f64);
type RuleRow = (Option<f64>, Option<f64>, f64, f64, bool);
type DailyRow = (i32, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64);
type FillRow = (i32, usize, i64, f64, f64, String);
type PositionRow = (i32, usize, i64, f64, f64);
type Output = (Vec<DailyRow>, Vec<FillRow>, Vec<PositionRow>);

#[pyfunction]
pub fn swing_impl(
    py: Python<'_>,
    bars: Vec<BarRow>,
    targets: Vec<(i32, usize, f64)>,
    capital: f64,
    costs: CostRow,
    rules: RuleRow,
    borrow: Option<Vec<(i32, usize, bool)>>,
    hedge: Option<Vec<(i32, f64, bool)>>,
) -> PyResult<Output> {
    let bars: Vec<_> = bars
        .into_iter()
        .map(|b| Bar {
            date: b.0,
            symbol: b.1,
            open: b.2,
            high: b.3,
            low: b.4,
            close: b.5,
            limit_up: b.6,
            limit_down: b.7,
            dividend: b.8,
        })
        .collect();
    let targets: Vec<_> = targets
        .into_iter()
        .map(|t| Target {
            date: t.0,
            symbol: t.1,
            weight: t.2,
        })
        .collect();
    let borrow = borrow.map(|rows| {
        rows.into_iter()
            .map(|b| Borrow {
                date: b.0,
                symbol: b.1,
                can_borrow: b.2,
            })
            .collect::<Vec<_>>()
    });
    let hedge = hedge.map(|rows| {
        rows.into_iter()
            .map(|h| Hedge {
                date: h.0,
                ret: h.1,
                roll: h.2,
            })
            .collect::<Vec<_>>()
    });
    let costs = SwingCosts {
        fee_rate: costs.0,
        fee_discount: costs.1,
        sell_tax: costs.2,
        short_sell_fee: costs.3,
        borrow_rate: costs.4,
        slippage: costs.5,
        hedge_cost: costs.6,
    };
    let rules = SwingRules {
        take_profit: rules.0,
        stop_loss: rules.1,
        exit_slippage: rules.2,
        hedge_ratio: rules.3,
        execution_close: rules.4,
    };
    let out = py
        .detach(|| {
            swing::run_swing(
                &bars,
                &targets,
                capital,
                &costs,
                &rules,
                borrow.as_deref(),
                hedge.as_deref(),
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        out.daily
            .into_iter()
            .map(|d| {
                (
                    d.date,
                    d.pnl,
                    d.costs,
                    d.equity,
                    d.long,
                    d.short,
                    d.hedge_pnl,
                    d.cash,
                    d.hedge_notional,
                    d.dividends,
                    d.borrow_cost,
                    d.hedge_cost,
                )
            })
            .collect(),
        out.fills
            .into_iter()
            .map(|f| {
                (
                    f.date,
                    f.symbol,
                    f.shares,
                    f.price,
                    f.cost,
                    f.reason.to_owned(),
                )
            })
            .collect(),
        out.positions
            .into_iter()
            .map(|p| (p.date, p.symbol, p.shares, p.entry, p.mark))
            .collect(),
    ))
}
