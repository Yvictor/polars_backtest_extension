//! Ledger DataFrames of a report
//!
//! Converts the btcore ledger to the frames exposed on `BacktestReport`.

use btcore::ledger::Ledger;
use polars::prelude::*;

/// Ledger tables carried by a report
#[derive(Clone)]
pub(crate) struct LedgerFrames {
    pub daily: DataFrame,
    pub fills: DataFrame,
}

fn date_column(dates: Vec<i32>) -> PolarsResult<Column> {
    Ok(Series::new("date".into(), dates)
        .cast(&DataType::Date)?
        .into_column())
}

fn float_column(name: &str, values: impl Iterator<Item = f64>) -> Column {
    Series::new(name.into(), values.collect::<Vec<f64>>()).into_column()
}

fn daily_frame(ledger: &Ledger<String, i32>) -> PolarsResult<DataFrame> {
    let rows = || ledger.daily.iter();
    let turnover: Vec<Option<f64>> = rows().map(|r| r.turnover).collect();

    DataFrame::new(vec![
        date_column(rows().map(|r| r.date).collect())?,
        float_column("equity", rows().map(|r| r.close.equity)),
        float_column("cash", rows().map(|r| r.close.cash)),
        float_column("commission", rows().map(|r| r.commission)),
        float_column("tax", rows().map(|r| r.tax)),
        float_column("slippage", rows().map(|r| r.slippage)),
        float_column("gross_long", rows().map(|r| r.close.gross_long)),
        float_column("gross_short", rows().map(|r| r.close.gross_short)),
        float_column(
            "net_exposure",
            rows().map(|r| r.close.gross_long - r.close.gross_short),
        ),
        float_column(
            "gross_exposure",
            rows().map(|r| r.close.gross_long + r.close.gross_short),
        ),
        float_column("traded_notional", rows().map(|r| r.traded_notional)),
        Series::new("turnover".into(), turnover).into_column(),
    ])
}

fn fills_frame(ledger: &Ledger<String, i32>) -> PolarsResult<DataFrame> {
    let rows = || ledger.fills.iter();
    let symbols: Vec<&str> = rows().map(|r| r.key.as_str()).collect();
    let reasons: Vec<&str> = rows().map(|r| r.fill.reason.as_str()).collect();

    DataFrame::new(vec![
        date_column(rows().map(|r| r.date).collect())?,
        Series::new("symbol".into(), symbols).into_column(),
        float_column("notional", rows().map(|r| r.fill.notional)),
        float_column("price", rows().map(|r| r.fill.price)),
        Series::new("reason".into(), reasons).into_column(),
    ])
}

/// Build the frames of a report from the btcore ledger
pub(crate) fn ledger_frames(ledger: &Ledger<String, i32>) -> PolarsResult<LedgerFrames> {
    Ok(LedgerFrames {
        daily: daily_frame(ledger)?,
        fills: fills_frame(ledger)?,
    })
}
