"""Synthetic inputs for the report bundle tests. Every name and price here is invented."""

from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl
import polars_backtest as pl_bt

SYMBOLS = ["FAKE001", "FAKE002", "FAKE003", "FAKE004", "FAKE005", "FAKE006"]
NAMES = {symbol: f"Fake Co {i + 1}" for i, symbol in enumerate(SYMBOLS)}
LEDGER_TABLES = ("ledger_daily", "fills", "blocks", "cashflows", "orders")


def weekdays(n: int, start: dt.date = dt.date(2021, 1, 4)) -> list[dt.date]:
    days: list[dt.date] = []
    day = start
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


def prices(n: int = 620) -> pl.DataFrame:
    """A fixed price path with limit prices and traded value, no random numbers."""
    rows: list[dict[str, Any]] = []
    for i, date in enumerate(weekdays(n)):
        for j, symbol in enumerate(SYMBOLS):
            wave = 0.04 * ((i * (j + 3)) % 17 - 8) / 8
            drift = 1 + 0.0009 * i * (1 if j % 3 else -0.4)
            close = (40.0 + 25.0 * j) * drift * (1 + wave)
            locked = j == 2 and i % 37 == 5
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close,
                    "high": close if locked else close * 1.01,
                    "low": close if locked else close * 0.99,
                    "close": close,
                    "limit_up": close if locked else close * 1.1,
                    "limit_down": close * 0.9,
                    "trading_value": 5e7 * (j + 1),
                    "weight": float((i // 21 + j) % 3 != 0),
                }
            )
    return pl.DataFrame(rows)


def benchmark(n: int = 620) -> pl.DataFrame:
    days = weekdays(n)
    level = [1 + 0.0004 * i + 0.03 * (((i * 5) % 23) - 11) / 11 for i in range(n)]
    return pl.DataFrame({"date": days, "creturn": level})


def long_report(n: int = 620) -> Any:
    return pl_bt.backtest_with_report(prices(n), resample="M", stop_loss=0.1)


class LedgerReport:
    """A report with ledger tables attached the way the engine exposes them.

    Delegates everything else to the wrapped report, so one test can run the same
    code path with any subset of ledger tables present.
    """

    def __init__(self, report: Any, **tables: Any) -> None:
        self._report = report
        self._tables = tables

    def __getattr__(self, name: str) -> Any:
        tables = self.__dict__["_tables"]
        if name in tables:
            return tables[name]
        return getattr(self.__dict__["_report"], name)


def huge_fills_report(report: Any, rows: int = 5_000_000) -> LedgerReport:
    """A report with millions of fills spread over its real calendar."""
    days = report.daily_creturn()["date"].cast(pl.Date)
    fills = (
        pl.DataFrame({"date": days, "n": [rows // days.len()] * days.len()})
        .select(pl.col("date").repeat_by("n").explode())
        .with_row_index("i")
        .with_columns(
            pl.format("FAKE{}", pl.col("i") % 977).alias("symbol"),
            ((pl.col("i") % 7).cast(pl.Float64) - 3.5).alias("notional"),
            (50.0 + (pl.col("i") % 11)).alias("price"),
            pl.lit("rebalance").alias("reason"),
        )
        .drop("i")
    )
    return LedgerReport(report, fills=fills)


def fake_ledger(report: Any) -> dict[str, pl.DataFrame]:
    """Ledger tables that reconcile exactly with the report's daily curve."""
    curve = report.daily_creturn().select(pl.col("date").cast(pl.Date), "creturn").sort("date")
    dates = curve.get_column("date").to_list()
    fills: list[dict[str, Any]] = []
    for i, date in enumerate(dates):
        if i % 19 == 4:
            for j, sign in enumerate((1.0, -1.0)):
                fills.append(
                    {
                        "date": date,
                        "symbol": SYMBOLS[(i + j) % len(SYMBOLS)],
                        "notional": sign * 0.1 * (j + 1),
                        "price": 50.0 + j,
                        "reason": "rebalance",
                        "cost": 0.0002 * (j + 1),
                    }
                )
    fills_df = pl.DataFrame(fills)
    by_day = fills_df.group_by("date").agg(
        pl.col("notional").abs().sum().alias("traded_notional"),
        pl.col("cost").sum().alias("cost"),
    )
    daily = (
        curve.join(by_day, on="date", how="left")
        .with_columns(
            pl.col("creturn").alias("equity"),
            pl.col("traded_notional").fill_null(0.0),
            pl.col("cost").fill_null(0.0),
        )
        .with_columns(
            (pl.col("cost") * 0.25).alias("commission"),
            (pl.col("cost") * 0.75).alias("tax"),
            (pl.col("equity") * 0.1).alias("cash"),
            (pl.col("equity") * 0.8).alias("gross_long"),
            (pl.col("equity") * 0.1).alias("gross_short"),
            (pl.col("equity") * 0.7).alias("net_exposure"),
            (pl.col("equity") * 0.9).alias("gross_exposure"),
            (pl.col("traded_notional") / pl.col("equity").shift(1)).alias("turnover"),
        )
        .drop("creturn", "cost")
    )
    blocks = pl.DataFrame(
        {
            "date": [dates[30], dates[30], dates[90]],
            "symbol": [SYMBOLS[0], SYMBOLS[1], SYMBOLS[0]],
            "reason": ["limit_up", "limit_up", "no_borrow"],
            "side": ["buy", "buy", "short"],
            "notional": [0.05, 0.04, 0.03],
        }
    )
    cashflows = pl.DataFrame(
        {
            "date": [dates[60]],
            "symbol": [SYMBOLS[3]],
            "kind": ["dividend"],
            "amount": [0.001],
        }
    )
    orders = pl.DataFrame(
        {"date": [dates[4]], "symbol": [SYMBOLS[0]], "status": ["filled"], "notional": [0.1]}
    )
    return {
        "daily": daily,
        "fills": fills_df,
        "blocks": blocks,
        "cashflows": cashflows,
        "orders": orders,
    }
