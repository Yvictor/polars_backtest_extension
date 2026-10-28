"""Whole-lot cash-book backtests backed by btcore, plus calendar-week book selection.

Prices, limits and dividends must share the same (unadjusted) share units. Map
a reference-price gap to dividend only when it approximates a cash distribution.
No data is loaded implicitly. The legacy weight engine is unaffected.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import astuple, dataclass
from typing import Literal

import polars as pl


@dataclass(frozen=True)
class SwingCosts:
    hedge_cost: float
    fee_rate: float = 0.001425
    fee_discount: float = 0.2
    sell_tax: float = 0.003
    short_sell_fee: float = 0.0008
    borrow_rate: float = 0.0
    slippage: float = 0.0


@dataclass(frozen=True)
class SwingRules:
    take_profit: float | None = None
    stop_loss: float | None = None
    exit_slippage: float = 0.001
    hedge_ratio: float = 0.0
    execution: Literal["open", "close"] = "open"


@dataclass(frozen=True)
class SwingResult:
    daily: pl.DataFrame
    fills: pl.DataFrame
    positions: pl.DataFrame


_DAILY = {
    "date": pl.Int32,
    **dict.fromkeys(
        [
            "pnl",
            "costs",
            "equity",
            "long",
            "short",
            "hedge_pnl",
            "cash",
            "hedge_notional",
            "dividends",
            "borrow_cost",
            "hedge_cost",
        ],
        pl.Float64,
    ),
}
_FILLS = {
    "date": pl.Int32,
    "symbol": pl.UInt64,
    "shares": pl.Int64,
    "price": pl.Float64,
    "cost": pl.Float64,
    "reason": pl.String,
}
_POSITIONS = {
    "date": pl.Int32,
    "symbol": pl.UInt64,
    "shares": pl.Int64,
    "entry": pl.Float64,
    "mark": pl.Float64,
}


def _frame(df: pl.DataFrame, columns: list[str], key: list[str]) -> pl.DataFrame:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.select(columns).with_columns(pl.col("date").cast(pl.Date))
    if df.select(pl.any_horizontal(pl.col(key).is_null()).any()).item():
        raise ValueError(f"Null keys: {key}")
    if df.select(key).is_duplicated().any():
        raise ValueError(f"Duplicate keys: {key}")
    if "code" in key and df.schema["code"] != pl.String:
        raise ValueError("code must be a string (preserve leading zeros)")
    return df.sort(key)


def _run_native(bars, targets, capital, costs, rules, borrow, hedge):
    from polars_backtest._polars_backtest import swing_impl

    return swing_impl(bars, targets, capital, costs, rules, borrow, hedge)


def backtest_swing(
    targets: pl.DataFrame,
    bars: pl.DataFrame,
    capital: float = 1e8,
    *,
    costs: SwingCosts,
    rules: SwingRules | None = None,
    borrowable: pl.DataFrame | None = None,
    hedge: pl.DataFrame | None = None,
    allow_zero_hedge_cost: bool = False,
) -> SwingResult:
    """Simulate signed targets decided after close, filled next session or later.

    targets: date, code, weight. A zero-weight row denotes an explicit empty book.
    bars: date, code, open, high, low, close, limit_up, limit_down[, dividend].
    borrowable: date, code, can_borrow; absent keys deny new shorts if supplied.
    hedge: date, ret, roll; ret must be the held contract's own daily return.
    costs is required: choose fee_discount explicitly in SwingCosts. Supply
    hedge_cost as the per-side futures cost (fraction of notional) from the
    caller's own cost model (use 0.0 without a hedge).
    A nonzero hedge_ratio requires a hedge frame. It also requires nonzero
    hedge_cost unless allow_zero_hedge_cost=True permits a zero-cost experiment.
    Target dates outside the bars' session calendar raise ValueError.
    swing_books rejects overlapping long/short selections with ValueError.
    Missing bar prices retry orders and retain the last finite close mark.
    Null dividends are treated as 0.0.
    Output positions contain only nonzero end-of-day shares.
    """
    if not isinstance(costs, SwingCosts):
        raise TypeError("costs must be an explicit SwingCosts instance")
    rules = rules or SwingRules()
    if rules.hedge_ratio != 0 and costs.hedge_cost == 0.0 and not allow_zero_hedge_cost:
        raise ValueError(
            "Nonzero hedge_ratio requires nonzero hedge_cost; "
            "set allow_zero_hedge_cost=True to explicitly allow zero hedge costs"
        )
    if rules.execution not in ("open", "close"):
        raise ValueError("execution must be open or close")
    if "dividend" not in bars.columns:
        bars = bars.with_columns(pl.lit(0.0).alias("dividend"))
    prices = ["open", "high", "low", "close", "limit_up", "limit_down", "dividend"]
    bars = _frame(bars, ["date", "code", *prices], ["date", "code"])
    bars = bars.with_columns(pl.col("dividend").fill_null(0.0)).with_columns(
        pl.col(prices).cast(pl.Float64).fill_null(float("nan"))
    )
    targets = _frame(targets, ["date", "code", "weight"], ["date", "code"])
    if targets["weight"].null_count():
        raise ValueError("weight cannot be null")
    targets = targets.with_columns(pl.col("weight").cast(pl.Float64))
    borrow = None
    if borrowable is not None:
        borrow = _frame(borrowable, ["date", "code", "can_borrow"], ["date", "code"])
        if borrow.schema["can_borrow"] != pl.Boolean or borrow["can_borrow"].null_count():
            raise ValueError("can_borrow must be nonnull Boolean")
    if hedge is not None:
        hedge = _frame(hedge, ["date", "ret", "roll"], ["date"])
        if (
            hedge.schema["roll"] != pl.Boolean
            or hedge["roll"].null_count()
            or hedge["ret"].null_count()
        ):
            raise ValueError("hedge needs nonnull Boolean roll and finite ret")
        hedge = hedge.with_columns(pl.col("ret").cast(pl.Float64))
    codes = sorted(
        set(bars["code"])
        | set(targets["code"])
        | (set(borrow["code"]) if borrow is not None else set())
    )
    ids = {code: i for i, code in enumerate(codes)}

    def rows(df, symbol=True):
        if df is None:
            return None
        records = df.with_columns(pl.col("date").cast(pl.Int32)).iter_rows()
        return [(r[0], ids[r[1]], *r[2:]) for r in records] if symbol else list(records)

    rule_row = (*astuple(rules)[:4], rules.execution == "close")
    daily_rows, fill_rows, position_rows = _run_native(
        rows(bars),
        rows(targets),
        capital,
        # Native ABI order is independent of the dataclass constructor order.
        (
            costs.fee_rate,
            costs.fee_discount,
            costs.sell_tax,
            costs.short_sell_fee,
            costs.borrow_rate,
            costs.slippage,
            costs.hedge_cost,
        ),
        rule_row,
        rows(borrow),
        rows(hedge, False),
    )

    def output(records, schema):
        df = pl.DataFrame(records, schema=schema, orient="row").with_columns(
            pl.col("date").cast(pl.Date)
        )
        if "symbol" in schema:
            df = df.with_columns(
                pl.Series("code", [codes[i] for i in df["symbol"]], dtype=pl.String)
            ).drop("symbol")
        return df

    daily = output(daily_rows, _DAILY).with_columns(
        (pl.col("pnl") / pl.col("equity").shift(1).fill_null(capital)).alias("ret")
    )
    return SwingResult(daily, output(fill_rows, _FILLS), output(position_rows, _POSITIONS))


def phase_rebalance_dates(
    days: list[dt.date],
    weekday: int = 4,
    every: int = 1,
    phase: int = 0,
) -> list[dt.date]:
    """Last session on/before weekday in every-k calendar weeks, including gaps."""
    if not 0 <= weekday <= 4 or every < 1 or not 0 <= phase < every:
        raise ValueError("weekday must be 0..4; every >= 1; 0 <= phase < every")
    if not days:
        return []
    origin = min(days) - dt.timedelta(days=min(days).weekday())
    weeks = {}
    for day in sorted(set(days)):
        week = (day - dt.timedelta(days=day.weekday()) - origin).days // 7
        if day.weekday() <= weekday and week % every == phase:
            weeks[week] = day
    return sorted(weeks.values())


def bench_book(days: list[dt.date], weekday=4, every=1, phase=0) -> pl.DataFrame:
    """One 0050 buy-and-hold target on the schedule's first rebalance."""
    schedule = phase_rebalance_dates(days, weekday, every, phase)
    return pl.DataFrame(
        {
            "date": schedule[:1],
            "code": ["0050"] if schedule else [],
            "weight": [1.0] if schedule else [],
        },
        schema={"date": pl.Date, "code": pl.String, "weight": pl.Float64},
    )


def swing_books(
    scores: pl.DataFrame,
    panel: pl.DataFrame,
    days: list[dt.date],
    weekday: int = 4,
    top: int = 15,
    kind: Literal["long", "ls"] = "long",
    hold: int = 0,
    every: int = 1,
    phase: int = 0,
) -> pl.DataFrame:
    """Top-N targets using next-session scores. hold is a rank-retention buffer.

    scores: decision_date, instrument_id (STK:code), score, available_at.
    panel: decision_date, instrument_id, top500, disposal, can_short.
    """
    if top < 1 or hold < 0 or kind not in ("long", "ls"):
        raise ValueError("top >= 1, hold >= 0 and kind long|ls required")
    for df in (scores, panel):
        if df.select("decision_date", "instrument_id").is_duplicated().any():
            raise ValueError("Duplicate decision_date/instrument_id")
    at = scores.schema.get("available_at")
    if not isinstance(at, pl.Datetime) or at.time_zone is None:
        raise ValueError("available_at must be timezone-aware")
    scores = scores.with_columns(
        pl.col("decision_date").cast(pl.Date),
        pl.col("available_at").dt.convert_time_zone("Asia/Taipei"),
    )
    deadline = pl.col("decision_date").cast(pl.Datetime).dt.replace_time_zone(
        "Asia/Taipei"
    ) + pl.duration(hours=8, minutes=30)
    if scores.filter(pl.col("available_at").is_null() | (pl.col("available_at") > deadline)).height:
        raise ValueError("Scores must be available by D 08:30 Asia/Taipei")
    panel = panel.with_columns(pl.col("decision_date").cast(pl.Date))
    joined = scores.select("decision_date", "instrument_id", "score").join(
        panel, on=["decision_date", "instrument_id"]
    )
    joined = joined.filter(pl.col("top500") & ~pl.col("disposal") & pl.col("score").is_finite())
    grouped = {
        key[0]: g.sort(["score", "instrument_id"], descending=[True, False])
        for key, g in joined.partition_by("decision_date", as_dict=True).items()
    }
    days = sorted(set(days))
    nxt = dict(zip(days, days[1:]))
    held_long, held_short = set(), set()
    out = []

    def select(ranked, held):
        keep = [c for c in ranked[: max(hold, top)] if c in held][:top]
        return keep + [c for c in ranked if c not in keep][: top - len(keep)]

    for rebalance in phase_rebalance_dates(days, weekday, every, phase):
        s = grouped.get(nxt.get(rebalance))
        if s is None or s.is_empty():
            continue
        longs = select(s["instrument_id"].to_list(), held_long)
        held_long = set(longs)
        w = 1.0 / (2 * top) if kind == "ls" else 1.0 / top
        out.extend((rebalance, code.removeprefix("STK:"), w) for code in longs)
        if kind == "ls":
            shorts = select(
                s.filter(pl.col("can_short"))["instrument_id"].to_list()[::-1], held_short
            )
            held_short = set(shorts)
            if set(longs) & set(shorts):
                raise ValueError(
                    "Long/short selections overlap; insufficient distinct eligible names"
                )
            out.extend((rebalance, code.removeprefix("STK:"), -w) for code in shorts)
    return pl.DataFrame(
        out, schema={"date": pl.Date, "code": pl.String, "weight": pl.Float64}, orient="row"
    )
