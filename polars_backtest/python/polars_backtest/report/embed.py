"""Choose what a page carries from a bundle.

The bundle stays complete. Only this step trims, and it trims detail rows, never daily
inputs or totals: the daily curve is always whole, and a day's fills are all in the
payload or all out of it, so no page shows half a day.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

import polars as pl

from polars_backtest import viz
from polars_backtest.report.bundle import ReportBundle
from polars_backtest.report.metrics import closed_trades
from polars_backtest.report.schema import SCHEMA_VERSION

SIGNIFICANT = 8
HIST_BINS = 60
STOPS = (0.05, 0.10, 0.20)
MAX_SYMBOLS = 5_000
SYMBOL_TAIL = 2_000
TRADE_TAIL = 1_000
TOP_DRAWDOWNS = 5
DRAWDOWN_MARGIN = 5
TOP_DAYS = 50

PAGE_LIMITS = {
    "trades": 50_000,
    "fills": 150_000,
    "blocks": 50_000,
    "orders": 30_000,
    "cashflows": 20_000,
}
NOTEBOOK_LIMITS = {
    "trades": 15_000,
    "fills": 40_000,
    "blocks": 15_000,
    "orders": 10_000,
    "cashflows": 5_000,
}
WHOLE_DAY_TABLES = ("fills", "blocks", "cashflows")
ROW_TABLES = ("trades", "symbol_index", "symbols", "fills", "blocks", "cashflows", "orders")

# Summary keys added within schema 1.x, so a bundle written earlier can lack them.
SUMMARY_DEFAULTS: dict[str, Any] = {"structure": {}}


def _round(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(f"{value:.{SIGNIFICANT}g}")


def _encode_column(series: pl.Series, exact: bool) -> Any:
    """A column as JSON. Repeated text and dates become a dictionary plus indexes."""
    if series.dtype == pl.Date:
        series = series.dt.strftime("%Y-%m-%d")
    if series.dtype == pl.String:
        values = series.to_list()
        words = sorted({v for v in values if v is not None})
        if len(words) * 4 < len(values):
            lookup = {word: i for i, word in enumerate(words)}
            return {"d": words, "i": [None if v is None else lookup[v] for v in values]}
        return values
    if series.dtype.is_float():
        values = series.to_list()
        if exact:
            return [v if v is not None and math.isfinite(v) else None for v in values]
        return [_round(v) for v in values]
    return series.to_list()


def _columns(frame: pl.DataFrame, exact: bool = False) -> dict[str, Any]:
    return {name: _encode_column(frame[name], exact) for name in frame.columns}


def _priority_days(bundle: ReportBundle) -> list[dt.date]:
    """Days in the order worth keeping. The same bundle always gives the same order."""
    index = bundle.table("day_index").collect()
    dates = index["date"].to_list()
    position = {date: i for i, date in enumerate(dates)}
    troughs = (
        bundle.table("drawdowns")
        .filter(pl.col("series") == "strategy")
        .sort("rank")
        .head(TOP_DRAWDOWNS)
        .collect()["trough"]
        .to_list()
    )
    order: list[dt.date] = []
    for trough in troughs:
        at = position.get(trough)
        if at is not None:
            order += dates[max(0, at - DRAWDOWN_MARGIN) : at + DRAWDOWN_MARGIN + 1]
    largest = index.sort(pl.col("ret").abs(), "date", descending=[True, False], nulls_last=True)
    order += largest.head(TOP_DAYS)["date"].to_list()
    if "n_blocks" in index.columns:
        blocked = index.filter(pl.col("n_blocks") > 0).sort(
            "n_blocks", "date", descending=[True, False]
        )
        order += blocked.head(TOP_DAYS)["date"].to_list()
    order += reversed(dates)
    return list(dict.fromkeys(order))


def _keep_whole_days(frame: pl.DataFrame, order: list[dt.date], limit: int) -> pl.DataFrame:
    """Take whole days in priority order until the row limit is reached."""
    counts = dict(frame.group_by("date").len().iter_rows())
    known = set(order)
    order = order + sorted(d for d in counts if d not in known)
    kept = []
    total = 0
    for day in order:
        rows = counts.get(day, 0)
        if rows and total + rows <= limit:
            kept.append(day)
            total += rows
    return frame.filter(pl.col("date").is_in(kept))


def _trim_trades(trades: pl.DataFrame, limit: int) -> pl.DataFrame:
    """Keep the largest gains and losses and every open trade, then an even spread by exit."""
    contrib = pl.col("ret").fill_null(0.0).fill_nan(0.0) * pl.col("weight").abs()
    frame = trades.with_row_index("_i").with_columns(contrib.alias("_c"))
    order = frame.sort("_c", "_i")
    keep = pl.concat([order.head(TRADE_TAIL)["_i"], order.tail(TRADE_TAIL)["_i"]])
    keep = pl.concat([keep, frame.filter(pl.col("status") != "closed")["_i"]]).unique()
    rest = frame.filter(~pl.col("_i").is_in(keep.implode())).sort("exit_date", "_i")
    room = max(0, limit - keep.len())
    if room and rest.height:
        step = rest.height / room
        picks = [int(i * step) for i in range(min(room, rest.height))]
        keep = pl.concat([keep, rest[picks]["_i"]])
    return frame.filter(pl.col("_i").is_in(keep.implode())).drop("_i", "_c")


def _trim_symbols(index: pl.DataFrame) -> pl.DataFrame:
    if index.height <= MAX_SYMBOLS:
        return index
    return pl.concat([index.head(SYMBOL_TAIL), index.tail(SYMBOL_TAIL)])


def _histogram(trades: pl.DataFrame) -> dict[str, Any] | None:
    """Trade returns binned from every trade, so trimming never reshapes the distribution."""
    rets = closed_trades(trades)["ret"]
    if rets.is_empty():
        return None
    low, high = rets.quantile(0.005), rets.quantile(0.995)
    if low is None or high is None or high <= low:
        low, high = rets.min(), rets.max()
    if high <= low:
        high = low + 1e-9
    width = (high - low) / HIST_BINS
    counts = [0] * HIST_BINS
    for value in rets.to_list():
        counts[min(HIST_BINS - 1, max(0, int((value - low) / width)))] += 1
    return {"low": low, "width": width, "counts": counts, "mean": rets.mean(), "n": rets.len()}


def _trade_days(trades: pl.DataFrame) -> pl.DataFrame:
    """Closed trades summed per exit day and side, from every trade.

    Trimming keeps extremes on purpose, so rows in a page are a biased sample. Window
    figures add these up instead: win rate, averages and stop replays stay exact for
    any date range, whatever the page's row limits.
    """
    ret, mae = pl.col("ret"), pl.col("mae")
    aggs = [
        pl.len().cast(pl.Int64).alias("n"),
        (ret > 0).sum().cast(pl.Int64).alias("wins"),
        ret.sum().alias("sum_ret"),
        (ret * pl.col("weight").abs()).sum().alias("contrib"),
    ]
    if "mae" in trades.columns:
        known = mae.is_not_null()
        aggs.append(known.sum().cast(pl.Int64).alias("n_mae"))
        for i, stop in enumerate(STOPS):
            hit = (mae <= -stop).fill_null(False)
            clipped = pl.when(hit).then(pl.lit(-stop)).otherwise(ret)
            aggs += [
                hit.sum().cast(pl.Int64).alias(f"hit_{i}"),
                ((clipped > 0) & known).sum().cast(pl.Int64).alias(f"win_{i}"),
                clipped.filter(known).sum().alias(f"sum_{i}"),
            ]
    return (
        closed_trades(trades)
        .group_by(pl.col("exit_date").alias("date"), "side")
        .agg(aggs)
        .sort("date", "side")
    )


def _hist_days(trades: pl.DataFrame, hist: dict[str, Any]) -> pl.DataFrame:
    """The histogram's bins per exit day, so a date range can sum its own distribution."""
    bins = ((pl.col("ret") - hist["low"]) / hist["width"]).floor().clip(0, HIST_BINS - 1)
    return (
        closed_trades(trades)
        .group_by(pl.col("exit_date").alias("date"), bins.cast(pl.Int64).alias("bin"))
        .agg(pl.len().cast(pl.Int64).alias("n"))
        .sort("date", "bin")
    )


def _metrics(bundle: ReportBundle) -> dict[str, dict[str, float]]:
    frame = bundle.table("metrics").collect()
    out: dict[str, dict[str, float]] = {"engine": {}, "stats": {}}
    for section, key, value in frame.select("section", "key", "value").iter_rows():
        out["stats" if section == "stats" else "engine"][key] = value
    return out


def _rows(bundle: ReportBundle, limits: dict[str, int], detail: str) -> dict[str, pl.DataFrame]:
    """The row tables that go in the page, within their limits."""
    frames: dict[str, pl.DataFrame] = {}
    if detail == "summary":
        return frames
    for name in ROW_TABLES:
        if bundle.has(name):
            frames[name] = bundle.table(name).collect()
    if "trades" in frames and frames["trades"].height > limits["trades"]:
        frames["trades"] = _trim_trades(frames["trades"], limits["trades"])
    if "symbol_index" in frames:
        frames["symbol_index"] = _trim_symbols(frames["symbol_index"])
    if "orders" in frames and frames["orders"].height > limits["orders"]:
        frames["orders"] = frames["orders"].tail(limits["orders"])
    order = None
    for name in WHOLE_DAY_TABLES:
        if name in frames and frames[name].height > limits[name]:
            order = order if order is not None else _priority_days(bundle)
            frames[name] = _keep_whole_days(frames[name], order, limits[name])
    return frames


def _embed_notes(
    bundle: ReportBundle, frames: dict[str, pl.DataFrame], detail: str
) -> dict[str, Any]:
    """What each row table holds against the bundle, so a page can state it from counts."""
    notes = {}
    for name in ROW_TABLES:
        entry = bundle.manifest["tables"].get(name, {})
        if "rows" not in entry:
            continue
        rows = frames[name].height if name in frames else 0
        if detail == "summary":
            policy = "left out"
        elif rows == entry["rows"]:
            policy = "complete"
        elif name in WHOLE_DAY_TABLES:
            policy = "whole days"
        else:
            policy = "tails kept"
        notes[name] = {"rows_total": entry["rows"], "rows_embedded": rows, "policy": policy}
    return notes


def build_payload(
    bundle: ReportBundle,
    *,
    limits: dict[str, int] = PAGE_LIMITS,
    detail: str = "full",
    capital: float | None = None,
    day: str | None = None,
) -> dict[str, Any]:
    """Everything a page reads. Built only from the bundle, so no engine is needed.

    Daily inputs are complete at any length: ranges and rolling figures are computed
    from them, so thinning belongs to drawing, never here.
    """
    if detail not in ("full", "summary"):
        raise ValueError(f"detail must be 'full' or 'summary', got {detail!r}")
    frames = _rows(bundle, limits, detail)
    metrics = _metrics(bundle)
    manifest = bundle.manifest
    trades = bundle.table("trades").collect() if bundle.has("trades") else None
    hist = _histogram(trades) if trades is not None else None
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "library_version": manifest["library_version"],
        "run": manifest["run"],
        "params": manifest.get("params", []),
        "detail": detail,
        "capital": capital,
        "day": day,
        "summary": {**SUMMARY_DEFAULTS, **bundle.summary()},
        "manifest_checks": manifest["checks"],
        "coverage": bundle.coverage,
        "quality": viz._quality(metrics["engine"]),
        "metrics": metrics["engine"],
        "stats": metrics["stats"],
        "curve": _columns(bundle.table("curve").collect(), exact=True),
        "monthly": _columns(bundle.table("monthly").collect(), exact=True),
        "yearly": _columns(bundle.table("yearly").collect(), exact=True),
        "drawdowns": _columns(bundle.table("drawdowns").collect()),
        "day_index": _columns(bundle.table("day_index").collect()),
        "hist": hist,
        "stops": list(STOPS),
        "embed": _embed_notes(bundle, frames, detail),
        "tables": {name: _columns(frame) for name, frame in frames.items()},
    }
    if trades is not None:
        payload["trade_days"] = _columns(_trade_days(trades), exact=True)
    if hist is not None:
        payload["hist_days"] = _columns(_hist_days(trades, hist))
    if bundle.has("ledger_daily"):
        payload["ledger_daily"] = _columns(bundle.table("ledger_daily").collect())
    if bundle.has("fill_scenarios"):
        payload["fill_scenarios"] = _columns(bundle.table("fill_scenarios").collect())
    return payload
