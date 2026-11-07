"""Turn a report object into bundle tables. Works for long and wide reports alike."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Any

import polars as pl

from polars_backtest import viz
from polars_backtest.report import metrics as M
from polars_backtest.report.bundle import BundleCheckError, ReportBundle
from polars_backtest.report.schema import LEDGER_TABLES, SCHEMA_VERSION, TABLES, conform
from polars_backtest.report.summary import _safe, build_summary

# Engine attribute that holds each ledger table.
LEDGER_ATTRS = {
    "ledger_daily": "daily",
    "fills": "fills",
    "blocks": "blocks",
    "cashflows": "cashflows",
    "orders": "orders",
}
METRIC_SECTIONS = ("backtest", "profitability", "risk", "ratio", "winrate", "liquidity")
CONFIG_FIELDS = (
    "resample",
    "fee_ratio",
    "tax_ratio",
    "stop_loss",
    "take_profit",
    "trail_stop",
    "trade_at",
)
TRADE_DROP = ("period", "entry_price", "exit_price")
DRAWDOWNS_PER_SERIES = 10


def _ledger_frame(report: Any, name: str) -> pl.DataFrame | None:
    frame = getattr(report, LEDGER_ATTRS[name], None)
    if not isinstance(frame, pl.DataFrame):
        return None
    if "date" in frame.columns:
        frame = frame.with_columns(pl.col("date").cast(pl.Date))
    return frame


def _curve(report: Any, ledger: pl.DataFrame | None) -> pl.DataFrame:
    curve = viz._daily_curve(report).with_columns(
        (pl.col("creturn") / pl.col("creturn").shift(1) - 1).alias("ret"),
        (pl.col("creturn") / pl.col("creturn").cum_max() - 1).alias("drawdown"),
    )
    bench = viz._benchmark_series(report, curve)
    if bench is not None and any(v is not None for v in bench):
        curve = curve.with_columns(pl.Series("bench_creturn", bench, dtype=pl.Float64))
        peak = pl.col("bench_creturn").forward_fill().cum_max()
        curve = curve.with_columns((pl.col("bench_creturn") / peak - 1).alias("bench_drawdown"))
    return conform("curve", M.with_gross_curve(curve, ledger))


def _period_returns(curve: pl.DataFrame, column: str, keys: list[str]) -> pl.DataFrame:
    """Compounded return per period, each from the previous period's last level."""
    return (
        curve.drop_nulls(column)
        .group_by(keys, maintain_order=True)
        .agg(pl.col(column).last().alias("_last"))
        .with_columns((pl.col("_last") / pl.col("_last").shift(1)).alias("_step"))
    )


def _with_first_period(periods: pl.DataFrame, curve: pl.DataFrame, column: str) -> pl.DataFrame:
    first = curve[column].drop_nulls()[0]
    return periods.with_columns(
        (pl.col("_step").fill_null(pl.col("_last") / first) - 1).alias(column)
    ).drop("_last", "_step")


def _periods(curve: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    frame = curve.with_columns(
        pl.col("date").dt.year().cast(pl.Int64).alias("year"),
        pl.col("date").dt.month().cast(pl.Int64).alias("month"),
    )
    ret = _with_first_period(_period_returns(frame, "creturn", keys), frame, "creturn")
    ret = ret.rename({"creturn": "ret"})
    if "bench_creturn" in frame.columns:
        bench = _with_first_period(
            _period_returns(frame, "bench_creturn", keys), frame, "bench_creturn"
        )
        ret = ret.join(bench.rename({"bench_creturn": "bench_ret"}), on=keys, how="left")
    return ret


def _year_drawdowns(curve: pl.DataFrame) -> pl.DataFrame:
    """Each year's worst fall, counted from highs inside the year.

    The previous year's last close is the year's opening level, so a fall on the first
    trading day counts.
    """
    frame = curve.select("date", "creturn").with_columns(
        pl.col("date").dt.year().cast(pl.Int64).alias("year"),
    )
    rows = []
    for year in frame["year"].unique().sort().to_list():
        before = frame.filter(pl.col("year") < year)["creturn"].tail(1).to_list()
        values = before + frame.filter(pl.col("year") == year)["creturn"].to_list()
        series = pl.Series(values)
        rows.append({"year": year, "mdd": (series / series.cum_max() - 1).min()})
    return pl.DataFrame(rows, schema={"year": pl.Int64, "mdd": pl.Float64})


def _drawdowns(curve: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    series_columns = [("strategy", "creturn")]
    if "bench_creturn" in curve.columns:
        series_columns.append(("benchmark", "bench_creturn"))
    for series, column in series_columns:
        frame = curve.select("date", column).drop_nulls()
        episodes = viz._drawdown_episodes(
            frame["date"].to_list(), frame[column].to_list(), DRAWDOWNS_PER_SERIES
        )
        for rank, episode in enumerate(episodes, start=1):
            rows.append({**episode, "rank": rank, "series": series})
    schema = {
        "rank": pl.Int64,
        "start": pl.Date,
        "trough": pl.Date,
        "end": pl.Date,
        "depth": pl.Float64,
        "days": pl.Int64,
        "recovery_days": pl.Int64,
        "series": pl.String,
    }
    return conform("drawdowns", pl.DataFrame(rows, schema=schema))


def _segments(raw: pl.DataFrame) -> pl.DataFrame:
    frame = raw.drop([c for c in TRADE_DROP if c in raw.columns])
    return frame.with_columns(
        pl.when(pl.col("position") >= 0)
        .then(pl.lit("long"))
        .otherwise(pl.lit("short"))
        .alias("side"),
        pl.when(pl.col("entry_date").is_null())
        .then(pl.lit("pending"))
        .when(pl.col("exit_date").is_null())
        .then(pl.lit("open"))
        .otherwise(pl.lit("closed"))
        .alias("status"),
    )


def _trades(
    report: Any, input_df: pl.DataFrame | None
) -> tuple[pl.DataFrame | None, pl.DataFrame | None, str]:
    """Round trips, the engine's segments when they differ, and the granularity."""
    raw = getattr(report, "trades", None)
    if not isinstance(raw, pl.DataFrame) or raw.is_empty():
        return None, None, "segment_merged"
    raw = viz._dated_trades(raw)
    legacy = getattr(report, "execution_model", "legacy") == "legacy"
    merged = viz._merge_rolled_trades(raw) if legacy else raw
    if "n_segments" not in merged.columns:
        merged = merged.with_columns(pl.lit(1).alias("n_segments"))
    if "entry_raw_price" in merged.columns:
        merged = viz._with_limit_flags(merged, input_df)
        merged = viz._with_limit_kinds(merged, input_df)
    unknown = [
        c
        for c in ("lim_entry", "lim_exit", "entry_kind", "exit_kind")
        if c in merged.columns and merged[c].null_count() == merged.height
    ]
    trades = conform(
        "trades",
        _segments(merged.drop(unknown)).with_columns(
            pl.col("n_segments").fill_null(1), pl.lit(0).alias("trade_id")
        ),
    ).with_row_index("_i")
    trades = trades.with_columns(pl.col("_i").cast(pl.Int64).alias("trade_id")).drop("_i")
    segments = None
    if legacy and merged.height != raw.height:
        segments = conform("trade_segments", _segments(raw))
    return trades, segments, "segment_merged" if legacy else "episode"


def _day_index(
    curve: pl.DataFrame, trades: pl.DataFrame | None, ledger: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    index = curve.select("date", pl.col("ret").fill_null(0.0))
    counts = [("entry_date", "n_entries"), ("exit_date", "n_exits")]
    for column, name in counts:
        per_day = pl.DataFrame(schema={"date": pl.Date, name: pl.Int64})
        if trades is not None:
            per_day = (
                trades.drop_nulls(column)
                .group_by(pl.col(column).alias("date"))
                .agg(pl.len().cast(pl.Int64).alias(name))
            )
        index = index.join(per_day, on="date", how="left").with_columns(pl.col(name).fill_null(0))
    for table, name in (
        ("fills", "n_fills"),
        ("blocks", "n_blocks"),
        ("cashflows", "n_cashflows"),
    ):
        if table in ledger:
            per_day = ledger[table].group_by("date").agg(pl.len().cast(pl.Int64).alias(name))
            index = index.join(per_day, on="date", how="left").with_columns(
                pl.col(name).fill_null(0)
            )
    daily = ledger.get("ledger_daily")
    if daily is not None:
        wanted = []
        if "traded_notional" in daily.columns:
            wanted.append(pl.col("traded_notional"))
        costs = M._ledger_costs(daily)
        if costs:
            wanted.append(pl.sum_horizontal(costs).alias("cost_total"))
        if wanted:
            index = index.join(daily.select("date", *wanted), on="date", how="left")
    return conform("day_index", index)


def _symbol_index(
    trades: pl.DataFrame | None,
    ledger: dict[str, pl.DataFrame],
    names: dict[str, str] | None,
) -> pl.DataFrame | None:
    if trades is None:
        return None
    closed = M.closed_trades(trades)
    index = (
        trades.select("symbol")
        .unique()
        .join(
            closed.group_by("symbol").agg(
                pl.len().cast(pl.Int64).alias("n_trades"),
                (pl.col("ret") > 0).mean().alias("win_rate"),
                (pl.col("ret") * pl.col("weight").abs()).sum().alias("contrib_proxy"),
            ),
            on="symbol",
            how="left",
        )
        .with_columns(pl.col("n_trades").fill_null(0), pl.col("contrib_proxy").fill_null(0.0))
    )
    if names:
        frame = pl.DataFrame(
            {"symbol": list(names), "name": list(names.values())},
            schema={"symbol": pl.String, "name": pl.String},
        )
        index = index.join(frame, on="symbol", how="left")
    fills = ledger.get("fills")
    if fills is not None:
        aggs = [pl.len().cast(pl.Int64).alias("n_fills")]
        if "notional" in fills.columns:
            aggs.append(pl.col("notional").abs().sum().alias("traded_notional"))
        if "cost" in fills.columns:
            aggs.append(pl.col("cost").sum().alias("cost_total"))
        index = index.join(fills.group_by("symbol").agg(aggs), on="symbol", how="left")
    blocks = ledger.get("blocks")
    if blocks is not None:
        counts = blocks.group_by("symbol").agg(pl.len().cast(pl.Int64).alias("n_blocks"))
        index = index.join(counts, on="symbol", how="left")
    frame = conform("symbol_index", index)
    return frame.sort(["contrib_proxy", "symbol"], maintain_order=True)


def _metrics_table(metrics: dict[str, Any], stats: dict[str, Any], report: Any) -> pl.DataFrame:
    rows = []
    sections: dict[str, list[str]] = {}
    if hasattr(report, "get_metrics"):
        try:
            for section in METRIC_SECTIONS:
                sections[section] = report.get_metrics(sections=[section]).columns
        except (ValueError, TypeError):
            sections = {}
    seen = {key for keys in sections.values() for key in keys}
    sections["extra"] = [key for key in metrics if key not in seen]
    for section, keys in sections.items():
        for key in keys:
            value = M._finite(metrics.get(key))
            if value is not None:
                rows.append((section, key, value, "", "engine"))
    for key, raw in stats.items():
        value = M._finite(raw)
        if value is not None:
            rows.append(("stats", key, value, "", "engine"))
    schema = {
        "section": pl.String,
        "key": pl.String,
        "value": pl.Float64,
        "unit": pl.String,
        "source": pl.String,
    }
    return conform("metrics", pl.DataFrame(rows, schema=schema, orient="row"))


def _table_entry(name: str, frame: pl.DataFrame | None) -> dict[str, Any]:
    table = TABLES[name]
    if frame is None:
        return {"state": "absent", "reason": table.absent_reason}
    entry: dict[str, Any] = {
        "state": "present" if frame.height else "empty",
        "rows": frame.height,
        "file": table.file,
        "sort": list(table.sort),
        "columns": frame.columns,
    }
    missing = [c for c in table.expected if c not in frame.columns]
    if missing and frame.height:
        entry["state"] = "partial"
        entry["columns_missing"] = missing
    return entry


def _config(report: Any) -> dict[str, Any]:
    return {name: getattr(report, name, None) for name in CONFIG_FIELDS}


def _config_hash(config: dict[str, Any]) -> str:
    """Hash of the parameters, never of results, so it is stable across reruns."""
    text = json.dumps(_safe(config), sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _library_version() -> str:
    from polars_backtest._polars_backtest import __version__

    return str(__version__)


def build_bundle(
    report: Any,
    *,
    title: str | None = None,
    input_df: pl.DataFrame | None = None,
    symbol_names: dict[str, str] | None = None,
    fill_scenarios: list[dict[str, Any]] | None = None,
    strict: bool = False,
) -> ReportBundle:
    """Build the in-memory bundle of a report. Nothing touches the disk."""
    ledger = {
        name: conform(name, frame)
        for name in LEDGER_TABLES
        if (frame := _ledger_frame(report, name)) is not None
    }
    daily = ledger.get("ledger_daily")
    curve = _curve(report, daily)
    trades, segments, granularity = _trades(report, input_df)
    stats = viz._stats_row(report)
    metrics = viz._metrics_row(report)
    trade_summary = M.trade_structure(
        trades, getattr(report, "fee_ratio", None), getattr(report, "tax_ratio", None)
    )
    if metrics.get("buyHigh") is None and "buy_high_ratio" in trade_summary:
        metrics["buyHigh"] = trade_summary["buy_high_ratio"]
    if metrics.get("sellLow") is None and "sell_low_ratio" in trade_summary:
        metrics["sellLow"] = trade_summary["sell_low_ratio"]

    tables: dict[str, pl.DataFrame] = {"curve": curve, **ledger}
    tables["monthly"] = conform("monthly", _periods(curve, ["year", "month"]))
    tables["yearly"] = conform(
        "yearly", _periods(curve, ["year"]).join(_year_drawdowns(curve), on="year", how="left")
    )
    tables["drawdowns"] = _drawdowns(curve)
    tables["metrics"] = _metrics_table(metrics, stats, report)
    tables["day_index"] = _day_index(curve, trades, ledger)
    symbol_index = _symbol_index(trades, ledger, symbol_names)
    if trades is not None:
        tables["trades"] = trades
        tables["symbol_index"] = symbol_index
    if segments is not None:
        tables["trade_segments"] = segments
    if symbol_names:
        tables["symbols"] = conform(
            "symbols",
            pl.DataFrame(
                {"symbol": list(symbol_names), "name": list(symbol_names.values())},
                schema={"symbol": pl.String, "name": pl.String},
            ),
        )
    if fill_scenarios:
        tables["fill_scenarios"] = conform(
            "fill_scenarios", pl.DataFrame(_safe(fill_scenarios), infer_schema_length=None)
        )

    checks = M.checks(curve, daily, ledger.get("fills"))
    failed = [c["id"] for c in checks if c["status"] == "fail"]
    if strict and failed:
        raise BundleCheckError(f"ledger does not reconcile: {', '.join(failed)}")

    rf = stats.get("rf", 0.02)
    config = _config(report)
    run = {
        "title": title or "Backtest Report",
        "config_hash": _config_hash(config),
        "execution_model": getattr(report, "execution_model", "legacy"),
        "start": curve["date"][0],
        "end": curve["date"][-1],
        "n_days": curve.height,
        "riskfree_rate": rf,
        "initial_cash": getattr(report, "initial_cash", None),
    }
    entries = {name: _table_entry(name, tables.get(name)) for name in TABLES}
    coverage = {name: entry["state"] for name, entry in entries.items()}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "library_version": _library_version(),
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run": _safe(run),
        "config": _safe(config),
        "params": viz._param_chips(report),
        "trade_granularity": granularity,
        "tables": entries,
        "checks": checks,
    }
    summary = build_summary(
        _safe(run), tables, metrics, trade_summary, checks, coverage, rf
    )
    return ReportBundle(manifest, summary, {k: v.lazy() for k, v in tables.items()})
