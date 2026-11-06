"""``summary.json`` and ``SUMMARY.md``.

The summary has fixed keys, so an agent never parses text. The Markdown is generated
from the dict and from nothing else, so the two cannot disagree.
"""

from __future__ import annotations

import json
import math
from typing import Any

import polars as pl

from polars_backtest.report import metrics as M
from polars_backtest.report.schema import SCHEMA_VERSION

MAX_REASONS = 8
WORST_DAYS = 5
STRUCTURE_KEYS = (
    "sides",
    "top10_contrib_ratio",
    "buy_high_n",
    "buy_high_ratio",
    "sell_low_n",
    "sell_low_ratio",
    "buy_high_contrib_ratio",
    "entry_locked_n",
    "entry_touched_n",
    "entry_locked_contrib_ratio",
    "exit_locked_n",
    "exit_touched_n",
)


def _safe(value: Any) -> Any:
    """Make a value strict-JSON: non-finite floats become null, dates become text."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return M.iso(value)


def _benchmark(
    tables: dict[str, pl.DataFrame], metrics: dict[str, Any]
) -> dict[str, Any] | None:
    curve = tables["curve"]
    if "bench_creturn" not in curve.columns:
        return None
    frame = curve.select("date", pl.col("bench_creturn").alias("c")).drop_nulls()
    if frame.height < 2:
        return None
    total = frame["c"][-1] / frame["c"][0] - 1
    years = (frame["date"][-1] - frame["date"][0]).days / 365.25
    return {
        "total_return": total,
        "cagr": M.annualise(1 + total, years),
        "max_drawdown": curve["bench_drawdown"].min(),
        "alpha": metrics.get("alpha"),
        "beta": metrics.get("beta"),
        "m12_win_rate": metrics.get("m12WinRate"),
    }


def _execution(tables: dict[str, pl.DataFrame]) -> dict[str, Any] | None:
    fills, blocks = tables.get("fills"), tables.get("blocks")
    if fills is None and blocks is None:
        return None
    out: dict[str, Any] = {"n_fills": fills.height if fills is not None else None}
    if blocks is not None:
        counts = blocks.group_by("reason").len().sort(["len", "reason"], descending=[True, False])
        out["n_block_sessions"] = blocks.select("date", "symbol").unique().height
        out["blocks_by_reason"] = dict(counts.head(MAX_REASONS).iter_rows())
    return out


def _structure(trade_summary: dict[str, Any]) -> dict[str, Any]:
    """Where the return came from, from every trade rather than the embedded rows."""
    return {key: trade_summary[key] for key in STRUCTURE_KEYS if key in trade_summary}


def _entry_date(curve: pl.DataFrame) -> dict[str, Any] | None:
    entry = M.entry_date_risk(curve)
    if entry is None:
        return None
    entry["level"] = "amber" if entry["worst"] < -0.10 else "green"
    if "bench_creturn" in curve.columns:
        bench = M.entry_date_risk(curve, "bench_creturn")
        if bench is not None:
            entry["bench_worst"] = bench["worst"]
            entry["bench_median"] = bench["median"]
    return entry


def build_summary(
    run: dict[str, Any],
    tables: dict[str, pl.DataFrame],
    metrics: dict[str, Any],
    trade_summary: dict[str, Any],
    checks: list[dict[str, Any]],
    coverage: dict[str, str],
    riskfree_rate: float,
) -> dict[str, Any]:
    curve = tables["curve"]
    ledger = tables.get("ledger_daily")
    performance = M.performance(curve, riskfree_rate)
    cost = M.costs(ledger, curve, trade_summary)
    worst = curve.drop_nulls("ret").sort("ret").head(WORST_DAYS).select("date", "ret")
    segment_win_rate = metrics.get("winRate")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run": {k: run.get(k) for k in ("title", "start", "end", "n_days")},
        "performance": performance,
        "benchmark": _benchmark(tables, metrics),
        "drawdown": M.underwater(curve),
        "entry_date": _entry_date(curve),
        "concentration": M.concentration(tables["monthly"], curve),
        "net_of_costs": M.net_of_costs(curve),
        "limit_fills": M.limit_fills(tables.get("trades")),
        "trades": M.trade_stats(tables.get("trades"), segment_win_rate),
        "structure": _structure(trade_summary),
        "costs": cost,
        "exposure": M.exposure(ledger),
        "turnover": M.turnover(ledger, trade_summary),
        "execution": _execution(tables),
        "flags": M.flags(curve, metrics, trade_summary, cost, performance["cagr"]),
        "worst_days": worst.to_dicts(),
        "coverage": coverage,
        "checks": {
            "pass": sum(c["status"] == "pass" for c in checks),
            "fail": sum(c["status"] == "fail" for c in checks),
        },
    }
    return _safe(summary)


def _pct(value: Any, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _num(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(summary: dict[str, Any]) -> str:
    """A short brief for a reader who has seconds. Everything comes from ``summary``."""
    perf = summary["performance"]
    run = summary["run"]
    dd = summary["drawdown"]
    trades = summary["trades"]
    lines = [
        f"# {run['title']}",
        "",
        f"{run['start']} to {run['end']}, {run['n_days']} days.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Annualised return | {_pct(perf['cagr'])} |",
        f"| Max drawdown | {_pct(perf['max_drawdown'])} |",
        f"| Longest under water | {dd['longest_underwater_days']} days |",
        f"| Sharpe | {_num(perf['sharpe'])} |",
        f"| Round trips | {trades['n_round_trips']} |",
        f"| Win rate (round trip) | {_pct(trades.get('win_rate'))} |",
        f"| Cost drag a year ({summary['costs']['source']}) "
        f"| {_pct(summary['costs'].get('annual_drag'), 2)} |",
    ]
    if summary["benchmark"]:
        lines.append(f"| Benchmark return | {_pct(summary['benchmark']['total_return'])} |")
    lines += ["", "## Flags", ""]
    for flag in summary["flags"]:
        value = flag["value"]
        shown = "n/a" if value is None else f"{value:.4g}"
        lines.append(f"- {flag['code']}: {flag['level']} ({shown})")
    missing = [k for k, state in summary["coverage"].items() if state != "present"]
    lines += ["", "## Coverage", ""]
    lines.append(f"Not complete: {', '.join(missing)}." if missing else "All tables present.")
    if summary["checks"]["fail"]:
        lines.append(f"Reconciliation failures: {summary['checks']['fail']}.")
    lines += [
        "",
        "## Next",
        "",
        "```",
        "python -m polars_backtest.report schema <bundle>",
        "python -m polars_backtest.report sql <bundle> \"select * from trades limit 5\"",
        "```",
        "",
    ]
    return "\n".join(lines)


def to_json(summary: dict[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
