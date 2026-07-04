"""Interactive HTML report for backtest results — zero extra dependencies.

Renders a self-contained HTML dashboard (inline SVG + vanilla JS, no external
assets or CDN) from a :class:`BacktestReport` (long format) or the wide-format
``Report``. The file works offline, in a browser tab, or inside a notebook.

Example:
    >>> import polars_backtest as pl_bt
    >>> report = df.bt.backtest_with_report(resample="M")
    >>> pl_bt.viz.save_html(report, "report.html")
    >>> pl_bt.viz.show(report)  # opens browser / renders inline in notebooks
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import tempfile
import webbrowser
from pathlib import Path
from typing import Any

import polars as pl

from polars_backtest._viz_template import TEMPLATE

__all__ = ["report_data", "report_html", "save_html", "show"]

# Trades beyond this count are evenly subsampled before embedding, to keep the
# HTML file size and scatter-plot DOM manageable.
MAX_EMBEDDED_TRADES = 8000


def _clean(obj: Any) -> Any:
    """Make a payload JSON-safe: non-finite floats -> None, dates -> ISO strings."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (_dt.date, _dt.datetime)):
        return obj.isoformat()[:10]
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _stats_row(report: Any) -> dict[str, Any]:
    stats = getattr(report, "stats", None)
    if stats is None and hasattr(report, "get_stats"):
        stats = report.get_stats()
    if isinstance(stats, pl.DataFrame):
        return stats.to_dicts()[0] if stats.height else {}
    return dict(stats) if stats else {}


def _daily_curve(report: Any) -> pl.DataFrame:
    if hasattr(report, "daily_creturn"):
        curve = report.daily_creturn()
    else:
        curve = report.creturn
    if not isinstance(curve, pl.DataFrame) or curve.is_empty():
        raise ValueError("report has no equity curve (empty creturn)")
    return curve.select(
        pl.col("date").cast(pl.Date),
        pl.col("creturn").cast(pl.Float64),
    ).sort("date")


def _benchmark_series(report: Any, curve: pl.DataFrame) -> list[float | None] | None:
    bench = getattr(report, "benchmark", None)
    if not isinstance(bench, pl.DataFrame) or bench.is_empty():
        return None
    if "date" not in bench.columns or "creturn" not in bench.columns:
        return None
    joined = curve.select("date").join(
        bench.select(pl.col("date").cast(pl.Date), pl.col("creturn").alias("bench")),
        on="date",
        how="left",
    )
    return joined.get_column("bench").to_list()


def _return_table(report: Any) -> list[dict[str, Any]]:
    if not hasattr(report, "get_return_table"):
        return []
    table = report.get_return_table()
    if not isinstance(table, pl.DataFrame) or table.is_empty():
        return []
    month_cols = [c for c in table.columns if c != "year"]
    rows = []
    for row in table.sort("year").to_dicts():
        months = [row.get(str(m)) for m in range(1, 13)] if set(month_cols) >= {"1"} else [
            row[c] for c in month_cols
        ]
        rows.append({"year": row["year"], "months": months})
    return rows


_LIMIT_EPS = 1e-6


def _round4(values: list) -> list:
    return [None if v is None else round(v, 4) for v in values]


def _with_limit_flags(trades: pl.DataFrame, input_df: pl.DataFrame | None) -> pl.DataFrame:
    """Flag trades that entered at limit-up / exited at limit-down (direction-aware).

    Uses raw prices with the same 1e-6 relative tolerance as the engine's
    liquidity metrics. Requires `limit_up`/`limit_down` columns in input_df;
    missing data leaves the flags null.
    """
    trades = trades.with_columns(
        pl.lit(None, dtype=pl.Boolean).alias("lim_entry"),
        pl.lit(None, dtype=pl.Boolean).alias("lim_exit"),
    )
    if input_df is None:
        return trades
    have = {c for c in ("limit_up", "limit_down") if c in input_df.columns}
    if not have or "date" not in input_df.columns or "symbol" not in input_df.columns:
        return trades
    limits = input_df.select(
        pl.col("date").cast(pl.Date),
        pl.col("symbol"),
        *[pl.col(c).cast(pl.Float64) for c in sorted(have)],
    ).unique(subset=["date", "symbol"], keep="first")

    def _leg(df: pl.DataFrame, date_col: str, suffix: str) -> pl.DataFrame:
        return df.join(
            limits.rename({"date": date_col, "symbol": "stock_id"}),
            on=[date_col, "stock_id"],
            how="left",
            suffix=suffix,
        )

    trades = _leg(trades, "entry_date", "_e")
    trades = _leg(trades, "exit_date", "_x")
    lu_e = "limit_up" if "limit_up" in trades.columns else None
    ld_e = "limit_down" if "limit_down" in trades.columns else None
    lu_x = f"{lu_e}_x" if lu_e and f"{lu_e}_x" in trades.columns else lu_e
    ld_x = f"{ld_e}_x" if ld_e and f"{ld_e}_x" in trades.columns else ld_e
    is_long = pl.col("position") >= 0
    at_up = lambda px, lim: pl.col(px) >= pl.col(lim) * (1 - _LIMIT_EPS)  # noqa: E731
    at_dn = lambda px, lim: pl.col(px) <= pl.col(lim) * (1 + _LIMIT_EPS)  # noqa: E731
    unknown = pl.lit(None, dtype=pl.Boolean)
    if lu_e or ld_e:
        # A side whose limit column is absent is UNKNOWN (null), never False —
        # otherwise one-sided data would fabricate passing 0% ratios.
        # adverse entry: long buys at limit-up, short sells at limit-down
        entry = pl.when(is_long).then(
            at_up("entry_raw_price", lu_e) if lu_e else unknown
        ).otherwise(
            at_dn("entry_raw_price", ld_e) if ld_e else unknown
        )
        # adverse exit: long sells at limit-down, short covers at limit-up
        exit_ = pl.when(is_long).then(
            at_dn("exit_raw_price", ld_x) if ld_x else unknown
        ).otherwise(
            at_up("exit_raw_price", lu_x) if lu_x else unknown
        )
        trades = trades.with_columns(entry.alias("lim_entry"), exit_.alias("lim_exit"))
    drop = [c for c in trades.columns if c.endswith("_x") or c in ("limit_up", "limit_down")]
    return trades.drop([c for c in drop if c not in ("lim_entry", "lim_exit")])


def _with_limit_kinds(trades: pl.DataFrame, input_df: pl.DataFrame | None) -> pl.DataFrame:
    """Add entry_kind/exit_kind ("locked" 一字 / "touched" 盤中 / null) when OHLC exists."""
    if (
        input_df is None
        or not {"open", "high", "low", "date", "symbol"} <= set(input_df.columns)
        or not ({"limit_up", "limit_down"} & set(input_df.columns))
    ):
        return trades.with_columns(
            pl.lit(None, dtype=pl.String).alias("entry_kind"),
            pl.lit(None, dtype=pl.String).alias("exit_kind"),
        )
    from polars_backtest.liquidity import classify_limit_trades

    return classify_limit_trades(trades, input_df)


def _trades_payload(
    report: Any,
    input_df: pl.DataFrame | None = None,
    symbol_names: dict[str, str] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    trades = getattr(report, "trades", None)
    if not isinstance(trades, pl.DataFrame) or trades.is_empty():
        return None, {}
    total = trades.height
    closed = trades.filter(pl.col("return").is_not_null() & pl.col("return").is_finite())
    rets = closed.get_column("return")
    summary: dict[str, Any] = {"n": total}
    if closed.height:
        wins = rets.filter(rets > 0)
        losses = rets.filter(rets < 0)
        loss_sum = abs(losses.sum() or 0.0)
        summary.update(
            win_rate=wins.len() / closed.height,
            profit_factor=(wins.sum() or 0.0) / loss_sum if loss_sum > 0 else None,
            expectancy=rets.mean(),
            avg_days=closed.get_column("pdays").mean() if "pdays" in closed.columns else None,
            avg_mae=closed.get_column("mae").mean() if "mae" in closed.columns else None,
        )

    if closed.height:
        # Long/short decomposition and return-concentration statistics.
        # Contribution proxy = return x |position| (portfolio-weighted trade P&L).
        contrib = closed.select(
            (pl.col("return") * pl.col("position").abs()).alias("c"),
            (pl.col("position") >= 0).alias("is_long"),
            pl.col("return").alias("r"),
        )
        total_c = contrib.get_column("c").sum() or 0.0
        sides = {}
        for side, flag in (("long", True), ("short", False)):
            grp = contrib.filter(pl.col("is_long") == flag)
            if grp.height:
                rets_g = grp.get_column("r")
                sides[side] = {
                    "n": grp.height,
                    "win_rate": (rets_g > 0).sum() / grp.height,
                    "avg_ret": rets_g.mean(),
                    "contrib": grp.get_column("c").sum(),
                }
        summary["sides"] = sides
        if abs(total_c) > 0:
            top = contrib.sort("c", descending=True).get_column("c")
            summary["top10_contrib_ratio"] = top.head(10).sum() / total_c
            summary["total_contrib"] = total_c

    # Annual turnover estimate and cost drag: each entry deploys |position| of
    # NAV; a round trip pays fee twice plus tax once (TW long convention).
    if closed.height and "pdays" in closed.columns:
        entered_pos = trades.filter(pl.col("entry_date").is_not_null())
        if entered_pos.height:
            dmin = entered_pos.get_column("entry_date").min()
            dmax = (
                trades.get_column("exit_date").max()
                or entered_pos.get_column("entry_date").max()
            )
            years = max(((dmax - dmin).days or 1) / 365.25, 1 / 365.25)
            turnover = (entered_pos.get_column("position").abs().sum() or 0.0) / years
            summary["annual_turnover"] = turnover
            fee = getattr(report, "fee_ratio", None)
            tax = getattr(report, "tax_ratio", None)
            if fee is not None and tax is not None:
                summary["cost_drag_annual"] = turnover * (2 * fee + tax)

    if "entry_raw_price" in trades.columns:
        trades = _with_limit_flags(trades, input_df)
        trades = _with_limit_kinds(trades, input_df)
        entered = trades.filter(pl.col("entry_date").is_not_null())
        exited = trades.filter(pl.col("exit_date").is_not_null())
        # Denominators count only trades whose flag is KNOWN (non-null) — same
        # population the engine uses (trades with a joinable limit price), and
        # one-sided limit data cannot fabricate a passing 0% ratio.
        lim_e = entered.get_column("lim_entry").drop_nulls() if entered.height else None
        lim_x = exited.get_column("lim_exit").drop_nulls() if exited.height else None
        if lim_e is not None and lim_e.len():
            summary["buy_high_n"] = int(lim_e.sum() or 0)
            summary["buy_high_ratio"] = (lim_e.sum() or 0) / lim_e.len()
        if lim_x is not None and lim_x.len():
            summary["sell_low_n"] = int(lim_x.sum() or 0)
            summary["sell_low_ratio"] = (lim_x.sum() or 0) / lim_x.len()
        # How much of the strategy's P&L rides on limit-up entries — the honest
        # answer to "would this survive not getting filled at limit-up?"
        tot = None
        closed_flagged = trades.filter(
            pl.col("exit_date").is_not_null() & pl.col("return").is_not_null()
        )
        flags_known = (
            closed_flagged.height
            and closed_flagged.get_column("lim_entry").null_count() < closed_flagged.height
        )
        if flags_known:
            call = closed_flagged.select(
                (pl.col("return") * pl.col("position").abs()).alias("c"),
                pl.col("lim_entry").fill_null(False).alias("f"),
            )
            tot = call.get_column("c").sum() or 0.0
            bh = call.filter(pl.col("f")).get_column("c").sum() or 0.0
            if abs(tot) > 0:
                summary["buy_high_contrib"] = bh
                summary["buy_high_contrib_ratio"] = bh / tot
        if "entry_kind" in trades.columns:
            kinds = trades.filter(pl.col("entry_date").is_not_null())
            if kinds.height and kinds.get_column("entry_kind").null_count() < kinds.height:
                locked = kinds.filter(pl.col("entry_kind") == "locked")
                touched = kinds.filter(pl.col("entry_kind") == "touched")
                summary["entry_locked_n"] = locked.height
                summary["entry_touched_n"] = touched.height
                closed_locked = locked.filter(pl.col("return").is_not_null())
                if closed_locked.height and tot is not None and abs(tot) > 0:
                    locked_c = closed_locked.select(
                        (pl.col("return") * pl.col("position").abs()).sum()
                    ).item() or 0.0
                    summary["entry_locked_contrib_ratio"] = locked_c / tot
            exits = trades.filter(pl.col("exit_date").is_not_null())
            if exits.height and exits.get_column("exit_kind").null_count() < exits.height:
                summary["exit_locked_n"] = exits.filter(pl.col("exit_kind") == "locked").height
                summary["exit_touched_n"] = exits.filter(pl.col("exit_kind") == "touched").height

    sampled = trades
    if total > MAX_EMBEDDED_TRADES:
        step = total / MAX_EMBEDDED_TRADES
        sampled = trades[[int(i * step) for i in range(MAX_EMBEDDED_TRADES)]]
    def _opt(name: str, alias: str | None = None) -> pl.Expr:
        alias = alias or name
        if name in sampled.columns:
            return pl.col(name).alias(alias)
        return pl.lit(None).alias(alias)

    cols = sampled.select(
        pl.col("stock_id"),
        pl.col("entry_date"),
        pl.col("exit_date"),
        pl.col("return").alias("ret"),
        _opt("mae"),
        _opt("pdays"),
        _opt("position", "pos"),
        _opt("entry_raw_price", "entry_px"),
        _opt("exit_raw_price", "exit_px"),
        _opt("gmfe"),
        _opt("bmfe"),
        _opt("mdd"),
        _opt("lim_entry"),
        _opt("lim_exit"),
        _opt("entry_kind"),
        _opt("exit_kind"),
    )
    payload = {
        "stock": cols.get_column("stock_id").to_list(),
        "entry": cols.get_column("entry_date").to_list(),
        "exit": cols.get_column("exit_date").to_list(),
        "ret": cols.get_column("ret").to_list(),
        "mae": cols.get_column("mae").to_list(),
        "pdays": cols.get_column("pdays").to_list(),
        "pos": cols.get_column("pos").to_list(),
        "entry_px": _round4(cols.get_column("entry_px").to_list()),
        "exit_px": _round4(cols.get_column("exit_px").to_list()),
        "gmfe": cols.get_column("gmfe").to_list(),
        "bmfe": cols.get_column("bmfe").to_list(),
        "mdd": cols.get_column("mdd").to_list(),
        "lim_entry": cols.get_column("lim_entry").to_list(),
        "lim_exit": cols.get_column("lim_exit").to_list(),
        "entry_kind": cols.get_column("entry_kind").to_list(),
        "exit_kind": cols.get_column("exit_kind").to_list(),
        "sampled": total > MAX_EMBEDDED_TRADES,
        "total": total,
    }
    if symbol_names:
        payload["name"] = [symbol_names.get(sym) for sym in payload["stock"]]
    return payload, summary


def _drawdown_episodes(
    dates: list, creturn: list[float], top_n: int = 10
) -> list[dict[str, Any]]:
    """Extract drawdown episodes (peak -> trough -> recovery), deepest first."""
    episodes: list[dict[str, Any]] = []
    pairs = [(d, v) for d, v in zip(dates, creturn) if v is not None and v > 0]
    if not pairs:
        return episodes
    peak = pairs[0][1]
    peak_date = pairs[0][0]
    cur: dict[str, Any] | None = None
    for d, v in pairs:
        if v >= peak:
            if cur is not None:
                cur["end"] = d
                episodes.append(cur)
                cur = None
            peak = v
            peak_date = d
        else:
            dd = v / peak - 1
            if cur is None:
                cur = {"start": peak_date, "trough": d, "depth": dd, "end": None}
            elif dd < cur["depth"]:
                cur["depth"] = dd
                cur["trough"] = d
    if cur is not None:
        episodes.append(cur)  # ongoing, end stays None
    last = pairs[-1][0]
    for ep in episodes:
        end = ep["end"] or last
        ep["days"] = (end - ep["start"]).days
        ep["recovery_days"] = (ep["end"] - ep["trough"]).days if ep["end"] else None
    episodes.sort(key=lambda e: e["depth"])
    return episodes[:top_n]


def _metrics_row(report: Any) -> dict[str, Any]:
    if not hasattr(report, "get_metrics"):
        return {}
    try:
        metrics = report.get_metrics()
    except ValueError:
        # engine raises "Insufficient data for metrics" on <2-point curves;
        # the report can still render from stats/creturn
        return {}
    if not isinstance(metrics, pl.DataFrame) or not metrics.height:
        return {}
    row = metrics.to_dicts()[0]
    # Stricter capacity estimates (min-leg / ADV-based), when the engine has them
    if row.get("capacity") is not None and hasattr(report, "capacity"):
        for method, key in (("min_leg", "capacityMinLeg"), ("adv", "capacityAdv")):
            try:
                row[key] = report.capacity(method=method)
            except TypeError:  # older builds without the method= signature
                row[key] = None
    return row


# FinLab-exact quality checks (specs/VIZ_V2_FINLAB_PARITY_SPEC.md §2-3).
# Tuple: (metrics key, zh label, predicate, threshold caption, format, weight).
# Score = round(100 * passed_weight / total_weight); a check whose value is
# missing/non-finite is dropped from the denominator ("corrected" mode — FinLab
# itself fails missing values, which we consider a data gap, not a strategy flaw).
_QUALITY_CHECKS: dict[str, list[tuple]] = {
    "profitability": [
        ("annualReturn", "年度回報", lambda v: v > 0.15, "需 ≥ 15%", "pct", 1),
        ("alpha", "Alpha", lambda v: v > 0.10, "需 ≥ 10%", "pct", 1),
        ("beta", "Beta", lambda v: 0 < v < 0.8, "需介於 0–0.8", "num", 1),
        ("avgNStock", "平均持有", lambda v: v >= 5, "需 ≥ 5 檔", "int", 1),
        ("maxNStock", "最多持有", lambda v: v <= 20, "需 ≤ 20 檔", "int", 1),
    ],
    "risk": [
        ("maxDrawdown", "最大回檔", lambda v: v > -0.30, "需 < 30%", "pct", 1),
        ("avgDrawdown", "平均回檔幅度", lambda v: v > -0.10, "需 < 10%", "pct", 1),
        ("avgDrawdownDays", "平均回檔時間", lambda v: v < 40, "需 < 40 天", "days", 1),
        ("volatility", "波動性", lambda v: v < 0.20, "需 < 20%", "pct", 1),
        ("valueAtRisk", "Value at Risk", lambda v: v > -0.07, "需 < 7%", "pct", 1),
        ("cvalueAtRisk", "Conditional VaR", lambda v: v > -0.10, "需 < 10%", "pct", 1),
    ],
    "ratio": [
        ("sharpeRatio", "夏普值", lambda v: v > 1.3, "需 > 1.3", "num", 1),
        ("sortinoRatio", "Sortino Ratio", lambda v: v > 1.8, "需 > 1.8", "num", 1),
        ("calmarRatio", "Calmar Ratio", lambda v: v > 0.9, "需 > 0.9", "num", 1),
        ("profitFactor", "Profit Factor", lambda v: v > 1.5, "需 > 1.5", "num", 1),
        ("tailRatio", "Tail Ratio", lambda v: v > 1.0, "需 > 1", "num", 1),
    ],
    "winrate": [
        ("winRate", "逐筆交易勝率", lambda v: v > 0.55, "需 ≥ 55%", "pct", 1),
        ("m12WinRate", "12個月勝大盤", lambda v: v > 0.70, "需 ≥ 70%", "pct", 1),
        ("expectancy", "期望值", lambda v: v > 0.02, "需 ≥ 2%", "pct", 1),
        ("mae", "最大不利偏移", lambda v: v > -0.10, "需 < 10%", "pct", 1),
        ("mfe", "最大有利偏移", lambda v: v > 0.10, "需 ≥ 10%", "pct", 1),
    ],
    "liquidity": [
        ("capacity", "胃納量", lambda v: v > 500_000, "需 > 50 萬", "wan", 11),
        ("buyHigh", "買在漲停", lambda v: v < 0.05, "需 < 5%", "pct", 1),
        ("sellLow", "賣在跌停", lambda v: v < 0.05, "需 < 5%", "pct", 1),
        ("disposalStockRatio", "處置股", lambda v: v < 0.05, "需 < 5%", "pct", 1),
        ("warningStockRatio", "警示股", lambda v: v < 0.05, "需 < 5%", "pct", 1),
        ("fullDeliveryStockRatio", "全額交割股", lambda v: v < 0.05, "需 < 5%", "pct", 1),
    ],
}

_QUALITY_LABELS = {
    "profitability": "獲利",
    "risk": "風險",
    "ratio": "報酬比",
    "winrate": "勝率",
    "liquidity": "流動性",
}


def _quality(metrics: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the FinLab quality checks and 0-100 dimension scores."""

    def _num(v: Any) -> float | None:
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
        return None

    out: dict[str, Any] = {}
    for dim, checks in _QUALITY_CHECKS.items():
        results = []
        passed_w = total_w = 0
        for key, label, pred, caption, fmt, weight in checks:
            value = _num(metrics.get(key))
            ok: bool | None = None
            if value is not None:
                ok = bool(pred(value))
                total_w += weight
                if ok:
                    passed_w += weight
            results.append({
                "key": key, "label": label, "value": value,
                "pass": ok, "caption": caption, "fmt": fmt,
            })
        out[dim] = {
            "label": _QUALITY_LABELS[dim],
            "score": round(100 * passed_w / total_w) if total_w else None,
            "checks": results,
        }
    return out


def _param_chips(report: Any) -> list[str]:
    chips = []

    def _get(name: str) -> Any:
        return getattr(report, name, None)

    if _get("resample"):
        chips.append(f"resample {_get('resample')}")
    if _get("fee_ratio") is not None:
        chips.append(f"fee {_get('fee_ratio') * 100:.4g}%")
    if _get("tax_ratio") is not None:
        chips.append(f"tax {_get('tax_ratio') * 100:.4g}%")
    for name, label in (("stop_loss", "SL"), ("take_profit", "TP"), ("trail_stop", "TS")):
        value = _get(name)
        disabled = (
            value is None or not math.isfinite(value)
            or (name == "stop_loss" and value >= 1.0)
        )
        if not disabled:
            chips.append(f"{label} {value * 100:.4g}%")
    if _get("trade_at"):
        chips.append(f"trade@{_get('trade_at')}")
    return chips


def _fmt_pct(v: Any, digits: int = 2) -> str:
    return "–" if v is None else f"{v * 100:.{digits}f}%"


def _fmt_num(v: Any, digits: int = 2) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def _stat_groups(
    stats: dict[str, Any], monthly: dict[str, Any], trade_summary: dict[str, Any]
) -> list[list[list[str]]]:
    perf = [
        ["Total return", _fmt_pct(stats.get("total_return"))],
        ["CAGR", _fmt_pct(stats.get("cagr"))],
        ["Max drawdown", _fmt_pct(stats.get("max_drawdown"))],
        ["Avg drawdown", _fmt_pct(stats.get("avg_drawdown"))],
        ["Best day", _fmt_pct(stats.get("best_day"))],
        ["Worst day", _fmt_pct(stats.get("worst_day"))],
        ["Daily vol (ann.)", _fmt_pct(stats.get("daily_vol"))],
    ]
    ratios = [
        ["Sharpe (daily)", _fmt_num(stats.get("daily_sharpe"))],
        ["Sortino (daily)", _fmt_num(stats.get("daily_sortino"))],
        ["Calmar", _fmt_num(stats.get("calmar"))],
        ["Win ratio", _fmt_pct(stats.get("win_ratio"), 1)],
        ["Sharpe (monthly)", _fmt_num(monthly.get("monthly_sharpe"))],
        ["Best month", _fmt_pct(monthly.get("best_month"))],
        ["Worst month", _fmt_pct(monthly.get("worst_month"))],
    ]
    trades = [
        ["Trades", str(trade_summary.get("n", "–"))],
        ["Trade win rate", _fmt_pct(trade_summary.get("win_rate"), 1)],
        ["Profit factor", _fmt_num(trade_summary.get("profit_factor"))],
        ["Expectancy", _fmt_pct(trade_summary.get("expectancy"))],
        ["Avg MAE", _fmt_pct(trade_summary.get("avg_mae"))],
        ["Avg holding days", _fmt_num(trade_summary.get("avg_days"), 1)],
        ["Risk-free rate", _fmt_pct(stats.get("rf"), 1)],
    ]
    return [perf, ratios, trades]


def report_data(
    report: Any,
    *,
    title: str = "Backtest Report",
    input_df: pl.DataFrame | None = None,
    symbol_names: dict[str, str] | None = None,
    fill_scenarios: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract a JSON-serializable payload from a report object.

    Works with both the long-format ``BacktestReport`` and the wide-format
    ``Report`` (anything exposing ``creturn``/``stats``; other sections are
    included when available). This payload is also the data contract for
    external dashboards/services (``schema`` versions it).

    Args:
        input_df: the backtest input frame; when it carries ``limit_up``/
            ``limit_down`` columns, per-trade limit-hit flags (漲停買不到 /
            跌停賣不掉) are derived from it.
        symbol_names: optional ``{symbol: display name}`` mapping for the
            trade table.

    Note: ``stat_groups`` contains pre-formatted display strings for the
    embedded template and is NOT part of the stable contract — external
    consumers should format from ``stats``/``metrics`` themselves.
    """
    from polars_backtest._polars_backtest import __version__

    curve = _daily_curve(report)
    stats = _stats_row(report)
    monthly: dict[str, Any] = {}
    if hasattr(report, "get_monthly_stats"):
        monthly_df = report.get_monthly_stats()
        if isinstance(monthly_df, pl.DataFrame) and monthly_df.height:
            monthly = monthly_df.to_dicts()[0]
    trades, trade_summary = _trades_payload(report, input_df, symbol_names)

    dates = curve.get_column("date").to_list()
    creturn = curve.get_column("creturn").to_list()
    benchmark = _benchmark_series(report, curve)
    bench_episodes = None
    if benchmark is not None:
        pairs = [(d, v) for d, v in zip(dates, benchmark) if v is not None]
        if pairs:
            bench_episodes = _drawdown_episodes([p[0] for p in pairs], [p[1] for p in pairs])

    quality_metrics = dict(payload_metrics := _metrics_row(report))
    # limit-hit ratios derived from input_df fill in for missing engine metrics
    if quality_metrics.get("buyHigh") is None and "buy_high_ratio" in trade_summary:
        quality_metrics["buyHigh"] = trade_summary["buy_high_ratio"]
    if quality_metrics.get("sellLow") is None and "sell_low_ratio" in trade_summary:
        quality_metrics["sellLow"] = trade_summary["sell_low_ratio"]

    payload = {
        "schema": 3,
        "version": str(__version__),
        "title": title,
        "params": _param_chips(report),
        "stats": stats,
        "metrics": payload_metrics,
        "quality": _quality(quality_metrics),
        "trade_summary": trade_summary,
        "daily": {
            "dates": dates,
            "creturn": creturn,
            "benchmark": benchmark,
            # multi-strategy-ready shape; "creturn" above is the compat alias
            "series": [{"name": "strategy", "creturn": creturn}],
        },
        "return_table": _return_table(report),
        "dd_episodes": _drawdown_episodes(dates, creturn),
        "benchmark_dd_episodes": bench_episodes,
        "trades": trades,
        "fill_scenarios": fill_scenarios,
        "stat_groups": _stat_groups(stats, monthly, trade_summary),
    }
    return _clean(payload)


def report_html(
    report: Any,
    *,
    title: str = "Backtest Report",
    input_df: pl.DataFrame | None = None,
    symbol_names: dict[str, str] | None = None,
    fill_scenarios: list[dict[str, Any]] | None = None,
) -> str:
    """Render a report to a self-contained HTML string."""
    from polars_backtest._polars_backtest import __version__

    payload = report_data(
        report,
        title=title,
        input_df=input_df,
        symbol_names=symbol_names,
        fill_scenarios=fill_scenarios,
    )
    # "<" must not appear raw inside the <script> block: "</script>" in any
    # payload string (title, symbol names, ...) would terminate the element at
    # HTML-parse time. \u003c is the JSON-native escape; ensure_ascii already
    # covers U+2028/U+2029.
    payload_json = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    # Each placeholder occurs once in the template and, in template order,
    # before any user-controlled substitution — replace with count=1 so user
    # strings containing a placeholder name are never re-substituted.
    html = TEMPLATE.replace("__TITLE__", _escape(title), 1)
    html = html.replace("__VERSION__", str(__version__), 1)
    return html.replace("__PAYLOAD__", payload_json, 1)


def save_html(
    report: Any,
    path: str | Path,
    *,
    title: str = "Backtest Report",
    input_df: pl.DataFrame | None = None,
    symbol_names: dict[str, str] | None = None,
    fill_scenarios: list[dict[str, Any]] | None = None,
) -> Path:
    """Render a report and write it to ``path``. Returns the written path."""
    out = Path(path)
    out.write_text(
        report_html(
            report,
            title=title,
            input_df=input_df,
            symbol_names=symbol_names,
            fill_scenarios=fill_scenarios,
        ),
        encoding="utf-8",
    )
    return out


def show(
    report: Any,
    *,
    title: str = "Backtest Report",
    height: int = 1400,
    input_df: pl.DataFrame | None = None,
    symbol_names: dict[str, str] | None = None,
) -> None:
    """Display the report: inline iframe in notebooks, browser tab otherwise."""
    html = report_html(report, title=title, input_df=input_df, symbol_names=symbol_names)
    try:  # notebook path
        import IPython
        from IPython.display import HTML, display

        if IPython.get_ipython() is not None:
            escaped = html.replace("&", "&amp;").replace('"', "&quot;")
            display(
                HTML(
                    f'<iframe srcdoc="{escaped}" style="width:100%;height:{height}px;'
                    'border:none;border-radius:8px;"></iframe>'
                )
            )
            return
    except ImportError:
        pass
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", prefix="polars_backtest_", delete=False, encoding="utf-8"
    ) as f:
        f.write(html)
    webbrowser.open(f"file://{f.name}")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
