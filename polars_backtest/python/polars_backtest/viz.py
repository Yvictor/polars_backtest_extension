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


def _trades_payload(report: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
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

    sampled = trades
    if total > MAX_EMBEDDED_TRADES:
        step = total / MAX_EMBEDDED_TRADES
        sampled = trades[[int(i * step) for i in range(MAX_EMBEDDED_TRADES)]]
    cols = sampled.select(
        pl.col("stock_id"),
        pl.col("entry_date"),
        pl.col("exit_date"),
        pl.col("return").alias("ret"),
        pl.col("mae") if "mae" in sampled.columns else pl.lit(None).alias("mae"),
        pl.col("pdays") if "pdays" in sampled.columns else pl.lit(None).alias("pdays"),
    )
    payload = {
        "stock": cols.get_column("stock_id").to_list(),
        "entry": cols.get_column("entry_date").to_list(),
        "exit": cols.get_column("exit_date").to_list(),
        "ret": cols.get_column("ret").to_list(),
        "mae": cols.get_column("mae").to_list(),
        "pdays": cols.get_column("pdays").to_list(),
        "sampled": total > MAX_EMBEDDED_TRADES,
        "total": total,
    }
    return payload, summary


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


def report_data(report: Any, *, title: str = "Backtest Report") -> dict[str, Any]:
    """Extract a JSON-serializable payload from a report object.

    Works with both the long-format ``BacktestReport`` and the wide-format
    ``Report`` (anything exposing ``creturn``/``stats``; other sections are
    included when available). This payload is also the data contract for
    external dashboards/services.
    """
    curve = _daily_curve(report)
    stats = _stats_row(report)
    monthly: dict[str, Any] = {}
    if hasattr(report, "get_monthly_stats"):
        monthly_df = report.get_monthly_stats()
        if isinstance(monthly_df, pl.DataFrame) and monthly_df.height:
            monthly = monthly_df.to_dicts()[0]
    trades, trade_summary = _trades_payload(report)

    payload = {
        "title": title,
        "params": _param_chips(report),
        "stats": stats,
        "trade_summary": trade_summary,
        "daily": {
            "dates": curve.get_column("date").to_list(),
            "creturn": curve.get_column("creturn").to_list(),
            "benchmark": _benchmark_series(report, curve),
        },
        "return_table": _return_table(report),
        "trades": trades,
        "stat_groups": _stat_groups(stats, monthly, trade_summary),
    }
    return _clean(payload)


def report_html(report: Any, *, title: str = "Backtest Report") -> str:
    """Render a report to a self-contained HTML string."""
    from polars_backtest._polars_backtest import __version__

    payload = report_data(report, title=title)
    return (
        TEMPLATE.replace("__TITLE__", _escape(title))
        .replace("__VERSION__", str(__version__))
        .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    )


def save_html(report: Any, path: str | Path, *, title: str = "Backtest Report") -> Path:
    """Render a report and write it to ``path``. Returns the written path."""
    out = Path(path)
    out.write_text(report_html(report, title=title), encoding="utf-8")
    return out


def show(report: Any, *, title: str = "Backtest Report", height: int = 1400) -> None:
    """Display the report: inline iframe in notebooks, browser tab otherwise."""
    html = report_html(report, title=title)
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
