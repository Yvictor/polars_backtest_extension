"""Limit-lock classification and fill-stress scenario backtests.

Taiwan price limits create two very different fill situations:

- 盤中觸及 ("touched"): the stock traded below limit-up at some point that
  day (open < limit or low < limit) — a monitored order could still fill.
- 一字鎖死 ("locked"): open == low == limit-up — buy orders queue all day
  and never fill. This is the class of trades a live portfolio simply
  cannot enter (symmetrically for limit-down exits).

``classify_limit_trades`` splits every at-limit fill into those classes,
and ``limit_stress`` re-runs the backtest with unfillable entries removed
(their weight zeroed at the signal date, so the cash stays undeployed) to
answer: what do CAGR / drawdown / Sharpe become if those fills never happen?
"""

from __future__ import annotations

from typing import Any

import polars as pl

from polars_backtest.namespace import backtest_with_report

__all__ = ["classify_limit_trades", "limit_stress"]

_EPS = 1e-6

_DAY_COLS = ("open", "high", "low", "limit_up", "limit_down", "factor")


def _day_frame(input_df: pl.DataFrame, date_alias: str) -> pl.DataFrame:
    """Per-(date, symbol) raw OHL + limit prices, aliased for a trade-leg join."""
    cols: list[pl.Expr] = [
        pl.col("date").cast(pl.Date).alias(date_alias),
        pl.col("symbol").alias("stock_id"),
    ]
    have_factor = "factor" in input_df.columns
    for name in ("open", "high", "low"):
        if name in input_df.columns:
            expr = pl.col(name).cast(pl.Float64)
            if have_factor:
                # OHLC in adjusted space, limits in raw space -> de-adjust
                expr = expr / pl.col("factor").cast(pl.Float64)
            cols.append(expr.alias(f"{name}_{date_alias}"))
    for name in ("limit_up", "limit_down"):
        if name in input_df.columns:
            cols.append(pl.col(name).cast(pl.Float64).alias(f"{name}_{date_alias}"))
    return input_df.select(cols).unique(subset=[date_alias, "stock_id"], keep="first")


def classify_limit_trades(trades: pl.DataFrame, input_df: pl.DataFrame) -> pl.DataFrame:
    """Add ``entry_kind`` / ``exit_kind`` columns: "locked", "touched", or null.

    Direction-aware: a long entry is adverse at limit-UP, a short entry at
    limit-DOWN; exits are the mirror. "locked" means the whole session traded
    at the limit (no fill chance); "touched" means the price was at the limit
    when the backtest filled, but the session traded through other prices.

    Requires ``input_df`` with date/symbol, open/high/low (adjusted, with a
    ``factor`` column to recover raw prices — or already raw without one)
    and ``limit_up``/``limit_down`` raw price columns.
    """
    required = {"date", "symbol"}
    if not required <= set(input_df.columns):
        raise ValueError("input_df must have 'date' and 'symbol' columns")

    out = trades.join(_day_frame(input_df, "entry_date"), on=["entry_date", "stock_id"], how="left")
    out = out.join(_day_frame(input_df, "exit_date"), on=["exit_date", "stock_id"], how="left")

    is_long = pl.col("position") >= 0

    def at_up(px: str, lim: str) -> pl.Expr:
        return pl.col(px) >= pl.col(lim) * (1 - _EPS)

    def at_dn(px: str, lim: str) -> pl.Expr:
        return pl.col(px) <= pl.col(lim) * (1 + _EPS)

    def kind(at_limit: pl.Expr, locked: pl.Expr) -> pl.Expr:
        return (
            pl.when(at_limit & locked)
            .then(pl.lit("locked"))
            .when(at_limit)
            .then(pl.lit("touched"))
            .otherwise(pl.lit(None, dtype=pl.String))
        )

    exprs = []
    if "limit_up_entry_date" in out.columns and "open_entry_date" in out.columns:
        long_at = at_up("entry_raw_price", "limit_up_entry_date")
        long_locked = at_up("open_entry_date", "limit_up_entry_date") & at_up(
            "low_entry_date", "limit_up_entry_date"
        )
        short_at = at_dn("entry_raw_price", "limit_down_entry_date")
        short_locked = at_dn("open_entry_date", "limit_down_entry_date") & at_dn(
            "high_entry_date", "limit_down_entry_date"
        )
        exprs.append(
            pl.when(is_long)
            .then(kind(long_at, long_locked))
            .otherwise(kind(short_at, short_locked))
            .alias("entry_kind")
        )
        long_exit_at = at_dn("exit_raw_price", "limit_down_exit_date")
        long_exit_locked = at_dn("open_exit_date", "limit_down_exit_date") & at_dn(
            "high_exit_date", "limit_down_exit_date"
        )
        short_exit_at = at_up("exit_raw_price", "limit_up_exit_date")
        short_exit_locked = at_up("open_exit_date", "limit_up_exit_date") & at_up(
            "low_exit_date", "limit_up_exit_date"
        )
        exprs.append(
            pl.when(pl.col("exit_date").is_null())
            .then(pl.lit(None, dtype=pl.String))
            .when(is_long)
            .then(kind(long_exit_at, long_exit_locked))
            .otherwise(kind(short_exit_at, short_exit_locked))
            .alias("exit_kind")
        )
    else:
        exprs = [
            pl.lit(None, dtype=pl.String).alias("entry_kind"),
            pl.lit(None, dtype=pl.String).alias("exit_kind"),
        ]

    out = out.with_columns(exprs)
    keep = list(trades.columns) + ["entry_kind", "exit_kind"]
    return out.select(keep)


def _scenario_stats(report: Any) -> dict[str, Any]:
    row = report.stats.to_dicts()[0] if report.stats.height else {}
    return {
        "total_return": row.get("total_return"),
        "cagr": row.get("cagr"),
        "max_drawdown": row.get("max_drawdown"),
        "daily_sharpe": row.get("daily_sharpe"),
        "calmar": row.get("calmar"),
    }


def limit_stress(
    df: pl.DataFrame,
    report: Any,
    *,
    position: str = "weight",
    scenarios: tuple[str, ...] = ("locked", "at_limit"),
    **backtest_kwargs: Any,
) -> list[dict[str, Any]]:
    """Re-run the backtest with unfillable entries removed, per scenario.

    Scenarios:
        "locked"   — remove only 一字鎖死 entries (open == low == limit):
                     the fills a monitored live order could NOT have gotten.
        "at_limit" — remove every entry filled at the limit price
                     (conservative: assume none of them fill).

    Blocking works by zeroing the ``position`` column at the blocked trade's
    (entry_sig_date, stock_id); with float weights the undeployed money stays
    in cash (finlab normalization does not scale the rest up). Boolean signal
    columns would redistribute the slot — pass float weights for exact
    semantics. One-pass approximation: blocking is decided from the baseline
    run's trades. Exits are NOT deferred (an unfillable exit day is reported
    by ``classify_limit_trades`` but the scenario still exits at that price).

    ``backtest_kwargs`` must be the same arguments used for the baseline
    ``backtest_with_report`` call (resample, benchmark, fees, ...).

    Returns one dict per scenario: name, blocked trade count/contribution,
    and the re-run's headline stats next to the baseline's.
    """
    flagged = classify_limit_trades(report.trades, df)
    baseline = {"name": "baseline", "blocked_n": 0, "blocked_contrib": 0.0}
    baseline.update(_scenario_stats(report))
    results = [baseline]

    for scenario in scenarios:
        if scenario == "locked":
            blocked = flagged.filter(pl.col("entry_kind") == "locked")
        elif scenario == "at_limit":
            blocked = flagged.filter(pl.col("entry_kind").is_not_null())
        else:
            raise ValueError(f"Unknown scenario '{scenario}' (use 'locked' or 'at_limit')")

        entry = {
            "name": scenario,
            "blocked_n": blocked.height,
            "blocked_contrib": (
                blocked.filter(pl.col("return").is_not_null() & pl.col("return").is_not_nan())
                .select((pl.col("return") * pl.col("position").abs()).sum())
                .item()
                or 0.0
            ),
        }
        if blocked.height == 0:
            entry.update(_scenario_stats(report))
            results.append(entry)
            continue

        keys = blocked.select(
            pl.col("entry_sig_date").alias("date"), pl.col("stock_id").alias("symbol")
        ).unique()
        stressed = df.join(
            keys.with_columns(pl.lit(True).alias("_blocked")),
            on=["date", "symbol"],
            how="left",
        ).with_columns(
            pl.when(pl.col("_blocked"))
            .then(0.0)
            .otherwise(pl.col(position).cast(pl.Float64))
            .alias(position)
        ).drop("_blocked")

        rerun = backtest_with_report(stressed, position=position, **backtest_kwargs)
        entry.update(_scenario_stats(rerun))
        results.append(entry)

    return results
