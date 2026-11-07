"""Derived numbers: the single definition the page, the summary and agents all read.

Formulas match ``get_stats`` where the two overlap. The JavaScript in the dashboard
repeats a few of them for live ranges, and a test holds the two copies together.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

import polars as pl

NPERIODS = 252
ROLLING_WINDOW = 252
MIN_ROLLING_DAYS = 300
ALPHA_DECAY_MIN_DAYS = 756
ALPHA_DECAY_RECENT_DAYS = 504
CONCENTRATION_SHARE = 0.05


def _finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def _span_years(curve: pl.DataFrame) -> float:
    start, end = curve["date"][0], curve["date"][-1]
    return (end - start).days / 365.25


def annualise(growth: float, years: float) -> float | None:
    if years <= 0 or growth <= 0:
        return None
    return growth ** (1 / years) - 1


def performance(curve: pl.DataFrame, riskfree_rate: float) -> dict[str, Any]:
    """Whole-period figures of a curve table.

    As in ``get_stats``, the first day counts as a zero return in the daily series.
    """
    rets = curve["ret"].fill_null(0.0)
    rf = (1 + riskfree_rate) ** (1 / NPERIODS) - 1
    excess = rets - rf
    downside = excess.clip(upper_bound=0.0)
    total = curve["creturn"][-1] / curve["creturn"][0] - 1
    cagr = annualise(1 + total, _span_years(curve))
    mdd = _finite(curve["drawdown"].min())
    vol = rets.std(ddof=1) if rets.len() > 1 else None
    excess_sd = excess.std(ddof=1) if rets.len() > 1 else None
    downside_sd = downside.std(ddof=1) if rets.len() > 1 else None
    root = math.sqrt(NPERIODS)
    sharpe = _ratio(excess.mean() if rets.len() else None, excess_sd)
    sortino = _ratio(excess.mean() if rets.len() else None, downside_sd)
    return {
        "total_return": _finite(total),
        "cagr": _finite(cagr),
        "volatility": _finite(vol * root) if vol is not None else None,
        "sharpe": _finite(sharpe * root) if sharpe is not None else None,
        "sortino": _finite(sortino * root) if sortino is not None else None,
        "calmar": _finite(_ratio(cagr, abs(mdd) if mdd is not None else None)),
        "max_drawdown": mdd,
    }


def underwater(curve: pl.DataFrame) -> dict[str, Any]:
    """Time below a prior high: depth alone hides how long a fall lasted.

    The longest stretch runs from the high to the day it was regained, or to the last
    day when it never was. Days are calendar days.
    """
    dates = curve["date"].to_list()
    values = curve["creturn"].to_list()
    peak = 0
    best, best_start, best_end = 0, dates[0], dates[0]
    below = 0
    for i, value in enumerate(values):
        under = value < values[peak]
        below += under
        days = (dates[i] - dates[peak]).days
        if (under or i - peak > 1) and days > best:
            best, best_start, best_end = days, dates[peak], dates[i]
        if not under:
            peak = i
    return {
        "longest_underwater_days": best,
        "underwater_start": best_start if best else None,
        "underwater_end": best_end if best else None,
        "underwater_ongoing": bool(best) and best_end == dates[-1] and values[-1] < values[peak],
        "share_below_high": below / len(values),
    }


def entry_date_risk(
    curve: pl.DataFrame, column: str = "creturn", window: int = ROLLING_WINDOW
) -> dict[str, Any] | None:
    """Outcome of every ``window``-day hold, one per start date."""
    frame = curve.select("date", pl.col(column).alias("c")).filter(pl.col("c").is_not_null())
    if frame.height <= window:
        return None
    rolled = frame.with_columns((pl.col("c").shift(-window) / pl.col("c") - 1).alias("r")).filter(
        pl.col("r").is_not_null()
    )
    rets = rolled["r"]
    worst_at = rolled.filter(pl.col("r") == rets.min())["date"][0]
    return {
        "window_days": window,
        "n_windows": rets.len(),
        "worst": rets.min(),
        "median": rets.median(),
        "best": rets.max(),
        "positive_share": (rets > 0).sum() / rets.len(),
        "worst_start": worst_at,
    }


def concentration(monthly: pl.DataFrame, curve: pl.DataFrame) -> dict[str, Any]:
    """Annualised return with the best months treated as flat, over the same years."""
    rets = monthly["ret"].sort(descending=True)
    k = math.ceil(rets.len() * CONCENTRATION_SHARE)
    growth = curve["creturn"][-1] / curve["creturn"][0]
    years = _span_years(curve)
    without = growth
    for r in rets.head(k).to_list():
        without /= 1 + r
    cagr = annualise(growth, years)
    excluded = annualise(without, years)
    level = "na"
    if cagr is not None and excluded is not None and cagr > 0:
        level = "red" if excluded <= 0 else "amber" if excluded < cagr / 2 else "green"
    return {
        "n_months": rets.len(),
        "k": k,
        "cagr": cagr,
        "cagr_ex_best": excluded,
        "level": level,
    }


def _ledger_costs(ledger: pl.DataFrame) -> list[str]:
    return [
        c
        for c in ("commission", "tax", "surcharge", "borrow_cost", "hedge_cost")
        if c in ledger.columns
    ]


def with_gross_curve(curve: pl.DataFrame, ledger: pl.DataFrame | None) -> pl.DataFrame:
    """Add ``gross_creturn``: daily costs added back to the net return, then compounded.

    Slippage is already inside fill prices, so only booked costs come back.
    """
    if (
        ledger is None
        or ledger.is_empty()
        or "equity" not in ledger.columns
        or not _ledger_costs(ledger)
    ):
        return curve
    costs = ledger.select(
        "date",
        (pl.sum_horizontal(_ledger_costs(ledger)) / pl.col("equity").shift(1)).alias("_cost"),
    )
    joined = curve.join(costs, on="date", how="left")
    gross = (1 + pl.col("ret").fill_null(0.0) + pl.col("_cost").fill_null(0.0)).cum_prod()
    first = pl.col("creturn").first()
    return joined.with_columns((gross * first).alias("gross_creturn")).drop("_cost")


def costs(
    ledger: pl.DataFrame | None,
    curve: pl.DataFrame,
    trade_summary: dict[str, Any],
) -> dict[str, Any]:
    """Annual cost drag: booked from the ledger when it can be, else estimated from trades."""
    years = _span_years(curve)
    estimate = {"source": "estimate", "annual_drag": trade_summary.get("cost_drag_annual")}
    if ledger is None or not _ledger_costs(ledger) or "equity" not in ledger.columns:
        return estimate
    in_range = ledger.filter(pl.col("date") >= curve["date"][0])
    avg_equity = in_range["equity"].mean()
    # A ledger with no rows, or no positive equity, has booked nothing to divide.
    if avg_equity is None or avg_equity <= 0 or years <= 0:
        return estimate
    share = {c: (in_range[c].sum() or 0.0) / avg_equity / years for c in _ledger_costs(ledger)}
    return {"source": "ledger", "annual_drag": sum(share.values()), **share}


def net_of_costs(curve: pl.DataFrame) -> dict[str, Any] | None:
    if "gross_creturn" not in curve.columns:
        return None
    years = _span_years(curve)
    gross = annualise(curve["gross_creturn"][-1] / curve["gross_creturn"][0], years)
    net = annualise(curve["creturn"][-1] / curve["creturn"][0], years)
    if gross is None or net is None:
        return None
    share = (gross - net) / gross if gross > 0 else None
    level = "na" if share is None else "amber" if share > 0.25 else "green"
    return {"gross_cagr": gross, "net_cagr": net, "cost_share": share, "level": level}


def exposure(ledger: pl.DataFrame | None) -> dict[str, Any]:
    needed = {"equity", "gross_exposure", "net_exposure", "cash"}
    if ledger is None or not needed <= set(ledger.columns):
        return {"source": "absent"}
    equity = pl.col("equity")
    if ledger.filter(equity > 0).is_empty():
        return {"source": "absent"}
    row = ledger.filter(equity > 0).select(
        (pl.col("gross_exposure") / equity).mean().alias("avg_gross"),
        (pl.col("net_exposure") / equity).mean().alias("avg_net"),
        (pl.col("cash") / equity).mean().alias("avg_cash"),
    )
    return {"source": "ledger", **row.to_dicts()[0]}


def turnover(ledger: pl.DataFrame | None, trade_summary: dict[str, Any]) -> dict[str, Any]:
    mean = None
    if ledger is not None and "turnover" in ledger.columns:
        mean = ledger["turnover"].mean()
    if mean is not None:
        return {"source": "ledger", "annual_two_sided": mean * NPERIODS}
    return {"source": "estimate", "annual_entry_weight": trade_summary.get("annual_turnover")}


def closed_trades(trades: pl.DataFrame) -> pl.DataFrame:
    """The round trips every trade figure counts: closed, with a finite return."""
    return trades.filter((pl.col("status") == "closed") & pl.col("ret").is_finite())


def _side_figures(frame: pl.DataFrame) -> dict[str, Any]:
    rets = frame["ret"]
    return {
        "n": frame.height,
        "win_rate": (rets > 0).sum() / frame.height,
        "avg_ret": rets.mean(),
        "contrib": frame["_c"].sum(),
    }


def _limit_structure(trades: pl.DataFrame, closed: pl.DataFrame) -> dict[str, Any]:
    """Limit-price counts and contribution, from the bundle's own trades table."""
    out: dict[str, Any] = {}
    entered = trades.filter(pl.col("status") != "pending")
    exited = trades.filter(pl.col("exit_date").is_not_null())
    legs = (("limit_entry", entered, "buy_high"), ("limit_exit", exited, "sell_low"))
    for column, frame, name in legs:
        known = frame[column].drop_nulls() if column in frame.columns else None
        if known is not None and known.len():
            out[f"{name}_n"] = int(known.sum() or 0)
            out[f"{name}_ratio"] = (known.sum() or 0) / known.len()
    total = closed["_c"].sum() or 0.0
    if "limit_entry" in closed.columns and closed["limit_entry"].drop_nulls().len():
        at_limit = closed.filter(pl.col("limit_entry").fill_null(False))["_c"].sum() or 0.0
        if abs(total) > 0:
            out["buy_high_contrib"] = at_limit
            out["buy_high_contrib_ratio"] = at_limit / total
    if "entry_kind" in trades.columns:
        out["entry_locked_n"] = entered.filter(pl.col("entry_kind") == "locked").height
        out["entry_touched_n"] = entered.filter(pl.col("entry_kind") == "touched").height
        locked = closed.filter(pl.col("entry_kind") == "locked")
        if locked.height and abs(total) > 0:
            out["entry_locked_contrib_ratio"] = (locked["_c"].sum() or 0.0) / total
    if "exit_kind" in trades.columns:
        out["exit_locked_n"] = exited.filter(pl.col("exit_kind") == "locked").height
        out["exit_touched_n"] = exited.filter(pl.col("exit_kind") == "touched").height
    return out


def trade_structure(
    trades: pl.DataFrame | None, fee_ratio: float | None, tax_ratio: float | None
) -> dict[str, Any]:
    """Where the return came from, measured on the bundle's own trades table.

    Flags and the summary read this, so they describe the same round trips as the
    ``trades`` table. Contribution is ``ret × |weight|``, closed round trips only.
    """
    if trades is None:
        return {}
    closed = closed_trades(trades).with_columns(
        (pl.col("ret") * pl.col("weight").abs()).alias("_c")
    )
    out: dict[str, Any] = {}
    if closed.height:
        sides = {}
        for side in ("long", "short"):
            frame = closed.filter(pl.col("side") == side)
            if frame.height:
                sides[side] = _side_figures(frame)
        out["sides"] = sides
        total = closed["_c"].sum() or 0.0
        if abs(total) > 0:
            out["top10_contrib_ratio"] = closed["_c"].sort(descending=True).head(10).sum() / total
            out["total_contrib"] = total
        entered = trades.filter(pl.col("status") != "pending")
        first = entered["entry_date"].min()
        last = trades["exit_date"].max() or entered["entry_date"].max()
        years = max(((last - first).days or 1) / 365.25, 1 / 365.25)
        out["annual_turnover"] = (entered["weight"].abs().sum() or 0.0) / years
        if fee_ratio is not None and tax_ratio is not None:
            out["cost_drag_annual"] = out["annual_turnover"] * (2 * fee_ratio + tax_ratio)
    return {**out, **_limit_structure(trades, closed)}


def limit_fills(trades: pl.DataFrame | None) -> dict[str, Any] | None:
    """How much of the summed trade return sits in fills that may not have been possible.

    ``ret × |weight|`` is an approximation of P&L, not booked P&L.
    """
    if trades is None or not {"limit_entry", "limit_exit"} & set(trades.columns):
        return None
    flags = [c for c in ("limit_entry", "limit_exit") if c in trades.columns]
    closed = closed_trades(trades)
    if not closed.height:
        return None
    frame = closed.select(
        (pl.col("ret") * pl.col("weight").abs()).alias("c"),
        pl.any_horizontal([pl.col(c).fill_null(False) for c in flags]).alias("flagged"),
    )
    total = frame["c"].sum() or 0.0
    share = _ratio(frame.filter(pl.col("flagged"))["c"].sum() or 0.0, total)
    level = "na" if share is None else "amber" if share > 0.10 else "green"
    return {
        "n": int(frame["flagged"].sum() or 0),
        "share_trades": (frame["flagged"].sum() or 0) / frame.height,
        "share_return": share,
        "level": level,
    }


def trade_stats(trades: pl.DataFrame | None, segment_win_rate: float | None) -> dict[str, Any]:
    """Round-trip figures. ``win_rate`` counts closed round trips only."""
    out: dict[str, Any] = {"n_round_trips": 0, "segment_win_rate": segment_win_rate}
    if trades is None:
        return out
    entered = trades.filter(pl.col("status") != "pending")
    closed = closed_trades(trades)
    rets = closed["ret"]
    wins = rets.filter(rets > 0)
    losses = rets.filter(rets < 0)
    loss_sum = abs(losses.sum() or 0.0)
    out.update(
        n_round_trips=entered.height,
        n_closed=closed.height,
        n_segments=int(trades["n_segments"].sum() or 0),
        win_rate=wins.len() / rets.len() if rets.len() else None,
        profit_factor=(wins.sum() or 0.0) / loss_sum if loss_sum > 0 else None,
        avg_hold_days=closed["pdays"].mean() if "pdays" in closed.columns else None,
    )
    return out


def alpha_decay(curve: pl.DataFrame) -> dict[str, Any] | None:
    """Mean rolling 1-year Sharpe over the last two years against the whole period."""
    n = curve.height
    if n < MIN_ROLLING_DAYS or n < ALPHA_DECAY_MIN_DAYS:
        return None
    rfd = 0.02 / NPERIODS
    rolled = (
        curve.select(pl.col("ret").fill_null(0.0).alias("r"))
        .with_row_index("i")
        .with_columns(
            pl.col("r").rolling_mean(ROLLING_WINDOW).alias("m"),
            pl.col("r").rolling_std(ROLLING_WINDOW, ddof=0).alias("s"),
        )
        .filter((pl.col("i") >= ROLLING_WINDOW) & (pl.col("s") > 1e-12))
        .with_columns(((pl.col("m") - rfd) / pl.col("s") * math.sqrt(NPERIODS)).alias("sharpe"))
    )
    recent = rolled.filter(pl.col("i") >= n - ALPHA_DECAY_RECENT_DAYS)
    if rolled.height < 10 or recent.height < 10:
        return None
    full = rolled["sharpe"].mean()
    rec = recent["sharpe"].mean()
    return {"full": full, "recent": rec, "ratio": rec / full if full > 1e-9 else None}


def _level(value: float, green: float, amber: float) -> str:
    return "green" if value <= green else "amber" if value <= amber else "red"


def _flag(code: str, level: str, value: Any, rule: str, drill: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "level": level, "value": value, "rule": rule, "drill": drill, **extra}


def _capacity_flag(metrics: dict[str, Any]) -> dict[str, Any]:
    capacity = metrics.get("capacity")
    if capacity is None:
        level = "na"
    else:
        level = "green" if capacity >= 5e7 else "amber" if capacity >= 1e7 else "red"
    return _flag(
        "capacity", level, capacity, "capacity", "metrics", green_min=5e7, amber_min=1e7
    )


def _limit_flag(metrics: dict[str, Any], ts: dict[str, Any]) -> dict[str, Any]:
    locked = ts.get("entry_locked_contrib_ratio")
    contrib = ts.get("buy_high_contrib_ratio")
    count = metrics.get("buyHigh")
    if count is None:
        count = ts.get("buy_high_ratio")
    if locked is not None:
        value, rule, green, amber = locked, "locked_entry_contrib / total_contrib", 0.05, 0.15
    elif contrib is not None:
        value, rule, green, amber = contrib, "limit_entry_contrib / total_contrib", 0.10, 0.25
    else:
        value, rule, green, amber = count, "limit_entries / entries", 0.05, 0.10
    level = "na" if value is None else _level(value, green, amber)
    return _flag(
        "limit_dependence", level, value, rule, "trades", green_max=green, amber_max=amber
    )


def _cost_flag(drag: float | None, source: str, cagr: float | None) -> dict[str, Any]:
    value = None
    level = "na"
    if drag is not None and cagr is not None and cagr > 0:
        value = drag / cagr
        level = _level(value, 0.2, 0.5)
    return _flag(
        "cost_sensitivity",
        level,
        value,
        "annual_drag / cagr",
        "ledger_daily",
        green_max=0.2,
        amber_max=0.5,
        source=source,
    )


def _decay_flag(decay: dict[str, Any] | None) -> dict[str, Any]:
    ratio = decay["ratio"] if decay else None
    if ratio is None:
        level = "na"
    else:
        level = "green" if ratio >= 0.7 else "amber" if ratio >= 0.4 else "red"
    return _flag(
        "alpha_decay",
        level,
        ratio,
        "recent_rolling_sharpe / full_rolling_sharpe",
        "curve",
        green_min=0.7,
        amber_min=0.4,
        recent=decay["recent"] if decay else None,
        full=decay["full"] if decay else None,
    )


def _balance_flag(ts: dict[str, Any]) -> dict[str, Any] | None:
    sides = ts.get("sides") or {}
    if "short" not in sides:
        return None
    long_contrib = sides.get("long", {}).get("contrib", 0.0)
    short_contrib = sides["short"]["contrib"]
    total = ts.get("total_contrib")
    total = long_contrib + short_contrib if total is None else total
    if abs(total) > 1e-12 and short_contrib < -0.10 * abs(total):
        level = "red"
    elif long_contrib > 0 and short_contrib > 0:
        level = "green"
    else:
        level = "amber"
    return _flag(
        "long_short_balance",
        level,
        short_contrib,
        "short_contrib against total_contrib",
        "trades",
        long_contrib=long_contrib,
        total_contrib=total,
    )


def flags(
    curve: pl.DataFrame,
    metrics: dict[str, Any],
    trade_summary: dict[str, Any],
    cost: dict[str, Any],
    cagr: float | None,
) -> list[dict[str, Any]]:
    """Whole-period judgement flags. They never move with a date range on the page."""
    out = [
        _capacity_flag(metrics),
        _limit_flag(metrics, trade_summary),
        _cost_flag(cost.get("annual_drag"), cost["source"], cagr),
        _decay_flag(alpha_decay(curve)),
    ]
    balance = _balance_flag(trade_summary)
    return out + [balance] if balance else out


def _worst(frame: pl.DataFrame) -> float:
    return float(frame["diff"].abs().max() or 0.0)


def _check(check_id: str, worst: float, tol: float) -> dict[str, Any]:
    status = "pass" if worst <= tol else "fail"
    return {"id": check_id, "status": status, "worst": worst, "tol": tol}


def checks(
    curve: pl.DataFrame,
    ledger: pl.DataFrame | None,
    fills: pl.DataFrame | None,
) -> list[dict[str, Any]]:
    """Reconcile ledger tables against each other. A check runs only if its tables exist."""
    out: list[dict[str, Any]] = []
    if ledger is None or ledger.is_empty():
        return out
    if "equity" in ledger.columns:
        joined = curve.select("date", "creturn").join(
            ledger.select("date", "equity"), on="date", how="inner"
        )
        if joined.height:
            scale = joined["equity"][0] / joined["creturn"][0]
            frame = joined.select(
                (pl.col("equity") / (pl.col("creturn") * scale) - 1).alias("diff")
            )
            out.append(_check("curve_vs_equity", _worst(frame), 1e-9))
    if fills is not None and "traded_notional" in ledger.columns and "notional" in fills.columns:
        per_day = fills.group_by("date").agg(pl.col("notional").abs().sum().alias("fills"))
        frame = ledger.join(per_day, on="date", how="left").select(
            (pl.col("traded_notional") - pl.col("fills").fill_null(0.0)).alias("diff")
        )
        out.append(_check("traded_vs_fills", _worst(frame), 1e-6))
    cost_columns = _ledger_costs(ledger)
    if fills is not None and "cost" in fills.columns and cost_columns:
        per_day = fills.group_by("date").agg(pl.col("cost").sum().alias("fills"))
        frame = ledger.join(per_day, on="date", how="left").select(
            (pl.sum_horizontal(cost_columns) - pl.col("fills").fill_null(0.0)).alias("diff")
        )
        out.append(_check("daily_cost_vs_fills", _worst(frame), 1e-6))
    if {"equity", "pnl"} <= set(ledger.columns):
        frame = ledger.select((pl.col("equity").diff() - pl.col("pnl")).alias("diff")).drop_nulls()
        out.append(_check("equity_identity", _worst(frame), 1e-6))
    return out


def iso(value: Any) -> Any:
    """ISO text for a date, anything else unchanged."""
    return value.isoformat() if isinstance(value, dt.date) else value
