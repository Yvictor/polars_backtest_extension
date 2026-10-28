"""Synthetic swing fixtures; optional SWING_EXTENSION loads an isolated binding.

Normal CI exercises the installed polars_backtest extension.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def engine():
    extension = os.environ.get("SWING_EXTENSION")
    if extension:
        root = Path(__file__).resolve().parents[1]
        adapter = load_module("swing_test_adapter", root / "python/polars_backtest/swing.py")
        native = load_module("_swing_core", Path(extension))
        adapter._run_native = native.swing_impl
        return adapter
    import polars_backtest.swing as swing

    return swing


def frames():
    days = [dt.date(2019, 2, i) for i in range(1, 5)]
    bars = pl.DataFrame(
        {
            "date": days,
            "code": ["0050"] * 4,
            **dict.fromkeys(["open", "high", "low", "close"], [100.0] * 4),
            "limit_up": [110.0] * 4,
            "limit_down": [90.0] * 4,
        }
    )
    targets = pl.DataFrame({"date": days[:1], "code": ["0050"], "weight": [0.5]})
    return bars, targets


def test_typed_output_and_cash_accounting(engine):
    b, t = frames()
    result = engine.backtest_swing(t, b, capital=1e6, costs=engine.SwingCosts(hedge_cost=0.0))
    assert result.fills["code"].to_list() == ["0050"]
    assert result.fills["shares"].to_list() == [5000]
    assert result.daily["equity"].to_list() == [1e6, 999857.5, 999857.5, 999857.5]
    assert result.positions["shares"].to_list() == [5000] * 3
    assert result.daily["date"].dtype == pl.Date
    assert result.daily["ret"][1] == -142.5 / 1e6


def test_empty_frames_have_public_schemas(engine):
    b, t = frames()
    result = engine.backtest_swing(t.head(0), b.head(0), costs=engine.SwingCosts(hedge_cost=0.0))
    assert result.daily.is_empty() and result.fills.is_empty() and result.positions.is_empty()
    assert result.fills.schema["code"] == pl.String
    assert result.fills.schema["shares"] == pl.Int64
    assert result.daily.schema["date"] == pl.Date


@pytest.mark.parametrize("which", ["bars", "targets", "borrow", "hedge"])
def test_duplicate_keys_rejected(engine, which):
    b, t = frames()
    borrow = b.select("date", "code").with_columns(pl.lit(True).alias("can_borrow"))
    hedge = b.select("date").with_columns(pl.lit(0.0).alias("ret"), pl.lit(False).alias("roll"))
    if which == "bars":
        b = pl.concat([b, b.head(1)])
    elif which == "targets":
        t = pl.concat([t, t])
    elif which == "borrow":
        borrow = pl.concat([borrow, borrow.head(1)])
    else:
        hedge = pl.concat([hedge, hedge.head(1)])
    with pytest.raises(ValueError, match="Duplicate"):
        engine.backtest_swing(
            t, b, costs=engine.SwingCosts(hedge_cost=0.0), borrowable=borrow, hedge=hedge
        )


@pytest.mark.parametrize("case", ["execution", "weight", "capital", "fee", "hedge"])
def test_invalid_configuration_rejected(engine, case):
    b, t = frames()
    kwargs = {"costs": engine.SwingCosts(hedge_cost=0.0)}
    if case == "execution":
        kwargs["rules"] = engine.SwingRules(execution="mid")
    elif case == "weight":
        t = t.with_columns(pl.lit(float("nan")).alias("weight"))
    elif case == "capital":
        kwargs["capital"] = float("inf")
    elif case == "fee":
        kwargs["costs"] = engine.SwingCosts(fee_discount=-0.2, hedge_cost=0.0)
    else:
        kwargs["costs"] = engine.SwingCosts(hedge_cost=0.0001)
        kwargs["rules"] = engine.SwingRules(hedge_ratio=1.0)
    with pytest.raises(ValueError):
        engine.backtest_swing(t, b, **kwargs)


def test_auction_retry_and_borrow_permission(engine):
    b, t = frames()
    t = t.with_columns(-pl.col("weight"))
    borrow = b.select("date", "code").with_columns(
        (pl.col("date").dt.day() >= 4).alias("can_borrow")
    )
    result = engine.backtest_swing(
        t, b, capital=1e6, costs=engine.SwingCosts(hedge_cost=0.0), borrowable=borrow
    )
    assert result.fills["date"].to_list() == [dt.date(2019, 2, 4)]
    assert result.fills["shares"].to_list() == [-5000]
    assert result.daily["short"].to_list() == [0.0, 0.0, 0.0, -500_000.0]


def test_every_k_calendar_weeks_does_not_shift_after_empty_week(engine):
    days = [dt.date(2019, 2, 4), dt.date(2019, 2, 13), dt.date(2019, 2, 18), dt.date(2019, 2, 25)]
    assert engine.phase_rebalance_dates(days, 0, 2, 0) == days[::2]
    assert engine.phase_rebalance_dates(days, 0, 2, 1) == days[3:]
    assert engine.phase_rebalance_dates([], 4) == []


@pytest.mark.parametrize("weekday", range(5))
def test_bench_is_one_target_at_first_schedule_date(engine, weekday):
    days = [dt.date(2019, 2, 4) + dt.timedelta(days=i) for i in range(5)]
    book = engine.bench_book(days, weekday)
    assert book.rows() == [(days[weekday], "0050", 1.0)]


def selection_frames():
    days = [dt.date(2019, 2, 4) + dt.timedelta(days=i) for i in range(12)]
    dates = [days[1]] * 4 + [days[8]] * 4
    ids = ["STK:A", "STK:B", "STK:C", "STK:D"] * 2
    scores = pl.DataFrame(
        {
            "decision_date": dates,
            "instrument_id": ids,
            "score": [4.0, 3.0, 2.0, 1.0, 2.0, 4.0, 3.0, 1.0],
            "available_at": [
                dt.datetime.combine(d, dt.time(8, 30), tzinfo=dt.timezone(dt.timedelta(hours=8)))
                for d in dates
            ],
        }
    )
    panel = scores.select("decision_date", "instrument_id").with_columns(
        pl.lit(True).alias("top500"),
        pl.lit(False).alias("disposal"),
        pl.lit(True).alias("can_short"),
    )
    return scores, panel, days


def test_hold_is_a_rank_buffer_and_top_weights_stay_fixed(engine):
    scores, panel, days = selection_frames()
    rebuilt = engine.swing_books(scores, panel, days, weekday=0, top=1)
    held = engine.swing_books(scores, panel, days, weekday=0, top=1, hold=3)
    assert rebuilt["code"].to_list() == ["A", "B"]
    assert held["code"].to_list() == ["A", "A"]
    assert held["weight"].to_list() == [1.0, 1.0]


def test_selector_ties_disposal_and_short_eligibility(engine):
    scores, panel, days = selection_frames()
    scores = scores.with_columns(pl.lit(1.0).alias("score"))
    panel = panel.with_columns(
        (pl.col("instrument_id") == "STK:B").alias("disposal"),
        (pl.col("instrument_id") == "STK:C").alias("can_short"),
    )
    book = engine.swing_books(scores, panel, days, weekday=0, top=1, kind="ls")
    assert book["code"].to_list() == ["A", "C", "A", "C"]
    assert book["weight"].to_list() == [0.5, -0.5, 0.5, -0.5]


def test_late_and_naive_scores_rejected(engine):
    scores, panel, days = selection_frames()
    for expr in [
        pl.col("available_at") + pl.duration(minutes=1),
        pl.col("available_at").dt.replace_time_zone(None),
    ]:
        with pytest.raises(ValueError, match="available|08:30"):
            engine.swing_books(scores.with_columns(expr), panel, days)


def test_costs_and_caller_hedge_cost_are_required(engine):
    b, t = frames()
    with pytest.raises(TypeError, match="costs"):
        engine.backtest_swing(t, b)
    with pytest.raises(TypeError, match="hedge_cost"):
        engine.SwingCosts()
    with pytest.raises(TypeError, match="SwingCosts"):
        engine.backtest_swing(t, b, costs=None)


def test_public_import_and_costs_without_dataclass_kw_only():
    command = [sys.executable, str(Path(__file__).with_name("swing_import_smoke.py"))]
    if os.environ.get("SWING_EXTENSION"):
        command.append("--stub-native")
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1] / "python")
    env["PYTHONPATH"] = os.pathsep.join([source, env.get("PYTHONPATH", "")])
    subprocess.run(command, env=env, check=True)


@pytest.mark.parametrize("hedge_ratio", [1.0, 0.5])
def test_zero_hedge_cost_requires_explicit_allow_flag(engine, hedge_ratio):
    b, t = frames()
    hedge = b.select("date").with_columns(pl.lit(0.01).alias("ret"), pl.lit(False).alias("roll"))
    kwargs = dict(
        capital=1e6,
        costs=engine.SwingCosts(hedge_cost=0.0),
        rules=engine.SwingRules(hedge_ratio=hedge_ratio),
        hedge=hedge,
    )
    with pytest.raises(ValueError, match="hedge_ratio.*hedge_cost.*allow_zero_hedge_cost=True"):
        engine.backtest_swing(t, b, **kwargs)
    result = engine.backtest_swing(t, b, **kwargs, allow_zero_hedge_cost=True)
    assert result.daily["hedge_cost"].to_list() == [0.0] * b.height
    assert result.daily["hedge_notional"][1] == -500_000.0 * hedge_ratio
    assert result.daily["hedge_pnl"][2] == -5_000.0 * hedge_ratio


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_fills_positions_and_exposures_are_exact(engine, sign):
    b, t = frames()
    t = t.with_columns(pl.col("weight") * sign)
    result = engine.backtest_swing(
        t,
        b,
        capital=1e6,
        costs=engine.SwingCosts(fee_rate=0.0, sell_tax=0.0, short_sell_fee=0.0, hedge_cost=0.0),
    )
    assert result.fills.select("date", "code", "shares", "price", "cost").rows() == [
        (dt.date(2019, 2, 2), "0050", int(sign * 5000), 100.0, 0.0)
    ]
    assert result.positions.select("date", "code", "shares", "entry", "mark").rows() == [
        (dt.date(2019, 2, day), "0050", int(sign * 5000), 100.0, 100.0) for day in (2, 3, 4)
    ]
    assert result.daily["long"].to_list() == [0.0] + [max(0.0, sign * 500_000)] * 3
    assert result.daily["short"].to_list() == [0.0] + [min(0.0, sign * 500_000)] * 3


def test_null_dividend_matches_zero_without_losing_real_dividend(engine):
    b, t = frames()
    b = b.with_columns(pl.Series("dividend", [None, None, 2.0, None], dtype=pl.Float64))
    costs = engine.SwingCosts(hedge_cost=0.0)
    actual = engine.backtest_swing(t, b, capital=1e6, costs=costs)
    expected = engine.backtest_swing(
        t, b.with_columns(pl.col("dividend").fill_null(0.0)), capital=1e6, costs=costs
    )
    assert actual.daily.equals(expected.daily)
    assert actual.daily["dividends"].to_list() == [0.0, 0.0, 10000.0, 0.0]


def test_non_session_target_rejected(engine):
    b, t = frames()
    t = t.with_columns(pl.col("date") - pl.duration(days=1))
    with pytest.raises(ValueError, match="session"):
        engine.backtest_swing(t, b, costs=engine.SwingCosts(hedge_cost=0.0))


def test_overlapping_long_short_selection_rejected(engine):
    scores, panel, days = selection_frames()
    with pytest.raises(ValueError, match="overlap"):
        engine.swing_books(scores, panel, days, weekday=0, top=3, kind="ls")


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_limit_auction_retries_until_fill(engine, sign):
    b, t = frames()
    limit = "limit_up" if sign > 0 else "limit_down"
    b = b.with_columns(
        pl.when(pl.col("date").dt.day() == 2).then(100.0).otherwise(pl.col(limit)).alias(limit)
    )
    result = engine.backtest_swing(
        t.with_columns(pl.col("weight") * sign),
        b,
        capital=1e6,
        costs=engine.SwingCosts(hedge_cost=0.0),
    )
    assert result.fills["date"].to_list() == [dt.date(2019, 2, 3)]
    assert result.fills["shares"].to_list() == [int(sign * 5000)]
    exposure = "long" if sign > 0 else "short"
    assert result.daily[exposure].to_list() == [0.0, 0.0, sign * 500_000, sign * 500_000]


@pytest.mark.parametrize("execution, exit_day", [("open", 2), ("close", 3)])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_ambiguous_bar_stops_first_after_entry(engine, execution, exit_day, sign):
    b, t = frames()
    b = b.with_columns(pl.lit(120.0).alias("high"), pl.lit(80.0).alias("low"))
    result = engine.backtest_swing(
        t.with_columns(pl.col("weight") * sign),
        b,
        capital=1e6,
        costs=engine.SwingCosts(fee_rate=0.0, sell_tax=0.0, short_sell_fee=0.0, hedge_cost=0.0),
        rules=engine.SwingRules(execution=execution, take_profit=0.1, stop_loss=0.1),
    )
    assert result.fills["date"].to_list() == [dt.date(2019, 2, 2), dt.date(2019, 2, exit_day)]
    assert result.fills["shares"].to_list() == [int(sign * 5000), int(-sign * 5000)]
    assert result.fills["price"][1] == pytest.approx(100 * (1 - sign * 0.1) * (1 - sign * 0.001))


@pytest.mark.parametrize("per_side_cost", [0.0001, 0.0002, 0.001])
def test_caller_futures_cost_charges_resize_and_both_roll_sides(engine, per_side_cost):
    b, t = frames()
    hedge = b.select("date").with_columns(
        pl.lit(0.0).alias("ret"), (pl.col("date").dt.day() == 3).alias("roll")
    )
    result = engine.backtest_swing(
        t,
        b,
        capital=1e6,
        costs=engine.SwingCosts(fee_rate=0.0, hedge_cost=per_side_cost),
        rules=engine.SwingRules(hedge_ratio=1.0),
        hedge=hedge,
    )
    assert result.daily["hedge_notional"].to_list() == [0.0] + [-500_000.0] * 3
    assert result.daily["hedge_cost"].to_list() == [
        0.0,
        500_000 * per_side_cost,
        1_000_000 * per_side_cost,
        0.0,
    ]
    assert result.daily["equity"][-1] == 1e6 - 1_500_000 * per_side_cost
