"""Tests for limit-lock classification and fill-stress scenarios."""

import datetime

import polars as pl
import polars_backtest as pl_bt
import pytest
from polars_backtest import liquidity


@pytest.fixture(scope="module")
def limit_fixture():
    """Three symbols; on the February entry day:

    - LOCK: 一字漲停 (open == low == close == limit_up) — unfillable
    - TOUCH: closes at limit_up but traded below it intraday — fillable
    - FREE: nowhere near the limit
    """
    rows = []
    start = datetime.date(2024, 1, 1)
    entry_day = datetime.date(2024, 2, 1)
    for i in range(90):
        date = start + datetime.timedelta(days=i)
        if date.weekday() >= 5:
            continue
        held = date < datetime.date(2024, 2, 15)
        for symbol in ("LOCK", "TOUCH", "FREE"):
            base = 100.0 * (1 + 0.001 * i)
            if date == entry_day:
                limit = round(base * 1.1, 2)
                if symbol == "LOCK":
                    o = h = lo = c = limit
                elif symbol == "TOUCH":
                    o, lo, h, c = base, base * 0.99, limit, limit
                else:
                    o, lo, h, c = base, base * 0.99, base * 1.02, base * 1.01
            else:
                o, lo, h, c = base * 0.995, base * 0.99, base * 1.01, base
                limit = round(c * 1.1, 2)
            rows.append({
                "date": date, "symbol": symbol,
                "open": o, "high": h, "low": lo, "close": c,
                "limit_up": limit, "limit_down": round(c * 0.9, 2),
                "weight": (1 / 3 if held else 0.0),
            })
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def report(limit_fixture):
    return pl_bt.backtest_with_report(limit_fixture, resample="M")


def test_classification(report, limit_fixture):
    flagged = liquidity.classify_limit_trades(report.trades, limit_fixture)
    kinds = {
        r["stock_id"]: r["entry_kind"]
        for r in flagged.filter(pl.col("entry_date").is_not_null()).to_dicts()
    }

    assert kinds["LOCK"] == "locked"
    assert kinds["TOUCH"] == "touched"
    assert kinds["FREE"] is None


def test_limit_stress_scenarios(report, limit_fixture):
    results = liquidity.limit_stress(limit_fixture, report, resample="M")
    by_name = {r["name"]: r for r in results}

    assert set(by_name) == {"baseline", "locked", "at_limit"}
    assert by_name["locked"]["blocked_n"] == 1      # only LOCK
    assert by_name["at_limit"]["blocked_n"] == 2    # LOCK + TOUCH
    # every scenario reports comparable headline stats
    for r in results:
        assert r["cagr"] is not None and r["max_drawdown"] is not None
    # blocking entries must change the equity path
    assert by_name["locked"]["total_return"] != by_name["baseline"]["total_return"]


def test_stress_blocks_the_right_symbol(report, limit_fixture):
    results = liquidity.limit_stress(
        limit_fixture, report, scenarios=("locked",), resample="M"
    )
    del results
    # re-run manually to inspect trades
    flagged = liquidity.classify_limit_trades(report.trades, limit_fixture)
    blocked = flagged.filter(pl.col("entry_kind") == "locked")
    keys = blocked.select(
        pl.col("entry_sig_date").alias("date"), pl.col("stock_id").alias("symbol")
    ).unique()
    stressed = limit_fixture.join(
        keys.with_columns(pl.lit(True).alias("_b")), on=["date", "symbol"], how="left"
    ).with_columns(
        pl.when(pl.col("_b")).then(0.0).otherwise(pl.col("weight")).alias("weight")
    ).drop("_b")
    rerun = pl_bt.backtest_with_report(stressed, resample="M")
    entered = set(
        rerun.trades.filter(pl.col("entry_date").is_not_null()).get_column("stock_id").to_list()
    )

    assert "LOCK" not in entered
    assert {"TOUCH", "FREE"} <= entered


def test_unknown_scenario_rejected(report, limit_fixture):
    with pytest.raises(ValueError, match="scenario"):
        liquidity.limit_stress(limit_fixture, report, scenarios=("wat",), resample="M")


def test_viz_payload_kinds_and_scenarios(report, limit_fixture):
    from polars_backtest import viz

    scenarios = liquidity.limit_stress(limit_fixture, report, resample="M")
    data = viz.report_data(report, input_df=limit_fixture, fill_scenarios=scenarios)

    # only executed entries — pending re-entry rows have entry=None
    kinds = {
        s: k
        for s, k, e in zip(
            data["trades"]["stock"], data["trades"]["entry_kind"], data["trades"]["entry"]
        )
        if e is not None
    }
    assert kinds["LOCK"] == "locked" and kinds["TOUCH"] == "touched"
    assert data["trade_summary"]["entry_locked_n"] == 1
    assert data["trade_summary"]["entry_touched_n"] == 1
    assert len(data["fill_scenarios"]) == 3
    import json

    json.dumps(data, allow_nan=False)
