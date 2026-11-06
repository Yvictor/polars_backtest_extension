"""Synthetic next-session fills, closing marks and adverse slippage."""

import datetime as dt
import json

import polars as pl
import polars_backtest as bt
import pytest
from legacy_fixtures import frame
from multi_name_fixtures import CASES, inputs
from polars.testing import assert_frame_equal


def prices(
    weights=(1.0, 1.0, 0.0, 0.0),
    opens=(100.0, 100.0, 120.0, 130.0),
    closes=(90.0, 110.0, 125.0, 140.0),
):
    return pl.DataFrame(
        {
            "date": [dt.date(2024, 1, d) for d in (2, 3, 4, 5)],
            "symbol": ["A"] * 4,
            "open": opens,
            "close": closes,
            "weight": weights,
        }
    )


def run(df, **options):
    return bt.backtest_with_report(
        df, resample=None, fee_ratio=0.0, tax_ratio=0.0, ledger=True, **options
    )


def test_open_fill_close_mark_including_entry_and_final_day():
    df = prices()
    report = run(df, trade_at_price="open", mark_price="close")
    assert report.creturn["creturn"].to_list() == pytest.approx([1.0, 1.1, 1.25, 1.3])
    assert report.fills["price"].to_list() == [100.0, 130.0]
    assert report.trades["entry_price"][0] == 100.0
    assert report.trades["exit_price"][0] == 130.0
    assert report.trade_at == "open"
    assert_frame_equal(
        bt.backtest(
            df,
            resample=None,
            fee_ratio=0.0,
            tax_ratio=0.0,
            trade_at_price="open",
            mark_price="close",
        ),
        report.creturn,
        check_exact=True,
    )


@pytest.mark.parametrize("name", CASES)
def test_explicit_defaults_and_same_price_keep_all_bits(name):
    kind, options = CASES[name]
    df = inputs(kind)
    plain = bt.backtest_with_report(df, **options)
    explicit = bt.backtest_with_report(
        df, mark_price=pl.col("close"), slippage_ratio=0.0, **options
    )
    for prop in ("creturn", "trades"):
        assert json.dumps(frame(getattr(plain, prop), keyed=True)) == json.dumps(
            frame(getattr(explicit, prop), keyed=True)
        )
    assert_frame_equal(
        bt.backtest(df, **options),
        bt.backtest(df, mark_price="close", slippage_ratio=0.0, **options),
        check_exact=True,
    )


@pytest.mark.parametrize("weight", [1.0, -1.0])
def test_slippage_round_trip_both_directions(weight):
    df = prices(weights=(weight, weight, 0.0, 0.0), opens=(100.0,) * 4, closes=(100.0,) * 4)
    report = run(df, slippage_ratio=0.1)
    expected = [110.0, 90.0] if weight > 0 else [90.0, 110.0]
    assert report.fills["price"].to_list() == pytest.approx(expected)
    assert report.trades["entry_price"][0] == pytest.approx(expected[0])
    assert report.trades["exit_price"][0] == pytest.approx(expected[1])
    final = 0.9 / 1.1 if weight > 0 else 0.8
    assert report.creturn["creturn"][-1] == pytest.approx(final)
    assert report.daily["slippage"].sum() == pytest.approx(1.0 - final)
    assert report.trades["return"][0] < 0.0


def test_slippage_fees_use_filled_notional():
    df = prices(opens=(100.0,) * 4, closes=(100.0,) * 4)
    report = bt.backtest_with_report(
        df, resample=None, slippage_ratio=0.1, fee_ratio=0.01, tax_ratio=0.02, ledger=True
    )
    assert report.daily["commission"].sum() == pytest.approx(
        report.fills["notional"].abs().sum() * 0.01
    )
    assert report.daily["tax"].sum() == pytest.approx(abs(report.fills["notional"][-1]) * 0.02)
    assert report.daily["equity"][-1] == pytest.approx(
        1.0 - report.daily.select(pl.sum_horizontal("commission", "tax", "slippage").sum()).item()
    )


def test_close_stop_fills_next_open_with_slippage():
    df = prices(
        weights=(1.0,) * 4, opens=(100.0, 100.0, 100.0, 80.0), closes=(100.0, 90.0, 90.0, 80.0)
    )
    report = run(df, trade_at_price="open", mark_price="close", stop_loss=0.05, slippage_ratio=0.01)
    stop = report.fills.filter(pl.col("reason") == "stop")
    assert stop["date"].to_list() == [dt.date(2024, 1, 4)]
    assert stop["price"].to_list() == pytest.approx([99.0])


def test_slippage_is_part_of_stop_basis():
    df = prices(weights=(1.0,) * 4, opens=(100.0,) * 4, closes=(100.0,) * 4)
    report = run(df, mark_price="close", slippage_ratio=0.1, stop_loss=0.05)
    stop = report.fills.filter(pl.col("reason") == "stop")
    assert stop["date"].to_list() == [dt.date(2024, 1, 5)]


def test_mark_expression_and_missing_mark():
    df = prices().with_columns(
        pl.when(pl.col("date").dt.day() == 4).then(None).otherwise(pl.col("close")).alias("mark")
    )
    report = run(df, trade_at_price="open", mark_price=pl.col("mark"))
    assert report.creturn["creturn"].to_list() == pytest.approx([1.0, 1.1, 1.2, 1.3])


def test_final_missing_execution_price_still_marks_existing_position():
    df = prices(weights=(1.0,) * 4, opens=(100.0, 100.0, 120.0, None))
    report = run(df, trade_at_price="open", mark_price="close")
    assert report.creturn["creturn"].to_list() == pytest.approx([1.0, 1.1, 1.25, 1.4])


@pytest.mark.parametrize("value", [-0.1, 1.0, float("inf"), float("nan")])
@pytest.mark.parametrize("api", [bt.backtest, bt.backtest_with_report])
def test_rejects_invalid_slippage(value, api):
    with pytest.raises(ValueError, match="slippage_ratio"):
        api(prices(), slippage_ratio=value)


def test_missing_mark_column_is_an_error():
    with pytest.raises(ValueError, match="Missing required"):
        bt.backtest(prices(), mark_price="absent")


def test_touched_exit_with_separate_marks_uses_touch_then_slippage():
    df = prices(
        weights=(1.0,) * 4, opens=(100.0,) * 4, closes=(100.0, 102.0, 100.0, 100.0)
    ).with_columns(pl.lit(104.0).alias("high"), pl.lit(90.0).alias("low"))
    report = run(
        df,
        trade_at_price="open",
        mark_price="close",
        touched_exit=True,
        stop_loss=0.05,
        slippage_ratio=0.01,
    )
    stop = report.fills.filter(pl.col("reason") == "stop")
    assert stop["date"][0] == dt.date(2024, 1, 3)
    assert stop["price"][0] == pytest.approx(101.0 * 0.95 * 0.99)


@pytest.mark.parametrize("touched", [False, True])
def test_collecting_ledger_with_open_marks_changes_no_bits(touched):
    df = inputs("signed")
    options = dict(
        trade_at_price="open",
        mark_price="close",
        slippage_ratio=0.005,
        resample="W",
        touched_exit=touched,
        stop_loss=0.04,
    )
    plain = bt.backtest_with_report(df, **options)
    ledger = bt.backtest_with_report(df, ledger=True, **options)
    assert_frame_equal(plain.creturn, ledger.creturn, check_exact=True)
    assert_frame_equal(plain.trades, ledger.trades, check_exact=True)
