"""Short-sale and calendar-day carrying costs in the existing long loop."""

import datetime as dt
import json

import polars as pl
import polars_backtest as bt
import pytest
from legacy_fixtures import frame
from multi_name_fixtures import CASES, inputs
from polars.testing import assert_frame_equal


def prices(weights=(-1.0, -1.0, 0.0, 0.0)):
    return pl.DataFrame(
        {
            "date": [dt.date(2024, 1, d) for d in (4, 5, 8, 9)],
            "symbol": ["A"] * 4,
            "close": [100.0] * 4,
            "weight": weights,
        }
    )


def run(df, **options):
    return bt.backtest_with_report(
        df, resample=None, fee_ratio=0.0, tax_ratio=0.0, ledger=True, **options
    )


def test_borrow_cost_uses_previous_mark_and_calendar_days():
    df = prices().with_columns(pl.Series("close", [100.0, 100.0, 110.0, 110.0]))
    report = run(df, borrow_fee_ratio=0.365)
    assert report.daily["borrow_cost"].to_list() == pytest.approx([0.0, 0.0, 0.003, 0.0011])
    assert report.creturn["creturn"][-1] == pytest.approx(0.8959)
    assert report.cashflows["amount"].to_list() == pytest.approx([-0.003, -0.0011])
    assert report.cashflows["kind"].to_list() == ["borrow_fee", "borrow_fee"]
    assert report.cashflows["date"].to_list() == [dt.date(2024, 1, 8), dt.date(2024, 1, 9)]


def test_short_sale_fee_only_charges_new_short_notional():
    report = run(prices(), short_sale_fee_ratio=0.01)
    assert report.daily["surcharge"].to_list() == pytest.approx([0.0, 0.01, 0.0, 0.0])
    assert report.creturn["creturn"][-1] == pytest.approx(0.99)


def test_sell_tax_charges_short_sale_and_not_cover():
    report = bt.backtest_with_report(
        prices(), resample=None, fee_ratio=0.0, tax_ratio=0.02, tax_side="sell", ledger=True
    )
    assert report.daily["tax"].to_list() == pytest.approx([0.0, 0.02, 0.0, 0.0])
    assert report.creturn["creturn"][-1] == pytest.approx(0.98)


@pytest.mark.parametrize("weights", [(1.0, -1.0, -1.0, -1.0), (-1.0, 1.0, 1.0, 1.0)])
def test_reversal_records_close_then_new_position(weights):
    report = run(prices(weights), short_sale_fee_ratio=0.01, tax_side="sell")
    reversal = report.fills.filter(pl.col("date") == dt.date(2024, 1, 8))
    assert reversal.height == 2
    if weights[0] > 0:
        assert reversal["notional"].to_list() == pytest.approx([-1.0, -1.0])
        assert report.daily["surcharge"].sum() == pytest.approx(0.01)
    else:
        assert reversal["notional"].to_list() == pytest.approx([1.0, 0.99])
        assert report.daily["surcharge"].sum() == pytest.approx(0.01)


def test_long_positions_have_no_short_costs():
    report = run(prices((1.0, 1.0, 0.0, 0.0)), short_sale_fee_ratio=0.01, borrow_fee_ratio=0.365)
    assert report.daily["surcharge"].sum() == 0.0
    assert report.daily["borrow_cost"].sum() == 0.0
    assert report.cashflows.is_empty()


def test_slippage_combines_with_short_fees_and_sell_tax():
    report = bt.backtest_with_report(
        prices(),
        resample=None,
        fee_ratio=0.01,
        tax_ratio=0.02,
        slippage_ratio=0.1,
        short_sale_fee_ratio=0.03,
        borrow_fee_ratio=0.365,
        tax_side="sell",
        ledger=True,
    )
    assert report.fills["price"].to_list() == pytest.approx([90.0, 110.0])
    assert report.daily["surcharge"].sum() == pytest.approx(0.9 * 0.03)
    assert report.daily["tax"].sum() == pytest.approx(0.9 * 0.02)
    assert report.daily["commission"].sum() == pytest.approx(2.0 * 0.01)
    assert report.daily["borrow_cost"].sum() == pytest.approx(0.004)
    costs = report.daily.select(
        pl.sum_horizontal("commission", "tax", "slippage", "surcharge", "borrow_cost").sum()
    ).item()
    assert report.creturn["creturn"][-1] == pytest.approx(1.0 - costs)


@pytest.mark.parametrize("name", CASES)
def test_explicit_fee_defaults_are_bit_exact(name):
    kind, options = CASES[name]
    df = inputs(kind)
    plain = bt.backtest_with_report(df, **options)
    explicit = bt.backtest_with_report(
        df, short_sale_fee_ratio=0.0, borrow_fee_ratio=0.0, tax_side="exit", **options
    )
    assert frame(plain.creturn) == frame(explicit.creturn)
    assert json.dumps(frame(plain.trades, keyed=True)) == json.dumps(
        frame(explicit.trades, keyed=True)
    )
    assert_frame_equal(
        bt.backtest(df, **options),
        bt.backtest(df, short_sale_fee_ratio=0.0, borrow_fee_ratio=0.0, tax_side="exit", **options),
        check_exact=True,
    )


@pytest.mark.parametrize("option", ["short_sale_fee_ratio", "borrow_fee_ratio"])
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
@pytest.mark.parametrize("api", [bt.backtest, bt.backtest_with_report])
def test_invalid_fee_ratios(option, value, api):
    with pytest.raises(ValueError, match=option):
        api(prices(), **{option: value})


def test_invalid_tax_side():
    with pytest.raises(ValueError, match="tax_side"):
        bt.backtest(prices(), tax_side="both")


def test_partial_cover_has_no_new_short_fee_or_sell_tax():
    report = bt.backtest_with_report(
        prices((-1.0, -0.5, -0.5, -0.5)),
        resample=None,
        fee_ratio=0.0,
        tax_ratio=0.02,
        tax_side="sell",
        short_sale_fee_ratio=0.01,
        ledger=True,
    )
    assert report.fills["notional"].to_list() == pytest.approx([-1.0, 0.515])
    assert report.daily["surcharge"].sum() == pytest.approx(0.01)
    assert report.daily["tax"].sum() == pytest.approx(0.02)


def test_short_increase_charges_only_the_added_notional():
    report = run(prices((-0.5, -1.0, -1.0, -1.0)), short_sale_fee_ratio=0.01)
    assert report.fills["notional"].to_list() == pytest.approx([-0.5, -0.495])
    assert report.daily["surcharge"].sum() == pytest.approx(0.00995)


def test_short_stop_cover_is_exempt_from_sell_tax():
    df = prices().with_columns(pl.Series("close", [100.0, 100.0, 110.0, 110.0]))
    report = bt.backtest_with_report(
        df,
        resample=None,
        fee_ratio=0.0,
        tax_ratio=0.02,
        tax_side="sell",
        short_sale_fee_ratio=0.01,
        stop_loss=0.05,
        ledger=True,
    )
    assert report.fills["reason"].to_list() == ["rebalance", "stop"]
    assert report.daily["tax"].sum() == pytest.approx(0.02)
    assert report.daily["surcharge"].sum() == pytest.approx(0.01)


def test_borrow_cost_uses_close_mark_with_open_execution():
    df = prices().with_columns(
        pl.lit(100.0).alias("open"), pl.Series("close", [100.0, 120.0, 120.0, 120.0])
    )
    report = run(df, trade_at_price="open", mark_price="close", borrow_fee_ratio=0.365)
    assert report.daily["borrow_cost"].to_list() == pytest.approx([0.0, 0.0, 0.0036, 0.0012])


def test_final_missing_price_still_accrues_borrow_cost():
    df = prices().with_columns(pl.Series("close", [100.0, 100.0, 100.0, float("nan")]))
    report = run(df, borrow_fee_ratio=0.365)
    assert report.daily["borrow_cost"].to_list() == pytest.approx([0.0, 0.0, 0.003, 0.001])


def test_borrow_cashflows_are_opt_in():
    assert bt.backtest_with_report(prices(), borrow_fee_ratio=0.365).cashflows is None


@pytest.mark.parametrize("touched", [False, True])
def test_fee_options_match_simple_api_and_ledger_off(touched):
    df = inputs("signed")
    options = dict(
        trade_at_price="open",
        mark_price="close",
        slippage_ratio=0.005,
        short_sale_fee_ratio=0.01,
        borrow_fee_ratio=0.05,
        tax_side="sell",
        touched_exit=touched,
        stop_loss=0.04,
        resample="W",
    )
    plain = bt.backtest_with_report(df, **options)
    ledger = bt.backtest_with_report(df, ledger=True, **options)
    assert_frame_equal(plain.creturn, ledger.creturn, check_exact=True)
    assert_frame_equal(plain.trades, ledger.trades, check_exact=True)
    assert_frame_equal(bt.backtest(df, **options), ledger.creturn, check_exact=True)
