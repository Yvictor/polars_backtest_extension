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


def test_stop_timing_is_prefix_invariant():
    df = prices(weights=(1.0,) * 4, opens=(100.0,) * 4, closes=(100.0,) * 4).with_columns(
        pl.lit(105.0).alias("high"), pl.lit(99.0).alias("low")
    )
    options = dict(trade_at_price="open", mark_price="close", touched_exit=True, take_profit=0.04)
    original = run(df, **options)
    changed = run(df.with_columns(pl.Series("close", [100.0, 100.0, 100.0, 101.0])), **options)
    prefix = run(df.head(3), **options)
    for report in (changed, prefix):
        assert_frame_equal(original.daily.head(3), report.daily.head(3), check_exact=True)
        assert_frame_equal(
            original.fills.filter(pl.col("date") <= dt.date(2024, 1, 4)),
            report.fills.filter(pl.col("date") <= dt.date(2024, 1, 4)),
            check_exact=True,
        )


def test_touched_gap_exit_uses_open_after_prior_gain():
    df = prices(
        weights=(1.0,) * 4, opens=(100.0, 100.0, 80.0, 80.0), closes=(100.0, 100.0, 90.0, 90.0)
    ).with_columns(
        pl.Series("high", [105.0, 105.0, 95.0, 95.0]),
        pl.Series("low", [99.0, 99.0, 79.0, 79.0]),
    )
    report = run(df, trade_at_price="open", mark_price="close", touched_exit=True, stop_loss=0.1)
    stop = report.fills.filter(pl.col("reason") == "stop")
    assert stop["price"][0] == pytest.approx(80.0)
    assert report.creturn["creturn"][-1] == pytest.approx(0.8)


def test_touched_exit_metrics_exclude_the_later_closing_mark():
    df = prices(
        weights=(1.0,) * 4, opens=(100.0, 100.0, 80.0, 80.0), closes=(100.0, 100.0, 200.0, 200.0)
    ).with_columns(
        pl.Series("high", [105.0, 105.0, 210.0, 210.0]),
        pl.Series("low", [99.0, 99.0, 79.0, 79.0]),
    )
    report = run(df, trade_at_price="open", mark_price="close", touched_exit=True, stop_loss=0.1)
    trade = report.trades.row(0, named=True)
    assert trade["return"] == pytest.approx(-0.2)
    assert trade["mae"] == pytest.approx(-0.2)
    assert trade["mdd"] == pytest.approx(-0.2)
    assert trade["gmfe"] == pytest.approx(0.0)


@pytest.mark.parametrize("stopped", [False, True])
def test_exit_observation_is_in_metrics_without_later_close(stopped):
    df = prices(
        weights=(1.0, 0.0, 0.0, 0.0),
        opens=(100.0, 100.0, 50.0, 50.0),
        closes=(100.0, 100.0, 200.0, 200.0),
    )
    if stopped:
        df = df.with_columns(pl.Series("close", [100.0, 90.0, 200.0, 200.0]))
    report = run(df, trade_at_price="open", mark_price="close", stop_loss=0.05 if stopped else 1.0)
    trade = report.trades.row(0, named=True)
    assert trade["return"] == pytest.approx(-0.5)
    assert trade["mae"] == pytest.approx(-0.5)
    assert trade["mdd"] == pytest.approx(-0.5)
    assert trade["gmfe"] == pytest.approx(0.0)


@pytest.mark.parametrize("weight", [0.5, 0.9])
def test_continuing_position_resets_to_unslipped_basis(weight):
    df = prices(weights=(0.8, weight, weight, weight), opens=(100.0,) * 4, closes=(100.0,) * 4)
    df = df.vstack(df.tail(1).with_columns(pl.lit(dt.date(2024, 1, 6)).alias("date")))
    report = run(df, slippage_ratio=0.1, take_profit=0.05, retain_cost_when_rebalance=False)
    assert report.fills.filter(pl.col("reason") == "stop").is_empty()
    assert report.trades["entry_price"][1] == 100.0


def test_buy_fee_base_is_continuous_at_zero_slippage():
    df = prices(opens=(100.0,) * 4, closes=(100.0,) * 4)
    options = dict(resample=None, fee_ratio=0.01, tax_ratio=0.02, ledger=True)
    plain = bt.backtest_with_report(df, **options)
    tiny = bt.backtest_with_report(df, slippage_ratio=1e-12, **options)
    assert tiny.daily["commission"].to_list() == pytest.approx(
        plain.daily["commission"].to_list(), abs=1e-12
    )
    assert tiny.creturn["creturn"].to_list() == pytest.approx(
        plain.creturn["creturn"].to_list(), abs=3e-12
    )


@pytest.mark.parametrize(
    "weights",
    [(0.8, 0.5, 0.0, 0.0), (0.5, 0.8, 0.0, 0.0), (1.0, -1.0, 0.0, 0.0), (-1.0, 1.0, 0.0, 0.0)],
)
def test_net_leg_costs_reconcile_with_flat_equity(weights):
    df = prices(weights=weights, opens=(100.0,) * 4, closes=(100.0,) * 4)
    report = bt.backtest_with_report(
        df, resample=None, fee_ratio=0.01, tax_ratio=0.02, slippage_ratio=0.1, ledger=True
    )
    costs = report.daily.select(pl.sum_horizontal("commission", "tax", "slippage")).to_series()
    assert report.daily["equity"].to_list() == pytest.approx((1.0 - costs.cum_sum()).to_list())
    assert report.daily["traded_notional"].sum() == pytest.approx(
        report.fills["notional"].abs().sum()
    )


@pytest.mark.parametrize("resample", [None, "D", "W"])
@pytest.mark.parametrize("slippage", [0.0, 0.01])
@pytest.mark.parametrize(
    "stop_options",
    [
        {"trail_stop": 0.1},
        {"trail_stop": 0.1, "stop_loss": 0.2, "take_profit": 0.3},
        {"trail_stop": 0.1, "stop_loss": 0.1, "take_profit": 0.2},
    ],
)
def test_intraday_fills_ignore_same_day_close_and_future_rows(resample, slippage, stop_options):
    # Friday signals enter on Monday, so weekly and daily rebalances share the open.
    df = prices(weights=(1.0,) * 4, opens=(100.0,) * 4, closes=(100.0,) * 4).with_columns(
        pl.Series("date", [dt.date(2024, 1, d) for d in (5, 8, 9, 10)]),
        pl.Series("high", [100.0, 125.0, 100.0, 100.0]),
        pl.Series("low", [100.0, 99.0, 100.0, 100.0]),
    )
    options = dict(
        trade_at_price="open",
        mark_price="close",
        touched_exit=True,
        resample=resample,
        slippage_ratio=slippage,
        fee_ratio=0.0,
        tax_ratio=0.0,
        ledger=True,
        **stop_options,
    )
    original = bt.backtest_with_report(df, **options)
    variants = [df.head(2)]
    for close in (100.0, 120.0):
        variants.append(
            df.with_columns(
                pl.Series("close", [100.0, close, 1000.0, 1.0]),
                pl.Series("open", [100.0, 100.0, 500.0, 2.0]),
                pl.Series("high", [100.0, 125.0, 1500.0, 10.0]),
                pl.Series("low", [100.0, 99.0, 400.0, 0.5]),
                pl.Series("weight", [1.0, 1.0, -1.0, 0.0]),
            )
        )
    cutoff = dt.date(2024, 1, 8)
    fills = original.fills.filter(pl.col("date") <= cutoff)
    # High then low: the high raises the trail, then the low crosses it.
    expected = (125.0 - 0.1 * (100.0 * (1.0 + slippage))) * (1.0 - slippage)
    if stop_options.get("take_profit") == 0.2:
        expected = 100.0 * (1.0 + slippage) * 1.2 * (1.0 - slippage)
    assert fills.filter(pl.col("reason") == "stop")["price"].to_list() == pytest.approx([expected])
    for variant in variants:
        report = bt.backtest_with_report(variant, **options)
        assert_frame_equal(fills, report.fills.filter(pl.col("date") <= cutoff), check_exact=True)
        assert_frame_equal(original.daily.head(2), report.daily.head(2), check_exact=True)
        completed = pl.col("exit_date") <= cutoff
        assert_frame_equal(
            original.trades.filter(completed), report.trades.filter(completed), check_exact=True
        )
        assert_frame_equal(
            bt.backtest(variant, **{k: v for k, v in options.items() if k != "ledger"}),
            report.creturn,
            check_exact=True,
        )


@pytest.mark.parametrize("resample", [None, "D", "W"])
@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("slippage", [0.0, 0.01])
@pytest.mark.parametrize(
    "stop_options,gap,expected",
    [
        ({"stop_loss": 0.1}, 80.0, 80.0),
        ({"trail_stop": 0.1}, 80.0, 80.0),
        ({"take_profit": 0.1}, 120.0, 120.0),
        ({"stop_loss": 0.1, "trail_stop": 0.1, "take_profit": 0.1}, 80.0, 80.0),
    ],
)
def test_carried_opening_gap_precedes_rebalance(
    resample, retain, slippage, stop_options, gap, expected
):
    df = prices(
        weights=(1.0,) * 4,
        opens=(100.0, 100.0, gap, 100.0),
        closes=(100.0, 100.0, 90.0 if gap < 100.0 else 110.0, 100.0),
    )
    df = df.with_columns(
        pl.Series("date", [dt.date(2024, 1, d) for d in (5, 8, 15, 16)]),
        (pl.max_horizontal("open", "close") + 1.0).alias("high"),
        (pl.min_horizontal("open", "close") - 1.0).alias("low"),
    )
    report = bt.backtest_with_report(
        df,
        trade_at_price="open",
        mark_price="close",
        touched_exit=True,
        resample=resample,
        retain_cost_when_rebalance=retain,
        slippage_ratio=slippage,
        fee_ratio=0.0,
        tax_ratio=0.0,
        ledger=True,
        **stop_options,
    )
    stops = report.fills.filter(pl.col("reason") == "stop")
    assert stops["date"].to_list() == [dt.date(2024, 1, 15)]
    assert stops["price"].to_list() == pytest.approx([expected * (1.0 - slippage)])
    assert report.daily.filter(pl.col("date") == dt.date(2024, 1, 15))["gross_long"][0] == 0.0


@pytest.mark.parametrize("separate_marks", [False, True])
@pytest.mark.parametrize("exit_kind", ["rebalance", "close_stop", "touched_stop"])
def test_slipped_final_fill_is_observed_without_separate_marks(separate_marks, exit_kind):
    df = prices(weights=(1.0, 0.0, 0.0, 0.0), opens=(100.0,) * 4, closes=(100.0,) * 4)
    options = dict(slippage_ratio=0.1)
    if separate_marks:
        options.update(trade_at_price="open", mark_price="close")
    if exit_kind != "rebalance":
        options["stop_loss"] = 0.05
    if exit_kind == "touched_stop":
        options["touched_exit"] = True
        df = df.with_columns(pl.lit(100.0).alias("high"), pl.lit(100.0).alias("low"))
    report = run(df, **options)
    trade = report.trades.filter(pl.col("exit_date").is_not_null()).row(0, named=True)
    assert trade["entry_price"] == pytest.approx(110.0)
    assert trade["exit_price"] == pytest.approx(90.0)
    expected = 90.0 / 110.0 - 1.0
    assert trade["return"] == pytest.approx(expected)
    assert trade["mae"] == pytest.approx(expected)
    assert trade["mdd"] == pytest.approx(expected)


@pytest.mark.parametrize("touched", [False, True])
@pytest.mark.parametrize("separate_marks", [False, True])
@pytest.mark.parametrize("resample", ["D", "W"])
@pytest.mark.parametrize("slippage", [0.0, 0.01])
def test_combined_stops_marks_and_slippage_reconcile(touched, separate_marks, resample, slippage):
    df = prices(weights=(1.0, 0.0, 0.0, 0.0), opens=(100.0,) * 4, closes=(100.0,) * 4)
    df = df.with_columns(
        pl.Series("date", [dt.date(2024, 1, d) for d in (5, 8, 15, 16)]),
        pl.lit(104.0).alias("high"),
        pl.lit(96.0).alias("low"),
    )
    options = dict(
        trade_at_price="open",
        mark_price="close" if separate_marks else None,
        touched_exit=touched,
        resample=resample,
        slippage_ratio=slippage,
        stop_loss=0.2,
        take_profit=0.3,
        trail_stop=0.1,
        fee_ratio=0.0,
        tax_ratio=0.0,
    )
    report = bt.backtest_with_report(df, ledger=True, **options)
    plain = bt.backtest_with_report(df, **options)
    expected = (1.0 - slippage) / (1.0 + slippage)
    assert report.creturn["creturn"][-1] == pytest.approx(expected)
    assert report.fills["price"].to_list() == pytest.approx(
        [100.0 * (1.0 + slippage), 100.0 * (1.0 - slippage)]
    )
    assert report.fills.filter(pl.col("reason") == "stop").is_empty()
    assert report.daily["slippage"].sum() == pytest.approx(1.0 - expected)
    assert_frame_equal(report.creturn, bt.backtest(df, **options), check_exact=True)
    assert_frame_equal(report.creturn, plain.creturn, check_exact=True)
    assert_frame_equal(report.trades, plain.trades, check_exact=True)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("target", [0.05, 1.0])
@pytest.mark.parametrize("separate_marks", [False, True])
@pytest.mark.parametrize("costs", [(0.0, 0.0), (0.01, 0.02)])
def test_slipped_reversal_closes_before_opposite_entry(direction, target, separate_marks, costs):
    df = prices(
        weights=(direction, -direction * target, -direction * target, -direction * target),
        opens=(100.0,) * 4,
        closes=(100.0,) * 4,
    )
    options = dict(
        resample=None,
        fee_ratio=costs[0],
        tax_ratio=costs[1],
        slippage_ratio=0.1,
        trade_at_price="open",
        mark_price="close" if separate_marks else None,
    )
    report = bt.backtest_with_report(df, ledger=True, **options)
    reversal = report.daily.row(2, named=True)
    opposite = "gross_short" if direction < 0 else "gross_long"
    entered = "gross_long" if direction < 0 else "gross_short"
    assert reversal[opposite] == 0.0
    budget = report.daily["equity"][1] * target
    expected = budget * (1.0 - costs[0]) / 1.1 if direction < 0 else budget
    assert reversal[entered] == pytest.approx(expected)
    legs = report.fills.filter(pl.col("date") == dt.date(2024, 1, 4))
    assert legs.height == 2
    # The reversal keeps the legacy entry tax classification for both legs.
    assert reversal["tax"] == 0.0
    old_value = report.daily["gross_short" if direction < 0 else "gross_long"][1]
    assert abs(legs["notional"][0]) == pytest.approx(
        old_value * (1.1 if direction < 0 else 0.9)
    )
    assert legs["price"].to_list() == pytest.approx([110.0 if direction < 0 else 90.0] * 2)
    assert report.trades["position"][1] == -direction * target
    assert report.trades["entry_price"][1] == pytest.approx(110.0 if direction < 0 else 90.0)
    daily_cost = report.daily.select(pl.sum_horizontal("commission", "tax", "slippage"))
    assert report.daily["equity"].to_list() == pytest.approx(
        (1.0 - daily_cost.to_series().cum_sum()).to_list()
    )
    plain = bt.backtest_with_report(df, **options)
    assert_frame_equal(report.creturn, bt.backtest(df, **options), check_exact=True)
    assert_frame_equal(report.creturn, plain.creturn, check_exact=True)
    assert_frame_equal(report.trades, plain.trades, check_exact=True)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_adverse_intraday_extremes_cannot_loosen_marked_trail(direction):
    df = prices(weights=(direction,) * 4, opens=(100.0,) * 4, closes=(100.0,) * 4)
    df = df.with_columns(
        pl.Series("high", [100.0, 109.0, 115.0, 100.0] if direction < 0 else [100.0] * 4),
        pl.Series("low", [100.0] * 4 if direction < 0 else [100.0, 91.0, 85.0, 100.0]),
    )
    options = dict(trade_at_price="open", mark_price="close", touched_exit=True, trail_stop=0.1)
    report = run(df, **options)
    stops = report.fills.filter(pl.col("reason") == "stop")
    assert stops["date"].to_list() == [dt.date(2024, 1, 4)]
    assert stops["price"].to_list() == pytest.approx([110.0 if direction < 0 else 90.0])
    assert report.daily["equity"][-1] == pytest.approx(0.9)
    assert_frame_equal(
        report.creturn,
        bt.backtest(df, resample=None, fee_ratio=0.0, tax_ratio=0.0, **options),
        check_exact=True,
    )


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("favorable", [False, True])
def test_closing_marks_keep_directional_trailing_extreme(direction, favorable):
    closes = [100.0, 85.0, 96.0, 100.0] if favorable else [100.0, 109.0, 115.0, 100.0]
    if direction > 0:
        closes = [200.0 - price for price in closes]
    df = prices(weights=(direction,) * 4, opens=(100.0,) * 4, closes=closes)
    report = run(df, trade_at_price="open", mark_price="close", trail_stop=0.1)
    stops = report.fills.filter(pl.col("reason") == "stop")
    assert stops["date"].to_list() == [dt.date(2024, 1, 5)]
    assert stops["price"].to_list() == [100.0]


@pytest.mark.parametrize("slippage", [0.0, 0.01])
@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("resample", [None, "D", "W"])
def test_short_favorable_low_carries_to_next_high_before_closing_mark(slippage, retain, resample):
    df = prices(
        weights=(-1.0,) * 4,
        opens=(100.0, 100.0, 89.0, 100.0),
        closes=(100.0, 85.0, 94.0, 100.0),
    ).with_columns(
        pl.Series("date", [dt.date(2024, 1, d) for d in (5, 8, 15, 16)]),
        pl.Series("high", [100.0, 100.0, 96.0, 100.0]),
        pl.Series("low", [100.0, 80.0, 85.0, 100.0]),
    )
    options = dict(
        trade_at_price="open", mark_price="close", touched_exit=True, trail_stop=0.1,
        slippage_ratio=slippage, retain_cost_when_rebalance=retain, resample=resample,
        fee_ratio=0.0, tax_ratio=0.0,
    )
    report = bt.backtest_with_report(df, ledger=True, **options)
    stops = report.fills.filter(pl.col("reason") == "stop")
    if resample is None or retain:
        # The low tightens only after the first high; its threshold survives the close.
        expected = (80.0 + 0.1 * 100.0 * (1.0 - slippage)) * (1.0 + slippage)
        assert stops["date"].to_list() == [dt.date(2024, 1, 15)]
        assert stops["price"].to_list() == pytest.approx([expected])
    else:
        # Resetting at the second open intentionally starts a fresh trailing basis.
        assert stops.filter(pl.col("date") <= dt.date(2024, 1, 15)).is_empty()
    assert_frame_equal(report.creturn, bt.backtest(df, **options), check_exact=True)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("at_open", [False, True])
def test_favorable_open_or_mark_tightens_trail_before_later_rebound(direction, at_open):
    favorable = 80.0 if direction < 0 else 120.0
    rebound = 95.0 if direction < 0 else 105.0
    df = prices(weights=(direction,) * 4, opens=(100.0,) * 4, closes=(100.0,) * 4)
    df = df.with_columns(
        pl.Series("open", [100.0, 100.0, favorable if at_open else 100.0, 100.0]),
        pl.Series("close", [100.0, 100.0, rebound if at_open else favorable, rebound]),
    )
    df = df.vstack(df.tail(1).with_columns(
        pl.lit(dt.date(2024, 1, 6)).alias("date"),
        pl.lit(100.0).alias("open"), pl.lit(100.0).alias("close"),
    ))
    report = run(df, trade_at_price="open", mark_price="close", trail_stop=0.1)
    stops = report.fills.filter(pl.col("reason") == "stop")
    expected_date = dt.date(2024, 1, 5 if at_open else 6)
    assert stops["date"].to_list() == [expected_date]
    assert stops["price"].to_list() == [100.0]


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_slipped_entry_does_not_loosen_initial_marked_trail(direction):
    df = prices(weights=(direction,) * 4, opens=(100.0,) * 4, closes=(100.0,) * 4)
    df = df.with_columns(pl.lit(100.0).alias("high"), pl.lit(100.0).alias("low"))
    report = run(
        df.head(2), trade_at_price="open", mark_price="close", touched_exit=True,
        slippage_ratio=0.1, trail_stop=0.05,
    )
    stops = report.fills.filter(pl.col("reason") == "stop")
    assert stops["date"].to_list() == [dt.date(2024, 1, 3)]
    assert stops["price"].to_list() == pytest.approx([110.0 if direction < 0 else 90.0])


@pytest.mark.parametrize("resample", [None, "D", "W"])
@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("slippage", [0.0, 0.01])
def test_short_adverse_opening_gap_cannot_loosen_trail_before_rebalance(resample, retain, slippage):
    df = prices(
        weights=(-1.0,) * 4,
        opens=(100.0, 100.0, 115.0, 100.0),
        closes=(100.0, 100.0, 105.0, 100.0),
    ).with_columns(
        pl.Series("date", [dt.date(2024, 1, d) for d in (5, 8, 15, 16)]),
        pl.max_horizontal("open", "close").alias("high"),
        pl.min_horizontal("open", "close").alias("low"),
    )
    report = bt.backtest_with_report(
        df, trade_at_price="open", mark_price="close", touched_exit=True,
        resample=resample, retain_cost_when_rebalance=retain, slippage_ratio=slippage,
        trail_stop=0.1, fee_ratio=0.0, tax_ratio=0.0, ledger=True,
    )
    stops = report.fills.filter(pl.col("reason") == "stop")
    assert stops["date"].to_list() == [dt.date(2024, 1, 15)]
    assert stops["price"].to_list() == pytest.approx([115.0 * (1.0 + slippage)])
    assert report.daily.filter(pl.col("date") == dt.date(2024, 1, 15))["gross_short"][0] == 0.0
