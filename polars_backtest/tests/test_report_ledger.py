"""The ledger reports what was booked, and collecting it changes nothing."""

import datetime as dt

import polars as pl
import pytest
from multi_name_fixtures import CASES, STOPS, inputs
from polars.testing import assert_frame_equal

import polars_backtest as bt

DAYS = [dt.date(2020, 1, d) for d in (2, 3, 6, 7, 8)]
FEES = {"fee_ratio": 0.01, "tax_ratio": 0.02}


def one_name(weights, closes):
    return pl.DataFrame(
        {
            "date": DAYS[: len(weights)],
            "symbol": ["A"] * len(weights),
            "close": closes,
            "weight": weights,
        }
    )


def test_ledger_is_off_by_default():
    report = bt.backtest_with_report(inputs())
    assert report.daily is None
    assert report.fills is None


def test_tables_hold_only_real_columns():
    report = bt.backtest_with_report(inputs(), resample="W", ledger=True)
    assert report.fills.columns == ["date", "symbol", "notional", "price", "reason"]
    assert report.daily.columns == [
        "date",
        "equity",
        "cash",
        "commission",
        "tax",
        "gross_long",
        "gross_short",
        "net_exposure",
        "gross_exposure",
        "traded_notional",
        "turnover",
    ]


@pytest.mark.parametrize("name", CASES)
def test_ledger_changes_no_result(name):
    kind, options = CASES[name]
    df = inputs(kind)
    plain = bt.backtest_with_report(df, **options)
    ledger = bt.backtest_with_report(df, ledger=True, **options)
    assert_frame_equal(plain.creturn, ledger.creturn, check_exact=True)
    assert_frame_equal(plain.trades, ledger.trades, check_exact=True)


def test_hand_computed_round_trip():
    df = one_name([1.0, 1.0, 0.0, 0.0], [100.0, 100.0, 110.0, 110.0])
    report = bt.backtest_with_report(df, resample=None, ledger=True, **FEES)

    fills = report.fills.select("date", "symbol", "price", "reason")
    assert fills.rows() == [
        (DAYS[1], "A", 100.0, "rebalance"),
        (DAYS[3], "A", 110.0, "rebalance"),
    ]
    assert report.fills["notional"].to_list() == pytest.approx([1.0, -1.089])

    entry = report.daily.filter(pl.col("date") == DAYS[1]).row(0, named=True)
    assert entry["commission"] == pytest.approx(0.01)
    assert entry["tax"] == 0.0

    daily = report.daily.filter(pl.col("date") == DAYS[3]).row(0, named=True)
    assert daily["cash"] == pytest.approx(1.089 - 0.01089 - 0.02178)
    assert daily["gross_long"] == 0.0
    assert daily["commission"] == pytest.approx(0.01089)
    assert daily["tax"] == pytest.approx(0.02178)
    assert daily["traded_notional"] == pytest.approx(1.089)
    assert daily["turnover"] == pytest.approx(1.089 / 1.089)


def test_buy_to_cover_pays_tax_and_opening_a_short_does_not():
    df = one_name([-1.0, -1.0, 0.0, 0.0], [100.0, 100.0, 100.0, 100.0])
    report = bt.backtest_with_report(df, resample=None, ledger=True, **FEES)
    assert report.fills["notional"].to_list() == pytest.approx([-1.0, 1.0])

    opening = report.daily.filter(pl.col("date") == DAYS[1]).row(0, named=True)
    assert opening["commission"] == pytest.approx(0.01)
    assert opening["tax"] == 0.0

    cover = report.daily.filter(pl.col("date") == DAYS[3]).row(0, named=True)
    assert cover["commission"] == pytest.approx(0.01)
    assert cover["tax"] == pytest.approx(0.02)


def test_partial_cover_pays_tax_on_the_covered_part():
    df = one_name([-1.0, -1.0, -0.5, -0.5], [100.0, 100.0, 100.0, 100.0])
    report = bt.backtest_with_report(df, resample=None, ledger=True, **FEES)
    day = report.daily.filter(pl.col("date") == DAYS[3]).row(0, named=True)
    assert day["traded_notional"] > 0.0
    assert day["tax"] == pytest.approx(day["traded_notional"] * 0.02)


def test_short_position_shows_as_gross_short():
    df = one_name([-0.5, -0.5, -0.5], [100.0, 100.0, 100.0])
    report = bt.backtest_with_report(
        df, resample=None, ledger=True, fee_ratio=0.0, tax_ratio=0.0
    )
    last = report.daily.row(-1, named=True)
    assert last["gross_short"] == pytest.approx(0.5)
    assert last["gross_long"] == 0.0
    assert last["net_exposure"] == pytest.approx(-0.5)
    assert last["gross_exposure"] == pytest.approx(0.5)
    assert last["cash"] == pytest.approx(1.5)
    assert report.fills["notional"].to_list() == pytest.approx([-0.5])


def test_stop_exit_is_a_stop_fill():
    df = one_name([1.0] * 5, [100.0, 100.0, 90.0, 90.0, 90.0])
    report = bt.backtest_with_report(
        df, resample=None, stop_loss=0.05, ledger=True, fee_ratio=0.0, tax_ratio=0.0
    )
    assert report.fills["reason"].to_list() == ["rebalance", "stop"]
    assert report.fills["date"].to_list() == [DAYS[1], DAYS[3]]


def test_touched_exit_fills_at_the_touched_price():
    df = one_name([1.0] * 4, [100.0, 100.0, 100.0, 100.0]).with_columns(
        open=pl.lit(100.0), high=pl.lit(100.0), low=pl.Series([100.0, 100.0, 90.0, 100.0])
    )
    report = bt.backtest_with_report(
        df, resample=None, stop_loss=0.05, touched_exit=True, ledger=True
    )
    stop = report.fills.filter(pl.col("reason") == "stop").row(0, named=True)
    assert stop["price"] == pytest.approx(95.0)


def reconstruct(df, report, fee_ratio, tax_ratio):
    """Replay the fills over the prices, independently of the engine's positions."""
    closes = {(d, s): c for d, s, c in df.select("date", "symbol", "close").rows()}
    fills = report.fills.partition_by("date", as_dict=True)
    held, cash, previous = {}, 1.0, {}
    for row in report.daily.iter_rows(named=True):
        day = row["date"]
        pnl = commission = tax = 0.0
        for symbol in held:
            ratio = closes[day, symbol] / closes[previous[symbol], symbol]
            pnl += held[symbol] * (ratio - 1.0)
            held[symbol] *= ratio
            previous[symbol] = day
        for fill in fills.get((day,), pl.DataFrame()).iter_rows(named=True):
            symbol, notional = fill["symbol"], fill["notional"]
            after = held.get(symbol, 0.0) + notional
            full_exit = abs(after) <= 1e-12
            adds = not full_exit and notional * after > 0
            leg_tax = 0.0 if adds else abs(notional) * tax_ratio
            cost = abs(notional) * fee_ratio + leg_tax
            commission += abs(notional) * fee_ratio
            tax += leg_tax
            # A full exit and a sell pay from cash; a buy that keeps a position
            # pays from the position.
            from_position = notional > 0 and not full_exit
            held[symbol] = after - (cost if from_position else 0.0)
            previous[symbol] = day
            cash -= notional + (0.0 if from_position else cost)
        held = {s: v for s, v in held.items() if abs(v) > 1e-12}
        yield row, pnl, commission, tax, cash, held


@pytest.mark.parametrize(
    "kind,options",
    [
        ("long", {"resample": "W"}),
        ("long", {}),
        ("long", {"resample": "W", **STOPS}),
        ("signed", {"resample": "W", **STOPS}),
    ],
)
def test_daily_identities_hold_on_many_names(kind, options):
    df = inputs(kind)
    report = bt.backtest_with_report(df, ledger=True, **FEES, **options)
    previous_equity = 1.0
    replay = reconstruct(df, report, FEES["fee_ratio"], FEES["tax_ratio"])
    for row, pnl, commission, tax, cash, held in replay:
        assert row["commission"] == pytest.approx(commission, abs=1e-12)
        assert row["tax"] == pytest.approx(tax, abs=1e-12)
        change = pnl - commission - tax
        assert row["equity"] - previous_equity == pytest.approx(change, abs=1e-12)
        assert row["cash"] == pytest.approx(cash, abs=1e-12)
        assert row["equity"] == pytest.approx(cash + sum(held.values()), abs=1e-12)
        assert row["net_exposure"] == pytest.approx(sum(held.values()), abs=1e-12)
        previous_equity = row["equity"]


def test_daily_equity_is_the_uncut_creturn():
    report = bt.backtest_with_report(inputs(), resample="W", ledger=True)
    joined = report.creturn.join(report.daily, on="date")
    assert joined.height == report.creturn.height
    assert_frame_equal(
        joined.select(pl.col("equity").alias("creturn")),
        report.creturn.select("creturn"),
        check_exact=True,
    )


def test_daily_sums_the_fills():
    report = bt.backtest_with_report(inputs(), resample="W", ledger=True)
    by_day = report.fills.group_by("date").agg(pl.col("notional").abs().sum())
    joined = report.daily.join(by_day, on="date", how="inner")
    assert (joined["notional"] - joined["traded_notional"]).abs().max() < 1e-12
    assert report.daily["traded_notional"].sum() == pytest.approx(
        report.fills["notional"].abs().sum()
    )


def test_rebalance_every_trades_only_in_selected_weeks():
    report = bt.backtest_with_report(
        inputs(),
        resample="W",
        rebalance_every=2,
        rebalance_anchor=dt.date(2021, 1, 4),
        ledger=True,
    )
    traded = report.fills["date"].unique().sort()
    assert (traded.diff().dt.total_days().drop_nulls() >= 13).all()


def test_ledger_with_no_trades_is_empty():
    report = bt.backtest_with_report(
        one_name([0.0, 0.0], [100.0, 100.0]), resample=None, ledger=True
    )
    assert report.fills.is_empty()
    assert report.daily.is_empty()
