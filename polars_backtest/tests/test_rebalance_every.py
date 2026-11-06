"""Rebalancing every k calendar weeks."""

import datetime as dt

import polars as pl
import pytest
from multi_name_fixtures import inputs
from polars.testing import assert_frame_equal

import polars_backtest as bt

ANCHOR = dt.date(2020, 1, 6)  # a Monday
PLAIN = {"resample": "W", "fee_ratio": 0.0, "tax_ratio": 0.0}


def days(first=dt.date(2020, 1, 1), count=45):
    dates = (first + dt.timedelta(days=i) for i in range(count))
    return [d for d in dates if d.weekday() < 5]


def data(dates=None):
    dates = dates or days()
    return pl.DataFrame(
        {
            "date": dates,
            "symbol": ["A"] * len(dates),
            "close": [100.0 + i for i in range(len(dates))],
            "weight": [0.5] * len(dates),
        }
    )


def signal_dates(df, **options):
    report = bt.backtest_with_report(df, **{**PLAIN, **options})
    return sorted(set(report.trades["entry_sig_date"].to_list()))


@pytest.mark.parametrize(
    "phase,expected",
    [
        (0, [dt.date(2020, 1, 10), dt.date(2020, 1, 24), dt.date(2020, 2, 7)]),
        (1, [dt.date(2020, 1, 3), dt.date(2020, 1, 17), dt.date(2020, 1, 31)]),
    ],
)
def test_phases_pick_alternating_weeks(phase, expected):
    options = {"rebalance_every": 2, "rebalance_phase": phase, "rebalance_anchor": ANCHOR}
    assert signal_dates(data(), **options) == expected


def test_phases_do_not_overlap_and_cover_every_week():
    picked = [
        set(signal_dates(data(), rebalance_every=3, rebalance_phase=p, rebalance_anchor=ANCHOR))
        for p in range(3)
    ]
    assert sum(len(p) for p in picked) == len(set.union(*picked))
    assert set.union(*picked) == set(signal_dates(data()))


def test_any_date_in_the_anchor_week_is_the_same_anchor():
    assert signal_dates(
        data(), rebalance_every=2, rebalance_anchor=dt.date(2020, 1, 11)
    ) == signal_dates(data(), rebalance_every=2, rebalance_anchor=ANCHOR)


def test_slicing_the_input_keeps_the_phase():
    options = {"rebalance_every": 2, "rebalance_anchor": ANCHOR}
    full = signal_dates(data(), **options)
    start = dt.date(2020, 1, 15)
    sliced = signal_dates(data().filter(pl.col("date") >= start), **options)
    assert sliced == [d for d in full if d >= start]


def test_anchor_after_the_data_counts_backwards():
    options = {"rebalance_every": 2, "rebalance_anchor": ANCHOR + dt.timedelta(days=700)}
    assert signal_dates(data(), **options) == signal_dates(
        data(), rebalance_every=2, rebalance_anchor=ANCHOR
    )


def test_a_closed_week_counts_and_produces_no_signal():
    closed = [d for d in days() if not dt.date(2020, 1, 20) <= d <= dt.date(2020, 1, 24)]
    assert signal_dates(data(closed), rebalance_every=2, rebalance_anchor=ANCHOR) == [
        dt.date(2020, 1, 10),
        dt.date(2020, 2, 7),
    ]


def test_long_holiday_keeps_the_calendar_phase():
    # Closed for two full weeks, like a lunar new year: phase is calendar based
    closed = [d for d in days() if not dt.date(2020, 1, 13) <= d <= dt.date(2020, 1, 24)]
    assert signal_dates(data(closed), rebalance_every=2, rebalance_anchor=ANCHOR) == [
        dt.date(2020, 1, 10),
        dt.date(2020, 2, 7),
    ]


def test_weekday_resample_selects_the_same_weeks():
    assert signal_dates(
        data(), resample="W-WED", rebalance_every=2, rebalance_anchor=ANCHOR
    ) == [dt.date(2020, 1, 8), dt.date(2020, 1, 22), dt.date(2020, 2, 5)]


def test_offset_delays_the_selected_boundary():
    # Week 0 ends on Sunday Jan 12; +3 days is Wednesday Jan 15
    assert signal_dates(
        data(), resample_offset="3D", rebalance_every=2, rebalance_anchor=ANCHOR
    ) == [dt.date(2020, 1, 15), dt.date(2020, 1, 29), dt.date(2020, 2, 12)]


def test_offset_finds_a_selected_week_inside_a_gap():
    # Jan 10 sits in an unselected week. The selected week after it is closed, but
    # its boundary Jan 19 + 7D = Jan 26 still falls due, so Jan 10 is the signal.
    closed = [d for d in days() if not dt.date(2020, 1, 13) <= d <= dt.date(2020, 1, 24)]
    assert signal_dates(
        data(closed),
        resample_offset="7D",
        rebalance_every=3,
        rebalance_phase=1,
        rebalance_anchor=ANCHOR,
    ) == [dt.date(2020, 1, 10)]


def test_nothing_trades_before_the_first_selected_week():
    options = {**PLAIN, "rebalance_every": 2, "rebalance_anchor": ANCHOR}
    # Signal Jan 10 enters on Jan 13, and creturn starts at the entry day
    assert bt.backtest(data(), **options)["date"][0] == dt.date(2020, 1, 13)


def test_every_one_is_the_default_path():
    df = inputs()
    plain = bt.backtest_with_report(df, resample="W")
    scheduled = bt.backtest_with_report(
        df, resample="W", rebalance_every=1, rebalance_phase=0, rebalance_anchor=ANCHOR
    )
    assert_frame_equal(plain.creturn, scheduled.creturn, check_exact=True)
    assert_frame_equal(plain.trades, scheduled.trades, check_exact=True)


def test_report_and_backtest_agree():
    options = {"resample": "W", "rebalance_every": 2, "rebalance_anchor": ANCHOR}
    df = inputs()
    assert_frame_equal(
        bt.backtest(df, **options),
        bt.backtest_with_report(df, **options).creturn,
        check_exact=True,
    )


def test_every_two_holds_positions_through_the_skipped_week():
    report = bt.backtest_with_report(
        data(), **PLAIN, rebalance_every=2, rebalance_anchor=ANCHOR
    )
    closed = report.trades.filter(pl.col("exit_date").is_not_null())
    gaps = (closed["exit_date"] - closed["entry_date"]).dt.total_days().to_list()
    assert gaps == [14, 14]


@pytest.mark.parametrize(
    "phase,expected", [(0, dt.date(2020, 2, 21)), (1, dt.date(2020, 2, 28))]
)
def test_next_weight_date_is_the_next_selected_week(phase, expected):
    df = data(days(count=52))
    report = bt.backtest_with_report(
        df, **PLAIN, rebalance_every=2, rebalance_phase=phase, rebalance_anchor=ANCHOR
    )
    assert report.next_weights()["date"][0] == expected


def test_next_weight_date_with_offset_and_a_holiday():
    # Week 4 ends Sunday Feb 9 and +3D is Wednesday Feb 12, which is closed, so the
    # signal is Tuesday Feb 11. The next date comes from the calendar, week 6 ending
    # Sunday Feb 23 plus 3D, and does not repeat the holiday's shift.
    closed = [d for d in days(count=52) if d != dt.date(2020, 2, 12)]
    report = bt.backtest_with_report(
        data(closed),
        **PLAIN,
        resample_offset="3D",
        rebalance_every=2,
        rebalance_anchor=ANCHOR,
    )
    assert report.next_weights()["date"][0] == dt.date(2020, 2, 26)
    assert report.weights()["date"][0] == dt.date(2020, 2, 11)


def test_next_weight_date_is_not_before_the_reporting_date():
    # The Jan 10 signal comes from a closed week and executes on Jan 27. The next
    # selected week ends Sunday Feb 9, and +7D is after the data ends on Feb 14.
    closed = [d for d in days() if not dt.date(2020, 1, 13) <= d <= dt.date(2020, 1, 24)]
    report = bt.backtest_with_report(
        data(closed),
        **PLAIN,
        resample_offset="7D",
        rebalance_every=3,
        rebalance_phase=1,
        rebalance_anchor=ANCHOR,
    )
    assert report.trades["entry_date"][0] == dt.date(2020, 1, 27)
    assert report.next_weights()["date"][0] == dt.date(2020, 2, 16)


def test_input_inside_an_unselected_week_gives_no_result():
    df = data([dt.date(2020, 1, 1), dt.date(2020, 1, 2), dt.date(2020, 1, 3)])
    options = {**PLAIN, "rebalance_every": 2, "rebalance_anchor": ANCHOR}
    report = bt.backtest_with_report(df, **options)
    assert report.creturn.is_empty()
    assert report.trades.is_empty()
    assert bt.backtest(df, **options).is_empty()


def test_input_inside_a_selected_week_starts_like_every_week():
    df = data([dt.date(2020, 1, 1), dt.date(2020, 1, 2), dt.date(2020, 1, 3)])
    scheduled = bt.backtest_with_report(
        df, **PLAIN, rebalance_every=2, rebalance_phase=1, rebalance_anchor=ANCHOR
    )
    plain = bt.backtest_with_report(df, **PLAIN)
    assert_frame_equal(scheduled.creturn, plain.creturn, check_exact=True)
    # The pending entry is dated by the schedule: Jan 3 is the selected boundary itself
    assert scheduled.trades["entry_sig_date"].to_list() == [dt.date(2020, 1, 3)]


@pytest.mark.parametrize("phase,rows", [(0, 1), (1, 0)])
def test_final_day_fallback_uses_the_week_before_the_offset(phase, rows):
    # Jan 14 minus 3D is Jan 11, in week 0: its signal serves the boundary of Jan 12
    df = data([dt.date(2020, 1, 13), dt.date(2020, 1, 14)])
    report = bt.backtest_with_report(
        df,
        **PLAIN,
        resample_offset="3D",
        rebalance_every=2,
        rebalance_phase=phase,
        rebalance_anchor=ANCHOR,
    )
    assert report.creturn.height == rows


@pytest.mark.parametrize(
    "options,message",
    [
        ({"rebalance_every": True, "rebalance_anchor": ANCHOR}, "int"),
        ({"rebalance_every": 2.0, "rebalance_anchor": ANCHOR}, "int"),
        ({"rebalance_phase": True, "rebalance_every": 2, "rebalance_anchor": ANCHOR}, "int"),
        ({"rebalance_every": 2**31, "rebalance_anchor": ANCHOR}, "out of range"),
        ({"rebalance_every": 5201, "rebalance_anchor": ANCHOR}, "at most"),
        ({"rebalance_phase": -(2**31) - 1, "rebalance_anchor": ANCHOR}, "out of range"),
        ({"rebalance_every": 2, "rebalance_anchor": dt.datetime(2020, 1, 6)}, "date"),
        ({"rebalance_every": 0, "rebalance_anchor": ANCHOR}, "positive"),
        ({"rebalance_every": 2, "rebalance_phase": 2, "rebalance_anchor": ANCHOR}, "phase"),
        ({"rebalance_every": 2, "rebalance_phase": -1, "rebalance_anchor": ANCHOR}, "phase"),
        ({"rebalance_every": 2}, "anchor"),
        ({"rebalance_every": 2, "rebalance_anchor": "2020-01-06"}, "date"),
        ({"rebalance_every": 2, "rebalance_anchor": ANCHOR, "resample": "M"}, "weekly"),
        ({"rebalance_every": 2, "rebalance_anchor": ANCHOR, "resample": None}, "weekly"),
        ({"rebalance_anchor": ANCHOR, "resample": "D"}, "weekly"),
    ],
)
def test_invalid_schedules_raise(options, message):
    with pytest.raises(ValueError, match=message):
        bt.backtest(data(), **{"resample": "W", **options})
    with pytest.raises(ValueError, match=message):
        bt.backtest_with_report(data(), **{"resample": "W", **options})
