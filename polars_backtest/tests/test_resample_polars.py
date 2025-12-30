"""Test polars-based resample functionality.

These tests verify the pure polars implementation of resampling,
without requiring pandas or finlab.
"""

from datetime import date, timedelta

import polars as pl
import pytest
from polars_backtest import backtest_wide, backtest_with_report_wide
from polars_backtest.utils import (
    _get_period_end_dates,
    _parse_offset,
    _parse_resample_freq,
)

# =============================================================================
# parse_resample_freq tests
# =============================================================================


@pytest.mark.parametrize("freq_str,expected_freq,expected_weekday", [
    ("D", "1d", None),
    ("W", "1w", 7),
    ("W-FRI", "1w", 5),
    ("W-MON", "1w", 1),
    ("M", "1mo", None),
    ("ME", "1mo", None),
    ("MS", "1mo_start", None),
    ("Q", "3mo", None),
    ("QE", "3mo", None),
    ("QS", "3mo_start", None),
    ("Y", "1y", None),
    ("YE", "1y", None),
    ("A", "1y", None),
])
def test_parse_resample_freq(freq_str, expected_freq, expected_weekday):
    freq, weekday = _parse_resample_freq(freq_str)
    assert freq == expected_freq
    assert weekday == expected_weekday


def test_parse_resample_freq_invalid():
    with pytest.raises(ValueError, match="Invalid resample frequency"):
        _parse_resample_freq("X")


# =============================================================================
# parse_offset tests
# =============================================================================


@pytest.mark.parametrize("offset_str,expected", [
    ("1D", timedelta(days=1)),
    ("-1D", timedelta(days=-1)),
    ("2W", timedelta(weeks=2)),
    ("3H", timedelta(hours=3)),
    ("", timedelta(0)),
])
def test_parse_offset(offset_str, expected):
    assert _parse_offset(offset_str) == expected


def test_parse_offset_invalid():
    with pytest.raises(ValueError, match="Invalid offset format"):
        _parse_offset("invalid")


# =============================================================================
# get_period_end_dates tests
# =============================================================================


def test_get_period_end_dates_weekly_sunday():
    start = date(2024, 1, 1)  # Monday
    end = date(2024, 1, 31)
    dates = _get_period_end_dates(start, end, "1w", weekday=7)

    # Should have Sundays: Jan 7, 14, 21, 28
    assert len(dates) == 4
    for d in dates:
        assert d.isoweekday() == 7  # Sunday


def test_get_period_end_dates_weekly_friday():
    start = date(2024, 1, 1)
    end = date(2024, 1, 31)
    dates = _get_period_end_dates(start, end, "1w", weekday=5)

    # Should have Fridays: Jan 5, 12, 19, 26
    assert len(dates) == 4
    for d in dates:
        assert d.isoweekday() == 5  # Friday


def test_get_period_end_dates_monthly_end():
    start = date(2024, 1, 1)
    end = date(2024, 6, 30)
    dates = _get_period_end_dates(start, end, "1mo")

    expected = [
        date(2024, 1, 31),
        date(2024, 2, 29),  # Leap year
        date(2024, 3, 31),
        date(2024, 4, 30),
        date(2024, 5, 31),
        date(2024, 6, 30),
    ]
    assert dates == expected


def test_get_period_end_dates_monthly_start():
    start = date(2024, 1, 15)  # Mid-January
    end = date(2024, 6, 15)
    dates = _get_period_end_dates(start, end, "1mo_start")

    expected = [
        date(2024, 2, 1),
        date(2024, 3, 1),
        date(2024, 4, 1),
        date(2024, 5, 1),
        date(2024, 6, 1),
    ]
    assert dates == expected


def test_get_period_end_dates_quarterly_end():
    start = date(2024, 1, 1)
    end = date(2024, 12, 31)
    dates = _get_period_end_dates(start, end, "3mo")

    expected = [
        date(2024, 3, 31),
        date(2024, 6, 30),
        date(2024, 9, 30),
        date(2024, 12, 31),
    ]
    assert dates == expected


def test_get_period_end_dates_yearly_end():
    start = date(2022, 6, 1)
    end = date(2024, 6, 30)
    dates = _get_period_end_dates(start, end, "1y")

    expected = [date(2022, 12, 31), date(2023, 12, 31)]
    assert dates == expected


# =============================================================================
# Resample backtest tests
# =============================================================================


def test_resample_daily_no_change():
    close = pl.DataFrame({
        "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "2330": [100.0, 101.0, 102.0],
    })
    position = pl.DataFrame({"date": ["2024-01-02"], "2330": [1.0]})

    result = backtest_wide(close, position, resample="D")
    assert len(result) == 3


def test_resample_weekly():
    dates = [
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
    ]
    close = pl.DataFrame({"date": dates, "2330": [100.0 + i for i in range(len(dates))]})
    position = pl.DataFrame({"date": dates, "2330": [1.0] * len(dates)})

    result = backtest_wide(close, position, resample="W")
    assert len(result) == len(dates)


def test_resample_monthly():
    dates = ["2024-01-15", "2024-01-31", "2024-02-15", "2024-02-29", "2024-03-15", "2024-03-29"]
    close = pl.DataFrame({"date": dates, "2330": [100.0 + i for i in range(len(dates))]})
    position = pl.DataFrame({"date": dates, "2330": [1.0] * len(dates)})

    result = backtest_wide(close, position, resample="M")
    assert len(result) == len(dates)


def test_resample_offset_positive():
    dates = [
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
    ]
    close = pl.DataFrame({"date": dates, "2330": [100.0 + i for i in range(len(dates))]})
    position = pl.DataFrame({"date": dates, "2330": [1.0] * len(dates)})

    result = backtest_wide(close, position, resample="W", resample_offset="1D")
    assert len(result) == len(dates)


def test_resample_offset_negative():
    dates = [
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
    ]
    close = pl.DataFrame({"date": dates, "2330": [100.0 + i for i in range(len(dates))]})
    position = pl.DataFrame({"date": dates, "2330": [1.0] * len(dates)})

    result = backtest_wide(close, position, resample="W", resample_offset="-1D",
                           fee_ratio=0.0, tax_ratio=0.0)
    assert len(result) == len(dates)


def test_resample_invalid_raises():
    close = pl.DataFrame({"date": ["2024-01-01", "2024-01-02"], "2330": [100.0, 102.0]})
    position = pl.DataFrame({"date": ["2024-01-01"], "2330": [1.0]})

    with pytest.raises(ValueError, match="Invalid resample frequency"):
        backtest_wide(close, position, resample="X")


# =============================================================================
# backtest_with_report resample tests
# =============================================================================


def test_resample_monthly_with_report():
    dates = ["2024-01-15", "2024-01-31", "2024-02-15", "2024-02-29", "2024-03-15", "2024-03-29"]
    close = pl.DataFrame({"date": dates, "2330": [100.0, 102.0, 103.0, 105.0, 106.0, 108.0]})
    position = pl.DataFrame({"date": dates, "2330": [1.0] * len(dates)})

    report = backtest_with_report_wide(close=close, position=position, resample="M")

    assert isinstance(report.creturn, pl.DataFrame)
    assert isinstance(report.trades, pl.DataFrame)


def test_resample_weekly_with_report():
    dates = [
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
    ]
    close = pl.DataFrame({"date": dates, "2330": [100.0 + i for i in range(len(dates))]})
    position = pl.DataFrame({"date": dates, "2330": [1.0] * len(dates)})

    report = backtest_with_report_wide(close=close, position=position, resample="W")

    assert isinstance(report.creturn, pl.DataFrame)
    assert isinstance(report.trades, pl.DataFrame)


# =============================================================================
# resample=None tests
# =============================================================================


def test_resample_none_basic():
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    close = pl.DataFrame({"date": dates, "2330": [100.0, 101.0, 102.0, 103.0, 104.0]})
    position = pl.DataFrame({"date": dates, "2330": [1.0, 1.0, 1.0, 0.5, 0.5]})

    result = backtest_wide(close, position, resample=None)
    assert len(result) == len(dates)


def test_resample_none_with_report():
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    close = pl.DataFrame({"date": dates, "2330": [100.0, 101.0, 102.0, 103.0, 104.0]})
    position = pl.DataFrame({"date": dates, "2330": [1.0, 1.0, 1.0, 0.5, 0.5]})

    report = backtest_with_report_wide(close=close, position=position, resample=None)

    assert isinstance(report.creturn, pl.DataFrame)
    assert isinstance(report.trades, pl.DataFrame)


# =============================================================================
# Edge cases
# =============================================================================


def test_single_date_position():
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    close = pl.DataFrame({"date": dates, "2330": [100.0, 101.0, 102.0]})
    position = pl.DataFrame({"date": ["2024-01-01"], "2330": [1.0]})

    result = backtest_wide(close, position, resample="D")
    assert len(result) == len(dates)


def test_position_dates_subset_of_price_dates():
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    close = pl.DataFrame({"date": dates, "2330": [100.0, 101.0, 102.0, 103.0, 104.0]})
    position = pl.DataFrame({"date": ["2024-01-01", "2024-01-03"], "2330": [1.0, 0.5]})

    result = backtest_wide(close, position, resample="D")
    assert len(result) == len(dates)


def test_multi_stock_resample():
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
             "2024-01-08", "2024-01-09", "2024-01-10"]
    close = pl.DataFrame({
        "date": dates,
        "2330": [100.0 + i for i in range(len(dates))],
        "2317": [50.0 + i * 0.5 for i in range(len(dates))],
    })
    position = pl.DataFrame({
        "date": dates,
        "2330": [0.5] * len(dates),
        "2317": [0.5] * len(dates),
    })

    result = backtest_wide(close, position, resample="W")
    assert len(result) == len(dates)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
