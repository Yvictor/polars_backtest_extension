"""Test polars-based resample functionality.

These tests verify the pure polars implementation of resampling,
without requiring pandas or finlab.
"""

import pytest
import polars as pl
from datetime import date, timedelta

from polars_backtest import backtest_wide, backtest_with_report_wide


class TestParseResampleFreq:
    """Test _parse_resample_freq function."""

    def test_daily(self):
        """Test daily frequency."""
        from polars_backtest.wide import _parse_resample_freq
        freq, weekday = _parse_resample_freq('D')
        assert freq == '1d'
        assert weekday is None

    def test_weekly_default(self):
        """Test weekly frequency defaults to Sunday."""
        from polars_backtest.wide import _parse_resample_freq
        freq, weekday = _parse_resample_freq('W')
        assert freq == '1w'
        assert weekday == 7  # Sunday

    def test_weekly_friday(self):
        """Test W-FRI anchors to Friday."""
        from polars_backtest.wide import _parse_resample_freq
        freq, weekday = _parse_resample_freq('W-FRI')
        assert freq == '1w'
        assert weekday == 5

    def test_weekly_monday(self):
        """Test W-MON anchors to Monday."""
        from polars_backtest.wide import _parse_resample_freq
        freq, weekday = _parse_resample_freq('W-MON')
        assert freq == '1w'
        assert weekday == 1

    def test_monthly(self):
        """Test monthly frequency."""
        from polars_backtest.wide import _parse_resample_freq
        for freq_str in ('M', 'ME'):
            freq, weekday = _parse_resample_freq(freq_str)
            assert freq == '1mo'
            assert weekday is None

    def test_monthly_start(self):
        """Test month start frequency."""
        from polars_backtest.wide import _parse_resample_freq
        freq, weekday = _parse_resample_freq('MS')
        assert freq == '1mo_start'
        assert weekday is None

    def test_quarterly(self):
        """Test quarterly frequency."""
        from polars_backtest.wide import _parse_resample_freq
        for freq_str in ('Q', 'QE'):
            freq, weekday = _parse_resample_freq(freq_str)
            assert freq == '3mo'
            assert weekday is None

    def test_quarterly_start(self):
        """Test quarter start frequency."""
        from polars_backtest.wide import _parse_resample_freq
        freq, weekday = _parse_resample_freq('QS')
        assert freq == '3mo_start'
        assert weekday is None

    def test_yearly(self):
        """Test yearly frequency."""
        from polars_backtest.wide import _parse_resample_freq
        for freq_str in ('Y', 'YE', 'A'):
            freq, weekday = _parse_resample_freq(freq_str)
            assert freq == '1y'
            assert weekday is None

    def test_invalid(self):
        """Test invalid frequency raises ValueError."""
        from polars_backtest.wide import _parse_resample_freq
        with pytest.raises(ValueError, match="Invalid resample frequency"):
            _parse_resample_freq('X')


class TestParseOffset:
    """Test _parse_offset function."""

    def test_positive_days(self):
        """Test positive day offset."""
        from polars_backtest.wide import _parse_offset
        offset = _parse_offset('1D')
        assert offset == timedelta(days=1)

    def test_negative_days(self):
        """Test negative day offset."""
        from polars_backtest.wide import _parse_offset
        offset = _parse_offset('-1D')
        assert offset == timedelta(days=-1)

    def test_weeks(self):
        """Test week offset."""
        from polars_backtest.wide import _parse_offset
        offset = _parse_offset('2W')
        assert offset == timedelta(weeks=2)

    def test_hours(self):
        """Test hour offset."""
        from polars_backtest.wide import _parse_offset
        offset = _parse_offset('3H')
        assert offset == timedelta(hours=3)

    def test_empty(self):
        """Test empty offset returns zero timedelta."""
        from polars_backtest.wide import _parse_offset
        offset = _parse_offset('')
        assert offset == timedelta(0)

    def test_invalid(self):
        """Test invalid offset raises ValueError."""
        from polars_backtest.wide import _parse_offset
        with pytest.raises(ValueError, match="Invalid offset format"):
            _parse_offset('invalid')


class TestGetPeriodEndDates:
    """Test _get_period_end_dates function."""

    def test_weekly_sunday(self):
        """Test weekly dates ending on Sunday."""
        from polars_backtest.wide import _get_period_end_dates
        start = date(2024, 1, 1)  # Monday
        end = date(2024, 1, 31)
        dates = _get_period_end_dates(start, end, '1w', weekday=7)

        # Should have Sundays: Jan 7, 14, 21, 28
        assert len(dates) == 4
        for d in dates:
            assert d.isoweekday() == 7  # Sunday

    def test_weekly_friday(self):
        """Test weekly dates ending on Friday."""
        from polars_backtest.wide import _get_period_end_dates
        start = date(2024, 1, 1)  # Monday
        end = date(2024, 1, 31)
        dates = _get_period_end_dates(start, end, '1w', weekday=5)

        # Should have Fridays: Jan 5, 12, 19, 26
        assert len(dates) == 4
        for d in dates:
            assert d.isoweekday() == 5  # Friday

    def test_monthly_end(self):
        """Test monthly end dates."""
        from polars_backtest.wide import _get_period_end_dates
        start = date(2024, 1, 1)
        end = date(2024, 6, 30)
        dates = _get_period_end_dates(start, end, '1mo')

        # Should have end of Jan, Feb, Mar, Apr, May, Jun
        expected = [
            date(2024, 1, 31),
            date(2024, 2, 29),  # Leap year
            date(2024, 3, 31),
            date(2024, 4, 30),
            date(2024, 5, 31),
            date(2024, 6, 30),
        ]
        assert dates == expected

    def test_monthly_start(self):
        """Test monthly start dates."""
        from polars_backtest.wide import _get_period_end_dates
        start = date(2024, 1, 15)  # Mid-January
        end = date(2024, 6, 15)
        dates = _get_period_end_dates(start, end, '1mo_start')

        # Should have start of Feb, Mar, Apr, May, Jun
        expected = [
            date(2024, 2, 1),
            date(2024, 3, 1),
            date(2024, 4, 1),
            date(2024, 5, 1),
            date(2024, 6, 1),
        ]
        assert dates == expected

    def test_quarterly_end(self):
        """Test quarterly end dates."""
        from polars_backtest.wide import _get_period_end_dates
        start = date(2024, 1, 1)
        end = date(2024, 12, 31)
        dates = _get_period_end_dates(start, end, '3mo')

        # Should have end of Q1, Q2, Q3, Q4
        expected = [
            date(2024, 3, 31),
            date(2024, 6, 30),
            date(2024, 9, 30),
            date(2024, 12, 31),
        ]
        assert dates == expected

    def test_yearly_end(self):
        """Test yearly end dates."""
        from polars_backtest.wide import _get_period_end_dates
        start = date(2022, 6, 1)
        end = date(2024, 6, 30)
        dates = _get_period_end_dates(start, end, '1y')

        # Should have Dec 31 of 2022 and 2023
        expected = [
            date(2022, 12, 31),
            date(2023, 12, 31),
        ]
        assert dates == expected


class TestResamplePosition:
    """Test _resample_position function with polars implementation."""

    def test_daily_no_change(self):
        """Test that resample='D' returns position unchanged."""
        close = pl.DataFrame({
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "2330": [100.0, 101.0, 102.0],
        })
        position = pl.DataFrame({
            "date": ["2024-01-02"],
            "2330": [1.0],
        })

        result = backtest_wide(close, position, resample='D')
        assert len(result) == 3

    def test_weekly_resample(self):
        """Test weekly resampling."""
        # Create 2 weeks of data
        dates = [
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",  # Week 1 (Tue-Fri)
            "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",  # Week 2
        ]
        close = pl.DataFrame({
            "date": dates,
            "2330": [100.0 + i for i in range(len(dates))],
        })
        position = pl.DataFrame({
            "date": dates,
            "2330": [1.0] * len(dates),
        })

        result = backtest_wide(close, position, resample='W')
        assert len(result) == len(dates)

    def test_monthly_resample(self):
        """Test monthly resampling."""
        # Create data spanning 3 months
        dates = [
            "2024-01-15", "2024-01-31",  # January
            "2024-02-15", "2024-02-29",  # February
            "2024-03-15", "2024-03-29",  # March
        ]
        close = pl.DataFrame({
            "date": dates,
            "2330": [100.0 + i for i in range(len(dates))],
        })
        position = pl.DataFrame({
            "date": dates,
            "2330": [1.0] * len(dates),
        })

        result = backtest_wide(close, position, resample='M')
        assert len(result) == len(dates)

    def test_resample_offset_positive(self):
        """Test resample with positive offset."""
        dates = [
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
        ]
        close = pl.DataFrame({
            "date": dates,
            "2330": [100.0 + i for i in range(len(dates))],
        })
        position = pl.DataFrame({
            "date": dates,
            "2330": [1.0] * len(dates),
        })

        # Should not crash
        result = backtest_wide(close, position, resample='W', resample_offset='1D')
        assert len(result) == len(dates)

    def test_resample_offset_negative(self):
        """Test resample with negative offset."""
        dates = [
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
        ]
        close = pl.DataFrame({
            "date": dates,
            "2330": [100.0 + i for i in range(len(dates))],
        })
        position = pl.DataFrame({
            "date": dates,
            "2330": [1.0] * len(dates),
        })

        # Should not crash
        result = backtest_wide(close, position, resample='W', resample_offset='-1D',
                         fee_ratio=0.0, tax_ratio=0.0)
        assert len(result) == len(dates)

    def test_invalid_resample_raises(self):
        """Test that invalid resample raises ValueError."""
        close = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "2330": [100.0, 102.0],
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "2330": [1.0],
        })

        with pytest.raises(ValueError, match="Invalid resample frequency"):
            backtest_wide(close, position, resample='X')


class TestBacktestWithReportResample:
    """Test backtest_with_report with resample parameter."""

    def test_resample_monthly_with_report(self):
        """Test monthly resample with backtest_with_report."""
        dates = [
            "2024-01-15", "2024-01-31",
            "2024-02-15", "2024-02-29",
            "2024-03-15", "2024-03-29",
        ]
        close = pl.DataFrame({
            "date": dates,
            "2330": [100.0, 102.0, 103.0, 105.0, 106.0, 108.0],
        })
        position = pl.DataFrame({
            "date": dates,
            "2330": [1.0] * len(dates),
        })

        report = backtest_with_report_wide(
            close=close,
            position=position,
            resample='M',
        )

        assert isinstance(report.creturn, pl.DataFrame)
        assert isinstance(report.trades, pl.DataFrame)

    def test_resample_weekly_with_report(self):
        """Test weekly resample with backtest_with_report."""
        dates = [
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
        ]
        close = pl.DataFrame({
            "date": dates,
            "2330": [100.0 + i for i in range(len(dates))],
        })
        position = pl.DataFrame({
            "date": dates,
            "2330": [1.0] * len(dates),
        })

        report = backtest_with_report_wide(
            close=close,
            position=position,
            resample='W',
        )

        assert isinstance(report.creturn, pl.DataFrame)
        assert isinstance(report.trades, pl.DataFrame)


class TestResampleNone:
    """Test resample=None (only rebalance on position change)."""

    def test_resample_none_basic(self):
        """Test resample=None filters to changed positions only."""
        dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        close = pl.DataFrame({
            "date": dates,
            "2330": [100.0, 101.0, 102.0, 103.0, 104.0],
        })
        # Position changes only on day 1 and day 4
        position = pl.DataFrame({
            "date": dates,
            "2330": [1.0, 1.0, 1.0, 0.5, 0.5],  # Change at index 0 and 3
        })

        result = backtest_wide(close, position, resample=None)
        assert len(result) == len(dates)

    def test_resample_none_with_report(self):
        """Test resample=None with backtest_with_report."""
        dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        close = pl.DataFrame({
            "date": dates,
            "2330": [100.0, 101.0, 102.0, 103.0, 104.0],
        })
        position = pl.DataFrame({
            "date": dates,
            "2330": [1.0, 1.0, 1.0, 0.5, 0.5],
        })

        report = backtest_with_report_wide(
            close=close,
            position=position,
            resample=None,
        )

        assert isinstance(report.creturn, pl.DataFrame)
        assert isinstance(report.trades, pl.DataFrame)


class TestEdgeCases:
    """Test edge cases for resample functionality."""

    def test_single_date(self):
        """Test with single date position."""
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        close = pl.DataFrame({
            "date": dates,
            "2330": [100.0, 101.0, 102.0],
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "2330": [1.0],
        })

        result = backtest_wide(close, position, resample='D')
        assert len(result) == len(dates)

    def test_position_dates_subset_of_price_dates(self):
        """Test when position has fewer dates than price data."""
        dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        close = pl.DataFrame({
            "date": dates,
            "2330": [100.0, 101.0, 102.0, 103.0, 104.0],
        })
        position = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-03"],
            "2330": [1.0, 0.5],
        })

        result = backtest_wide(close, position, resample='D')
        assert len(result) == len(dates)

    def test_multi_stock_resample(self):
        """Test resample with multiple stocks."""
        dates = [
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-08", "2024-01-09", "2024-01-10",
        ]
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

        result = backtest_wide(close, position, resample='W')
        assert len(result) == len(dates)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
