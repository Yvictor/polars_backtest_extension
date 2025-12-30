"""Tests for utility functions."""

import polars as pl

from polars_backtest.utils import long_to_wide


def test_long_to_wide_conversion():
    """Test that pivot conversion preserves data correctly."""
    df = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "symbol": ["AAPL", "GOOG", "AAPL", "GOOG"],
        "close": [100.0, 200.0, 102.0, 198.0],
    })

    wide = long_to_wide(df, "close")

    assert "date" in wide.columns
    assert "AAPL" in wide.columns
    assert "GOOG" in wide.columns
    assert len(wide) == 2

    row1 = wide.filter(pl.col("date") == "2024-01-01")
    assert row1["AAPL"][0] == 100.0
    assert row1["GOOG"][0] == 200.0
