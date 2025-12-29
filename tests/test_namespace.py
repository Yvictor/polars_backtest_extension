"""Tests for long format backtest API (df.bt and pl_bt.backtest)."""

import datetime
import pytest
import polars as pl

import polars_backtest as pl_bt
from polars_backtest.namespace import _long_to_wide


@pytest.fixture(scope="module")
def sample_data():
    """Sample long format data for testing."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    symbols = ["AAPL", "GOOG"]

    rows = []
    for date in dates:
        for symbol in symbols:
            base_price = 100.0 if symbol == "AAPL" else 200.0
            day_idx = dates.index(date)
            price = base_price * (1 + 0.01 * day_idx)

            rows.append({
                "date": date,
                "symbol": symbol,
                "open": price * 0.99,
                "high": price * 1.02,
                "low": price * 0.98,
                "close": price,
                "weight": 0.5,
            })

    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def monthly_data():
    """Data spanning multiple months for resample tests."""
    rows = []
    start = datetime.date(2024, 1, 1)
    for i in range(90):
        date = start + datetime.timedelta(days=i)
        if date.weekday() >= 5:
            continue

        for symbol in ["AAPL", "GOOG"]:
            base = 100.0 if symbol == "AAPL" else 200.0
            price = base * (1 + 0.001 * i)
            rows.append({
                "date": str(date),
                "symbol": symbol,
                "close": price,
                "weight": 0.5,
            })

    return pl.DataFrame(rows)


# =============================================================================
# Namespace API Tests (df.bt)
# =============================================================================

def test_namespace_exists(sample_data):
    """Test that bt namespace is accessible."""
    assert hasattr(sample_data, "bt")
    assert hasattr(sample_data.bt, "backtest")
    assert hasattr(sample_data.bt, "backtest_with_report")


def test_namespace_basic(sample_data):
    """Test basic backtest via namespace."""
    result = sample_data.bt.backtest(trade_at_price="close", position="weight", resample="D")

    assert isinstance(result, pl.DataFrame)
    assert "date" in result.columns
    assert "creturn" in result.columns
    assert len(result) == 5


def test_namespace_with_report(sample_data):
    """Test backtest_with_report via namespace."""
    report = sample_data.bt.backtest_with_report(trade_at_price="close", position="weight")

    assert hasattr(report, "creturn")
    assert hasattr(report, "trades")


# =============================================================================
# Standalone Function API Tests (pl_bt.backtest)
# =============================================================================

def test_function_basic(sample_data):
    """Test basic backtest via standalone function."""
    result = pl_bt.backtest(sample_data, trade_at_price="close", position="weight", resample="D")

    assert isinstance(result, pl.DataFrame)
    assert "date" in result.columns
    assert "creturn" in result.columns
    assert len(result) == 5


def test_function_with_report(sample_data):
    """Test backtest_with_report via standalone function."""
    report = pl_bt.backtest_with_report(sample_data, trade_at_price="close", position="weight")

    assert hasattr(report, "creturn")
    assert hasattr(report, "trades")


def test_function_equals_namespace(sample_data):
    """Verify standalone function and namespace produce same results."""
    result_ns = sample_data.bt.backtest(trade_at_price="close", position="weight", resample="D")
    result_fn = pl_bt.backtest(sample_data, trade_at_price="close", position="weight", resample="D")

    assert result_ns["creturn"].to_list() == result_fn["creturn"].to_list()


# =============================================================================
# Feature Tests
# =============================================================================

def test_with_fees(sample_data):
    """Test backtest with transaction fees."""
    result_no_fee = pl_bt.backtest(
        sample_data, trade_at_price="close", position="weight",
        fee_ratio=0.0, tax_ratio=0.0,
    )
    result_with_fee = pl_bt.backtest(
        sample_data, trade_at_price="close", position="weight",
        fee_ratio=0.001425, tax_ratio=0.003,
    )

    assert result_with_fee["creturn"][-1] <= result_no_fee["creturn"][-1]


def test_with_stop_loss(sample_data):
    """Test backtest with stop loss."""
    result = pl_bt.backtest(sample_data, trade_at_price="close", position="weight", stop_loss=0.05)

    assert isinstance(result, pl.DataFrame)
    assert len(result) == 5


def test_missing_columns(sample_data):
    """Test that missing columns raise error."""
    with pytest.raises(ValueError, match="Missing required columns"):
        pl_bt.backtest(sample_data, trade_at_price="nonexistent", position="weight")


def test_bool_signals():
    """Test backtest with boolean signals."""
    df = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "symbol": ["AAPL", "GOOG", "AAPL", "GOOG"],
        "close": [100.0, 200.0, 102.0, 198.0],
        "signal": [True, True, True, False],
    })

    result = pl_bt.backtest(df, trade_at_price="close", position="signal")
    assert isinstance(result, pl.DataFrame)
    assert "creturn" in result.columns


# =============================================================================
# Resample Tests
# =============================================================================

def test_monthly_resample(monthly_data):
    """Test monthly rebalancing."""
    result = pl_bt.backtest(monthly_data, trade_at_price="close", position="weight", resample="M")

    assert isinstance(result, pl.DataFrame)
    assert "creturn" in result.columns
    # With 90 days of data and monthly resample, we get ~40-50 days from first month-end
    assert len(result) > 30


def test_weekly_resample(monthly_data):
    """Test weekly rebalancing."""
    result = pl_bt.backtest(monthly_data, trade_at_price="close", position="weight", resample="W")

    assert isinstance(result, pl.DataFrame)


# =============================================================================
# Edge Cases
# =============================================================================

def test_single_stock():
    """Test with single stock."""
    df = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "symbol": ["AAPL", "AAPL", "AAPL"],
        "close": [100.0, 102.0, 101.0],
        "weight": [1.0, 1.0, 1.0],
    })

    result = pl_bt.backtest(df, trade_at_price="close", position="weight")
    assert len(result) == 3


def test_single_date():
    """Test with single date (multiple stocks)."""
    df = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-01"],
        "symbol": ["AAPL", "GOOG"],
        "close": [100.0, 200.0],
        "weight": [0.5, 0.5],
    })

    result = pl_bt.backtest(df, trade_at_price="close", position="weight")
    assert len(result) == 1


def test_zero_weights():
    """Test with zero weights (no position)."""
    df = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "symbol": ["AAPL", "GOOG", "AAPL", "GOOG"],
        "close": [100.0, 200.0, 102.0, 198.0],
        "weight": [0.0, 0.0, 0.0, 0.0],
    })

    result = pl_bt.backtest(df, trade_at_price="close", position="weight")
    assert all(r == pytest.approx(1.0) for r in result["creturn"].to_list())


def test_negative_weights_short():
    """Test with negative weights (short positions)."""
    df = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "symbol": ["AAPL", "GOOG", "AAPL", "GOOG"],
        "close": [100.0, 200.0, 102.0, 198.0],
        "weight": [-0.5, -0.5, -0.5, -0.5],
    })

    result = pl_bt.backtest(df, trade_at_price="close", position="weight")
    assert isinstance(result, pl.DataFrame)


# =============================================================================
# Internal Function Tests
# =============================================================================

def test_long_to_wide_conversion():
    """Test that pivot conversion preserves data correctly."""
    df = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "symbol": ["AAPL", "GOOG", "AAPL", "GOOG"],
        "close": [100.0, 200.0, 102.0, 198.0],
    })

    wide = _long_to_wide(df, "close")

    assert "date" in wide.columns
    assert "AAPL" in wide.columns
    assert "GOOG" in wide.columns
    assert len(wide) == 2

    row1 = wide.filter(pl.col("date") == "2024-01-01")
    assert row1["AAPL"][0] == 100.0
    assert row1["GOOG"][0] == 200.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
