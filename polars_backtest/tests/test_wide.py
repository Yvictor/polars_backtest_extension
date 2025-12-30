"""Tests for polars_backtest Wide format API."""

import polars as pl
import pytest

from polars_backtest import (
    BacktestConfig,
    backtest_wide,
    cumulative_returns,
    daily_returns,
    drawdown_series,
    equal_weights,
    max_drawdown,
    portfolio_return,
    sharpe_ratio,
    sortino_ratio,
)

# Alias for backward compatibility
backtest = backtest_wide


# =============================================================================
# Core backtest tests
# =============================================================================


def test_backtest_equal_weight_basic():
    """Test basic equal-weight backtest without fees (T+1 execution)."""
    prices = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "A": [100.0, 110.0, 121.0, 133.1],  # +10% each day
        "B": [100.0, 90.0, 81.0, 72.9],     # -10% each day
    })
    position = pl.DataFrame({"date": ["2024-01-01"], "A": [True], "B": [True]})

    result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0)

    assert len(result) == 4
    assert abs(result["creturn"][0] - 1.0) < 1e-10
    assert abs(result["creturn"][1] - 1.0) < 1e-10
    assert abs(result["creturn"][2] - 1.0) < 1e-10
    assert abs(result["creturn"][3] - 1.01) < 1e-10


def test_backtest_custom_weights():
    """Test backtest with custom weights (T+1 execution)."""
    prices = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "A": [100.0, 100.0, 110.0],
        "B": [100.0, 100.0, 100.0],
    })
    position = pl.DataFrame({"date": ["2024-01-01"], "A": [0.7], "B": [0.3]})

    result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0)

    expected = 1.07
    assert abs(result["creturn"][2] - expected) < 1e-10


def test_backtest_with_fees():
    """Test backtest with transaction fees (T+1 execution)."""
    prices = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "A": [100.0, 100.0, 100.0],
    })
    position = pl.DataFrame({"date": ["2024-01-01"], "A": [True]})

    fee_ratio = 0.001425
    result = backtest(prices, position, fee_ratio=fee_ratio, tax_ratio=0.0)

    assert abs(result["creturn"][0] - 1.0) < 1e-10
    expected = 1.0 / (1.0 + fee_ratio)
    assert abs(result["creturn"][1] - expected) < 1e-5


def test_backtest_weight_normalization():
    """Test that weights > 1.0 are normalized (T+1 execution)."""
    prices = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "A": [100.0, 100.0, 110.0],
        "B": [100.0, 100.0, 110.0],
    })
    position = pl.DataFrame({"date": ["2024-01-01"], "A": [1.0], "B": [1.0]})

    result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0)

    expected = 1.10
    assert abs(result["creturn"][2] - expected) < 1e-10


def test_backtest_partial_allocation():
    """Test that weights < 1.0 are not normalized up (T+1 execution)."""
    prices = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "A": [100.0, 100.0, 120.0],
    })
    position = pl.DataFrame({"date": ["2024-01-01"], "A": [0.3]})

    result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0)

    expected = 1.06
    assert abs(result["creturn"][2] - expected) < 1e-10


# =============================================================================
# BacktestConfig tests
# =============================================================================


def test_config_default():
    config = BacktestConfig()
    assert config.fee_ratio == 0.001425
    assert config.tax_ratio == 0.003
    assert config.stop_loss == 1.0
    assert config.position_limit == 1.0


def test_config_custom():
    config = BacktestConfig(
        fee_ratio=0.01, tax_ratio=0.02, stop_loss=0.1, position_limit=0.3
    )
    assert config.fee_ratio == 0.01
    assert config.tax_ratio == 0.02
    assert config.stop_loss == 0.1
    assert config.position_limit == 0.3


# =============================================================================
# Statistics tests
# =============================================================================


def test_daily_returns():
    df = pl.DataFrame({"price": [100.0, 102.0, 99.0]})
    df = df.with_columns(daily_returns("price").alias("ret"))

    assert df["ret"][0] is None
    assert abs(df["ret"][1] - 0.02) < 1e-10
    assert abs(df["ret"][2] - (99.0 / 102.0 - 1)) < 1e-10


def test_cumulative_returns():
    df = pl.DataFrame({"ret": [None, 0.02, -0.03]})
    df = df.with_columns(cumulative_returns("ret").alias("cret"))

    assert abs(df["cret"][0] - 1.0) < 1e-10
    assert abs(df["cret"][1] - 1.02) < 1e-10
    assert abs(df["cret"][2] - 1.02 * 0.97) < 1e-10


def test_max_drawdown():
    df = pl.DataFrame({"cret": [1.0, 1.1, 0.9, 1.0, 0.8]})
    result = df.select(max_drawdown("cret").alias("mdd"))

    expected = (0.8 - 1.1) / 1.1
    assert abs(result["mdd"][0] - expected) < 1e-6


def test_drawdown_series():
    df = pl.DataFrame({"cret": [1.0, 1.1, 1.0, 1.05]})
    df = df.with_columns(drawdown_series("cret").alias("dd"))

    assert abs(df["dd"][0]) < 1e-10
    assert abs(df["dd"][1]) < 1e-10
    assert abs(df["dd"][2] - (1.0 / 1.1 - 1)) < 1e-10
    assert abs(df["dd"][3] - (1.05 / 1.1 - 1)) < 1e-10


def test_sharpe_ratio():
    df = pl.DataFrame({"ret": [0.01, 0.02, -0.01, 0.015, 0.01]})
    result = df.select(sharpe_ratio("ret").alias("sharpe"))

    assert result["sharpe"][0] > 0
    assert result["sharpe"][0] < 100


def test_sortino_ratio():
    df = pl.DataFrame({"ret": [0.01, 0.02, -0.01, 0.015, 0.01]})
    result = df.select(sortino_ratio("ret").alias("sortino"))

    assert result["sortino"][0] > 0


# =============================================================================
# Portfolio helpers tests
# =============================================================================


def test_equal_weights():
    df = pl.DataFrame({"signal": [True, True, False, True]})
    df = df.with_columns(equal_weights("signal").alias("weight"))

    expected = 1.0 / 3.0
    assert abs(df["weight"][0] - expected) < 1e-10
    assert abs(df["weight"][1] - expected) < 1e-10
    assert abs(df["weight"][2]) < 1e-10
    assert abs(df["weight"][3] - expected) < 1e-10


def test_portfolio_return():
    df = pl.DataFrame({"weight": [0.5, 0.3, 0.2], "ret": [0.10, -0.05, 0.02]})
    result = df.select(portfolio_return("weight", "ret").alias("port_ret"))

    expected = 0.5 * 0.10 + 0.3 * (-0.05) + 0.2 * 0.02
    assert abs(result["port_ret"][0] - expected) < 1e-10


# =============================================================================
# Stop loss / Take profit / Trail stop tests
# =============================================================================


def test_stop_loss_triggers_exit():
    """Test that stop loss triggers position exit when price drops (T+1 execution)."""
    prices = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "A": [100.0, 100.0, 85.0, 90.0, 95.0],
    })
    position = pl.DataFrame({"date": ["2024-01-01"], "A": [True]})

    result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0, stop_loss=0.10)

    assert len(result) == 5
    day3_return = result["creturn"][3]
    day4_return = result["creturn"][4]
    assert abs(day4_return - day3_return) < 1e-10


def test_stop_loss_no_trigger():
    """Test that stop loss doesn't trigger when price drop is below threshold."""
    prices = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "A": [100.0, 100.0, 92.0],
    })
    position = pl.DataFrame({"date": ["2024-01-01"], "A": [True]})

    result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0, stop_loss=0.10)

    assert abs(result["creturn"][2] - 0.92) < 1e-10


def test_take_profit_triggers_exit():
    """Test that take profit triggers position exit when price rises (T+1 execution)."""
    prices = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "A": [100.0, 100.0, 115.0, 130.0, 140.0],
    })
    position = pl.DataFrame({"date": ["2024-01-01"], "A": [True]})

    result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0, take_profit=0.10)

    assert len(result) == 5
    day3_return = result["creturn"][3]
    day4_return = result["creturn"][4]
    assert abs(day4_return - day3_return) < 1e-10


def test_take_profit_no_trigger():
    """Test that take profit doesn't trigger when price rise is below threshold."""
    prices = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "A": [100.0, 100.0, 108.0],
    })
    position = pl.DataFrame({"date": ["2024-01-01"], "A": [True]})

    result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0, take_profit=0.10)

    assert abs(result["creturn"][2] - 1.08) < 1e-10


def test_trail_stop_triggers_exit():
    """Test that trailing stop triggers exit when price drops from high (T+1 execution)."""
    prices = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"],
        "A": [100.0, 100.0, 120.0, 105.0, 110.0, 115.0],
    })
    position = pl.DataFrame({"date": ["2024-01-01"], "A": [True]})

    result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0, trail_stop=0.10)

    assert len(result) == 6
    day4_return = result["creturn"][4]
    day5_return = result["creturn"][5]
    assert abs(day5_return - day4_return) < 1e-10


def test_trail_stop_no_trigger():
    """Test that trailing stop doesn't trigger when drop from high is below threshold."""
    prices = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "A": [100.0, 100.0, 120.0, 112.0],
    })
    position = pl.DataFrame({"date": ["2024-01-01"], "A": [True]})

    result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0, trail_stop=0.10)

    assert abs(result["creturn"][3] - 1.12) < 1e-10


def test_stop_trading_next_period_true():
    """Test that stop_trading_next_period=True blocks re-entry after stop loss."""
    prices = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"],
        "A": [100.0, 100.0, 84.0, 100.0, 100.0, 110.0],
    })
    position = pl.DataFrame({"date": ["2024-01-01", "2024-01-04"], "A": [True, True]})

    result = backtest(
        prices, position,
        fee_ratio=0.0, tax_ratio=0.0,
        stop_loss=0.10, stop_trading_next_period=True
    )

    day4_return = result["creturn"][4]
    day5_return = result["creturn"][5]
    assert abs(day5_return - day4_return) < 1e-10


def test_stop_trading_next_period_false():
    """Test that stop_trading_next_period=False allows re-entry after stop loss."""
    prices = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"],
        "A": [100.0, 100.0, 84.0, 100.0, 100.0, 110.0],
    })
    position = pl.DataFrame({"date": ["2024-01-01", "2024-01-04"], "A": [True, True]})

    result = backtest(
        prices, position,
        fee_ratio=0.0, tax_ratio=0.0,
        stop_loss=0.10, stop_trading_next_period=False
    )

    day5_return = result["creturn"][5] / result["creturn"][4]
    assert day5_return > 1.05


def test_retain_cost_when_rebalance_false():
    """Test that retain_cost_when_rebalance=False resets entry price on rebalance."""
    prices = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"],
        "A": [100.0, 100.0, 105.0, 110.0, 110.0, 100.0],
    })
    position = pl.DataFrame({"date": ["2024-01-01", "2024-01-04"], "A": [True, True]})

    result = backtest(
        prices, position,
        fee_ratio=0.0, tax_ratio=0.0,
        stop_loss=0.10, retain_cost_when_rebalance=False
    )

    day5_change = result["creturn"][5] / result["creturn"][4]
    assert 0.85 < day5_change < 0.95


def test_retain_cost_when_rebalance_true():
    """Test that retain_cost_when_rebalance=True keeps original entry price."""
    prices = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"],
        "A": [100.0, 100.0, 105.0, 110.0, 110.0, 88.0],
    })
    position = pl.DataFrame({"date": ["2024-01-01", "2024-01-04"], "A": [True, True]})

    result = backtest(
        prices, position,
        fee_ratio=0.0, tax_ratio=0.0,
        stop_loss=0.10, retain_cost_when_rebalance=True
    )

    day5_return = result["creturn"][5]
    day4_return = result["creturn"][4]
    assert day5_return < day4_return * 0.92


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
