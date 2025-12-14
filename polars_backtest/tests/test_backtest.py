"""Tests for polars_backtest package."""

import polars as pl
import pytest
from polars_backtest import (
    BacktestConfig,
    backtest,
    backtest_signals,
    backtest_weights,
    cumulative_returns,
    daily_returns,
    drawdown_series,
    equal_weights,
    max_drawdown,
    portfolio_return,
    sharpe_ratio,
    sortino_ratio,
)


class TestBacktest:
    """Tests for the main backtest function."""

    def test_backtest_equal_weight_basic(self):
        """Test basic equal-weight backtest without fees."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "A": [100.0, 110.0, 121.0],  # +10%, +10%
            "B": [100.0, 90.0, 81.0],    # -10%, -10%
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "A": [True],
            "B": [True],
        })

        result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0)

        assert len(result) == 3
        assert abs(result["creturn"][0] - 1.0) < 1e-10  # Day 0: 1.0
        # Day 1: 0.5 * 10% + 0.5 * (-10%) = 0%
        assert abs(result["creturn"][1] - 1.0) < 1e-10
        # Day 2: Portfolio drifts - A grows to 0.55, B shrinks to 0.45
        # Day 2 return: 0.55 * 10% + 0.45 * (-10%) = 1%
        assert abs(result["creturn"][2] - 1.01) < 1e-10

    def test_backtest_custom_weights(self):
        """Test backtest with custom weights."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "A": [100.0, 110.0],  # +10%
            "B": [100.0, 100.0],  # flat
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "A": [0.7],
            "B": [0.3],
        })

        result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0)

        # Day 1: 0.7 * 10% + 0.3 * 0% = 7%
        expected = 1.07
        assert abs(result["creturn"][1] - expected) < 1e-10

    def test_backtest_with_fees(self):
        """Test backtest with transaction fees."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "A": [100.0, 100.0],  # flat
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "A": [True],
        })

        fee_ratio = 0.001425
        result = backtest(prices, position, fee_ratio=fee_ratio, tax_ratio=0.0)

        # Entry fee reduces initial value
        # effective = 1.0 / (1 + fee_ratio) for 100% allocation
        expected = 1.0 / (1.0 + fee_ratio)
        assert abs(result["creturn"][0] - expected) < 1e-6

    def test_backtest_weight_normalization(self):
        """Test that weights > 1.0 are normalized."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "A": [100.0, 110.0],  # +10%
            "B": [100.0, 110.0],  # +10%
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "A": [1.0],
            "B": [1.0],
        })

        result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0)

        # Weights sum to 2.0, should be normalized to 0.5, 0.5
        # Return: 0.5 * 10% + 0.5 * 10% = 10%
        expected = 1.10
        assert abs(result["creturn"][1] - expected) < 1e-10

    def test_backtest_partial_allocation(self):
        """Test that weights < 1.0 are not normalized up."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "A": [100.0, 120.0],  # +20%
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "A": [0.3],
        })

        result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0)

        # Only 30% invested, 70% cash
        # Return: 0.3 * 20% + 0.7 * 0% = 6%
        expected = 1.06
        assert abs(result["creturn"][1] - expected) < 1e-10


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig()
        assert config.fee_ratio == 0.001425
        assert config.tax_ratio == 0.003
        assert config.stop_loss == 1.0  # disabled
        assert config.position_limit == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            fee_ratio=0.01,
            tax_ratio=0.02,
            stop_loss=0.1,
            position_limit=0.3,
        )
        assert config.fee_ratio == 0.01
        assert config.tax_ratio == 0.02
        assert config.stop_loss == 0.1
        assert config.position_limit == 0.3


class TestStatistics:
    """Tests for statistics functions."""

    def test_daily_returns(self):
        """Test daily returns calculation."""
        df = pl.DataFrame({"price": [100.0, 102.0, 99.0]})
        df = df.with_columns(daily_returns("price").alias("ret"))

        assert df["ret"][0] is None  # First value is null
        assert abs(df["ret"][1] - 0.02) < 1e-10  # 102/100 - 1
        assert abs(df["ret"][2] - (99.0 / 102.0 - 1)) < 1e-10

    def test_cumulative_returns(self):
        """Test cumulative returns calculation."""
        df = pl.DataFrame({"ret": [None, 0.02, -0.03]})
        df = df.with_columns(cumulative_returns("ret").alias("cret"))

        assert abs(df["cret"][0] - 1.0) < 1e-10
        assert abs(df["cret"][1] - 1.02) < 1e-10
        assert abs(df["cret"][2] - 1.02 * 0.97) < 1e-10

    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        df = pl.DataFrame({"cret": [1.0, 1.1, 0.9, 1.0, 0.8]})
        result = df.select(max_drawdown("cret").alias("mdd"))

        # Max drawdown from 1.1 to 0.8 = -0.2727...
        expected = (0.8 - 1.1) / 1.1
        assert abs(result["mdd"][0] - expected) < 1e-6

    def test_drawdown_series(self):
        """Test drawdown series calculation."""
        df = pl.DataFrame({"cret": [1.0, 1.1, 1.0, 1.05]})
        df = df.with_columns(drawdown_series("cret").alias("dd"))

        assert abs(df["dd"][0]) < 1e-10  # No drawdown
        assert abs(df["dd"][1]) < 1e-10  # New high
        assert abs(df["dd"][2] - (1.0 / 1.1 - 1)) < 1e-10  # -9.09%
        assert abs(df["dd"][3] - (1.05 / 1.1 - 1)) < 1e-10  # -4.55%

    def test_sharpe_ratio(self):
        """Test Sharpe ratio is calculated and reasonable."""
        df = pl.DataFrame({"ret": [0.01, 0.02, -0.01, 0.015, 0.01]})
        result = df.select(sharpe_ratio("ret").alias("sharpe"))

        # Should be positive for overall positive returns
        assert result["sharpe"][0] > 0
        assert result["sharpe"][0] < 100  # Reasonable range

    def test_sortino_ratio(self):
        """Test Sortino ratio is calculated."""
        df = pl.DataFrame({"ret": [0.01, 0.02, -0.01, 0.015, 0.01]})
        result = df.select(sortino_ratio("ret").alias("sortino"))

        # Should be positive for overall positive returns
        assert result["sortino"][0] > 0


class TestPortfolioHelpers:
    """Tests for portfolio helper functions."""

    def test_equal_weights(self):
        """Test equal weights calculation."""
        df = pl.DataFrame({"signal": [True, True, False, True]})
        df = df.with_columns(equal_weights("signal").alias("weight"))

        # 3 True signals, each gets 1/3
        expected = 1.0 / 3.0
        assert abs(df["weight"][0] - expected) < 1e-10
        assert abs(df["weight"][1] - expected) < 1e-10
        assert abs(df["weight"][2]) < 1e-10  # False
        assert abs(df["weight"][3] - expected) < 1e-10

    def test_portfolio_return(self):
        """Test portfolio return calculation."""
        df = pl.DataFrame({
            "weight": [0.5, 0.3, 0.2],
            "ret": [0.10, -0.05, 0.02],
        })
        result = df.select(portfolio_return("weight", "ret").alias("port_ret"))

        # 0.5*10% + 0.3*(-5%) + 0.2*2% = 3.9%
        expected = 0.5 * 0.10 + 0.3 * (-0.05) + 0.2 * 0.02
        assert abs(result["port_ret"][0] - expected) < 1e-10


class TestLowLevelFunctions:
    """Tests for low-level backtest functions."""

    def test_backtest_signals_direct(self):
        """Test backtest_signals function directly."""
        prices = pl.DataFrame({
            "A": [100.0, 110.0],
            "B": [100.0, 90.0],
        })
        signals = pl.DataFrame({
            "A": [True],
            "B": [True],
        })

        creturn = backtest_signals(prices, signals, [0])

        assert len(creturn) == 2
        # Entry fee: effective = 1.0 / (1 + fee_ratio)
        assert abs(creturn[0] - 1.0 / 1.001425) < 1e-6

    def test_backtest_weights_direct(self):
        """Test backtest_weights function directly."""
        prices = pl.DataFrame({
            "A": [100.0, 110.0],
            "B": [100.0, 100.0],
        })
        weights = pl.DataFrame({
            "A": [0.7],
            "B": [0.3],
        })

        config = BacktestConfig(fee_ratio=0.0, tax_ratio=0.0)
        creturn = backtest_weights(prices, weights, [0], config)

        assert len(creturn) == 2
        assert abs(creturn[0] - 1.0) < 1e-10
        assert abs(creturn[1] - 1.07) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
