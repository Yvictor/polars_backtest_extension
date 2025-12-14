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
        """Test basic equal-weight backtest without fees (T+1 execution)."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "A": [100.0, 110.0, 121.0, 133.1],  # +10% each day
            "B": [100.0, 90.0, 81.0, 72.9],     # -10% each day
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "A": [True],
            "B": [True],
        })

        result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0)

        assert len(result) == 4
        assert abs(result["creturn"][0] - 1.0) < 1e-10  # Day 0: 1.0 (signal sent)
        # T+1: Position entered on day 1 at day 0 prices (100, 100)
        assert abs(result["creturn"][1] - 1.0) < 1e-10  # Day 1: 1.0 (just entered)
        # Day 2: 0.5 * (121/110-1) + 0.5 * (81/90-1) = 0.5 * 10% + 0.5 * (-10%) = 0%
        assert abs(result["creturn"][2] - 1.0) < 1e-10
        # Day 3: Portfolio drifts - A grows to 0.55, B shrinks to 0.45
        # Day 3 return: 0.55 * 10% + 0.45 * (-10%) = 1%
        assert abs(result["creturn"][3] - 1.01) < 1e-10

    def test_backtest_custom_weights(self):
        """Test backtest with custom weights (T+1 execution)."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "A": [100.0, 100.0, 110.0],  # flat, then +10%
            "B": [100.0, 100.0, 100.0],  # flat
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "A": [0.7],
            "B": [0.3],
        })

        result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0)

        # T+1: Position entered on day 1 at day 0 prices
        # Day 2 return: 0.7 * (110/100-1) + 0.3 * 0% = 7%
        expected = 1.07
        assert abs(result["creturn"][2] - expected) < 1e-10

    def test_backtest_with_fees(self):
        """Test backtest with transaction fees (T+1 execution)."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "A": [100.0, 100.0, 100.0],  # flat
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "A": [True],
        })

        fee_ratio = 0.001425
        result = backtest(prices, position, fee_ratio=fee_ratio, tax_ratio=0.0)

        # T+1: Position entered on day 1, fee applied then
        assert abs(result["creturn"][0] - 1.0) < 1e-10  # Day 0: no position yet
        # Entry fee reduces value when position is entered on day 1
        # effective = 1.0 / (1 + fee_ratio) for 100% allocation
        expected = 1.0 / (1.0 + fee_ratio)
        assert abs(result["creturn"][1] - expected) < 1e-5

    def test_backtest_weight_normalization(self):
        """Test that weights > 1.0 are normalized (T+1 execution)."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "A": [100.0, 100.0, 110.0],  # flat, then +10%
            "B": [100.0, 100.0, 110.0],  # flat, then +10%
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "A": [1.0],
            "B": [1.0],
        })

        result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0)

        # T+1: Position entered on day 1 at day 0 prices
        # Weights sum to 2.0, should be normalized to 0.5, 0.5
        # Day 2 return: 0.5 * 10% + 0.5 * 10% = 10%
        expected = 1.10
        assert abs(result["creturn"][2] - expected) < 1e-10

    def test_backtest_partial_allocation(self):
        """Test that weights < 1.0 are not normalized up (T+1 execution)."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "A": [100.0, 100.0, 120.0],  # flat, then +20%
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "A": [0.3],
        })

        result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0)

        # T+1: Position entered on day 1 at day 0 prices
        # Only 30% invested, 70% cash
        # Day 2 return: 0.3 * 20% + 0.7 * 0% = 6%
        expected = 1.06
        assert abs(result["creturn"][2] - expected) < 1e-10


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


class TestStopLossTakeProfit:
    """Tests for stop loss, take profit, and trailing stop functionality."""

    def test_stop_loss_triggers_exit(self):
        """Test that stop loss triggers position exit when price drops (T+1 execution)."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "A": [100.0, 100.0, 85.0, 90.0, 95.0],  # Day 2: -15% drop triggers 10% stop loss
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "A": [True],
        })

        result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0, stop_loss=0.10)

        assert len(result) == 5
        # T+1 execution:
        # Day 2: -15% drop triggers stop loss (signal to exit)
        # Day 3: Position experiences +5.88% return (85->90), then exits at 90
        # Day 4: Position is flat (no position held)
        day3_return = result["creturn"][3]  # After stop exit
        day4_return = result["creturn"][4]
        # Position should be exited on day 3, so day 4 return should equal day 3
        assert abs(day4_return - day3_return) < 1e-10, \
            f"Position should be flat after stop loss, but day3={day3_return}, day4={day4_return}"

    def test_stop_loss_no_trigger(self):
        """Test that stop loss doesn't trigger when price drop is below threshold."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "A": [100.0, 100.0, 92.0],  # Day 2: -8% drop, below 10% threshold
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "A": [True],
        })

        result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0, stop_loss=0.10)

        # Position should still be held (8% drop < 10% threshold)
        # Return should reflect the loss
        assert abs(result["creturn"][2] - 0.92) < 1e-10

    def test_take_profit_triggers_exit(self):
        """Test that take profit triggers position exit when price rises (T+1 execution)."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "A": [100.0, 100.0, 115.0, 130.0, 140.0],  # Day 2: +15% triggers 10% take profit
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "A": [True],
        })

        result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0, take_profit=0.10)

        assert len(result) == 5
        # T+1 execution:
        # Day 2: +15% gain triggers take profit (signal to exit)
        # Day 3: Position experiences +13.04% return (115->130), then exits
        # Day 4: Position is flat (no position held)
        day3_return = result["creturn"][3]  # After take profit exit
        day4_return = result["creturn"][4]
        # Position should be exited on day 3, so day 4 return should equal day 3
        assert abs(day4_return - day3_return) < 1e-10, \
            f"Position should be flat after take profit, but day3={day3_return}, day4={day4_return}"

    def test_take_profit_no_trigger(self):
        """Test that take profit doesn't trigger when price rise is below threshold."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "A": [100.0, 100.0, 108.0],  # Day 2: +8% gain, below 10% threshold
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "A": [True],
        })

        result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0, take_profit=0.10)

        # Position should still be held (8% gain < 10% threshold)
        # Return should reflect the gain
        assert abs(result["creturn"][2] - 1.08) < 1e-10

    def test_trail_stop_triggers_exit(self):
        """Test that trailing stop triggers exit when price drops from high (T+1 execution).

        Trail stop formula (Finlab): (max_price - current_price) / entry_price >= trail_stop
        """
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"],
            "A": [100.0, 100.0, 120.0, 105.0, 110.0, 115.0],  # Peak at 120, then drops to 105
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "A": [True],
        })

        result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0, trail_stop=0.10)

        assert len(result) == 6
        # Day 2: peak at 120 (+20%), max_price = 120
        # Day 3: price = 105, drawdown_from_entry = (120 - 105) / 100 = 15% > 10%, trail_stop triggered
        # T+1 execution:
        # Day 4: Position experiences +4.76% return (105->110), then exits
        # Day 5: Position is flat (no position held)
        day4_return = result["creturn"][4]  # After trail stop exit
        day5_return = result["creturn"][5]
        # Position should be exited on day 4, so day 5 return should equal day 4
        assert abs(day5_return - day4_return) < 1e-10, \
            f"Position should be flat after trail stop, but day4={day4_return}, day5={day5_return}"

    def test_trail_stop_no_trigger(self):
        """Test that trailing stop doesn't trigger when drop from high is below threshold."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "A": [100.0, 100.0, 120.0, 112.0],  # Peak at 120, drop to 112 (-6.7%)
        })
        position = pl.DataFrame({
            "date": ["2024-01-01"],
            "A": [True],
        })

        result = backtest(prices, position, fee_ratio=0.0, tax_ratio=0.0, trail_stop=0.10)

        # Position should still be held (6.7% drop from high < 10% threshold)
        # Return should reflect the position value (112/100 = 1.12)
        assert abs(result["creturn"][3] - 1.12) < 1e-10

    def test_stop_trading_next_period_true(self):
        """Test that stop_trading_next_period=True blocks re-entry after stop loss."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"],
            "A": [100.0, 100.0, 84.0, 100.0, 100.0, 110.0],  # Day 2: -16% triggers stop, Day 5: +10%
        })
        position = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-04"],
            "A": [True, True],  # Try to re-enter on Day 4
        })

        result = backtest(
            prices, position,
            fee_ratio=0.0, tax_ratio=0.0,
            stop_loss=0.10,
            stop_trading_next_period=True
        )

        # Day 5 and Day 6 should remain flat because re-entry is blocked
        day4_return = result["creturn"][4]
        day5_return = result["creturn"][5]
        assert abs(day5_return - day4_return) < 1e-10, \
            "Re-entry should be blocked when stop_trading_next_period=True"

    def test_stop_trading_next_period_false(self):
        """Test that stop_trading_next_period=False allows re-entry after stop loss."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"],
            "A": [100.0, 100.0, 84.0, 100.0, 100.0, 110.0],  # Day 2: -16% triggers stop, Day 5: +10%
        })
        position = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-04"],
            "A": [True, True],  # Re-enter on Day 4
        })

        result = backtest(
            prices, position,
            fee_ratio=0.0, tax_ratio=0.0,
            stop_loss=0.10,
            stop_trading_next_period=False
        )

        # Day 5→6 should show return because re-entry is allowed
        day5_return = result["creturn"][5] / result["creturn"][4]
        assert day5_return > 1.05, \
            f"Re-entry should be allowed when stop_trading_next_period=False, got ratio {day5_return}"

    def test_retain_cost_when_rebalance_false(self):
        """Test that retain_cost_when_rebalance=False resets entry price on rebalance."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"],
            "A": [100.0, 100.0, 105.0, 110.0, 110.0, 100.0],  # Entry at 100, rebalance at 110, drop to 100
        })
        position = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-04"],
            "A": [True, True],  # Rebalance on Day 4
        })

        result = backtest(
            prices, position,
            fee_ratio=0.0, tax_ratio=0.0,
            stop_loss=0.10,
            retain_cost_when_rebalance=False  # Reset entry price to 110 on rebalance
        )

        # Day 5: drop from 110 to 100 = -9%, should NOT trigger 10% stop loss
        # Position should still be held, showing the loss
        day5_change = result["creturn"][5] / result["creturn"][4]
        assert day5_change > 0.85 and day5_change < 0.95, \
            f"Should show ~9% loss without stop loss trigger, got ratio {day5_change}"

    def test_retain_cost_when_rebalance_true(self):
        """Test that retain_cost_when_rebalance=True keeps original entry price."""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"],
            "A": [100.0, 100.0, 105.0, 110.0, 110.0, 88.0],  # Entry at 100, rebalance at 110, drop to 88
        })
        position = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-04"],
            "A": [True, True],  # Rebalance on Day 4
        })

        result = backtest(
            prices, position,
            fee_ratio=0.0, tax_ratio=0.0,
            stop_loss=0.10,
            retain_cost_when_rebalance=True  # Keep entry price at 100
        )

        # Day 5: drop from original entry 100 to 88 = -12%, should trigger 10% stop loss
        # The return should reflect stop loss being triggered
        day5_return = result["creturn"][5]
        day4_return = result["creturn"][4]
        assert day5_return < day4_return * 0.92, \
            f"Should trigger stop loss with retain_cost=True, got ratio {day5_return/day4_return}"


class TestLowLevelFunctions:
    """Tests for low-level backtest functions."""

    def test_backtest_signals_direct(self):
        """Test backtest_signals function directly (T+1 execution)."""
        prices = pl.DataFrame({
            "A": [100.0, 100.0, 110.0],
            "B": [100.0, 100.0, 90.0],
        })
        signals = pl.DataFrame({
            "A": [True],
            "B": [True],
        })

        creturn = backtest_signals(prices, signals, [0])

        assert len(creturn) == 3
        # T+1: Day 0 has no position yet
        assert abs(creturn[0] - 1.0) < 1e-10
        # Day 1: Position entered with fee
        # Entry fee: effective = 1.0 / (1 + fee_ratio)
        assert abs(creturn[1] - 1.0 / 1.001425) < 1e-5

    def test_backtest_weights_direct(self):
        """Test backtest_weights function directly (T+1 execution)."""
        prices = pl.DataFrame({
            "A": [100.0, 100.0, 110.0],
            "B": [100.0, 100.0, 100.0],
        })
        weights = pl.DataFrame({
            "A": [0.7],
            "B": [0.3],
        })

        config = BacktestConfig(fee_ratio=0.0, tax_ratio=0.0)
        creturn = backtest_weights(prices, weights, [0], config)

        assert len(creturn) == 3
        assert abs(creturn[0] - 1.0) < 1e-10  # Day 0: Signal sent, no position yet
        assert abs(creturn[1] - 1.0) < 1e-10  # Day 1: Position entered, no return yet
        # Day 2: 0.7 * (110/100-1) + 0.3 * 0% = 7%
        assert abs(creturn[2] - 1.07) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
