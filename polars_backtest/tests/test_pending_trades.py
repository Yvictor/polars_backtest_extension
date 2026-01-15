"""Tests for pending trades and actions functionality.

This module tests the Finlab-compatible pending trades behavior:
- Pending entry: trades with entry_date=null, entry_sig_date=latest signal date
- Pending exit: trades with exit_date=null, exit_sig_date=latest signal date
- Actions: enter/exit/hold based on comparing current positions vs signal weights
"""

import datetime
import polars as pl
import pytest
import polars_backtest as pl_bt


@pytest.fixture
def simple_data_with_pending_entry():
    """Create test data where the last signal has new stock to enter.

    Long format DataFrame with:
    - A: signal from day 1
    - B: signal on day 3 (pending entry)
    """
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]

    # Long format: date, symbol, close, weight
    df = pl.DataFrame({
        "date": dates * 2,
        "symbol": ["A"] * 3 + ["B"] * 3,
        "close": [100.0, 101.0, 102.0, 50.0, 51.0, 52.0],
        "weight": [1.0, 1.0, 1.0, 0.0, 0.0, 1.0],  # B gets signal on day 3
    }).with_columns(pl.col("date").str.to_date()).sort(["date", "symbol"])

    return df


@pytest.fixture
def simple_data_with_pending_exit():
    """Create test data where the last signal removes a stock.

    Long format DataFrame with:
    - A: signal on all days
    - B: signal on day 1-2, exit signal on day 3
    """
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]

    df = pl.DataFrame({
        "date": dates * 2,
        "symbol": ["A"] * 3 + ["B"] * 3,
        "close": [100.0, 101.0, 102.0, 50.0, 51.0, 52.0],
        "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],  # B exit signal on day 3
    }).with_columns(pl.col("date").str.to_date()).sort(["date", "symbol"])

    return df


@pytest.fixture
def simple_data_hold_only():
    """Create test data where no changes happen on last day."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]

    df = pl.DataFrame({
        "date": dates,
        "symbol": ["A"] * 3,
        "close": [100.0, 101.0, 102.0],
        "weight": [1.0, 1.0, 1.0],  # Always holding A
    }).with_columns(pl.col("date").str.to_date())

    return df


class TestPendingEntry:
    """Test pending entry trades."""

    def test_pending_entry_creates_trade(self, simple_data_with_pending_entry):
        """Test that a new signal creates a pending entry trade."""
        df = simple_data_with_pending_entry

        report = df.bt.backtest_with_report(
            trade_at_price="close", position="weight", fee_ratio=0.0, tax_ratio=0.0, resample=None
        )

        trades = report.trades

        # Should have trades for both A and B
        stock_ids = trades["stock_id"].to_list()
        assert "A" in stock_ids
        assert "B" in stock_ids

        # B should have entry_date=null (pending)
        b_trade = trades.filter(pl.col("stock_id") == "B")
        assert b_trade.height == 1
        assert b_trade["entry_date"][0] is None, "Pending entry should have entry_date=null"
        assert b_trade["entry_sig_date"][0] is not None, "Pending entry should have entry_sig_date"

    def test_pending_entry_action_is_enter(self, simple_data_with_pending_entry):
        """Test that pending entry shows as 'enter' action."""
        df = simple_data_with_pending_entry

        report = df.bt.backtest_with_report(
            trade_at_price="close", position="weight", fee_ratio=0.0, tax_ratio=0.0, resample=None
        )

        actions = report.actions()

        # Convert to dict for easier checking
        action_dict = dict(zip(
            actions["symbol"].to_list(),
            actions["action"].to_list()
        ))

        assert action_dict.get("B") == "enter", f"B should be 'enter', got {action_dict}"


class TestPendingExit:
    """Test pending exit trades."""

    def test_pending_exit_has_exit_sig_date(self, simple_data_with_pending_exit):
        """Test that exit signal sets exit_sig_date."""
        df = simple_data_with_pending_exit

        report = df.bt.backtest_with_report(
            trade_at_price="close", position="weight", fee_ratio=0.0, tax_ratio=0.0, resample=None
        )

        trades = report.trades

        # B should have exit_sig_date set but exit_date=null
        b_trades = trades.filter(pl.col("stock_id") == "B")
        assert b_trades.height >= 1

        # Get the open trade (entry_date is set, exit_date is null)
        open_b = b_trades.filter(
            pl.col("entry_date").is_not_null() & pl.col("exit_date").is_null()
        )
        assert open_b.height == 1
        assert open_b["exit_sig_date"][0] is not None, "Pending exit should have exit_sig_date"

    def test_pending_exit_action_is_exit(self, simple_data_with_pending_exit):
        """Test that pending exit shows as 'exit' action."""
        df = simple_data_with_pending_exit

        report = df.bt.backtest_with_report(
            trade_at_price="close", position="weight", fee_ratio=0.0, tax_ratio=0.0, resample=None
        )

        actions = report.actions()

        action_dict = dict(zip(
            actions["symbol"].to_list(),
            actions["action"].to_list()
        ))

        assert action_dict.get("B") == "exit", f"B should be 'exit', got {action_dict}"


class TestHoldAction:
    """Test hold actions."""

    def test_hold_action(self, simple_data_hold_only):
        """Test that continuing position shows as 'hold' action."""
        df = simple_data_hold_only

        report = df.bt.backtest_with_report(
            trade_at_price="close", position="weight", fee_ratio=0.0, tax_ratio=0.0, resample=None
        )

        actions = report.actions()

        action_dict = dict(zip(
            actions["symbol"].to_list(),
            actions["action"].to_list()
        ))

        assert action_dict.get("A") == "hold", f"A should be 'hold', got {action_dict}"


class TestActionsCount:
    """Test actions count matches expected."""

    def test_actions_count_with_mixed(self, simple_data_with_pending_entry):
        """Test actions count with both enter and hold."""
        df = simple_data_with_pending_entry

        report = df.bt.backtest_with_report(
            trade_at_price="close", position="weight", fee_ratio=0.0, tax_ratio=0.0, resample=None
        )

        actions = report.actions()

        # Should have 2 actions: A=hold, B=enter
        assert actions.height == 2, f"Expected 2 actions, got {actions.height}"

        action_types = actions["action"].to_list()
        assert "hold" in action_types
        assert "enter" in action_types


class TestTradesCount:
    """Test trades count with pending trades."""

    def test_pending_entry_increases_trades_count(self, simple_data_with_pending_entry):
        """Test that pending entry creates additional trade record."""
        df = simple_data_with_pending_entry

        report = df.bt.backtest_with_report(
            trade_at_price="close", position="weight", fee_ratio=0.0, tax_ratio=0.0, resample=None
        )

        trades = report.trades

        # Should have at least 2 trades: A (open) and B (pending)
        assert trades.height >= 2, f"Expected at least 2 trades, got {trades.height}"
