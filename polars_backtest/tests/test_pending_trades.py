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


class TestWeights:
    """Test weights() and next_weights() methods."""

    def test_weights_returns_current_positions(self, simple_data_with_pending_entry):
        """Test that weights() returns only currently held positions."""
        df = simple_data_with_pending_entry

        report = df.bt.backtest_with_report(
            trade_at_price="close", position="weight", fee_ratio=0.0, tax_ratio=0.0, resample=None
        )

        weights = report.weights()

        # Only A is currently held (B is pending entry, not yet entered)
        symbols = weights["symbol"].to_list()
        assert "A" in symbols, f"A should be in weights, got {symbols}"

        # Weights should be normalized
        total = weights["weight"].sum()
        assert abs(total - 1.0) < 0.01 or total == 0.0, f"Weights should sum to 1, got {total}"

    def test_next_weights_includes_pending_entries(self, simple_data_with_pending_entry):
        """Test that next_weights() includes pending entry stocks."""
        df = simple_data_with_pending_entry

        report = df.bt.backtest_with_report(
            trade_at_price="close", position="weight", fee_ratio=0.0, tax_ratio=0.0, resample=None
        )

        next_weights = report.next_weights()

        # Both A (continuing) and B (entering) should be in next_weights
        symbols = next_weights["symbol"].to_list()
        assert "A" in symbols, f"A should be in next_weights, got {symbols}"
        assert "B" in symbols, f"B should be in next_weights (pending entry), got {symbols}"

        # Weights should be normalized
        total = next_weights["weight"].sum()
        assert abs(total - 1.0) < 0.01, f"Next weights should sum to 1, got {total}"

    def test_weights_excludes_pending_exit(self, simple_data_with_pending_exit):
        """Test weights with pending exit scenario."""
        df = simple_data_with_pending_exit

        report = df.bt.backtest_with_report(
            trade_at_price="close", position="weight", fee_ratio=0.0, tax_ratio=0.0, resample=None
        )

        weights = report.weights()
        next_weights = report.next_weights()

        # Current weights should include B (still holding)
        current_symbols = weights["symbol"].to_list()
        assert "B" in current_symbols, f"B should be in current weights, got {current_symbols}"

        # Next weights should NOT include B (exiting)
        next_symbols = next_weights["symbol"].to_list()
        assert "B" not in next_symbols, f"B should NOT be in next_weights (pending exit), got {next_symbols}"
        assert "A" in next_symbols, f"A should be in next_weights, got {next_symbols}"

    def test_weights_sum_normalized(self, simple_data_with_pending_entry):
        """Test that both weights and next_weights are properly normalized."""
        df = simple_data_with_pending_entry

        report = df.bt.backtest_with_report(
            trade_at_price="close", position="weight", fee_ratio=0.0, tax_ratio=0.0, resample=None
        )

        weights = report.weights()
        next_weights = report.next_weights()

        if weights.height > 0:
            weights_sum = weights["weight"].sum()
            assert weights_sum <= 1.0 + 0.01, f"Weights sum {weights_sum} should be <= 1"

        if next_weights.height > 0:
            next_sum = next_weights["weight"].sum()
            assert next_sum <= 1.0 + 0.01, f"Next weights sum {next_sum} should be <= 1"


class TestActionsWithWeights:
    """Test actions() with weight and next_weight columns."""

    def test_actions_has_weight_columns(self, simple_data_with_pending_entry):
        """Test that actions() returns weight and next_weight columns."""
        df = simple_data_with_pending_entry

        report = df.bt.backtest_with_report(
            trade_at_price="close", position="weight", fee_ratio=0.0, tax_ratio=0.0, resample=None
        )

        actions = report.actions()

        # Check columns exist
        assert "weight" in actions.columns, f"actions should have 'weight' column, got {actions.columns}"
        assert "next_weight" in actions.columns, f"actions should have 'next_weight' column, got {actions.columns}"

    def test_enter_action_has_zero_weight(self, simple_data_with_pending_entry):
        """Test that 'enter' action has weight=0 (not yet held)."""
        df = simple_data_with_pending_entry

        report = df.bt.backtest_with_report(
            trade_at_price="close", position="weight", fee_ratio=0.0, tax_ratio=0.0, resample=None
        )

        actions = report.actions()

        # B is entering, so weight should be 0
        b_action = actions.filter(pl.col("symbol") == "B")
        assert b_action.height == 1
        assert b_action["action"][0] == "enter"
        assert b_action["weight"][0] == 0.0, f"Enter stock should have weight=0, got {b_action['weight'][0]}"
        assert b_action["next_weight"][0] > 0.0, f"Enter stock should have next_weight>0, got {b_action['next_weight'][0]}"

    def test_hold_action_has_weight(self, simple_data_with_pending_entry):
        """Test that 'hold' action has both weight and next_weight."""
        df = simple_data_with_pending_entry

        report = df.bt.backtest_with_report(
            trade_at_price="close", position="weight", fee_ratio=0.0, tax_ratio=0.0, resample=None
        )

        actions = report.actions()

        # A is holding, should have both weights
        a_action = actions.filter(pl.col("symbol") == "A")
        assert a_action.height == 1
        assert a_action["action"][0] == "hold"
        assert a_action["weight"][0] > 0.0, f"Hold stock should have weight>0, got {a_action['weight'][0]}"
        assert a_action["next_weight"][0] > 0.0, f"Hold stock should have next_weight>0, got {a_action['next_weight'][0]}"

    def test_exit_action_has_zero_next_weight(self, simple_data_with_pending_exit):
        """Test that 'exit' action has next_weight=0 (leaving portfolio)."""
        df = simple_data_with_pending_exit

        report = df.bt.backtest_with_report(
            trade_at_price="close", position="weight", fee_ratio=0.0, tax_ratio=0.0, resample=None
        )

        actions = report.actions()

        # B is exiting, so next_weight should be 0
        b_action = actions.filter(pl.col("symbol") == "B")
        assert b_action.height == 1
        assert b_action["action"][0] == "exit"
        assert b_action["weight"][0] > 0.0, f"Exit stock should have weight>0, got {b_action['weight'][0]}"
        assert b_action["next_weight"][0] == 0.0, f"Exit stock should have next_weight=0, got {b_action['next_weight'][0]}"
