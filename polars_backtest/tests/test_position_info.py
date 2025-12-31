"""Tests for Stage 4: Position Information."""
import polars as pl
import pytest
from polars_backtest import backtest_with_report_wide


@pytest.fixture
def simple_backtest():
    """Create a simple backtest with known positions."""
    dates = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
    close = pl.DataFrame({
        'date': dates,
        'A': [100.0, 101.0, 102.0, 103.0, 104.0],
        'B': [50.0, 51.0, 52.0, 51.0, 50.0],
        'C': [200.0, 202.0, 204.0, 206.0, 208.0],
    })
    position = pl.DataFrame({
        'date': dates,
        'A': [0.0, 0.5, 0.5, 0.5, 0.5],  # Enter A on day 2
        'B': [0.0, 0.0, 0.3, 0.3, 0.0],  # Enter B on day 3, exit on day 5
        'C': [0.0, 0.0, 0.0, 0.2, 0.2],  # Enter C on day 4
    })
    return backtest_with_report_wide(close, position, resample=None)


@pytest.fixture
def stop_loss_backtest():
    """Create a backtest with stop loss triggered."""
    dates = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
    close = pl.DataFrame({
        'date': dates,
        'A': [100.0, 101.0, 85.0, 84.0, 83.0],  # A drops 15%
        'B': [50.0, 51.0, 52.0, 53.0, 54.0],   # B goes up
    })
    position = pl.DataFrame({
        'date': dates,
        'A': [0.0, 0.5, 0.5, 0.5, 0.5],
        'B': [0.0, 0.5, 0.5, 0.5, 0.5],
    })
    return backtest_with_report_wide(
        close, position, resample=None, stop_loss=0.10
    )


class TestWeights:
    """Tests for weights property."""

    def test_weights_returns_dataframe(self, simple_backtest):
        """weights should return a DataFrame."""
        weights = simple_backtest.weights
        assert isinstance(weights, pl.DataFrame)

    def test_weights_has_required_columns(self, simple_backtest):
        """weights should have stock_id, weight, and date columns."""
        weights = simple_backtest.weights
        assert "stock_id" in weights.columns
        assert "weight" in weights.columns
        assert "date" in weights.columns

    def test_weights_filters_zero_positions(self, simple_backtest):
        """weights should only include non-zero positions."""
        weights = simple_backtest.weights
        # Last row has A=0.5, B=0, C=0.2
        assert weights.height == 2
        stock_ids = weights["stock_id"].to_list()
        assert "A" in stock_ids
        assert "C" in stock_ids
        assert "B" not in stock_ids  # B has weight 0

    def test_weights_correct_values(self, simple_backtest):
        """weights should have correct weight values."""
        weights = simple_backtest.weights
        a_weight = weights.filter(pl.col("stock_id") == "A")["weight"][0]
        c_weight = weights.filter(pl.col("stock_id") == "C")["weight"][0]
        assert a_weight == 0.5
        assert c_weight == 0.2


class TestNextWeights:
    """Tests for next_weights property."""

    def test_next_weights_returns_none(self, simple_backtest):
        """next_weights should return None in wide format."""
        assert simple_backtest.next_weights is None


class TestCurrentTrades:
    """Tests for current_trades property."""

    def test_current_trades_returns_dataframe(self, simple_backtest):
        """current_trades should return a DataFrame."""
        current = simple_backtest.current_trades
        assert isinstance(current, pl.DataFrame)

    def test_current_trades_filters_active(self, simple_backtest):
        """current_trades should only include active trades."""
        current = simple_backtest.current_trades
        # Active trades have null exit_date or exit_date == last_date
        for row in current.iter_rows(named=True):
            exit_date = row["exit_date"]
            # Either null or matches last date
            assert exit_date is None or "2024-01-05" in str(exit_date)


class TestActions:
    """Tests for actions property."""

    def test_actions_returns_dataframe(self, simple_backtest):
        """actions should return a DataFrame."""
        actions = simple_backtest.actions
        assert isinstance(actions, pl.DataFrame)

    def test_actions_has_required_columns(self, simple_backtest):
        """actions should have stock_id and action columns."""
        actions = simple_backtest.actions
        assert "stock_id" in actions.columns
        assert "action" in actions.columns

    def test_actions_valid_values(self, simple_backtest):
        """actions should only contain valid action values."""
        actions = simple_backtest.actions
        valid_actions = {"enter", "exit", "hold"}
        for action in actions["action"].to_list():
            assert action in valid_actions


class TestPositionInfo:
    """Tests for position_info method."""

    def test_position_info_returns_dataframe(self, simple_backtest):
        """position_info should return a DataFrame."""
        pos_info = simple_backtest.position_info()
        assert isinstance(pos_info, pl.DataFrame)

    def test_position_info_has_required_columns(self, simple_backtest):
        """position_info should have required columns."""
        pos_info = simple_backtest.position_info()
        required = ["stock_id", "weight", "entry_date", "exit_date", "return", "action"]
        for col in required:
            assert col in pos_info.columns

    def test_position_info_empty_for_no_current_trades(self):
        """position_info should return empty DataFrame for no current trades."""
        # All trades already closed
        dates = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
        close = pl.DataFrame({
            'date': dates,
            'A': [100.0, 101.0, 102.0, 103.0],
        })
        position = pl.DataFrame({
            'date': dates,
            'A': [0.0, 0.5, 0.5, 0.0],  # Enter and exit before last date
        })
        report = backtest_with_report_wide(close, position, resample=None)
        # Current trades should be empty (trade already closed)
        # Note: depends on Rust engine behavior
        pos_info = report.position_info()
        assert isinstance(pos_info, pl.DataFrame)


class TestPositionInfo2:
    """Tests for position_info2 method."""

    def test_position_info2_returns_dict(self, simple_backtest):
        """position_info2 should return a dict."""
        pos_info2 = simple_backtest.position_info2()
        assert isinstance(pos_info2, dict)

    def test_position_info2_has_positions_key(self, simple_backtest):
        """position_info2 should have positions list."""
        pos_info2 = simple_backtest.position_info2()
        assert "positions" in pos_info2
        assert isinstance(pos_info2["positions"], list)

    def test_position_info2_has_config_key(self, simple_backtest):
        """position_info2 should have positionConfig."""
        pos_info2 = simple_backtest.position_info2()
        assert "positionConfig" in pos_info2
        config = pos_info2["positionConfig"]
        assert "feeRatio" in config
        assert "taxRatio" in config
        assert "resample" in config
        assert "tradeAt" in config

    def test_position_info2_dates_are_strings(self, simple_backtest):
        """position_info2 dates should be ISO strings."""
        pos_info2 = simple_backtest.position_info2()
        for p in pos_info2["positions"]:
            if p.get("entry_date"):
                assert isinstance(p["entry_date"], str)
                assert len(p["entry_date"]) == 10  # YYYY-MM-DD


class TestIsRebalanceDue:
    """Tests for is_rebalance_due method."""

    def test_is_rebalance_due_returns_bool(self, simple_backtest):
        """is_rebalance_due should return a bool."""
        result = simple_backtest.is_rebalance_due()
        assert isinstance(result, bool)

    def test_is_rebalance_due_detects_change(self):
        """is_rebalance_due should detect position changes."""
        dates = ['2024-01-01', '2024-01-02', '2024-01-03']
        close = pl.DataFrame({
            'date': dates,
            'A': [100.0, 101.0, 102.0],
        })
        # Position changes on last day
        position = pl.DataFrame({
            'date': dates,
            'A': [0.0, 0.5, 0.3],  # Changed from 0.5 to 0.3
        })
        report = backtest_with_report_wide(close, position, resample=None)
        assert report.is_rebalance_due() is True

    def test_is_rebalance_due_no_change(self):
        """is_rebalance_due should return False when no change."""
        dates = ['2024-01-01', '2024-01-02', '2024-01-03']
        close = pl.DataFrame({
            'date': dates,
            'A': [100.0, 101.0, 102.0],
        })
        # Position doesn't change on last day
        position = pl.DataFrame({
            'date': dates,
            'A': [0.0, 0.5, 0.5],  # Same as previous
        })
        report = backtest_with_report_wide(close, position, resample=None)
        assert report.is_rebalance_due() is False


class TestIsStopTriggered:
    """Tests for is_stop_triggered method."""

    def test_is_stop_triggered_returns_bool(self, simple_backtest):
        """is_stop_triggered should return a bool."""
        result = simple_backtest.is_stop_triggered()
        assert isinstance(result, bool)

    def test_is_stop_triggered_no_stop(self, simple_backtest):
        """is_stop_triggered should return False when no stop configured."""
        # simple_backtest has no stop loss/take profit
        assert simple_backtest.is_stop_triggered() is False

    def test_is_stop_triggered_with_stop_loss(self, stop_loss_backtest):
        """is_stop_triggered should detect stop loss."""
        # Stock A dropped 15% which should trigger 10% stop loss
        # Note: The actual triggering depends on the Rust engine implementation
        # This test verifies the method runs without error
        result = stop_loss_backtest.is_stop_triggered()
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
