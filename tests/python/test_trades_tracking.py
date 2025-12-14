"""Test trades tracking functionality.

Tests verify that:
1. backtest_with_report() returns proper trades records
2. trades use ORIGINAL prices (not adjusted) for entry/exit prices
3. The trades format matches Finlab's trades DataFrame structure
4. trade_at_price parameter works correctly
5. factor parameter correctly restores original prices
"""

import os
import pytest
import polars as pl
import pandas as pd
import numpy as np

# Skip all tests if finlab is not available
finlab = pytest.importorskip("finlab")

from dotenv import load_dotenv
load_dotenv()

finlab.login(os.getenv('FINLAB_API_TOKEN'))

from finlab import backtest as finlab_backtest
from finlab import data as finlab_data

from polars_backtest import backtest, backtest_with_report, Report


@pytest.fixture(scope="module")
def price_data():
    """Load price data from Finlab."""
    adj_close = finlab_data.get('etl:adj_close')
    close = finlab_data.get('price:收盤價')
    return adj_close, close


@pytest.fixture(scope="module")
def factor_data(price_data):
    """Calculate adjustment factor from price data.

    factor = adj_close / close
    So: close = adj_close / factor
    """
    adj_close, close = price_data
    # Align the dataframes
    common_dates = adj_close.index.intersection(close.index)
    common_cols = adj_close.columns.intersection(close.columns)

    adj = adj_close.loc[common_dates, common_cols]
    orig = close.loc[common_dates, common_cols]

    # factor = adj / original, so original = adj / factor
    factor = adj / orig
    return factor


def test_basic_trades_tracking(price_data, factor_data):
    """Test basic trades tracking with a simple single-stock backtest."""
    adj_close, close = price_data
    factor = factor_data

    stock = '2330'
    all_dates = adj_close.index
    dates = all_dates[(all_dates >= '2024-06-01') & (all_dates <= '2024-06-30')]

    # Create position (hold entire period)
    position_pd = pd.DataFrame({stock: [True] * len(dates)}, index=dates)

    fee_ratio = 0.001425
    tax_ratio = 0.003

    # Run Finlab
    finlab_report = finlab_backtest.sim(
        position_pd, resample='D',
        fee_ratio=fee_ratio, tax_ratio=tax_ratio,
        upload=False
    )

    # Prepare data for our backtest
    finlab_dates = finlab_report.creturn.index.tolist()
    date_strs = [str(d.date()) for d in finlab_dates]

    # Adjusted prices
    adj_close_pd = adj_close.loc[finlab_dates, [stock]]
    adj_prices_pl = pl.DataFrame({"date": date_strs, stock: adj_close_pd[stock].tolist()})

    # Factor for converting adj to original
    factor_pd = factor.loc[finlab_dates, [stock]]
    factor_pl = pl.DataFrame({"date": date_strs, stock: factor_pd[stock].tolist()})

    # Position weights
    finlab_position = finlab_report.position
    position_aligned = finlab_position.reindex(finlab_dates).ffill()[stock].fillna(0.0)

    # Find rebalance points
    change_mask = position_aligned.diff().abs() != 0
    change_mask.iloc[0] = True
    filtered_position = position_aligned[change_mask]
    rebalance_date_strs = [str(d.date()) for d in filtered_position.index.tolist()]
    weights_pl = pl.DataFrame({"date": rebalance_date_strs, stock: filtered_position.tolist()})

    # Run our backtest with factor
    report = backtest_with_report(
        close=adj_prices_pl,
        position=weights_pl,
        factor=factor_pl,
        fee_ratio=fee_ratio,
        tax_ratio=tax_ratio,
    )

    # Verify we got trades
    assert len(report.trades) >= 1, "Should have at least one trade"

    # Verify entry price uses original price (restored via factor)
    first_trade = report.trades.row(0, named=True)
    entry_date = first_trade.get('entry_date')
    entry_price = first_trade.get('trade_price@entry_date')

    expected_original = close.loc[pd.Timestamp(entry_date), stock]
    expected_adjusted = adj_close.loc[pd.Timestamp(entry_date), stock]

    # Entry price should match original, not adjusted
    assert abs(entry_price - expected_original) < 0.01, \
        f"Entry price {entry_price} should match original {expected_original}, not adjusted {expected_adjusted}"


def test_multi_stock_trades(price_data, factor_data):
    """Test trades tracking with multiple stocks and rebalancing."""
    adj_close, close = price_data
    factor = factor_data

    stocks = ['2330', '2317']
    all_dates = adj_close.index
    dates = all_dates[(all_dates >= '2024-01-01') & (all_dates <= '2024-03-31')]

    # Create position with switch: 2330 in Jan-Feb, 2317 in Mar
    position_dict = {}
    for d in dates:
        if d.month <= 2:
            position_dict[d] = {stocks[0]: 1.0, stocks[1]: 0.0}
        else:
            position_dict[d] = {stocks[0]: 0.0, stocks[1]: 1.0}

    position_pd = pd.DataFrame(position_dict).T
    position_pd = position_pd.reindex(dates).ffill()

    fee_ratio = 0.001425
    tax_ratio = 0.003

    # Run Finlab
    finlab_report = finlab_backtest.sim(
        position_pd, resample='D',
        fee_ratio=fee_ratio, tax_ratio=tax_ratio,
        upload=False
    )

    # Prepare data
    finlab_dates = finlab_report.creturn.index.tolist()
    date_strs = [str(d.date()) for d in finlab_dates]

    # Prices and factor
    adj_data = {"date": date_strs}
    factor_data_dict = {"date": date_strs}
    for s in stocks:
        adj_data[s] = adj_close.loc[finlab_dates, s].tolist()
        factor_data_dict[s] = factor.loc[finlab_dates, s].tolist()
    adj_prices_pl = pl.DataFrame(adj_data)
    factor_pl = pl.DataFrame(factor_data_dict)

    # Position
    finlab_position = finlab_report.position
    position_aligned = finlab_position.reindex(finlab_dates).ffill()[stocks].fillna(0.0)

    # Find rebalance points
    change_mask = position_aligned.diff().abs().sum(axis=1) != 0
    change_mask.iloc[0] = True
    filtered_weights = position_aligned[change_mask]
    rebalance_date_strs = [str(d.date()) for d in filtered_weights.index.tolist()]
    weights_data = {"date": rebalance_date_strs}
    for s in stocks:
        weights_data[s] = filtered_weights[s].tolist()
    weights_pl = pl.DataFrame(weights_data)

    # Run our backtest with factor
    report = backtest_with_report(
        close=adj_prices_pl,
        position=weights_pl,
        factor=factor_pl,
        fee_ratio=fee_ratio,
        tax_ratio=tax_ratio,
    )

    # Should have 2 trades (one for each stock)
    assert len(report.trades) == 2, f"Expected 2 trades, got {len(report.trades)}"

    # Verify all prices are original (restored via factor)
    for row in report.trades.iter_rows(named=True):
        stock_id = row['stock_id']
        entry_date = row['entry_date']
        entry_price = row['trade_price@entry_date']

        if entry_date:
            expected_original = close.loc[pd.Timestamp(entry_date), stock_id]
            expected_adjusted = adj_close.loc[pd.Timestamp(entry_date), stock_id]

            # Entry price should be closer to original than adjusted
            diff_original = abs(entry_price - expected_original)
            diff_adjusted = abs(entry_price - expected_adjusted)

            assert diff_original < diff_adjusted or diff_original < 1.0, \
                f"{stock_id} @ {entry_date}: entry={entry_price:.2f} should be original={expected_original:.2f}, not adjusted={expected_adjusted:.2f}"


def test_creturn_matches_standard(price_data, factor_data):
    """Verify that creturn from backtest_with_report matches standard backtest."""
    adj_close, close = price_data
    factor = factor_data

    stock = '2330'
    all_dates = adj_close.index
    dates = all_dates[(all_dates >= '2024-01-01') & (all_dates <= '2024-03-31')]

    date_strs = [str(d.date()) for d in dates]

    # Prices and factor
    adj_prices_pl = pl.DataFrame({"date": date_strs, stock: adj_close.loc[dates, stock].tolist()})
    factor_pl = pl.DataFrame({"date": date_strs, stock: factor.loc[dates, stock].tolist()})

    # Position (hold entire period)
    weights_pl = pl.DataFrame({"date": [date_strs[0]], stock: [1.0]})

    # Run standard backtest
    standard_result = backtest(adj_prices_pl, weights_pl)

    # Run backtest_with_report with factor
    report = backtest_with_report(
        close=adj_prices_pl,
        position=weights_pl,
        factor=factor_pl,
    )

    # Compare
    standard_creturn = standard_result["creturn"].to_list()
    report_creturn = report.creturn.to_list()

    max_diff = max(abs(s - r) for s, r in zip(standard_creturn, report_creturn))

    assert max_diff < 1e-10, f"creturn should match, but max diff is {max_diff}"


def test_trade_at_price_requires_ohlc():
    """Test that trade_at_price='open' requires open DataFrame."""
    close = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "2330": [100.0, 102.0],
    })
    position = pl.DataFrame({
        "date": ["2024-01-01"],
        "2330": [1.0],
    })

    with pytest.raises(ValueError, match="requires 'open' DataFrame"):
        backtest_with_report(
            close=close,
            position=position,
            trade_at_price='open',  # No open DataFrame provided
        )


def test_trade_at_price_invalid():
    """Test that invalid trade_at_price raises error."""
    close = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "2330": [100.0, 102.0],
    })
    position = pl.DataFrame({
        "date": ["2024-01-01"],
        "2330": [1.0],
    })

    with pytest.raises(ValueError, match="Invalid trade_at_price"):
        backtest_with_report(
            close=close,
            position=position,
            trade_at_price='invalid',
        )


def test_report_properties():
    """Test Report object properties."""
    close = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "2330": [100.0, 102.0, 105.0],
    })
    position = pl.DataFrame({
        "date": ["2024-01-01"],
        "2330": [1.0],
    })

    report = backtest_with_report(close=close, position=position)

    # Check creturn is a Series
    assert isinstance(report.creturn, pl.Series)
    assert len(report.creturn) == 3

    # Check position is a DataFrame
    assert isinstance(report.position, pl.DataFrame)

    # Check trades is a DataFrame
    assert isinstance(report.trades, pl.DataFrame)

    # Check trades has required columns
    expected_columns = [
        'stock_id', 'entry_date', 'exit_date',
        'entry_sig_date', 'exit_sig_date', 'position',
        'period', 'return', 'trade_price@entry_date', 'trade_price@exit_date'
    ]
    for col in expected_columns:
        assert col in report.trades.columns, f"Missing column: {col}"


def test_empty_trades():
    """Test that empty trades returns empty DataFrame with correct schema."""
    # Position with zero weight
    close = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "2330": [100.0, 102.0],
    })
    position = pl.DataFrame({
        "date": ["2024-01-01"],
        "2330": [0.0],  # No position
    })

    report = backtest_with_report(close=close, position=position)

    # trades should be empty but have correct columns
    assert len(report.trades) == 0
    assert 'stock_id' in report.trades.columns
    assert 'trade_price@entry_date' in report.trades.columns


def test_factor_restores_original_prices():
    """Test that factor correctly restores original prices from adjusted prices."""
    # Adjusted prices (e.g., after stock split: original / 2)
    adj_close = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "2330": [50.0, 51.0, 52.0],  # adjusted
    })

    # Factor = adj / original, so original = adj / factor
    # If original was [100, 102, 104] and adj is [50, 51, 52]
    # factor = [0.5, 0.5, 0.5]
    factor = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "2330": [0.5, 0.5, 0.5],
    })

    position = pl.DataFrame({
        "date": ["2024-01-01"],
        "2330": [1.0],
    })

    report = backtest_with_report(
        close=adj_close,
        position=position,
        factor=factor,
    )

    # T+1 execution: Signal on day 0 → Execute on day 1
    # Entry price should be original at day 1: 51 / 0.5 = 102
    first_trade = report.trades.row(0, named=True)
    entry_price = first_trade.get('trade_price@entry_date')

    expected_original = 102.0  # 51 / 0.5 (day 1 price)
    assert abs(entry_price - expected_original) < 0.01, \
        f"Entry price {entry_price} should be {expected_original} (adj[1] / factor)"


def test_factor_with_trade_at_price_open():
    """Test that factor works correctly with trade_at_price='open'."""
    # Adjusted close (for creturn)
    adj_close = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "2330": [50.0, 51.0, 52.0],
    })

    # Adjusted open (for trading)
    adj_open = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "2330": [49.5, 50.5, 51.5],
    })

    # Factor to restore original prices
    factor = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "2330": [0.5, 0.5, 0.5],
    })

    position = pl.DataFrame({
        "date": ["2024-01-01"],
        "2330": [1.0],
    })

    report = backtest_with_report(
        close=adj_close,
        position=position,
        trade_at_price='open',
        open=adj_open,
        factor=factor,
    )

    # Entry price should be original open: 49.5 / 0.5 = 99
    # (trade_at_price='open' means we use adj_open, then divide by factor)
    first_trade = report.trades.row(0, named=True)
    entry_price = first_trade.get('trade_price@entry_date')

    # Entry happens on day T+1 (index 1) with T+1 execution
    # Signal on day 0, execute on day 1
    expected_original_open = 50.5 / 0.5  # 101 (open price on day 1)
    assert abs(entry_price - expected_original_open) < 0.01, \
        f"Entry price {entry_price} should be {expected_original_open} (adj_open[1] / factor)"
