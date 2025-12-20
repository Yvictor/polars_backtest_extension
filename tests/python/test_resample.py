"""Test resample functionality.

Tests verify that:
1. resample='D' is the default (no change)
2. resample='W' creates weekly rebalance points
3. resample='M' creates monthly rebalance points
4. Results match Finlab's behavior
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

from polars_backtest import backtest, backtest_with_report


@pytest.fixture(scope="module")
def price_data():
    """Load price data from Finlab."""
    adj_close = finlab_data.get('etl:adj_close')
    return adj_close


def test_resample_daily_default(price_data):
    """Test that resample='D' is the default and doesn't change behavior."""
    adj_close = price_data

    stock = '2330'
    all_dates = adj_close.index
    dates = all_dates[(all_dates >= '2024-01-01') & (all_dates <= '2024-03-31')]

    # Create position
    position_pd = pd.DataFrame({stock: [True] * len(dates)}, index=dates)

    # Run Finlab with resample='D'
    finlab_report = finlab_backtest.sim(
        position_pd, resample='D',
        fee_ratio=0.001425, tax_ratio=0.003,
        upload=False
    )

    # Prepare for our backtest
    finlab_dates = finlab_report.creturn.index.tolist()
    date_strs = [str(d.date()) for d in finlab_dates]
    adj_prices_pl = pl.DataFrame({"date": date_strs, stock: adj_close.loc[finlab_dates, stock].tolist()})

    # Get weights
    finlab_position = finlab_report.position
    position_aligned = finlab_position.reindex(finlab_dates).ffill()[stock].fillna(0.0)
    change_mask = position_aligned.diff().abs() != 0
    change_mask.iloc[0] = True
    filtered_position = position_aligned[change_mask]
    rebalance_date_strs = [str(d.date()) for d in filtered_position.index.tolist()]
    weights_pl = pl.DataFrame({"date": rebalance_date_strs, stock: filtered_position.tolist()})

    # Run with explicit resample='D' and without (default)
    result_explicit = backtest(adj_prices_pl, weights_pl, resample='D')
    result_default = backtest(adj_prices_pl, weights_pl)

    # Should be identical
    max_diff = max(abs(a - b) for a, b in zip(
        result_explicit["creturn"].to_list(),
        result_default["creturn"].to_list()
    ))
    assert max_diff < 1e-10, f"resample='D' should match default, but diff is {max_diff}"


def test_resample_weekly(price_data):
    """Test resample='W' matches Finlab weekly rebalancing."""
    adj_close = price_data

    stock = '2330'
    all_dates = adj_close.index
    dates = all_dates[(all_dates >= '2024-01-01') & (all_dates <= '2024-03-31')]

    # Create daily position
    position_pd = pd.DataFrame({stock: [True] * len(dates)}, index=dates)

    # Run Finlab with weekly resample
    finlab_report = finlab_backtest.sim(
        position_pd, resample='W',
        fee_ratio=0.001425, tax_ratio=0.003,
        upload=False
    )

    # Prepare for our backtest
    finlab_dates = finlab_report.creturn.index.tolist()
    date_strs = [str(d.date()) for d in finlab_dates]
    adj_prices_pl = pl.DataFrame({"date": date_strs, stock: adj_close.loc[finlab_dates, stock].tolist()})

    # Create daily position for our backtest (resample will handle it)
    daily_weights_data = {"date": date_strs, stock: [1.0] * len(date_strs)}
    daily_weights_pl = pl.DataFrame(daily_weights_data)

    # Run our backtest with resample='W'
    result = backtest(adj_prices_pl, daily_weights_pl, resample='W')

    # Compare final returns
    finlab_final = finlab_report.creturn.iloc[-1]
    our_final = result["creturn"].to_list()[-1]

    diff_pct = abs(finlab_final - our_final) / finlab_final * 100
    assert diff_pct < 1.0, f"Weekly resample final return diff: {diff_pct:.4f}%"


def test_resample_monthly(price_data):
    """Test resample='M' matches Finlab monthly rebalancing."""
    adj_close = price_data

    stock = '2330'
    all_dates = adj_close.index
    dates = all_dates[(all_dates >= '2024-01-01') & (all_dates <= '2024-06-30')]

    # Create daily position
    position_pd = pd.DataFrame({stock: [True] * len(dates)}, index=dates)

    # Run Finlab with monthly resample
    finlab_report = finlab_backtest.sim(
        position_pd, resample='M',
        fee_ratio=0.001425, tax_ratio=0.003,
        upload=False
    )

    # Prepare for our backtest
    finlab_dates = finlab_report.creturn.index.tolist()
    date_strs = [str(d.date()) for d in finlab_dates]
    adj_prices_pl = pl.DataFrame({"date": date_strs, stock: adj_close.loc[finlab_dates, stock].tolist()})

    # Create daily position for our backtest
    daily_weights_data = {"date": date_strs, stock: [1.0] * len(date_strs)}
    daily_weights_pl = pl.DataFrame(daily_weights_data)

    # Run our backtest with resample='M'
    result = backtest(adj_prices_pl, daily_weights_pl, resample='M')

    # Compare final returns
    finlab_final = finlab_report.creturn.iloc[-1]
    our_final = result["creturn"].to_list()[-1]

    diff_pct = abs(finlab_final - our_final) / finlab_final * 100
    assert diff_pct < 1.0, f"Monthly resample final return diff: {diff_pct:.4f}%"


def test_resample_invalid():
    """Test that invalid resample raises error."""
    close = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "2330": [100.0, 102.0],
    })
    position = pl.DataFrame({
        "date": ["2024-01-01"],
        "2330": [1.0],
    })

    with pytest.raises(ValueError, match="Invalid resample frequency"):
        backtest(close, position, resample='X')


def test_resample_with_report(price_data):
    """Test that resample works with backtest_with_report."""
    adj_close = price_data

    stock = '2330'
    all_dates = adj_close.index
    dates = all_dates[(all_dates >= '2024-01-01') & (all_dates <= '2024-03-31')]

    date_strs = [str(d.date()) for d in dates]
    adj_prices_pl = pl.DataFrame({"date": date_strs, stock: adj_close.loc[dates, stock].tolist()})

    # Create daily position
    daily_weights_pl = pl.DataFrame({"date": date_strs, stock: [1.0] * len(date_strs)})

    # Run with monthly resample
    report = backtest_with_report(
        close=adj_prices_pl,
        position=daily_weights_pl,
        resample='M',
    )

    # Should have 3 trades (one per month) - actually likely just 1 since we hold through
    # The important thing is it doesn't crash
    assert len(report.creturn) == len(dates)
    assert isinstance(report.trades, pl.DataFrame)


def test_resample_weekly_friday(price_data):
    """Test resample='W-FRI' matches Finlab weekly Friday rebalancing."""
    adj_close = price_data

    stock = '2330'
    all_dates = adj_close.index
    dates = all_dates[(all_dates >= '2024-01-01') & (all_dates <= '2024-03-31')]

    # Create daily position
    position_pd = pd.DataFrame({stock: [True] * len(dates)}, index=dates)

    # Run Finlab with W-FRI resample
    finlab_report = finlab_backtest.sim(
        position_pd, resample='W-FRI',
        fee_ratio=0.001425, tax_ratio=0.003,
        upload=False
    )

    # Prepare for our backtest
    finlab_dates = finlab_report.creturn.index.tolist()
    date_strs = [str(d.date()) for d in finlab_dates]
    adj_prices_pl = pl.DataFrame({"date": date_strs, stock: adj_close.loc[finlab_dates, stock].tolist()})

    # Create daily position for our backtest
    daily_weights_data = {"date": date_strs, stock: [1.0] * len(date_strs)}
    daily_weights_pl = pl.DataFrame(daily_weights_data)

    # Run our backtest with resample='W-FRI'
    result = backtest(adj_prices_pl, daily_weights_pl, resample='W-FRI')

    # Compare final returns
    finlab_final = finlab_report.creturn.iloc[-1]
    our_final = result["creturn"].to_list()[-1]

    diff_pct = abs(finlab_final - our_final) / finlab_final * 100
    assert diff_pct < 1.0, f"W-FRI resample final return diff: {diff_pct:.4f}%"


def test_resample_offset(price_data):
    """Test resample_offset shifts rebalance dates."""
    adj_close = price_data

    stock = '2330'
    all_dates = adj_close.index
    dates = all_dates[(all_dates >= '2024-01-01') & (all_dates <= '2024-03-31')]

    # Create daily position
    position_pd = pd.DataFrame({stock: [True] * len(dates)}, index=dates)

    # Run Finlab with W and offset='1D'
    finlab_report = finlab_backtest.sim(
        position_pd, resample='W', resample_offset='1D',
        fee_ratio=0.001425, tax_ratio=0.003,
        upload=False
    )

    # Prepare for our backtest
    finlab_dates = finlab_report.creturn.index.tolist()
    date_strs = [str(d.date()) for d in finlab_dates]
    adj_prices_pl = pl.DataFrame({"date": date_strs, stock: adj_close.loc[finlab_dates, stock].tolist()})

    # Create daily position for our backtest
    daily_weights_data = {"date": date_strs, stock: [1.0] * len(date_strs)}
    daily_weights_pl = pl.DataFrame(daily_weights_data)

    # Run our backtest with resample='W' and resample_offset='1D'
    result = backtest(adj_prices_pl, daily_weights_pl, resample='W', resample_offset='1D')

    # Compare final returns
    finlab_final = finlab_report.creturn.iloc[-1]
    our_final = result["creturn"].to_list()[-1]

    diff_pct = abs(finlab_final - our_final) / finlab_final * 100
    # Allow 2% tolerance for resample_offset due to trading day boundary differences
    assert diff_pct < 2.0, f"resample_offset final return diff: {diff_pct:.4f}%"


def test_resample_offset_negative():
    """Test resample_offset with negative offset."""
    close = pl.DataFrame({
        "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                 "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"],
        "2330": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
    })
    position = pl.DataFrame({
        "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                 "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"],
        "2330": [1.0] * 9,
    })

    # Run with W and offset='-1D' (shift back one day)
    result = backtest(close, position, resample='W', resample_offset='-1D',
                     fee_ratio=0.0, tax_ratio=0.0)

    # Should not crash and return valid results
    assert len(result) == 9
    assert all(r >= 0 for r in result["creturn"].to_list())
