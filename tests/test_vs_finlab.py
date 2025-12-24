"""
Test comparing polars_backtest with Finlab backtest.sim

Gold standard test - compare creturn AND trades with Finlab.

NOTE on creturn differences with non-daily rebalance:
Finlab's balance calculation uses: pos * actual_close / adj_close
Our calculation uses only adjusted close (total return including dividends).
This causes a small difference (~0.5% over 17 years) for weekly/monthly rebalance.
The difference is the portfolio-weighted actual/adj price ratio, which reflects
historical dividend adjustments. Daily rebalance is unaffected since positions
reset daily.
"""

import os
import pytest
import numpy as np
import pandas as pd
import polars as pl

from dotenv import load_dotenv
load_dotenv()

import finlab
finlab.login(os.getenv('FINLAB_API_TOKEN'))

from finlab import backtest as finlab_backtest
from finlab import data as finlab_data
from polars_backtest import backtest_with_report


CRETURN_RTOL = 1e-6


@pytest.fixture(scope="module")
def price_data():
    """Load price data once for all tests."""
    close = finlab_data.get('price:收盤價')
    adj_close = finlab_data.get('etl:adj_close')
    return close, adj_close


def run_comparison(adj_close, position, test_name, **kwargs):
    """Run Finlab vs Polars comparison."""
    finlab_report = finlab_backtest.sim(position, upload=False, **kwargs)

    df_adj = pl.from_pandas(adj_close.reset_index()).with_columns(
        pl.col("date").cast(pl.Date).cast(pl.Utf8)
    )
    df_pos = pl.from_pandas(position.reset_index()).with_columns(
        pl.col("date").cast(pl.Date).cast(pl.Utf8)
    )
    polars_report = backtest_with_report(df_adj, df_pos, **kwargs)

    df_finlab = pl.DataFrame({
        "date": [str(d.date()) for d in finlab_report.creturn.index],
        "creturn_finlab": finlab_report.creturn.values
    })
    df_cmp = df_finlab.join(polars_report.creturn, on="date", how="inner")

    df_ne = df_cmp.filter(
        pl.col("creturn_finlab").round(6) != pl.col("creturn").round(6)
    )
    max_diff = df_cmp.select(
        ((pl.col("creturn_finlab") - pl.col("creturn")).abs().max()).alias("max_diff")
    ).get_column("max_diff")[0]


    print(f"\n=== {test_name} ===")
    print(f"{df_ne}")
    print(f"MaxDiff: {max_diff:.2e}")
    assert df_ne.is_empty()
    assert max_diff < CRETURN_RTOL
    return finlab_report, polars_report


# Resample
@pytest.mark.parametrize("resample", ['D', 'W', 'M'])
def test_resample(price_data, resample):
    close, adj_close = price_data
    position = close >= close.rolling(300).max()
    run_comparison(adj_close, position, f"resample={resample}", resample=resample)


# Fees
@pytest.mark.parametrize("fee_ratio,tax_ratio", [(0, 0), (0.001425, 0.003), (0.01, 0.005)])
def test_fees(price_data, fee_ratio, tax_ratio):
    close, adj_close = price_data
    position = close >= close.rolling(300).max()
    run_comparison(adj_close, position, f"fee={fee_ratio},tax={tax_ratio}",
                   resample='M', fee_ratio=fee_ratio, tax_ratio=tax_ratio)


# Position Limit
@pytest.mark.parametrize("position_limit", [0.2, 0.5, 1.0])
def test_position_limit(price_data, position_limit):
    close, adj_close = price_data
    position = close >= close.rolling(300).max()
    run_comparison(adj_close, position, f"position_limit={position_limit}",
                   resample='M', position_limit=position_limit)


# Stop Loss
@pytest.mark.parametrize("stop_loss", [0.05, 0.1])
def test_stop_loss(price_data, stop_loss):
    close, adj_close = price_data
    position = close >= close.rolling(300).max()
    run_comparison(adj_close, position, f"stop_loss={stop_loss}",
                   resample='M', stop_loss=stop_loss)


# Take Profit
@pytest.mark.parametrize("take_profit", [0.1, 0.2])
def test_take_profit(price_data, take_profit):
    close, adj_close = price_data
    position = close >= close.rolling(300).max()
    run_comparison(adj_close, position, f"take_profit={take_profit}",
                   resample='M', take_profit=take_profit)


# Trail Stop
@pytest.mark.parametrize("trail_stop", [0.1, 0.15])
def test_trail_stop(price_data, trail_stop):
    close, adj_close = price_data
    position = close >= close.rolling(300).max()
    run_comparison(adj_close, position, f"trail_stop={trail_stop}",
                   resample='M', trail_stop=trail_stop)


# Rebalance Behavior
def test_retain_cost_when_rebalance(price_data):
    close, adj_close = price_data
    position = close >= close.rolling(300).max()
    run_comparison(adj_close, position, "retain_cost=True",
                   resample='M', stop_loss=0.1, retain_cost_when_rebalance=True)


def test_stop_trading_next_period_false(price_data):
    close, adj_close = price_data
    position = close >= close.rolling(300).max()
    run_comparison(adj_close, position, "stop_trading_next_period=False",
                   resample='M', stop_loss=0.1, stop_trading_next_period=False)


# Trades Comparison
def test_trades_match(price_data):
    close, adj_close = price_data
    position = close >= close.rolling(300).max()
    finlab_report, polars_report = run_comparison(adj_close, position, "trades_match", resample='M')

    print(f"\nTrades: Finlab={len(finlab_report.trades)}, Polars={len(polars_report.trades)}")
    assert len(finlab_report.trades) == len(polars_report.trades)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
