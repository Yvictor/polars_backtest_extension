"""
Test comparing polars_backtest with Finlab backtest.sim

Gold standard test - compare creturn with Finlab using real strategy.
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


# Tolerance for creturn comparison
CRETURN_RTOL = 0.02  # 2% relative tolerance


@pytest.fixture(scope="module")
def price_data():
    """Load price data once for all tests."""
    close = finlab_data.get('price:收盤價')
    adj_close = finlab_data.get('etl:adj_close')
    return close, adj_close


class TestMonthlyResample:
    """Test monthly resample comparison with Finlab."""

    def test_rolling_max_strategy(self, price_data):
        """Test rolling max strategy with monthly resample."""
        close, adj_close = price_data

        # Create position: buy when price >= 300-day rolling max
        position = close >= close.rolling(300).max()

        # Run Finlab backtest
        finlab_report = finlab_backtest.sim(position, resample='M', upload=False)

        # Prepare data for polars backtest
        df_adj_close = pl.from_pandas(adj_close.reset_index()).with_columns(
            pl.col("date").cast(pl.Date).cast(pl.Utf8)
        )
        df_pos = pl.from_pandas(position.reset_index()).with_columns(
            pl.col("date").cast(pl.Date).cast(pl.Utf8)
        )

        # Run polars backtest
        polars_report = backtest_with_report(df_adj_close, df_pos, resample='M')

        # Compare creturn
        df_finlab = pl.DataFrame({
            "date": [str(d.date()) for d in finlab_report.creturn.index],
            "creturn_finlab": finlab_report.creturn.values
        })
        df_comparison = df_finlab.join(
            polars_report.creturn,
            on="date",
            how="inner"
        )

        # Calculate relative difference
        finlab_arr = df_comparison["creturn_finlab"].to_numpy()
        polars_arr = df_comparison["creturn"].to_numpy()

        rel_diff = np.abs(finlab_arr - polars_arr) / (np.abs(finlab_arr) + 1e-10)
        max_rel_diff = rel_diff.max()

        print("\n=== Rolling Max Strategy (resample='M') ===")
        print(f"Finlab final creturn: {finlab_arr[-1]:.6f}")
        print(f"Polars final creturn: {polars_arr[-1]:.6f}")
        print(f"Max relative diff: {max_rel_diff:.4%}")
        print(f"Dates compared: {len(df_comparison)}")

        assert max_rel_diff < CRETURN_RTOL, \
            f"creturn diff {max_rel_diff:.4%} exceeds tolerance {CRETURN_RTOL:.2%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
