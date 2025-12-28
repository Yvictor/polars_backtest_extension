"""
Test long format API (pl_bt.backtest) vs wide format API (backtest_with_report_wide).

Key test: Polars rolling operations return null (not False like pandas NaN),
which must be handled correctly by pl_bt.backtest.
"""

import os
import pytest
import polars as pl

from dotenv import load_dotenv
load_dotenv()

import finlab
finlab.login(os.getenv('FINLAB_API_TOKEN'))

from finlab import data as finlab_data
import polars_backtest as pl_bt
from polars_backtest import backtest_with_report_wide


CRETURN_RTOL = 1e-6


@pytest.fixture(scope="module")
def price_data():
    """Load price data once for all tests."""
    close = finlab_data.get('price:收盤價')
    adj_close = finlab_data.get('etl:adj_close')
    return close, adj_close


@pytest.mark.parametrize("resample", ['M'])
def test_long_format_polars_signals(price_data, resample):
    """Test long format with polars-calculated signals (null handling).

    This replicates the notebook example exactly:
    - Uses polars rolling_max (returns null for first N-1 rows)
    - Polars: null >= null returns null (not False like pandas NaN >= NaN)
    - pl_bt.backtest should handle these nulls correctly
    """
    close, adj_close = price_data

    # Prepare wide format data using pandas signals (baseline)
    # Note: backtest_with_report_wide expects String date for internal joins
    position_pandas = close >= close.rolling(300).max()
    df_adj = pl.from_pandas(adj_close.reset_index()).with_columns(
        pl.col("date").cast(pl.Date).cast(pl.Utf8)
    )
    df_pos = pl.from_pandas(position_pandas.reset_index()).with_columns(
        pl.col("date").cast(pl.Date).cast(pl.Utf8)
    )
    wide_report = backtest_with_report_wide(df_adj, df_pos, resample=resample)

    # Prepare long format data with polars-calculated signals (like notebook)
    # Use Date type directly (no Utf8 conversion) for zero-copy performance
    df_long = (
        pl.from_pandas(close.unstack().reset_index())
        .select(
            pl.col("symbol"),
            pl.col("date").cast(pl.Date),
            pl.col("0").alias("close"),
        )
        .join(
            pl.from_pandas(adj_close.unstack().reset_index())
            .select(
                pl.col("symbol"),
                pl.col("date").cast(pl.Date),
                pl.col("0").alias("adj_close"),
            ),
            on=["symbol", "date"],
            how="left",
        )
        .sort(["symbol", "date"])  # Sort by symbol first for correct rolling calculation
    )

    # Calculate weight using polars (this produces null values, not False)
    df_long = df_long.with_columns(
        (pl.col("close") >= pl.col("close").rolling_max(300).over("symbol")).alias("weight")
    )

    # Sort by date for backtest (enables skip-sort optimization in Rust)
    df_long = df_long.sort("date")

    # Check that we have null values (not False) - this is the root cause
    null_count = df_long.select(pl.col("weight").null_count()).item()
    print(f"\nNull count in polars weight: {null_count}")
    assert null_count > 0, "Expected null values from polars rolling_max comparison"

    # Run long format backtest
    long_result = pl_bt.backtest(
        df_long,
        price="adj_close",
        weight="weight",
        finlab_mode=True,
        resample=resample
    )

    # Compare Long vs Wide (ensure date types match)
    wide_creturn = wide_report.creturn.with_columns(pl.col("date").cast(pl.Date))
    long_creturn = long_result.rename({"creturn": "creturn_long"}).with_columns(pl.col("date").cast(pl.Date))
    df_cmp = wide_creturn.join(
        long_creturn,
        on="date",
        how="inner"
    )

    # Get final creturn values
    wide_final = df_cmp.get_column("creturn")[-1]
    long_final = df_cmp.get_column("creturn_long")[-1]

    print(f"\n=== Polars Signals Test (resample={resample}) ===")
    print(f"Wide format final creturn: {wide_final:.6f}")
    print(f"Long format final creturn: {long_final:.6f}")
    print(f"Difference: {abs(wide_final - long_final):.6f}")

    # This should fail if null handling is broken
    max_diff = df_cmp.select(
        ((pl.col("creturn") - pl.col("creturn_long")).abs().max()).alias("max_diff")
    ).get_column("max_diff")[0]

    assert max_diff < CRETURN_RTOL, (
        f"Long format with polars signals doesn't match Wide format!\n"
        f"Max difference: {max_diff:.6e}\n"
        f"Wide final: {wide_final}, Long final: {long_final}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
