"""
Test long format API vs wide format API with real market data.

Long format: pl_bt.backtest(), backtest_with_trades_long()
Wide format: backtest_with_report_wide() (used as reference)

Uses finlab to load real market data, but compares long vs wide format
(not finlab.sim) for faster testing.
"""

import os
import pytest
import polars as pl
import pandas as pd
import polars_backtest as pl_bt

from dotenv import load_dotenv
from polars_backtest import (
    backtest_with_report_wide,
    backtest_with_trades_long,
    BacktestConfig,
)

load_dotenv()


CRETURN_RTOL = 1e-6

# Cache directory for parquet files
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


# =============================================================================
# Data Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def price_data():
    """Load price data once for all tests."""
    import finlab
    from finlab import data as finlab_data

    finlab.login(os.getenv("FINLAB_API_TOKEN"))
    close = finlab_data.get("price:收盤價")
    adj_close = finlab_data.get("etl:adj_close")
    return close, adj_close


@pytest.fixture(scope="module")
def wide_format_df(price_data):
    """Convert to wide format polars DataFrames with parquet cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    adj_cache = os.path.join(CACHE_DIR, "wide_adj.parquet")
    close_cache = os.path.join(CACHE_DIR, "wide_close.parquet")

    if os.path.exists(adj_cache) and os.path.exists(close_cache):
        df_adj = pl.read_parquet(adj_cache)
        df_close = pl.read_parquet(close_cache)
        return df_adj, df_close

    close, adj_close = price_data

    df_adj = pl.from_pandas(adj_close.reset_index()).with_columns(
        pl.col("date").cast(pl.Date).cast(pl.Utf8)
    )
    df_close = pl.from_pandas(close.reset_index()).with_columns(
        pl.col("date").cast(pl.Date).cast(pl.Utf8)
    )

    df_adj.write_parquet(adj_cache)
    df_close.write_parquet(close_cache)

    return df_adj, df_close


@pytest.fixture(scope="module")
def long_format_df(price_data):
    """Convert to long format polars DataFrame with parquet cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "long_format.parquet")

    if os.path.exists(cache_path):
        return pl.read_parquet(cache_path)

    close, adj_close = price_data

    df_long = (
        pl.from_pandas(close.unstack().reset_index())
        .select(
            pl.col("symbol"),
            pl.col("date").cast(pl.Date),
            pl.col("0").alias("close"),
        )
        .join(
            pl.from_pandas(adj_close.unstack().reset_index()).select(
                pl.col("symbol"),
                pl.col("date").cast(pl.Date),
                pl.col("0").alias("adj_close"),
            ),
            on=["symbol", "date"],
            how="left",
        )
        .sort(["date", "symbol"])
        .rechunk()
    )

    df_long.write_parquet(cache_path)

    return df_long


@pytest.fixture(scope="module")
def position_bool(price_data):
    """Generate boolean position: price >= rolling max."""
    close, adj_close = price_data
    position = close >= close.rolling(300).max()
    return position


@pytest.fixture(scope="module")
def position_short(price_data):
    """Generate short position: price <= rolling min."""
    close, adj_close = price_data
    position = (close <= close.rolling(300).min()) * -1
    return position


# =============================================================================
# Helper Functions
# =============================================================================


def wide_position_to_pl(position: pd.DataFrame) -> pl.DataFrame:
    """Convert pandas position to polars DataFrame."""
    return pl.from_pandas(position.reset_index()).with_columns(
        pl.col("date").cast(pl.Date).cast(pl.Utf8)
    )


def add_weight_to_long(df_long: pl.DataFrame, window: int = 300) -> pl.DataFrame:
    """Add weight column to long format data based on rolling max signal.

    Note: polars rolling_max returns null for first N-1 rows, and comparing
    with null returns null (not False like pandas). We fill nulls with 0.0
    to match pandas behavior.
    """
    return df_long.with_columns(
        (pl.col("close") >= pl.col("close").rolling_max(window).over("symbol"))
        .cast(pl.Float64)
        .fill_null(0.0)
        .alias("weight")
    ).sort("date")


def add_short_weight_to_long(df_long: pl.DataFrame, window: int = 300) -> pl.DataFrame:
    """Add short weight column to long format data.

    Note: polars rolling_min returns null for first N-1 rows. Fill nulls with 0.0.
    """
    return df_long.with_columns(
        (
            (pl.col("close") <= pl.col("close").rolling_min(window).over("symbol"))
            .cast(pl.Float64)
            .fill_null(0.0)
            * -1
        ).alias("weight")
    ).sort("date")


def run_comparison(
    df_long: pl.DataFrame,
    df_adj: pl.DataFrame,
    df_position: pl.DataFrame,
    test_name: str,
    **kwargs,
):
    """Run comparison between long format and wide format backtest."""
    # Run wide format backtest (reference)
    wide_report = backtest_with_report_wide(df_adj, df_position, **kwargs)

    # Parse resample for long format
    resample = kwargs.get("resample", "D")

    # Run long format backtest
    long_result = pl_bt.backtest(
        df_long,
        trade_at_price="adj_close",
        position="weight",
        resample=resample,
        fee_ratio=kwargs.get("fee_ratio", 0.001425),
        tax_ratio=kwargs.get("tax_ratio", 0.003),
        stop_loss=kwargs.get("stop_loss", 1.0),
        take_profit=kwargs.get("take_profit", float("inf")),
        trail_stop=kwargs.get("trail_stop", float("inf")),
        position_limit=kwargs.get("position_limit", 1.0),
        retain_cost_when_rebalance=kwargs.get("retain_cost_when_rebalance", False),
        stop_trading_next_period=kwargs.get("stop_trading_next_period", True),
        finlab_mode=True,
    )

    # Compare
    wide_creturn = wide_report.creturn.with_columns(pl.col("date").cast(pl.Date))
    long_creturn = long_result.rename({"creturn": "creturn_long"}).with_columns(
        pl.col("date").cast(pl.Date)
    )

    df_cmp = wide_creturn.join(long_creturn, on="date", how="inner")

    max_diff = df_cmp.select(
        ((pl.col("creturn") - pl.col("creturn_long")).abs().max()).alias("max_diff")
    ).get_column("max_diff")[0]

    df_ne = df_cmp.filter(pl.col("creturn").round(6) != pl.col("creturn_long").round(6))

    print(f"\n=== {test_name} ===")
    print(f"Wide final: {wide_creturn.get_column('creturn')[-1]:.6f}")
    print(f"Long final: {long_creturn.get_column('creturn_long')[-1]:.6f}")
    print(f"Max diff: {max_diff:.2e}")
    if not df_ne.is_empty():
        print(f"Differences:\n{df_ne.head(5)}")

    assert df_ne.is_empty(), "Found differences in creturn"
    assert max_diff < CRETURN_RTOL, f"Max diff {max_diff} exceeds tolerance"

    return long_result, wide_report


def run_trades_comparison(
    df_long: pl.DataFrame,
    df_adj: pl.DataFrame,
    df_position: pl.DataFrame,
    test_name: str,
    **kwargs,
):
    """Run comparison with trades tracking."""
    # Run wide format backtest (reference)
    wide_report = backtest_with_report_wide(df_adj, df_position, **kwargs)

    # Parse resample for long format (backtest_with_trades_long uses None for daily)
    resample = kwargs.get("resample", "D")
    resample_long = None if resample == "D" else resample

    # Build config
    config = BacktestConfig(
        fee_ratio=kwargs.get("fee_ratio", 0.001425),
        tax_ratio=kwargs.get("tax_ratio", 0.003),
        stop_loss=kwargs.get("stop_loss", 1.0),
        take_profit=kwargs.get("take_profit", float("inf")),
        trail_stop=kwargs.get("trail_stop", float("inf")),
        position_limit=kwargs.get("position_limit", 1.0),
        retain_cost_when_rebalance=kwargs.get("retain_cost_when_rebalance", False),
        stop_trading_next_period=kwargs.get("stop_trading_next_period", True),
        finlab_mode=True,
    )

    # Run long format backtest with trades (uses Rust API directly)
    long_result = backtest_with_trades_long(
        df_long,
        trade_at_price="adj_close",
        position="weight",
        resample=resample_long,
        config=config,
    )

    print(f"\n=== {test_name} (trades) ===")
    print(f"Wide trades: {len(wide_report.trades)}")
    print(f"Long trades: {len(long_result.trades)}")

    # Trade count comparison (allow some tolerance due to date alignment)
    wide_trade_count = len(wide_report.trades)
    long_trade_count = len(long_result.trades)

    if wide_trade_count > 0:
        diff_ratio = abs(wide_trade_count - long_trade_count) / wide_trade_count
        assert diff_ratio < 0.1, (
            f"Trade count differs by >10%: wide={wide_trade_count}, long={long_trade_count}"
        )

    return long_result, wide_report


# =============================================================================
# Resample Tests
# =============================================================================


@pytest.mark.parametrize("resample", ["D", "W", "M", None])
def test_resample(wide_format_df, long_format_df, position_bool, resample):
    """Test different resample frequencies."""
    df_adj, df_close = wide_format_df
    df_long = long_format_df

    df_position = wide_position_to_pl(position_bool)
    df_long_with_weight = add_weight_to_long(df_long)

    run_comparison(
        df_long_with_weight,
        df_adj,
        df_position,
        f"resample={resample}",
        resample=resample,
    )


# =============================================================================
# Fee Tests
# =============================================================================


@pytest.mark.parametrize(
    "fee_ratio,tax_ratio",
    [
        (0, 0),
        (0.001425, 0.003),
        (0.01, 0.005),
    ],
)
def test_fees(wide_format_df, long_format_df, position_bool, fee_ratio, tax_ratio):
    """Test different fee configurations."""
    df_adj, df_close = wide_format_df
    df_long = long_format_df

    df_position = wide_position_to_pl(position_bool)
    df_long_with_weight = add_weight_to_long(df_long)

    run_comparison(
        df_long_with_weight,
        df_adj,
        df_position,
        f"fee={fee_ratio},tax={tax_ratio}",
        resample="M",
        fee_ratio=fee_ratio,
        tax_ratio=tax_ratio,
    )


# =============================================================================
# Position Limit Tests
# =============================================================================


@pytest.mark.parametrize("position_limit", [0.2, 0.5, 1.0])
def test_position_limit(wide_format_df, long_format_df, position_bool, position_limit):
    """Test position limit parameter."""
    df_adj, df_close = wide_format_df
    df_long = long_format_df

    df_position = wide_position_to_pl(position_bool)
    df_long_with_weight = add_weight_to_long(df_long)

    run_comparison(
        df_long_with_weight,
        df_adj,
        df_position,
        f"position_limit={position_limit}",
        resample="M",
        position_limit=position_limit,
    )


# =============================================================================
# Stop Loss Tests
# =============================================================================


@pytest.mark.parametrize("stop_loss", [0.05, 0.1])
def test_stop_loss(wide_format_df, long_format_df, position_bool, stop_loss):
    """Test stop loss parameter."""
    df_adj, df_close = wide_format_df
    df_long = long_format_df

    df_position = wide_position_to_pl(position_bool)
    df_long_with_weight = add_weight_to_long(df_long)

    run_comparison(
        df_long_with_weight,
        df_adj,
        df_position,
        f"stop_loss={stop_loss}",
        resample="M",
        stop_loss=stop_loss,
    )


# =============================================================================
# Take Profit Tests
# =============================================================================


@pytest.mark.parametrize("take_profit", [0.1, 0.2])
def test_take_profit(wide_format_df, long_format_df, position_bool, take_profit):
    """Test take profit parameter."""
    df_adj, df_close = wide_format_df
    df_long = long_format_df

    df_position = wide_position_to_pl(position_bool)
    df_long_with_weight = add_weight_to_long(df_long)

    run_comparison(
        df_long_with_weight,
        df_adj,
        df_position,
        f"take_profit={take_profit}",
        resample="M",
        take_profit=take_profit,
    )


# =============================================================================
# Trail Stop Tests
# =============================================================================


@pytest.mark.parametrize("trail_stop", [0.1, 0.15])
def test_trail_stop(wide_format_df, long_format_df, position_bool, trail_stop):
    """Test trailing stop parameter."""
    df_adj, df_close = wide_format_df
    df_long = long_format_df

    df_position = wide_position_to_pl(position_bool)
    df_long_with_weight = add_weight_to_long(df_long)

    run_comparison(
        df_long_with_weight,
        df_adj,
        df_position,
        f"trail_stop={trail_stop}",
        resample="M",
        trail_stop=trail_stop,
    )


# =============================================================================
# Rebalance Behavior Tests
# =============================================================================


def test_retain_cost_when_rebalance(wide_format_df, long_format_df, position_bool):
    """Test retain_cost_when_rebalance=True."""
    df_adj, df_close = wide_format_df
    df_long = long_format_df

    df_position = wide_position_to_pl(position_bool)
    df_long_with_weight = add_weight_to_long(df_long)

    run_comparison(
        df_long_with_weight,
        df_adj,
        df_position,
        "retain_cost=True",
        resample="M",
        stop_loss=0.1,
        retain_cost_when_rebalance=True,
    )


def test_stop_trading_next_period_false(wide_format_df, long_format_df, position_bool):
    """Test stop_trading_next_period=False."""
    df_adj, df_close = wide_format_df
    df_long = long_format_df

    df_position = wide_position_to_pl(position_bool)
    df_long_with_weight = add_weight_to_long(df_long)

    run_comparison(
        df_long_with_weight,
        df_adj,
        df_position,
        "stop_trading_next_period=False",
        resample="M",
        stop_loss=0.1,
        stop_trading_next_period=False,
    )


# =============================================================================
# Short Position Tests
# =============================================================================


def test_short_basic(wide_format_df, long_format_df, position_short):
    """Test basic short positions."""
    df_adj, df_close = wide_format_df
    df_long = long_format_df

    df_position = wide_position_to_pl(position_short)
    df_long_with_weight = add_short_weight_to_long(df_long)

    run_comparison(
        df_long_with_weight,
        df_adj,
        df_position,
        "short_basic",
        resample="M",
    )


@pytest.mark.parametrize("stop_loss", [0.05, 0.1])
def test_short_stop_loss(wide_format_df, long_format_df, position_short, stop_loss):
    """Test short positions with stop_loss."""
    df_adj, df_close = wide_format_df
    df_long = long_format_df

    df_position = wide_position_to_pl(position_short)
    df_long_with_weight = add_short_weight_to_long(df_long)

    run_comparison(
        df_long_with_weight,
        df_adj,
        df_position,
        f"short+stop_loss={stop_loss}",
        resample="M",
        stop_loss=stop_loss,
    )


# =============================================================================
# Trades Tests
# =============================================================================


def test_trades_match(wide_format_df, long_format_df, position_bool):
    """Test trades count matches between long and wide format."""
    df_adj, df_close = wide_format_df
    df_long = long_format_df

    df_position = wide_position_to_pl(position_bool)
    df_long_with_weight = add_weight_to_long(df_long)

    run_trades_comparison(
        df_long_with_weight,
        df_adj,
        df_position,
        "trades_match",
        resample="M",
    )


def test_trades_with_stops(wide_format_df, long_format_df, position_bool):
    """Test trades with stop_loss and take_profit."""
    df_adj, df_close = wide_format_df
    df_long = long_format_df

    df_position = wide_position_to_pl(position_bool)
    df_long_with_weight = add_weight_to_long(df_long)

    run_trades_comparison(
        df_long_with_weight,
        df_adj,
        df_position,
        "trades_with_stops",
        resample="M",
        stop_loss=0.1,
        take_profit=0.2,
    )


# =============================================================================
# Null Handling Tests
# =============================================================================


def test_polars_rolling_null(long_format_df):
    """Test that polars rolling operations with nulls work correctly.

    Polars rolling_max returns null for first N-1 rows, and comparing with null
    returns null (not False like pandas). pl_bt.backtest should handle this.
    """
    df_long = long_format_df
    window = 300

    # Add weight with polars rolling WITHOUT filling nulls
    # This produces nulls for first N-1 rows per symbol
    df_with_null_weight = df_long.with_columns(
        (pl.col("close") >= pl.col("close").rolling_max(window).over("symbol"))
        .cast(pl.Float64)
        .alias("weight")
    ).sort("date")

    # Check that nulls exist
    null_count = df_with_null_weight.select(pl.col("weight").null_count()).item()
    print(f"\nNull count in polars weight (before fill): {null_count}")
    assert null_count > 0, "Expected null values from polars rolling_max"

    # Backtest should handle null weights by treating them as 0
    # (Our add_weight_to_long fills nulls with 0.0 for this reason)
    df_with_weight = df_with_null_weight.with_columns(
        pl.col("weight").fill_null(0.0)
    )

    # Should complete without error
    result = pl_bt.backtest(
        df_with_null_weight,
        trade_at_price="adj_close",
        position="weight",
        resample="M",
        finlab_mode=True,
    )

    assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
