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
    backtest_with_report_long,
    BacktestConfig,
)
from polars.testing import assert_frame_equal

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


def long_trades_to_df(trades_list) -> pl.DataFrame:
    """Convert list of LongTradeRecord to DataFrame.

    Note: entry_date/exit_date are i32 (days since epoch), convert to Date then string
    to match wide format.
    """
    from datetime import date, timedelta

    def days_to_date_str(days: int | None) -> str | None:
        if days is None:
            return None
        # Convert days since epoch to date string
        epoch = date(1970, 1, 1)
        d = epoch + timedelta(days=days)
        return d.isoformat()

    if not trades_list:
        return pl.DataFrame({
            "stock_id": [],
            "entry_date": [],
            "exit_date": [],
            "entry_sig_date": [],
            "exit_sig_date": [],
            "position": [],
            "period": [],
            "return": [],
            "entry_price": [],
            "exit_price": [],
        })

    records = []
    for t in trades_list:
        records.append({
            "stock_id": t.symbol,
            "entry_date": days_to_date_str(t.entry_date),
            "exit_date": days_to_date_str(t.exit_date),
            "entry_sig_date": days_to_date_str(t.entry_sig_date),
            "exit_sig_date": days_to_date_str(t.exit_sig_date),
            "position": t.position_weight,
            "period": t.holding_days(),
            "return": t.trade_return,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
        })
    return pl.DataFrame(records)


def run_trades_comparison(
    df_long: pl.DataFrame,
    df_adj: pl.DataFrame,
    df_position: pl.DataFrame,
    test_name: str,
    **kwargs,
):
    """Run comparison with trades tracking and verify all trade contents match."""
    # Run wide format backtest (reference)
    wide_report = backtest_with_report_wide(df_adj, df_position, **kwargs)

    # Parse resample for long format (backtest_with_report_long uses None for daily)
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

    # Run long format backtest with report (trades as DataFrame from Rust)
    long_report = backtest_with_report_long(
        df_long,
        trade_at_price="adj_close",
        position="weight",
        resample=resample_long,
        config=config,
    )

    print(f"\n=== {test_name} (trades) ===")
    print(f"Wide trades: {len(wide_report.trades)}")
    print(f"Long trades: {len(long_report.trades)}")

    # Get trades as DataFrames (long format already returns DataFrame from Rust)
    long_trades_df = long_report.trades
    wide_trades_df = wide_report.trades

    # Filter out pending trades (null entry_date) and open trades (null exit_date)
    # Wide format tracks pending trades (signaled but never executed due to NaN price),
    # while Long format only tracks executed trades. This is a known difference.
    #
    # For comparison, we only compare COMPLETED trades (both entry_date and exit_date not null)
    wide_completed = wide_trades_df.filter(
        pl.col("entry_date").is_not_null() & pl.col("exit_date").is_not_null()
    )
    long_completed = long_trades_df.filter(
        pl.col("entry_date").is_not_null() & pl.col("exit_date").is_not_null()
    )

    wide_completed_count = len(wide_completed)
    long_completed_count = len(long_completed)

    print(f"Wide completed trades: {wide_completed_count}")
    print(f"Long completed trades: {long_completed_count}")

    assert wide_completed_count == long_completed_count, (
        f"Completed trade count mismatch: wide={wide_completed_count}, long={long_completed_count}"
    )

    if wide_completed_count == 0:
        return long_report, wide_report

    # Normalize both DataFrames to same schema for assert_frame_equal comparison
    # Wide format: dates as strings (Utf8), columns like "trade_price@entry_date"
    # Long format: dates as Date type, columns like "entry_price"

    # Normalize wide format: cast dates to Date type, rename columns
    wide_normalized = wide_completed.select([
        pl.col("stock_id"),
        pl.col("entry_date").str.to_date("%Y-%m-%d").alias("entry_date"),
        pl.col("exit_date").str.to_date("%Y-%m-%d").alias("exit_date"),
        pl.col("entry_sig_date").str.to_date("%Y-%m-%d").alias("entry_sig_date"),
        pl.col("exit_sig_date").str.to_date("%Y-%m-%d").alias("exit_sig_date"),
        pl.col("position"),
        pl.col("trade_price@entry_date").alias("entry_price"),
        pl.col("trade_price@exit_date").alias("exit_price"),
        pl.col("return"),
    ])

    # Normalize long format: select same columns in same order
    long_normalized = long_completed.select([
        pl.col("stock_id"),
        pl.col("entry_date"),
        pl.col("exit_date"),
        pl.col("entry_sig_date"),
        pl.col("exit_sig_date"),
        pl.col("position"),
        pl.col("entry_price"),
        pl.col("exit_price"),
        pl.col("return"),
    ])

    # Sort both DataFrames by (stock_id, entry_date, exit_date) for comparison
    sort_cols = ["stock_id", "entry_date", "exit_date"]
    wide_sorted = wide_normalized.sort(sort_cols)
    long_sorted = long_normalized.sort(sort_cols)

    # Use assert_frame_equal for comparison
    # Note: Wide format uses raw trade_prices (can be NaN for missing data),
    # Long format uses filtered prices with fallback. This causes exit_price
    # differences when wide has NaN. We handle this by comparing non-NaN rows.

    # First compare columns that should match exactly
    exact_cols = ["stock_id", "entry_date", "exit_date", "entry_sig_date", "exit_sig_date"]
    assert_frame_equal(
        wide_sorted.select(exact_cols),
        long_sorted.select(exact_cols),
        check_exact=True,
    )

    # Compare float columns with tolerance
    float_cols = ["position", "entry_price"]
    assert_frame_equal(
        wide_sorted.select(float_cols),
        long_sorted.select(float_cols),
        check_exact=False,
        abs_tol=1e-6,
    )

    # For exit_price and return, we need special handling due to NaN fallback difference
    # Wide format may have NaN exit_price when price data is missing
    # Long format uses fallback price instead
    # Filter to rows where both have non-NaN exit_price for strict comparison
    import math

    wide_exit_prices = wide_sorted["exit_price"].to_list()
    long_exit_prices = long_sorted["exit_price"].to_list()

    # Find rows with NaN in wide (known difference)
    nan_fallback_rows = []
    for i, (w, l) in enumerate(zip(wide_exit_prices, long_exit_prices)):
        if w is not None and math.isnan(w) and l is not None and not math.isnan(l):
            nan_fallback_rows.append(i)

    if nan_fallback_rows:
        print(f"Note: {len(nan_fallback_rows)} rows have NaN exit_price in wide but valid in long (expected)")

    # For non-NaN rows, verify exit_price and return match
    wide_returns = wide_sorted["return"].to_list()
    long_returns = long_sorted["return"].to_list()

    mismatches = 0
    for i, (w_ep, l_ep, w_ret, l_ret) in enumerate(
        zip(wide_exit_prices, long_exit_prices, wide_returns, long_returns)
    ):
        # Skip rows with NaN fallback difference
        if i in nan_fallback_rows:
            continue
        # Check exit_price
        if w_ep is None or l_ep is None:
            continue
        if not (math.isnan(w_ep) and math.isnan(l_ep)):
            if abs(w_ep - l_ep) > 1e-4:
                mismatches += 1
                if mismatches <= 3:
                    print(f"exit_price mismatch at {i}: wide={w_ep}, long={l_ep}")
        # Check return
        if w_ret is None or l_ret is None:
            continue
        if not (math.isnan(w_ret) and math.isnan(l_ret)):
            if abs(w_ret - l_ret) > 1e-6:
                mismatches += 1
                if mismatches <= 3:
                    print(f"return mismatch at {i}: wide={w_ret}, long={l_ret}")

    assert mismatches == 0, f"Found {mismatches} exit_price/return mismatches"

    print(f"  All {wide_completed_count} completed trades match exactly!")

    return long_report, wide_report


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
