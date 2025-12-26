"""
Load stock data from S3 and convert between long/wide formats.

Long format: Each row is (date, symbol, open, high, low, close, volume)
Wide format (finlab): Dates as rows, symbols as columns for each price field
"""

import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv


def get_s3_storage_options() -> dict:
    """Get S3 connection settings from environment variables."""
    project_root = Path(__file__).parent.parent.parent
    load_dotenv(project_root / ".env")

    return {
        "access_key_id": os.getenv("S3_ACCESS_KEY_ID"),
        "secret_access_key": os.getenv("S3_SECRET_ACCESS_KEY"),
        "endpoint_url": os.getenv("S3_ENDPOINT_URL"),
    }


def load_ohlcv(
    storage_options: dict | None = None,
    daily_basic_path: str = "s3://trading-data/stock/TwStkPriceDaily/*.parquet",
) -> pl.LazyFrame:
    """
    Load raw OHLCV data.

    Returns:
        LazyFrame with columns: date, symbol, market_id, open, high, low, close, volume
    """
    if storage_options is None:
        storage_options = get_s3_storage_options()

    lf = pl.scan_parquet(daily_basic_path, storage_options=storage_options)

    return (
        lf.select(
            pl.col("ymdOn").cast(pl.Date).alias("date"),
            pl.col("listCode").str.strip_chars(" ").alias("symbol"),
            pl.col("mtMarketC_id").alias("market_id"),
            pl.col("openPrice").alias("open"),
            pl.col("highPrice").alias("high"),
            pl.col("lowPrice").alias("low"),
            pl.col("closePrice").alias("close"),
            (pl.col("txnVolume") * 1000).alias("volume"),
        )
        .filter(pl.col("symbol").str.len_chars() == 4)
        .unique(subset=["symbol", "date"])
        .sort("symbol", "date")
    )


def load_ref_price(
    storage_options: dict | None = None,
    refp_path: str = "s3://trading-data/stock/TwStkRefPrice/*.parquet",
) -> pl.LazyFrame:
    """
    Load reference price data.

    Returns:
        LazyFrame with columns: date, symbol, ref_price
    """
    if storage_options is None:
        storage_options = get_s3_storage_options()

    lf = pl.scan_parquet(refp_path, storage_options=storage_options)

    return (
        lf.select(
            pl.col("ymdOn").cast(pl.Date).alias("date"),
            pl.col("listCode").str.strip_chars(" ").alias("symbol"),
            pl.col("refPrice").alias("ref_price"),
        )
        .filter(pl.col("symbol").str.len_chars() == 4)
        .unique(subset=["symbol", "date"])
        .sort("symbol", "date")
    )


def load_adjusted_ohlcv(
    storage_options: dict | None = None,
) -> pl.LazyFrame:
    """
    Load OHLCV data with adjusted prices (for splits/dividends).

    Returns:
        LazyFrame with columns: date, symbol, market_id, open, high, low, close, volume,
                                ref_price, return, close_raw, factor
    """
    if storage_options is None:
        storage_options = get_s3_storage_options()

    ohlcv = load_ohlcv(storage_options)
    ref_price = load_ref_price(storage_options)

    return (
        ohlcv.join(ref_price, on=["date", "symbol"], how="left", maintain_order="left")
        .with_columns(
            (pl.col("close") / pl.col("ref_price") - 1).alias("return"),
            pl.col("close").alias("close_raw"),
        )
        .with_columns(
            (pl.col("return") + 1).cum_prod().over("symbol").alias("close")
        )
        .with_columns((pl.col("close") / pl.col("close_raw")).alias("factor"))
        .with_columns(
            (pl.col("open") * pl.col("factor")).alias("open"),
            (pl.col("high") * pl.col("factor")).alias("high"),
            (pl.col("low") * pl.col("factor")).alias("low"),
        )
    )


def to_wide(
    df: pl.DataFrame | pl.LazyFrame,
    value_column: str,
    index_column: str = "date",
    pivot_column: str = "symbol",
) -> pl.DataFrame:
    """
    Convert to wide format (finlab style).

    Args:
        df: DataFrame in long format
        value_column: Column to pivot (e.g., "close", "open", "high", "low", "volume")
        index_column: Column to use as row index (default: "date")
        pivot_column: Column to pivot into columns (default: "symbol")

    Returns:
        DataFrame with dates as rows, symbols as columns
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    return (
        df.select([index_column, pivot_column, value_column])
        .pivot(on=pivot_column, index=index_column, values=value_column)
        .sort(index_column)
    )


def load_ohlcv_wide(
    adjusted: bool = True,
    storage_options: dict | None = None,
) -> dict[str, pl.DataFrame]:
    """
    Load OHLCV data in wide format (finlab style).

    Args:
        adjusted: If True, use adjusted prices for splits/dividends
        storage_options: S3 connection settings

    Returns:
        Dict with keys: "open", "high", "low", "close", "volume"
        Each value is a DataFrame with date as rows, symbols as columns
    """
    if storage_options is None:
        storage_options = get_s3_storage_options()

    if adjusted:
        df = load_adjusted_ohlcv(storage_options).collect()
    else:
        df = load_ohlcv(storage_options).collect()

    return {
        "open": to_wide(df, "open"),
        "high": to_wide(df, "high"),
        "low": to_wide(df, "low"),
        "close": to_wide(df, "close"),
        "volume": to_wide(df, "volume"),
    }


def to_long(
    dfs: dict[str, pl.DataFrame],
    index_column: str = "date",
) -> pl.DataFrame:
    """
    Convert wide format (finlab style) back to long format.

    Args:
        dfs: Dict with keys like "open", "high", "low", "close", "volume"
             Each value is a wide DataFrame (date as rows, symbols as columns)
        index_column: Column name for the date index

    Returns:
        DataFrame with columns: date, symbol, open, high, low, close, volume
    """
    result = None

    for field_name, wide_df in dfs.items():
        symbol_cols = [c for c in wide_df.columns if c != index_column]
        long_df = wide_df.unpivot(
            index=index_column,
            on=symbol_cols,
            variable_name="symbol",
            value_name=field_name,
        )

        if result is None:
            result = long_df
        else:
            result = result.join(
                long_df, on=[index_column, "symbol"], how="outer_coalesce"
            )

    return result.sort(index_column, "symbol")


if __name__ == "__main__":
    print("Loading OHLCV data...")
    storage_options = get_s3_storage_options()

    ohlcv = load_ohlcv(storage_options).collect()
    print(f"Long format shape: {ohlcv.shape}")
    print(ohlcv.head())

    print("\nConverting to wide format...")
    close_wide = to_wide(ohlcv, "close")
    print(f"Wide format shape: {close_wide.shape}")
    print(close_wide.head())

    print("\nLoading all OHLCV in wide format (adjusted)...")
    ohlcv_wide = load_ohlcv_wide(adjusted=True, storage_options=storage_options)
    for name, df in ohlcv_wide.items():
        print(f"  {name}: {df.shape}")
