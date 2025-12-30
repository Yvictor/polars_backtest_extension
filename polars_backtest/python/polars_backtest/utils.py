"""Utility functions for polars_backtest."""

from __future__ import annotations

import polars as pl


def long_to_wide(
    df: pl.DataFrame,
    value_col: str,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pl.DataFrame:
    """Convert long format to wide format.

    Args:
        df: Long format DataFrame
        value_col: Column to pivot as values
        date_col: Date column name
        symbol_col: Symbol column name

    Returns:
        Wide format DataFrame with date as first column, symbols as other columns
    """
    return (
        df.select([date_col, symbol_col, value_col])
        .pivot(on=symbol_col, index=date_col, values=value_col)
        .sort(date_col)
    )
