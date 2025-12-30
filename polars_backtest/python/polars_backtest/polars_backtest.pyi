"""Type stubs for polars DataFrame extension with bt namespace."""

import polars as pl

from polars_backtest.namespace import BacktestNamespace

class DataFrame(pl.DataFrame):
    """Extended DataFrame with bt namespace.

    Usage with cast() to avoid Pylance warnings:
        from typing import cast
        from polars_backtest import DataFrame

        result = cast(DataFrame, df).bt.backtest()
    """

    @property
    def bt(self) -> BacktestNamespace: ...
