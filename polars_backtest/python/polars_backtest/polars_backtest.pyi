"""Type stubs for polars DataFrame extension with bt namespace."""

import polars as pl

from polars_backtest.namespace import BacktestNamespace

class DataFrame(pl.DataFrame):
    """Extended DataFrame with bt namespace."""

    @property
    def bt(self) -> BacktestNamespace:
        """Backtest namespace for long format DataFrames."""
        ...
