"""Type stubs for polars DataFrame extension with bt namespace."""

from typing import Union

import polars as pl

from polars_backtest._polars_backtest import BacktestReport
from polars_backtest.namespace import BacktestNamespace

ColumnSpec = Union[str, pl.Expr]


class DataFrame(pl.DataFrame):
    """Extended DataFrame with bt namespace."""

    @property
    def bt(self) -> BacktestNamespace:
        """Backtest namespace for long format DataFrames."""
        ...
