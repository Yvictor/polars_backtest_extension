"""Type stubs for polars_backtest.namespace module."""

from __future__ import annotations

from typing import Union

import polars as pl

from polars_backtest.polars_backtest import BacktestReport

ColumnSpec = Union[str, pl.Expr]


class BacktestNamespace:
    """Backtest namespace for Polars DataFrames."""

    def __init__(self, df: pl.DataFrame) -> None: ...

    def backtest(
        self,
        trade_at_price: ColumnSpec = "close",
        position: ColumnSpec = "weight",
        date: ColumnSpec = "date",
        symbol: ColumnSpec = "symbol",
        open: ColumnSpec = "open",
        high: ColumnSpec = "high",
        low: ColumnSpec = "low",
        factor: str = "factor",
        resample: str | None = "D",
        resample_offset: str | None = None,
        fee_ratio: float = 0.001425,
        tax_ratio: float = 0.003,
        stop_loss: float = 1.0,
        take_profit: float = ...,
        trail_stop: float = ...,
        position_limit: float = 1.0,
        retain_cost_when_rebalance: bool = False,
        stop_trading_next_period: bool = True,
        finlab_mode: bool = False,
        touched_exit: bool = False,
    ) -> pl.DataFrame:
        """Run backtest on long format DataFrame.

        Returns:
            DataFrame with columns: date, creturn
        """
        ...

    def backtest_with_report(
        self,
        trade_at_price: ColumnSpec = "close",
        position: ColumnSpec = "weight",
        date: ColumnSpec = "date",
        symbol: ColumnSpec = "symbol",
        open: ColumnSpec = "open",
        high: ColumnSpec = "high",
        low: ColumnSpec = "low",
        factor: str = "factor",
        benchmark: str | pl.DataFrame | None = None,
        resample: str | None = "D",
        resample_offset: str | None = None,
        fee_ratio: float = 0.001425,
        tax_ratio: float = 0.003,
        stop_loss: float = 1.0,
        take_profit: float = ...,
        trail_stop: float = ...,
        position_limit: float = 1.0,
        retain_cost_when_rebalance: bool = False,
        stop_trading_next_period: bool = True,
        finlab_mode: bool = True,
        touched_exit: bool = False,
        limit_up: str = "limit_up",
        limit_down: str = "limit_down",
        trading_value: str = "trading_value",
    ) -> BacktestReport:
        """Run backtest with trade tracking.

        Returns:
            BacktestReport object with creturn, trades, and statistics methods.
        """
        ...


def backtest(
    df: pl.DataFrame,
    trade_at_price: ColumnSpec = "close",
    position: ColumnSpec = "weight",
    date: ColumnSpec = "date",
    symbol: ColumnSpec = "symbol",
    open: ColumnSpec = "open",
    high: ColumnSpec = "high",
    low: ColumnSpec = "low",
    factor: str = "factor",
    resample: str | None = "D",
    resample_offset: str | None = None,
    fee_ratio: float = 0.001425,
    tax_ratio: float = 0.003,
    stop_loss: float = 1.0,
    take_profit: float = ...,
    trail_stop: float = ...,
    position_limit: float = 1.0,
    retain_cost_when_rebalance: bool = False,
    stop_trading_next_period: bool = True,
    finlab_mode: bool = False,
    touched_exit: bool = False,
) -> pl.DataFrame:
    """Standalone function for backtest."""
    ...


def backtest_with_report(
    df: pl.DataFrame,
    trade_at_price: ColumnSpec = "close",
    position: ColumnSpec = "weight",
    date: ColumnSpec = "date",
    symbol: ColumnSpec = "symbol",
    open: ColumnSpec = "open",
    high: ColumnSpec = "high",
    low: ColumnSpec = "low",
    factor: str = "factor",
    benchmark: str | pl.DataFrame | None = None,
    resample: str | None = "D",
    resample_offset: str | None = None,
    fee_ratio: float = 0.001425,
    tax_ratio: float = 0.003,
    stop_loss: float = 1.0,
    take_profit: float = ...,
    trail_stop: float = ...,
    position_limit: float = 1.0,
    retain_cost_when_rebalance: bool = False,
    stop_trading_next_period: bool = True,
    finlab_mode: bool = True,
    touched_exit: bool = False,
    limit_up: str = "limit_up",
    limit_down: str = "limit_down",
    trading_value: str = "trading_value",
) -> BacktestReport:
    """Standalone function for backtest with report."""
    ...
