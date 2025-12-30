"""Polars extension for portfolio backtesting.

This package provides high-performance backtesting functionality
for Polars DataFrames using Rust-based computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

# Register DataFrame namespace (df.bt) and import main API functions
# Load the Rust extension - core types
from polars_backtest._polars_backtest import (
    BacktestConfig,
    BacktestReport,
    __version__,
)
from polars_backtest.namespace import (
    BacktestNamespace,
    backtest,
    backtest_with_report,
)

# Wide format API (optional, for Finlab compatibility)
from polars_backtest.wide import (
    Report,
    backtest_wide,
    backtest_with_report_wide,
)

__all__ = [
    "__version__",
    # Core types
    "BacktestConfig",
    "BacktestReport",
    "BacktestNamespace",
    # Main API (long format)
    "backtest",
    "backtest_with_report",
    # Wide format API
    "backtest_wide",
    "backtest_with_report_wide",
    "Report",
    # Statistics expressions
    "daily_returns",
    "cumulative_returns",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "drawdown_series",
    "portfolio_return",
    "equal_weights",
]

# Get the path to the shared library
_LIB_PATH = Path(__file__).parent


def _get_lib_path() -> str:
    """Get the path to the shared library for plugin registration."""
    return str(_LIB_PATH)


def daily_returns(expr: IntoExpr) -> pl.Expr:
    """Calculate daily returns from a price series.

    Args:
        expr: Price series expression

    Returns:
        Polars expression with daily returns (first value is null)

    Example:
        >>> df.with_columns(pl_daily_returns=daily_returns("close"))
    """
    return pl.col(expr).register_plugin(
        lib=_get_lib_path(),
        symbol="pl_daily_returns",
        is_elementwise=False,
    )


def cumulative_returns(expr: IntoExpr) -> pl.Expr:
    """Calculate cumulative returns from daily returns.

    Args:
        expr: Daily returns series expression

    Returns:
        Polars expression with cumulative returns starting at 1.0

    Example:
        >>> df.with_columns(creturn=cumulative_returns("daily_return"))
    """
    return pl.col(expr).register_plugin(
        lib=_get_lib_path(),
        symbol="pl_cumulative_returns",
        is_elementwise=False,
    )


def sharpe_ratio(expr: IntoExpr) -> pl.Expr:
    """Calculate annualized Sharpe ratio from returns.

    Uses rf=0 and 252 trading days for annualization.

    Args:
        expr: Returns series expression

    Returns:
        Polars expression with Sharpe ratio (scalar)

    Example:
        >>> df.select(sharpe=sharpe_ratio("daily_return"))
    """
    return pl.col(expr).register_plugin(
        lib=_get_lib_path(),
        symbol="pl_sharpe_ratio",
        is_elementwise=False,
    )


def sortino_ratio(expr: IntoExpr) -> pl.Expr:
    """Calculate annualized Sortino ratio from returns.

    Uses rf=0 and 252 trading days for annualization.
    Only considers downside risk.

    Args:
        expr: Returns series expression

    Returns:
        Polars expression with Sortino ratio (scalar)

    Example:
        >>> df.select(sortino=sortino_ratio("daily_return"))
    """
    return pl.col(expr).register_plugin(
        lib=_get_lib_path(),
        symbol="pl_sortino_ratio",
        is_elementwise=False,
    )


def max_drawdown(expr: IntoExpr) -> pl.Expr:
    """Calculate maximum drawdown from cumulative returns.

    Args:
        expr: Cumulative returns series expression

    Returns:
        Polars expression with max drawdown (negative value, scalar)

    Example:
        >>> df.select(mdd=max_drawdown("creturn"))
    """
    return pl.col(expr).register_plugin(
        lib=_get_lib_path(),
        symbol="pl_max_drawdown",
        is_elementwise=False,
    )


def drawdown_series(expr: IntoExpr) -> pl.Expr:
    """Calculate drawdown at each point from cumulative returns.

    Args:
        expr: Cumulative returns series expression

    Returns:
        Polars expression with drawdown series (negative values)

    Example:
        >>> df.with_columns(dd=drawdown_series("creturn"))
    """
    return pl.col(expr).register_plugin(
        lib=_get_lib_path(),
        symbol="pl_drawdown_series",
        is_elementwise=False,
    )


def portfolio_return(weights: IntoExpr, returns: IntoExpr) -> pl.Expr:
    """Calculate weighted portfolio return for a single period.

    Args:
        weights: Portfolio weights expression
        returns: Asset returns expression

    Returns:
        Polars expression with portfolio return (scalar)

    Example:
        >>> df.select(port_ret=portfolio_return("weights", "returns"))
    """
    return pl.col(weights).register_plugin(
        lib=_get_lib_path(),
        symbol="pl_portfolio_return",
        is_elementwise=False,
        args=[pl.col(returns)],
    )


def equal_weights(expr: IntoExpr) -> pl.Expr:
    """Calculate equal weights from boolean signals.

    Args:
        expr: Boolean signals expression (True = hold)

    Returns:
        Polars expression with equal weights (sum to 1.0 for True values)

    Example:
        >>> df.with_columns(weight=equal_weights("signal"))
    """
    return pl.col(expr).register_plugin(
        lib=_get_lib_path(),
        symbol="pl_equal_weights",
        is_elementwise=False,
    )
