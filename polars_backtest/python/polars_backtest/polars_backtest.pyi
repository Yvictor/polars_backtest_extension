"""Type stubs for polars_backtest extension."""

from __future__ import annotations

from typing import Literal

import polars as pl

from polars_backtest.namespace import BacktestNamespace

MetricsSection = Literal["backtest", "profitability", "risk", "ratio", "winrate", "liquidity"]


class DataFrame(pl.DataFrame):
    """Extended DataFrame with bt namespace.

    Usage with cast() to avoid Pylance warnings:
        from typing import cast
        from polars_backtest import DataFrame

        result = cast(DataFrame, df).bt.backtest()
    """

    @property
    def bt(self) -> BacktestNamespace: ...


class BacktestConfig:
    """Backtest configuration parameters."""

    fee_ratio: float
    tax_ratio: float
    stop_loss: float
    take_profit: float
    trail_stop: float

    def __init__(
        self,
        fee_ratio: float = 0.001425,
        tax_ratio: float = 0.003,
        stop_loss: float = 1.0,
        take_profit: float = ...,
        trail_stop: float = ...,
    ) -> None: ...


class BacktestReport:
    """Backtest report from Rust with trades and statistics.

    This class is created by backtest_with_report() and provides access to
    cumulative returns, trades, and various statistics methods.
    """

    # Properties (getters)
    @property
    def creturn(self) -> pl.DataFrame:
        """Cumulative returns DataFrame with date and creturn columns."""
        ...

    @property
    def trades(self) -> pl.DataFrame:
        """Trade records DataFrame with MAE/MFE metrics."""
        ...

    @property
    def fee_ratio(self) -> float:
        """Transaction fee ratio used in backtest."""
        ...

    @property
    def tax_ratio(self) -> float:
        """Transaction tax ratio used in backtest."""
        ...

    @property
    def stop_loss(self) -> float | None:
        """Stop loss threshold if configured, None otherwise."""
        ...

    @property
    def take_profit(self) -> float | None:
        """Take profit threshold if configured, None otherwise."""
        ...

    @property
    def trail_stop(self) -> float | None:
        """Trailing stop threshold if configured, None otherwise."""
        ...

    @property
    def trade_at(self) -> str:
        """Trade timing setting (e.g., 'close')."""
        ...

    @property
    def resample(self) -> str | None:
        """Resample frequency if configured."""
        ...

    @property
    def benchmark(self) -> pl.DataFrame | None:
        """Benchmark DataFrame if set (date, creturn columns)."""
        ...

    @benchmark.setter
    def benchmark(self, value: pl.DataFrame | None) -> None: ...

    @property
    def stats(self) -> pl.DataFrame:
        """Backtest statistics as single-row DataFrame (with default riskfree_rate=0.02).

        Columns: start, end, rf, total_return, cagr, max_drawdown, avg_drawdown,
                 daily_mean, daily_vol, daily_sharpe, daily_sortino,
                 best_day, worst_day, calmar, win_ratio.
        """
        ...

    # Methods
    def daily_creturn(self) -> pl.DataFrame:
        """Get daily resampled cumulative return DataFrame.

        Returns:
            DataFrame with date and creturn columns, resampled to daily.
        """
        ...

    def get_stats(self, riskfree_rate: float = 0.02) -> pl.DataFrame:
        """Get backtest statistics as a single-row DataFrame.

        Args:
            riskfree_rate: Annual risk-free rate for Sharpe/Sortino calculations.

        Returns:
            DataFrame with columns: total_return, cagr, max_drawdown, avg_drawdown,
            daily_mean, daily_vol, daily_sharpe, daily_sortino, best_day, worst_day,
            calmar, win_ratio.
        """
        ...

    def get_monthly_stats(self, riskfree_rate: float = 0.02) -> pl.DataFrame:
        """Get monthly statistics as a single-row DataFrame.

        Args:
            riskfree_rate: Annual risk-free rate for Sharpe/Sortino calculations.

        Returns:
            DataFrame with columns: monthly_mean, monthly_vol, monthly_sharpe,
            monthly_sortino, best_month, worst_month.
        """
        ...

    def get_return_table(self) -> pl.DataFrame:
        """Get monthly return table pivoted by year x month.

        Returns:
            DataFrame with year as rows and months (1-12) as columns.
        """
        ...

    def current_trades(self) -> pl.DataFrame:
        """Get active trades (positions without exit or exiting on last date).

        Returns:
            DataFrame with current/active trade records.
        """
        ...

    def actions(self) -> pl.DataFrame:
        """Get trade actions for current positions.

        Returns:
            DataFrame with stock_id and action columns.
            Action values: 'enter', 'exit', 'hold'.
        """
        ...

    def is_stop_triggered(self) -> bool:
        """Check if any trade was triggered by stop loss or take profit.

        Returns:
            True if stop loss or take profit was triggered.
        """
        ...

    def get_metrics(
        self,
        sections: list[MetricsSection] | None = None,
        riskfree_rate: float = 0.02,
    ) -> pl.DataFrame:
        """Get structured metrics as single-row DataFrame.

        Args:
            sections: List of sections to include. Options: 'backtest', 'profitability',
                     'risk', 'ratio', 'winrate', 'liquidity'. Defaults to all sections.
            riskfree_rate: Annual risk-free rate for Sharpe/Sortino calculations.

        Returns:
            Single-row DataFrame with metrics columns:
            - backtest: startDate, endDate, feeRatio, taxRatio, freq, tradeAt,
                       stopLoss, takeProfit, trailStop
            - profitability: annualReturn, avgNStock, maxNStock
            - risk: maxDrawdown, avgDrawdown, avgDrawdownDays, valueAtRisk, cvalueAtRisk
            - ratio: sharpeRatio, sortinoRatio, calmarRatio, volatility,
                    profitFactor, tailRatio
            - winrate: winRate, expectancy, mae, mfe
            - liquidity: buyHigh, sellLow, capacity (requires limit_up/limit_down/trading_value columns in input DataFrame)

            If benchmark is set (via setter or backtest_with_report), additional columns:
            - alpha, beta, m12WinRate (12-month rolling win rate vs benchmark)
        """
        ...

    def __repr__(self) -> str: ...




