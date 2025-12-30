"""DataFrame namespace extension for backtesting.

Provides df.bt.backtest() API for long format DataFrames.
"""

from __future__ import annotations

from typing import Union

import polars as pl

# Type alias for column specification (str or Expr)
ColumnSpec = Union[str, pl.Expr]

# Import from internal module to avoid circular imports
from polars_backtest._polars_backtest import (
    BacktestConfig,
    BacktestResult,
    BacktestReport,
    # Main API (long format, zero-copy)
    backtest as _rust_backtest,
    backtest_with_report as _rust_backtest_with_report,
)


def _resolve_column(
    df: pl.DataFrame,
    col_spec: ColumnSpec,
    temp_name: str,
) -> tuple[pl.DataFrame, str]:
    """Resolve a column specification (str or Expr) to a column name.

    Args:
        df: DataFrame to process
        col_spec: Column name (str) or expression (Expr)
        temp_name: Temporary column name to use if col_spec is an Expr

    Returns:
        Tuple of (possibly modified DataFrame, column name to use)
    """
    if isinstance(col_spec, pl.Expr):
        df = df.with_columns(col_spec.alias(temp_name))
        return df, temp_name
    return df, col_spec


def _long_to_wide(
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


@pl.api.register_dataframe_namespace("bt")
class BacktestNamespace:
    """Backtest namespace for Polars DataFrames.

    Usage:
        df = pl.DataFrame({
            "date": [...],
            "symbol": [...],
            "close": [...],
            "weight": [...],
        })

        result = df.bt.backtest(
            price="close",
            weight="weight",
            resample="M",
        )
    """

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def _validate_columns(
        self,
        date: str,
        symbol: str,
        trade_at_price: str,
        position: str,
    ) -> None:
        """Validate required columns exist (only for str column names)."""
        required = [date, symbol, trade_at_price, position]
        missing = [c for c in required if c not in self._df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def backtest(
        self,
        trade_at_price: ColumnSpec = "close",
        position: ColumnSpec = "weight",
        date: ColumnSpec = "date",
        symbol: ColumnSpec = "symbol",
        open: ColumnSpec = "open",
        high: ColumnSpec = "high",
        low: ColumnSpec = "low",
        resample: str | None = "D",
        resample_offset: str | None = None,
        fee_ratio: float = 0.001425,
        tax_ratio: float = 0.003,
        stop_loss: float = 1.0,
        take_profit: float = float("inf"),
        trail_stop: float = float("inf"),
        position_limit: float = 1.0,
        retain_cost_when_rebalance: bool = False,
        stop_trading_next_period: bool = True,
        finlab_mode: bool = False,
        touched_exit: bool = False,
    ) -> pl.DataFrame:
        """Run backtest on long format DataFrame.

        Args:
            trade_at_price: Price column name or Expr (default: "close")
            position: Position/weight column name or Expr (default: "weight")
            date: Date column name or Expr (default: "date")
            symbol: Symbol column name or Expr (default: "symbol")
            open: Open price column name or Expr (default: "open", for touched_exit)
            high: High price column name or Expr (default: "high", for touched_exit)
            low: Low price column name or Expr (default: "low", for touched_exit)
            resample: Rebalance frequency ('D', 'W', 'M', 'Q', 'Y', None)
            resample_offset: Optional offset for rebalance dates
            fee_ratio: Transaction fee ratio
            tax_ratio: Transaction tax ratio
            stop_loss: Stop loss threshold (1.0 = disabled)
            take_profit: Take profit threshold (inf = disabled)
            trail_stop: Trailing stop threshold (inf = disabled)
            position_limit: Maximum weight per stock
            retain_cost_when_rebalance: Retain costs when rebalancing
            stop_trading_next_period: Stop trading after stop triggered
            finlab_mode: Use Finlab-compatible calculation
            touched_exit: Use OHLC for intraday stop detection (requires open/high/low)

        Returns:
            DataFrame with columns: date, creturn
        """
        df = self._df

        # Resolve column specs (str or Expr) to column names
        df, date_col = _resolve_column(df, date, "_bt_date")
        df, symbol_col = _resolve_column(df, symbol, "_bt_symbol")
        df, price_col = _resolve_column(df, trade_at_price, "_bt_price")
        df, position_col = _resolve_column(df, position, "_bt_position")

        # Validate columns exist
        required = [date_col, symbol_col, price_col, position_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Resolve OHLC columns only when touched_exit is True
        open_col: str = "open"
        high_col: str = "high"
        low_col: str = "low"
        if touched_exit:
            df, open_col = _resolve_column(df, open, "_bt_open")
            df, high_col = _resolve_column(df, high, "_bt_high")
            df, low_col = _resolve_column(df, low, "_bt_low")

            # Validate OHLC columns exist
            ohlc_cols = [open_col, high_col, low_col]
            ohlc_missing = [c for c in ohlc_cols if c not in df.columns]
            if ohlc_missing:
                raise ValueError(
                    f"touched_exit=True requires open/high/low columns. Missing: {ohlc_missing}"
                )

        # Check if position column is boolean (signals)
        position_dtype = df.get_column(position_col).dtype
        is_bool_signal = position_dtype == pl.Boolean

        # Handle null values in position column
        # Polars rolling operations return null for first N-1 rows (unlike pandas NaN -> False)
        # Fill nulls with False/0.0, then cast bool to float for Rust path
        if is_bool_signal:
            # Cast bool to float (True -> 1.0, False -> 0.0)
            # Rust's normalize_weights_finlab will convert to equal weights
            df = df.with_columns(
                pl.col(position_col).fill_null(False).cast(pl.Float64)
            )
        else:
            df = df.with_columns(pl.col(position_col).fill_null(0.0))

        # Validate resample parameter
        # Supported: D, W, W-MON..W-SUN, M, MS, Q, QS, Y, None
        supported_resample = (
            None, "D",
            "W", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT", "W-SUN",
            "M", "ME", "MS",
            "Q", "QE", "QS",
            "Y", "YE", "A",
        )
        if resample not in supported_resample:
            raise ValueError(
                f"Unsupported resample '{resample}'. "
                f"Supported values: {', '.join(str(s) for s in supported_resample)}"
            )

        # Create config
        config = BacktestConfig(
            fee_ratio=fee_ratio,
            tax_ratio=tax_ratio,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trail_stop=trail_stop,
            position_limit=position_limit,
            retain_cost_when_rebalance=retain_cost_when_rebalance,
            stop_trading_next_period=stop_trading_next_period,
            finlab_mode=finlab_mode,
            touched_exit=touched_exit,
        )

        # Check if already sorted by date
        skip_sort = df.get_column(date_col).is_sorted()

        result = _rust_backtest(
            df,
            date_col,
            symbol_col,
            price_col,
            position_col,
            open_col,
            high_col,
            low_col,
            resample,
            resample_offset,
            config,
            skip_sort,
        )

        return result.creturn.rename({result.creturn.columns[0]: date_col})

    def backtest_with_report(
        self,
        trade_at_price: ColumnSpec = "close",
        position: ColumnSpec = "weight",
        date: ColumnSpec = "date",
        symbol: ColumnSpec = "symbol",
        open: ColumnSpec = "open",
        high: ColumnSpec = "high",
        low: ColumnSpec = "low",
        resample: str | None = "D",
        fee_ratio: float = 0.001425,
        tax_ratio: float = 0.003,
        stop_loss: float = 1.0,
        take_profit: float = float("inf"),
        trail_stop: float = float("inf"),
        position_limit: float = 1.0,
        retain_cost_when_rebalance: bool = False,
        stop_trading_next_period: bool = True,
        touched_exit: bool = False,
    ) -> BacktestReport:
        """Run backtest with trade tracking, returning a BacktestReport object.

        Uses the Rust long format implementation directly for performance.
        When touched_exit=True, falls back to wide format for OHLC support.

        Args:
            trade_at_price: Price column name or Expr (default: "close")
            position: Position/weight column name or Expr (default: "weight")
            date: Date column name or Expr (default: "date")
            symbol: Symbol column name or Expr (default: "symbol")
            open: Open price column name or Expr (default: "open", for touched_exit)
            high: High price column name or Expr (default: "high", for touched_exit)
            low: Low price column name or Expr (default: "low", for touched_exit)
            resample: Rebalance frequency ('D', 'W', 'M', or None)
            fee_ratio: Transaction fee ratio
            tax_ratio: Transaction tax ratio
            stop_loss: Stop loss threshold
            take_profit: Take profit threshold
            trail_stop: Trailing stop threshold
            position_limit: Maximum weight per stock
            retain_cost_when_rebalance: Retain costs when rebalancing
            stop_trading_next_period: Stop trading after stop triggered
            touched_exit: Use OHLC for intraday stop detection (requires open/high/low)

        Returns:
            BacktestReport object with creturn (list) and trades (DataFrame)
        """
        df = self._df

        # Resolve column specs (str or Expr) to column names
        df, date_col = _resolve_column(df, date, "_bt_date")
        df, symbol_col = _resolve_column(df, symbol, "_bt_symbol")
        df, price_col = _resolve_column(df, trade_at_price, "_bt_price")
        df, position_col = _resolve_column(df, position, "_bt_position")

        # Validate required columns exist
        required = [date_col, symbol_col, price_col, position_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Resolve OHLC columns only when touched_exit is True
        open_col: str | None = None
        high_col: str | None = None
        low_col: str | None = None
        if touched_exit:
            df, open_col = _resolve_column(df, open, "_bt_open")
            df, high_col = _resolve_column(df, high, "_bt_high")
            df, low_col = _resolve_column(df, low, "_bt_low")

            # Validate OHLC columns exist
            ohlc_cols = [open_col, high_col, low_col]
            ohlc_missing = [c for c in ohlc_cols if c not in df.columns]
            if ohlc_missing:
                raise ValueError(
                    f"touched_exit=True requires open/high/low columns. Missing: {ohlc_missing}"
                )

        # Check if position column is boolean (signals)
        position_dtype = df.get_column(position_col).dtype
        is_bool_signal = position_dtype == pl.Boolean

        # Handle null values in position column
        if is_bool_signal:
            df = df.with_columns(
                pl.col(position_col).fill_null(False).cast(pl.Float64)
            )
        else:
            df = df.with_columns(pl.col(position_col).fill_null(0.0))

        # Build config
        config = BacktestConfig(
            fee_ratio=fee_ratio,
            tax_ratio=tax_ratio,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trail_stop=trail_stop,
            position_limit=position_limit,
            retain_cost_when_rebalance=retain_cost_when_rebalance,
            stop_trading_next_period=stop_trading_next_period,
            finlab_mode=True,  # Use Finlab mode for report
            touched_exit=touched_exit,
        )

        # Check if already sorted
        skip_sort = df.get_column(date_col).is_sorted()

        # Use Rust backtest_with_report directly (returns BacktestReport with trades DataFrame)
        # OHLC columns are only used when touched_exit=True
        return _rust_backtest_with_report(
            df,
            date_col,
            symbol_col,
            price_col,
            position_col,
            open_col if open_col else "open",
            high_col if high_col else "high",
            low_col if low_col else "low",
            resample,
            None,  # resample_offset not supported for backtest_with_report yet
            config,
            skip_sort,
        )

# =============================================================================
# Standalone Function API
# =============================================================================


def backtest(
    df: pl.DataFrame,
    trade_at_price: ColumnSpec = "close",
    position: ColumnSpec = "weight",
    date: ColumnSpec = "date",
    symbol: ColumnSpec = "symbol",
    open: ColumnSpec = "open",
    high: ColumnSpec = "high",
    low: ColumnSpec = "low",
    resample: str | None = "D",
    resample_offset: str | None = None,
    fee_ratio: float = 0.001425,
    tax_ratio: float = 0.003,
    stop_loss: float = 1.0,
    take_profit: float = float("inf"),
    trail_stop: float = float("inf"),
    position_limit: float = 1.0,
    retain_cost_when_rebalance: bool = False,
    stop_trading_next_period: bool = True,
    finlab_mode: bool = False,
    touched_exit: bool = False,
) -> pl.DataFrame:
    """Run backtest on long format DataFrame.

    Args:
        df: Long format DataFrame with date, symbol, price, position columns
        trade_at_price: Price column name or Expr (default: "close")
        position: Position/weight column name or Expr (default: "weight")
        date: Date column name or Expr (default: "date")
        symbol: Symbol column name or Expr (default: "symbol")
        open: Open price column name or Expr (default: "open", for touched_exit)
        high: High price column name or Expr (default: "high", for touched_exit)
        low: Low price column name or Expr (default: "low", for touched_exit)
        resample: Rebalance frequency ('D', 'W', 'M', 'Q', 'Y', None)
        resample_offset: Optional offset for rebalance dates
        fee_ratio: Transaction fee ratio
        tax_ratio: Transaction tax ratio
        stop_loss: Stop loss threshold (1.0 = disabled)
        take_profit: Take profit threshold (inf = disabled)
        trail_stop: Trailing stop threshold (inf = disabled)
        position_limit: Maximum weight per stock
        retain_cost_when_rebalance: Retain costs when rebalancing
        stop_trading_next_period: Stop trading after stop triggered
        finlab_mode: Use Finlab-compatible calculation
        touched_exit: Use OHLC for intraday stop detection (requires open/high/low)

    Returns:
        DataFrame with columns: date, creturn

    Example:
        >>> import polars_backtest as pl_bt
        >>> result = pl_bt.backtest(df, trade_at_price="close", position="weight", resample="M")
    """
    return df.bt.backtest(
        trade_at_price=trade_at_price,
        position=position,
        date=date,
        symbol=symbol,
        open=open,
        high=high,
        low=low,
        resample=resample,
        resample_offset=resample_offset,
        fee_ratio=fee_ratio,
        tax_ratio=tax_ratio,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trail_stop=trail_stop,
        position_limit=position_limit,
        retain_cost_when_rebalance=retain_cost_when_rebalance,
        stop_trading_next_period=stop_trading_next_period,
        finlab_mode=finlab_mode,
        touched_exit=touched_exit,
    )


def backtest_with_report(
    df: pl.DataFrame,
    trade_at_price: ColumnSpec = "close",
    position: ColumnSpec = "weight",
    date: ColumnSpec = "date",
    symbol: ColumnSpec = "symbol",
    open: ColumnSpec = "open",
    high: ColumnSpec = "high",
    low: ColumnSpec = "low",
    resample: str | None = "D",
    fee_ratio: float = 0.001425,
    tax_ratio: float = 0.003,
    stop_loss: float = 1.0,
    take_profit: float = float("inf"),
    trail_stop: float = float("inf"),
    position_limit: float = 1.0,
    retain_cost_when_rebalance: bool = False,
    stop_trading_next_period: bool = True,
    touched_exit: bool = False,
) -> BacktestReport:
    """Run backtest with trade tracking on long format DataFrame.

    Uses the Rust long format implementation directly for performance.

    Args:
        df: Long format DataFrame with date, symbol, price, position columns
        trade_at_price: Price column name or Expr (default: "close")
        position: Position/weight column name or Expr (default: "weight")
        date: Date column name or Expr (default: "date")
        symbol: Symbol column name or Expr (default: "symbol")
        open: Open price column name or Expr (default: "open", for touched_exit)
        high: High price column name or Expr (default: "high", for touched_exit)
        low: Low price column name or Expr (default: "low", for touched_exit)
        resample: Rebalance frequency ('D', 'W', 'M', or None)
        fee_ratio: Transaction fee ratio
        tax_ratio: Transaction tax ratio
        stop_loss: Stop loss threshold
        take_profit: Take profit threshold
        trail_stop: Trailing stop threshold
        position_limit: Maximum weight per stock
        retain_cost_when_rebalance: Retain costs when rebalancing
        stop_trading_next_period: Stop trading after stop triggered
        touched_exit: Use OHLC for intraday stop detection (requires open/high/low)

    Returns:
        BacktestReport object with creturn (list) and trades (DataFrame)

    Example:
        >>> import polars_backtest as pl_bt
        >>> report = pl_bt.backtest_with_report(df, trade_at_price="close", position="weight")
        >>> report.creturn  # list of cumulative returns
        >>> report.trades   # DataFrame with trade records
    """
    return df.bt.backtest_with_report(
        trade_at_price=trade_at_price,
        position=position,
        date=date,
        symbol=symbol,
        open=open,
        high=high,
        low=low,
        resample=resample,
        fee_ratio=fee_ratio,
        tax_ratio=tax_ratio,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trail_stop=trail_stop,
        position_limit=position_limit,
        retain_cost_when_rebalance=retain_cost_when_rebalance,
        stop_trading_next_period=stop_trading_next_period,
        touched_exit=touched_exit,
    )
