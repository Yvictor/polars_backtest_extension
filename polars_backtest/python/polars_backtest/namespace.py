"""DataFrame namespace extension for backtesting.

Provides df.bt.backtest() API for long format DataFrames.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable, Union, cast

import polars as pl

from polars_backtest._polars_backtest import (
    BacktestConfig,
    BacktestReport,
)
from polars_backtest._polars_backtest import (
    backtest as _rust_backtest,
)
from polars_backtest._polars_backtest import (
    backtest_with_report as _rust_backtest_with_report,
)

if TYPE_CHECKING:
    from polars_backtest.polars_backtest import DataFrame as ExtDataFrame

# Type alias for column specification (str or Expr)
ColumnSpec = Union[str, pl.Expr]


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


# Offsets the long-format Rust engine actually parses: non-negative "<N>D" / "<N>W".
# Anything else (e.g. finlab-style "-1D") used to be *silently ignored* — reject instead.
_OFFSET_PATTERN = re.compile(r"^\d+[dDwW]$")

# NOTE: "MS"/"QS" are intentionally absent — the engine treated them as aliases of
# "M"/"Q" (month/quarter END), which silently diverged from pandas/finlab semantics.
_SUPPORTED_RESAMPLE = (
    None, "D",
    "W", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT", "W-SUN",
    "M", "ME",
    "Q", "QE",
    "Y", "YE", "A",
)


def _validate_resample(resample: str | None, resample_offset: str | None) -> None:
    """Reject resample values the engine would silently misinterpret."""
    if resample not in _SUPPORTED_RESAMPLE:
        hint = ""
        if resample in ("MS", "QS"):
            hint = (
                " Start-of-period frequencies are not supported by the long-format"
                " engine (they used to be silently treated as month/quarter end)."
            )
        raise ValueError(
            f"Unsupported resample '{resample}'. "
            f"Supported values: {', '.join(str(s) for s in _SUPPORTED_RESAMPLE)}.{hint}"
        )
    if resample_offset is not None and not _OFFSET_PATTERN.match(resample_offset.strip()):
        raise ValueError(
            f"Unsupported resample_offset '{resample_offset}'. The long-format engine"
            " supports only non-negative day/week offsets such as '5D' or '1W'."
            " Negative offsets like '-1D' are not supported (they used to be silently"
            " ignored); use the wide-format API if you need them."
        )


def _validate_no_nulls(df: pl.DataFrame, columns: Iterable[str]) -> None:
    """date/symbol nulls would reach the Rust engine as undefined buffer bytes."""
    for col in columns:
        null_count = df.get_column(col).null_count()
        if null_count:
            raise ValueError(
                f"Column '{col}' contains {null_count} null value(s);"
                " drop or fill them before backtesting"
            )


def _floats_with_nan_nulls(df: pl.DataFrame, columns: Iterable[str | None]) -> pl.DataFrame:
    """Ensure Float64 dtype and represent nulls as NaN (the engine's missing marker).

    The FFI layer reads raw value buffers without checking Arrow validity bitmaps,
    so a null slot would otherwise surface as arbitrary bytes.
    """
    exprs = []
    for col in dict.fromkeys(c for c in columns if c is not None):
        series = df.get_column(col)
        if series.dtype != pl.Float64 or series.null_count():
            exprs.append(pl.col(col).cast(pl.Float64).fill_null(float("nan")))
    return df.with_columns(exprs) if exprs else df


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
        factor: str = "factor",
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
            factor: Factor column name for raw price calculation (default: "factor").
                   raw_price = adj_price / factor. If column doesn't exist, uses 1.0.
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

        _validate_resample(resample, resample_offset)

        # factor column: pass column name if exists, else None (defaults to 1.0 in Rust)
        factor_col = factor if factor in df.columns else None

        # Nulls must not reach the FFI boundary (validity bitmaps are not checked there)
        _validate_no_nulls(df, (date_col, symbol_col))
        float_cols: list[str | None] = [price_col, position_col, factor_col]
        if touched_exit:
            float_cols += [open_col, high_col, low_col]
        df = _floats_with_nan_nulls(df, float_cols)

        # Create config
        # Note: the long-format engine always uses finlab-style accounting; the former
        # `finlab_mode` parameter had no effect on this path and was removed.
        config = BacktestConfig(
            fee_ratio=fee_ratio,
            tax_ratio=tax_ratio,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trail_stop=trail_stop,
            position_limit=position_limit,
            retain_cost_when_rebalance=retain_cost_when_rebalance,
            stop_trading_next_period=stop_trading_next_period,
            finlab_mode=True,
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
            factor_col,
            resample,
            resample_offset,
            config,
            skip_sort,
        )

        return result.creturn

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
        take_profit: float = float("inf"),
        trail_stop: float = float("inf"),
        position_limit: float = 1.0,
        retain_cost_when_rebalance: bool = False,
        stop_trading_next_period: bool = True,
        touched_exit: bool = False,
        limit_up: str = "limit_up",
        limit_down: str = "limit_down",
        trading_value: str = "trading_value",
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
            factor: Factor column name for raw price calculation (default: "factor").
                   raw_price = adj_price / factor. If column doesn't exist, uses 1.0.
            benchmark: Benchmark for alpha/beta/m12WinRate calculation. Can be:
                      - str: Symbol value (e.g., "0050"), uses that symbol's price as benchmark
                      - pl.DataFrame: DataFrame with 'date' and 'creturn' columns
                      If provided, alpha/beta/m12WinRate will be auto-calculated in get_metrics().
            resample: Rebalance frequency ('D', 'W', 'M', 'Q', 'Y', None)
            resample_offset: Optional offset for rebalance dates
            fee_ratio: Transaction fee ratio
            tax_ratio: Transaction tax ratio
            stop_loss: Stop loss threshold
            take_profit: Take profit threshold
            trail_stop: Trailing stop threshold
            position_limit: Maximum weight per stock
            retain_cost_when_rebalance: Retain costs when rebalancing
            stop_trading_next_period: Stop trading after stop triggered
            touched_exit: Use OHLC for intraday stop detection (requires open/high/low)
            limit_up: Limit-up price column name, used by get_metrics() liquidity section
            limit_down: Limit-down price column name, used by get_metrics() liquidity section
            trading_value: Trading value column name, used by get_metrics() capacity metric

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

        _validate_resample(resample, resample_offset)

        # factor column: pass column name, Rust checks if it exists (defaults to 1.0 if not)
        factor_col = factor if factor in df.columns else None

        # Nulls must not reach the FFI boundary (validity bitmaps are not checked there)
        _validate_no_nulls(df, (date_col, symbol_col))
        float_cols: list[str | None] = [price_col, position_col, factor_col]
        if touched_exit:
            float_cols += [open_col, high_col, low_col]
        df = _floats_with_nan_nulls(df, float_cols)

        # Build config
        # Note: the long-format engine always uses finlab-style accounting; the former
        # `finlab_mode` parameter had no effect on this path and was removed.
        config = BacktestConfig(
            fee_ratio=fee_ratio,
            tax_ratio=tax_ratio,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trail_stop=trail_stop,
            position_limit=position_limit,
            retain_cost_when_rebalance=retain_cost_when_rebalance,
            stop_trading_next_period=stop_trading_next_period,
            finlab_mode=True,
            touched_exit=touched_exit,
        )

        # Check if already sorted
        skip_sort = df.get_column(date_col).is_sorted()

        # Process benchmark parameter
        # - str: symbol 值，用該 symbol 的價格計算 creturn
        # - pl.DataFrame: 直接使用
        benchmark_arg: pl.DataFrame | None = None
        if benchmark is not None:
            if isinstance(benchmark, pl.DataFrame):
                # Validate DataFrame has required columns
                if "date" not in benchmark.columns or "creturn" not in benchmark.columns:
                    raise ValueError(
                        "Benchmark DataFrame must have 'date' and 'creturn' columns"
                    )
                benchmark_arg = benchmark
            elif isinstance(benchmark, str):
                # str = symbol 值，用該 symbol 的價格計算 creturn
                bm_df = (
                    df.filter(pl.col(symbol_col) == benchmark)
                    .select([pl.col(date_col).alias("date"), pl.col(price_col)])
                    .sort("date")
                    .with_columns(
                        (pl.col(price_col) / pl.col(price_col).first()).alias("creturn")
                    )
                    .select(["date", "creturn"])
                )
                if bm_df.height == 0:
                    raise ValueError(f"Symbol '{benchmark}' not found in DataFrame")
                benchmark_arg = bm_df
            else:
                raise TypeError(
                    f"benchmark must be str or pl.DataFrame, got {type(benchmark)}"
                )

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
            factor_col,
            resample,
            resample_offset,
            config,
            skip_sort,
            benchmark_arg,
            limit_up,
            limit_down,
            trading_value,
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
    factor: str = "factor",
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
        factor: Factor column name for raw price calculation (default: "factor").
               raw_price = adj_price / factor. If column doesn't exist, uses 1.0.
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
        touched_exit: Use OHLC for intraday stop detection (requires open/high/low)

    Returns:
        DataFrame with columns: date, creturn

    Example:
        >>> import polars_backtest as pl_bt
        >>> result = pl_bt.backtest(df, trade_at_price="close", position="weight", resample="M")
    """
    return cast("ExtDataFrame", df).bt.backtest(
        trade_at_price=trade_at_price,
        position=position,
        date=date,
        symbol=symbol,
        open=open,
        high=high,
        low=low,
        factor=factor,
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
    factor: str = "factor",
    benchmark: str | pl.DataFrame | None = None,
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
    touched_exit: bool = False,
    limit_up: str = "limit_up",
    limit_down: str = "limit_down",
    trading_value: str = "trading_value",
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
        factor: Factor column name for raw price calculation (default: "factor").
               raw_price = adj_price / factor. If column doesn't exist, uses 1.0.
        benchmark: Benchmark for alpha/beta/m12WinRate calculation. Can be:
                  - str: Symbol value (e.g., "0050"), uses that symbol's price as benchmark
                  - pl.DataFrame: DataFrame with 'date' and 'creturn' columns
                  If provided, alpha/beta/m12WinRate will be auto-calculated in get_metrics().
        resample: Rebalance frequency ('D', 'W', 'M', 'Q', 'Y', None)
        resample_offset: Optional offset for rebalance dates
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
    return cast("ExtDataFrame", df).bt.backtest_with_report(
        trade_at_price=trade_at_price,
        position=position,
        date=date,
        symbol=symbol,
        open=open,
        high=high,
        low=low,
        factor=factor,
        benchmark=benchmark,
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
        touched_exit=touched_exit,
        limit_up=limit_up,
        limit_down=limit_down,
        trading_value=trading_value,
    )
