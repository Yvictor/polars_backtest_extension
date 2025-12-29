"""DataFrame namespace extension for backtesting.

Provides df.bt.backtest() API for long format DataFrames.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

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
    # Wide format API (for validation)
    backtest_wide as _backtest_wide_rust,
)

if TYPE_CHECKING:
    from polars_backtest import Report


def _get_resample_helpers():
    """Lazy import of resample helpers to avoid circular imports."""
    import polars_backtest as pb
    return pb._resample_position, pb._filter_changed_positions, pb.Report


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

    def _prepare_wide_data(
        self,
        date_col: str,
        symbol_col: str,
        price_col: str,
        weight_col: str,
        open_col: str | None = None,
        high_col: str | None = None,
        low_col: str | None = None,
        df: pl.DataFrame | None = None,
    ) -> dict:
        """Convert long format data to wide format for Rust backend.

        Args:
            df: DataFrame to use (defaults to self._df if not provided)

        Returns:
            Dict with wide format DataFrames:
            - prices: price DataFrame
            - position: weight DataFrame
            - open/high/low: optional OHLC DataFrames
        """
        if df is None:
            df = self._df

        # Get unique dates from prices
        price_dates = (
            df.select(date_col)
            .unique()
            .sort(date_col)
            .get_column(date_col)
            .to_list()
        )

        # Convert to wide format
        prices_wide = _long_to_wide(df, price_col, date_col, symbol_col)
        position_wide = _long_to_wide(df, weight_col, date_col, symbol_col)

        result = {
            "prices": prices_wide,
            "position": position_wide,
            "price_dates": price_dates,
        }

        # Optional OHLC
        if open_col and open_col in df.columns:
            result["open"] = _long_to_wide(df, open_col, date_col, symbol_col)
        if high_col and high_col in df.columns:
            result["high"] = _long_to_wide(df, high_col, date_col, symbol_col)
        if low_col and low_col in df.columns:
            result["low"] = _long_to_wide(df, low_col, date_col, symbol_col)

        return result

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

        # Check if we can use Rust path (basic resample, no offset)
        use_rust = (
            resample in (None, "D", "W", "M", "Q", "Y")
            and resample_offset is None
        )

        if use_rust:
            # Use Rust backtest directly (does pivot internally)
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

            # Check if already sorted by date (pyo3-polars loses this flag during transfer)
            # skip_sort = df[date_col].flags.get("SORTED_ASC", False)
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
                config,
                skip_sort,
            )

            # Rust now returns creturn as DataFrame with date and creturn columns
            # Filtering and normalization is done in Rust
            return result.creturn.rename({result.creturn.columns[0]: date_col})

        # Fallback to Python path for complex resample patterns
        # Convert to wide format (using df with nulls filled)
        wide_data = self._prepare_wide_data(
            date_col, symbol_col, price_col, position_col, df=df
        )
        prices_wide = wide_data["prices"]
        position_wide = wide_data["position"]
        price_dates = wide_data["price_dates"]

        # Get stock columns (all except date)
        stock_cols = [c for c in prices_wide.columns if c != date_col]

        # Get resample helpers (lazy import)
        _resample_position, _filter_changed_positions, _ = _get_resample_helpers()

        # Apply resample
        if resample is None:
            position_wide = _filter_changed_positions(position_wide)
        elif resample != "D":
            position_wide = _resample_position(
                position_wide, price_dates, resample, resample_offset
            )

        # Align columns
        position_stock_cols = [c for c in position_wide.columns if c in stock_cols]
        if not position_stock_cols:
            raise ValueError("No common stock columns between prices and position")

        prices_data = prices_wide.select(position_stock_cols)
        position_data = position_wide.select(position_stock_cols)

        # Determine signal type
        first_col_dtype = position_data.dtypes[0]
        is_bool = first_col_dtype == pl.Boolean

        # Calculate rebalance indices
        pos_date_col = position_wide.columns[0]
        position_dates = position_wide.select(pos_date_col).to_series().to_list()

        rebalance_indices = []
        for pos_d in position_dates:
            try:
                idx = price_dates.index(pos_d)
                rebalance_indices.append(idx)
            except ValueError:
                pass

        if not rebalance_indices:
            raise ValueError("No matching dates between prices and position")

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

        # Run backtest (convert bool to float if needed)
        if is_bool:
            position_data = position_data.cast(pl.Float64)
        else:
            position_data = position_data.cast(pl.Float64)

        # Note: Wide format backtest doesn't support touched_exit yet
        # Would need to pass OHLC data as well
        creturn = _backtest_wide_rust(
            prices_data, position_data, rebalance_indices, config
        )

        # Find first rebalance index with any non-zero signals (like Finlab)
        first_signal_rebalance_idx = 0
        for i in range(len(position_data)):
            row = position_data[i]
            # Check if any value in this row is non-zero (True or > 0)
            has_signal = any(
                row[col][0] is not None and row[col][0] > 0
                for col in position_data.columns
            )
            if has_signal:
                first_signal_rebalance_idx = i
                break

        # Get first signal date index in price data
        first_signal_idx = rebalance_indices[first_signal_rebalance_idx] if rebalance_indices else 0

        # Slice from first signal date and normalize creturn to start at 1.0
        sliced_creturn = creturn[first_signal_idx:]
        if len(sliced_creturn) > 0 and sliced_creturn[0] != 0:
            normalizer = sliced_creturn[0]
            sliced_creturn = [c / normalizer for c in sliced_creturn]

        dates = prices_wide.get_column(date_col)

        # Build result
        return pl.DataFrame({
            date_col: dates[first_signal_idx:],
            "creturn": sliced_creturn,
        })

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
