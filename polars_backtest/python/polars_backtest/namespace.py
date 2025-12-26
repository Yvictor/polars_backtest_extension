"""DataFrame namespace extension for backtesting.

Provides df.bt.backtest() API for long format DataFrames.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

# Import from internal module to avoid circular imports
from polars_backtest._polars_backtest import (
    BacktestConfig,
    BacktestResult,
    backtest_signals,
    backtest_weights,
    backtest_with_trades as _backtest_with_trades,
)

if TYPE_CHECKING:
    from polars_backtest import Report


def _get_resample_helpers():
    """Lazy import of resample helpers to avoid circular imports."""
    import polars_backtest as pb
    return pb._resample_position, pb._filter_changed_positions, pb.Report


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
        date_col: str,
        symbol_col: str,
        price_col: str,
        weight_col: str,
    ) -> None:
        """Validate required columns exist."""
        required = [date_col, symbol_col, price_col, weight_col]
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
    ) -> dict:
        """Convert long format data to wide format for Rust backend.

        Returns:
            Dict with wide format DataFrames:
            - prices: price DataFrame
            - position: weight DataFrame
            - open/high/low: optional OHLC DataFrames
        """
        # Get unique dates from prices
        price_dates = (
            self._df.select(date_col)
            .unique()
            .sort(date_col)
            .get_column(date_col)
            .to_list()
        )

        # Convert to wide format
        prices_wide = _long_to_wide(self._df, price_col, date_col, symbol_col)
        position_wide = _long_to_wide(self._df, weight_col, date_col, symbol_col)

        result = {
            "prices": prices_wide,
            "position": position_wide,
            "price_dates": price_dates,
        }

        # Optional OHLC
        if open_col and open_col in self._df.columns:
            result["open"] = _long_to_wide(self._df, open_col, date_col, symbol_col)
        if high_col and high_col in self._df.columns:
            result["high"] = _long_to_wide(self._df, high_col, date_col, symbol_col)
        if low_col and low_col in self._df.columns:
            result["low"] = _long_to_wide(self._df, low_col, date_col, symbol_col)

        return result

    def backtest(
        self,
        price: str = "close",
        weight: str = "weight",
        date_col: str = "date",
        symbol_col: str = "symbol",
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
    ) -> pl.DataFrame:
        """Run backtest on long format DataFrame.

        Args:
            price: Price column name (default: "close")
            weight: Weight column name (default: "weight")
            date_col: Date column name (default: "date")
            symbol_col: Symbol column name (default: "symbol")
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

        Returns:
            DataFrame with columns: date, creturn
        """
        self._validate_columns(date_col, symbol_col, price, weight)

        # Convert to wide format
        wide_data = self._prepare_wide_data(
            date_col, symbol_col, price, weight
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
        )

        # Run backtest
        if is_bool:
            creturn = backtest_signals(
                prices_data, position_data, rebalance_indices, config
            )
        else:
            position_data = position_data.cast(pl.Float64)
            creturn = backtest_weights(
                prices_data, position_data, rebalance_indices, config
            )

        # Build result
        return pl.DataFrame({
            date_col: prices_wide.get_column(date_col),
            "creturn": creturn,
        })

    def backtest_with_report(
        self,
        price: str = "close",
        weight: str = "weight",
        date_col: str = "date",
        symbol_col: str = "symbol",
        open_col: str | None = None,
        high_col: str | None = None,
        low_col: str | None = None,
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
    ) -> Report:
        """Run backtest with trade tracking, returning a Report object.

        Args:
            price: Price column name (default: "close")
            weight: Weight column name (default: "weight")
            date_col: Date column name (default: "date")
            symbol_col: Symbol column name (default: "symbol")
            open_col: Open price column name (optional)
            high_col: High price column name (optional)
            low_col: Low price column name (optional)
            resample: Rebalance frequency
            resample_offset: Optional offset for rebalance dates
            fee_ratio: Transaction fee ratio
            tax_ratio: Transaction tax ratio
            stop_loss: Stop loss threshold
            take_profit: Take profit threshold
            trail_stop: Trailing stop threshold
            position_limit: Maximum weight per stock
            retain_cost_when_rebalance: Retain costs when rebalancing
            stop_trading_next_period: Stop trading after stop triggered
            touched_exit: Use OHLC for intraday stop detection

        Returns:
            Report object with creturn, position, and trades
        """
        self._validate_columns(date_col, symbol_col, price, weight)

        if touched_exit:
            if not all([open_col, high_col, low_col]):
                raise ValueError(
                    "touched_exit=True requires open_col, high_col, and low_col"
                )
            for col in [open_col, high_col, low_col]:
                if col not in self._df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")

        # Convert to wide format
        wide_data = self._prepare_wide_data(
            date_col, symbol_col, price, weight,
            open_col, high_col, low_col
        )
        prices_wide = wide_data["prices"]
        position_wide = wide_data["position"]
        price_dates = wide_data["price_dates"]

        stock_cols = [c for c in prices_wide.columns if c != date_col]

        # Get resample helpers (lazy import)
        _resample_position, _filter_changed_positions, Report = _get_resample_helpers()

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
        position_data = position_wide.select(position_stock_cols).cast(pl.Float64)

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

        # Find first signal index
        first_signal_index = rebalance_indices[0] if rebalance_indices else 0

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
            finlab_mode=True,  # Use Finlab mode for report
            touched_exit=touched_exit,
        )

        # Prepare OHLC data if needed
        open_data = wide_data.get("open")
        high_data = wide_data.get("high")
        low_data = wide_data.get("low")

        if open_data is not None:
            open_data = open_data.select(position_stock_cols)
        if high_data is not None:
            high_data = high_data.select(position_stock_cols)
        if low_data is not None:
            low_data = low_data.select(position_stock_cols)

        # Run backtest with trades
        # Note: Using adjusted prices for both adj and original (no factor available)
        result = _backtest_with_trades(
            adj_prices=prices_data,
            original_prices=prices_data,  # Same as adj without factor
            weights=position_data,
            rebalance_indices=rebalance_indices,
            config=config,
            open_prices=open_data,
            high_prices=high_data,
            low_prices=low_data,
        )

        return Report(
            creturn=result.creturn,
            trades=result.trades,
            dates=price_dates,
            stock_columns=position_stock_cols,
            position=position_wide,
            fee_ratio=fee_ratio,
            tax_ratio=tax_ratio,
            first_signal_index=first_signal_index,
        )


# =============================================================================
# Standalone Function API
# =============================================================================


def backtest(
    df: pl.DataFrame,
    price: str = "close",
    weight: str = "weight",
    date_col: str = "date",
    symbol_col: str = "symbol",
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
) -> pl.DataFrame:
    """Run backtest on long format DataFrame.

    Args:
        df: Long format DataFrame with date, symbol, price, weight columns
        price: Price column name (default: "close")
        weight: Weight column name (default: "weight")
        date_col: Date column name (default: "date")
        symbol_col: Symbol column name (default: "symbol")
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

    Returns:
        DataFrame with columns: date, creturn

    Example:
        >>> import polars_backtest as pl_bt
        >>> result = pl_bt.backtest(df, price="close", weight="weight", resample="M")
    """
    return df.bt.backtest(
        price=price,
        weight=weight,
        date_col=date_col,
        symbol_col=symbol_col,
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
    )


def backtest_with_report(
    df: pl.DataFrame,
    price: str = "close",
    weight: str = "weight",
    date_col: str = "date",
    symbol_col: str = "symbol",
    open_col: str | None = None,
    high_col: str | None = None,
    low_col: str | None = None,
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
) -> "Report":
    """Run backtest with trade tracking on long format DataFrame.

    Args:
        df: Long format DataFrame with date, symbol, price, weight columns
        price: Price column name (default: "close")
        weight: Weight column name (default: "weight")
        date_col: Date column name (default: "date")
        symbol_col: Symbol column name (default: "symbol")
        open_col: Open price column name (optional)
        high_col: High price column name (optional)
        low_col: Low price column name (optional)
        resample: Rebalance frequency
        resample_offset: Optional offset for rebalance dates
        fee_ratio: Transaction fee ratio
        tax_ratio: Transaction tax ratio
        stop_loss: Stop loss threshold
        take_profit: Take profit threshold
        trail_stop: Trailing stop threshold
        position_limit: Maximum weight per stock
        retain_cost_when_rebalance: Retain costs when rebalancing
        stop_trading_next_period: Stop trading after stop triggered
        touched_exit: Use OHLC for intraday stop detection

    Returns:
        Report object with creturn, position, and trades

    Example:
        >>> import polars_backtest as pl_bt
        >>> report = pl_bt.backtest_with_report(df, price="close", weight="weight")
        >>> report.creturn  # DataFrame
        >>> report.trades   # DataFrame
    """
    return df.bt.backtest_with_report(
        price=price,
        weight=weight,
        date_col=date_col,
        symbol_col=symbol_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
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
