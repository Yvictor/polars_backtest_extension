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

# Load the Rust extension
from polars_backtest._polars_backtest import (
    __version__,
    BacktestConfig,
    TradeRecord,
    BacktestResult,
    backtest_signals,
    backtest_weights,
    backtest_with_trades as _backtest_with_trades,
)

__all__ = [
    "__version__",
    "BacktestConfig",
    "TradeRecord",
    "BacktestResult",
    "Report",
    "backtest_signals",
    "backtest_weights",
    "backtest",
    "backtest_with_report",
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


def _resample_position(
    position: pl.DataFrame,
    price_dates: list,
    resample: str,
    resample_offset: str | None = None,
) -> pl.DataFrame:
    """Resample position DataFrame to target frequency.

    Takes the last position value in each period, using trading days.

    Args:
        position: Position DataFrame with date column and stock columns.
        price_dates: List of all trading dates from price DataFrame.
        resample: Resample frequency. Supports:
            - Simple: 'D', 'W', 'M', 'Q', 'Y'
            - With anchor: 'W-FRI' (weekly on Friday), 'W-MON' (weekly on Monday)
            - Month variants: 'MS' (month start), 'ME' (month end)
            - Quarter variants: 'QS' (quarter start), 'QE' (quarter end)
        resample_offset: Optional time offset to shift rebalance dates.
            Examples: '1D' (shift 1 day forward), '-1D' (shift 1 day back)

    Returns:
        Resampled position DataFrame with dates at period boundaries (trading days).
    """
    import pandas as pd
    from pandas.tseries.offsets import DateOffset
    from pandas.tseries.frequencies import to_offset

    if resample == 'D':
        return position

    date_col = position.columns[0]
    stock_cols = position.columns[1:]

    # Convert to pandas for resampling
    pos_dates = position[date_col].to_list()
    pos_pd = pd.DataFrame(
        {col: position[col].to_list() for col in stock_cols},
        index=pd.DatetimeIndex([pd.Timestamp(d) for d in pos_dates])
    )

    # Forward fill position to all trading dates first
    all_dates_pd = pd.DatetimeIndex([pd.Timestamp(d) for d in price_dates])
    pos_filled = pos_pd.reindex(all_dates_pd).ffill()

    # Normalize resample string for pandas 2.2+ compatibility
    freq = resample
    if pd.__version__ >= '2.2.0':
        # Old strings need 'E' suffix for end-of-period
        old_resample_strings = ['M', 'BM', 'SM', 'CBM', 'Q', 'BQ', 'A', 'Y', 'BY']
        if freq in old_resample_strings:
            freq += 'E'

    # Generate rebalance dates using pandas date_range
    start_date = all_dates_pd[0]
    end_date = all_dates_pd[-1]

    try:
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    except ValueError as e:
        raise ValueError(f"Invalid resample frequency: {resample}. Error: {e}")

    # Apply resample_offset if provided
    if resample_offset is not None:
        try:
            offset = to_offset(resample_offset)
            rebalance_dates = rebalance_dates + offset
        except ValueError as e:
            raise ValueError(f"Invalid resample_offset: {resample_offset}. Error: {e}")

    # Filter to valid trading dates and find the last trading day <= each rebalance date
    trading_dates_set = set(all_dates_pd)
    selected_dates = []

    for rebal_date in rebalance_dates:
        # Find the last trading day that is <= rebalance date
        valid_dates = [d for d in all_dates_pd if d <= rebal_date]
        if valid_dates:
            last_trading_day = max(valid_dates)
            if last_trading_day not in selected_dates:
                selected_dates.append(last_trading_day)

    if not selected_dates:
        # If no valid dates found, return original position
        return position

    # Get position values at selected dates
    pos_at_dates = pos_filled.loc[pos_filled.index.isin(selected_dates)]

    # Remove duplicates and sort
    pos_at_dates = pos_at_dates[~pos_at_dates.index.duplicated(keep='last')]
    pos_at_dates = pos_at_dates.sort_index()

    # Convert back to Polars
    resampled_dates = [str(d.date()) for d in pos_at_dates.index.tolist()]
    result_data = {date_col: resampled_dates}
    for col in stock_cols:
        result_data[col] = pos_at_dates[col].tolist()

    return pl.DataFrame(result_data)


def backtest(
    prices: pl.DataFrame,
    position: pl.DataFrame,
    resample: str = 'D',
    resample_offset: str | None = None,
    rebalance_indices: list[int] | None = None,
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
    """Run portfolio backtest simulation (T+1 execution, Finlab-compatible).

    This is the main backtest function that simulates a portfolio over time
    based on position signals or weights. Uses T+1 execution mode to match
    Finlab's behavior.

    T+1 Execution Model:
        - Signal on Day T
        - Trade executes at Day T's close price (entry price)
        - Position experiences Day T→T+1 return on Day T+1
        - Entry fee applied on Day T+1

    Args:
        prices: DataFrame with dates as index (first column) and stock prices as columns.
                Must be sorted by date in ascending order. Should use adjusted prices.
        position: DataFrame with rebalance dates as index and position signals/weights
                  as columns. Column names must match price columns.
                  - Boolean values: treated as equal-weight signals (True = hold)
                  - Float values: treated as custom weights (will be normalized)
        resample: Rebalance frequency. Supports:
                  - 'D': Daily (default, no resampling)
                  - 'W': Weekly (rebalance on last trading day of each week)
                  - 'W-FRI': Weekly on Friday
                  - 'W-MON': Weekly on Monday
                  - 'M': Monthly (rebalance on last trading day of each month)
                  - 'MS': Month start
                  - 'Q': Quarterly
                  - 'QS': Quarter start
                  - 'Y': Yearly
        resample_offset: Optional time offset to shift rebalance dates.
                        Examples: '1D' (shift 1 day forward), '-1D' (shift 1 day back)
        rebalance_indices: List of row indices in prices where rebalancing occurs.
                          If None, will be computed from position dates.
        fee_ratio: Transaction fee ratio (default: 0.001425 for Taiwan stocks)
        tax_ratio: Transaction tax ratio (default: 0.003 for Taiwan stocks)
        stop_loss: Stop loss threshold as decimal (e.g., 0.1 = 10% loss triggers exit).
                  Default 1.0 disables stop loss.
        take_profit: Take profit threshold as decimal. Default inf disables.
        trail_stop: Trailing stop threshold as decimal. Default inf disables.
        position_limit: Maximum weight per stock. Default 1.0.
        retain_cost_when_rebalance: If True, retain transaction costs when rebalancing.
                                   Default False.
        stop_trading_next_period: If True, stop trading the stock next period after
                                 stop loss/take profit triggers. Default True.
        finlab_mode: If True, use Finlab-compatible calculation mode. This exactly
                    replicates Finlab's backtest_core calculation for perfect matching.
                    Default False uses standard portfolio calculation.

    Returns:
        DataFrame with columns: date, creturn (cumulative return)

    Example:
        >>> # Boolean signals (equal weight)
        >>> prices = pl.DataFrame({
        ...     "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        ...     "AAPL": [100.0, 102.0, 105.0],
        ...     "GOOG": [200.0, 198.0, 202.0],
        ... })
        >>> position = pl.DataFrame({
        ...     "date": ["2024-01-01"],
        ...     "AAPL": [True],
        ...     "GOOG": [True],
        ... })
        >>> result = backtest(prices, position)

        >>> # Custom weights
        >>> position = pl.DataFrame({
        ...     "date": ["2024-01-01"],
        ...     "AAPL": [0.7],
        ...     "GOOG": [0.3],
        ... })
        >>> result = backtest(prices, position)
    """
    # Get the date column (first column)
    date_col = prices.columns[0]
    stock_cols = prices.columns[1:]
    price_dates = prices.select(date_col).to_series().to_list()

    # Ensure position has same stock columns
    position_stock_cols = [c for c in position.columns if c in stock_cols]
    if not position_stock_cols:
        raise ValueError("Position and prices must have common stock columns")

    # Apply resample if needed
    if resample != 'D':
        position = _resample_position(position, price_dates, resample, resample_offset)

    # Select only common stocks and reorder
    prices_data = prices.select(position_stock_cols)
    position_data = position.select(position_stock_cols)

    # Determine if signals (bool) or weights (float)
    first_col_dtype = position_data.dtypes[0]
    is_bool = first_col_dtype == pl.Boolean

    # Calculate rebalance indices if not provided
    if rebalance_indices is None:
        # Get position dates
        pos_date_col = position.columns[0]
        position_dates = position.select(pos_date_col).to_series().to_list()

        rebalance_indices = []
        for pd in position_dates:
            try:
                idx = price_dates.index(pd)
                rebalance_indices.append(idx)
            except ValueError:
                # Date not found, skip
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
        creturn = backtest_signals(prices_data, position_data, rebalance_indices, config)
    else:
        # Cast to float if needed
        position_data = position_data.cast(pl.Float64)
        creturn = backtest_weights(prices_data, position_data, rebalance_indices, config)

    # Build result DataFrame
    dates = prices.select(date_col).to_series()
    result = pl.DataFrame({
        date_col: dates,
        "creturn": creturn,
    })

    return result


class Report:
    """Backtest report with trades and statistics.

    This class matches Finlab's Report structure, providing:
    - creturn: Cumulative return series
    - position: Position weights DataFrame
    - trades: Trade records DataFrame with original prices

    Attributes:
        creturn: Polars Series with cumulative returns
        position: Polars DataFrame with position weights
        trades: Polars DataFrame with trade records
        fee_ratio: Transaction fee ratio used
        tax_ratio: Transaction tax ratio used
        dates: List of dates from price data
        stock_columns: List of stock column names
    """

    def __init__(
        self,
        creturn: list[float],
        trades: list[TradeRecord],
        dates: list,
        stock_columns: list[str],
        position: pl.DataFrame,
        fee_ratio: float,
        tax_ratio: float,
    ):
        """Initialize Report.

        Args:
            creturn: List of cumulative returns
            trades: List of TradeRecord objects from Rust
            dates: List of dates from price data
            stock_columns: List of stock column names
            position: Position weights DataFrame
            fee_ratio: Transaction fee ratio
            tax_ratio: Transaction tax ratio
        """
        self._creturn_list = creturn
        self._trades_raw = trades
        self._dates = dates
        self._stock_columns = stock_columns
        self._position = position
        self.fee_ratio = fee_ratio
        self.tax_ratio = tax_ratio

    @property
    def creturn(self) -> pl.DataFrame:
        """Cumulative return DataFrame with date column."""
        return pl.DataFrame({
            "date": self._dates,
            "creturn": self._creturn_list,
        })

    @property
    def position(self) -> pl.DataFrame:
        """Position weights DataFrame."""
        return self._position

    @property
    def trades(self) -> pl.DataFrame:
        """Trade records DataFrame matching Finlab's format.

        Columns:
            stock_id: Stock symbol
            entry_date: Actual entry date
            exit_date: Actual exit date
            entry_sig_date: Signal date for entry
            exit_sig_date: Signal date for exit
            position: Position weight
            period: Holding period in days
            return: Trade return (with fees)
            trade_price@entry_date: Entry price (ORIGINAL, not adjusted)
            trade_price@exit_date: Exit price (ORIGINAL, not adjusted)
        """
        if not self._trades_raw:
            return pl.DataFrame({
                "stock_id": [],
                "entry_date": [],
                "exit_date": [],
                "entry_sig_date": [],
                "exit_sig_date": [],
                "position": [],
                "period": [],
                "return": [],
                "trade_price@entry_date": [],
                "trade_price@exit_date": [],
            })

        # Convert TradeRecord objects to DataFrame
        records = []
        for t in self._trades_raw:
            stock_id = self._stock_columns[t.stock_id] if t.stock_id < len(self._stock_columns) else str(t.stock_id)
            entry_date = self._dates[t.entry_index] if t.entry_index < len(self._dates) else None
            exit_date = self._dates[t.exit_index] if t.exit_index is not None and t.exit_index < len(self._dates) else None
            entry_sig_date = self._dates[t.entry_sig_index] if t.entry_sig_index < len(self._dates) else None
            exit_sig_date = self._dates[t.exit_sig_index] if t.exit_sig_index is not None and t.exit_sig_index < len(self._dates) else None
            period = t.holding_period()

            records.append({
                "stock_id": stock_id,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_sig_date": entry_sig_date,
                "exit_sig_date": exit_sig_date,
                "position": t.position_weight,
                "period": period,
                "return": t.trade_return,
                "trade_price@entry_date": t.entry_price,
                "trade_price@exit_date": t.exit_price,
            })

        return pl.DataFrame(records)

    def __repr__(self) -> str:
        return f"Report(creturn_len={len(self._creturn_list)}, trades_count={len(self._trades_raw)})"


def backtest_with_report(
    close: pl.DataFrame,
    position: pl.DataFrame,
    resample: str = 'D',
    resample_offset: str | None = None,
    trade_at_price: str | pl.DataFrame = "close",
    open: pl.DataFrame | None = None,
    high: pl.DataFrame | None = None,
    low: pl.DataFrame | None = None,
    factor: pl.DataFrame | None = None,
    rebalance_indices: list[int] | None = None,
    fee_ratio: float = 0.001425,
    tax_ratio: float = 0.003,
    stop_loss: float = 1.0,
    take_profit: float = float("inf"),
    trail_stop: float = float("inf"),
    position_limit: float = 1.0,
    retain_cost_when_rebalance: bool = False,
    stop_trading_next_period: bool = True,
) -> Report:
    """Run backtest with trades tracking, returning a Report object.

    This function matches Finlab's backtest.sim() interface with full OHLC support.

    Key features:
        - Uses adjusted prices for return calculation (creturn)
        - Uses original prices for trade records (restored via factor)
        - Supports trade_at_price parameter like Finlab
        - Supports resample parameter like Finlab
        - Returns trades DataFrame matching Finlab's trades format

    Price handling:
        - close: Adjusted close prices (for return calculation)
        - trade_at_price: Determines which price to use for trading
            - 'close' (default): Use close prices
            - 'open': Use open prices
            - 'high': Use high prices
            - 'low': Use low prices
            - DataFrame: Use custom prices
        - factor: Adjustment factor to restore original prices from adjusted prices.
                 Original price = Adjusted price / factor
                 If None, trades record uses adjusted prices.

    Args:
        close: DataFrame with dates as index and adjusted close prices as columns.
               Used for cumulative return calculation.
        position: DataFrame with rebalance dates as index and position weights as columns.
        resample: Rebalance frequency. Supports:
                  - 'D': Daily (default, no resampling)
                  - 'W': Weekly (rebalance on last trading day of each week)
                  - 'W-FRI': Weekly on Friday
                  - 'W-MON': Weekly on Monday
                  - 'M': Monthly (rebalance on last trading day of each month)
                  - 'MS': Month start
                  - 'Q': Quarterly
                  - 'QS': Quarter start
                  - 'Y': Yearly
        resample_offset: Optional time offset to shift rebalance dates.
                        Examples: '1D' (shift 1 day forward), '-1D' (shift 1 day back)
        trade_at_price: Price type for trading ('close', 'open', 'high', 'low')
                       or a custom DataFrame with trading prices.
        open: DataFrame with open prices (optional, required if trade_at_price='open').
        high: DataFrame with high prices (optional, for stop loss/take profit).
        low: DataFrame with low prices (optional, for stop loss/take profit).
        factor: DataFrame with adjustment factors (same shape as close).
               Used to restore original prices: original = adjusted / factor.
               If None, trades record uses adjusted prices directly.
        rebalance_indices: List of row indices in prices where rebalancing occurs.
        fee_ratio: Transaction fee ratio (default: 0.001425 for Taiwan stocks)
        tax_ratio: Transaction tax ratio (default: 0.003 for Taiwan stocks)
        stop_loss: Stop loss threshold as decimal. Default 1.0 disables.
        take_profit: Take profit threshold as decimal. Default inf disables.
        trail_stop: Trailing stop threshold as decimal. Default inf disables.
        position_limit: Maximum weight per stock. Default 1.0.
        retain_cost_when_rebalance: If True, retain transaction costs when rebalancing.
        stop_trading_next_period: If True, stop trading the stock next period after
                                 stop loss/take profit triggers.

    Returns:
        Report object with creturn, position, and trades

    Example:
        >>> # Basic usage with adjusted close only (trades use adjusted prices)
        >>> report = backtest_with_report(adj_close, position)

        >>> # Monthly rebalancing
        >>> report = backtest_with_report(adj_close, position, resample='M')

        >>> # With factor to restore original prices for trades
        >>> report = backtest_with_report(
        ...     close=adj_close,
        ...     position=position,
        ...     factor=adj_factor,  # original = adj / factor
        ... )

        >>> # With trade_at_price='open' and factor
        >>> report = backtest_with_report(
        ...     close=adj_close,
        ...     position=position,
        ...     trade_at_price='open',
        ...     open=adj_open,
        ...     factor=adj_factor,
        ... )
    """
    # Resolve trade_at_price to a DataFrame
    if isinstance(trade_at_price, str):
        if trade_at_price == "close":
            trade_prices = close
        elif trade_at_price == "open":
            if open is None:
                raise ValueError("trade_at_price='open' requires 'open' DataFrame")
            trade_prices = open
        elif trade_at_price == "high":
            if high is None:
                raise ValueError("trade_at_price='high' requires 'high' DataFrame")
            trade_prices = high
        elif trade_at_price == "low":
            if low is None:
                raise ValueError("trade_at_price='low' requires 'low' DataFrame")
            trade_prices = low
        else:
            raise ValueError(f"Invalid trade_at_price: {trade_at_price}. "
                           "Must be 'close', 'open', 'high', 'low', or a DataFrame")
    else:
        trade_prices = trade_at_price

    # Calculate original prices for trade records
    # Original price = Adjusted price / factor
    if factor is not None:
        date_col = trade_prices.columns[0]
        stock_cols = trade_prices.columns[1:]
        factor_stock_cols = [c for c in factor.columns if c in stock_cols]

        # Divide adjusted prices by factor to get original prices
        original_data = {date_col: trade_prices[date_col]}
        for col in stock_cols:
            if col in factor_stock_cols:
                original_data[col] = trade_prices[col] / factor[col]
            else:
                # No factor for this column, use adjusted price
                original_data[col] = trade_prices[col]
        original_prices = pl.DataFrame(original_data)
    else:
        # No factor provided, use adjusted prices for trade records
        original_prices = trade_prices

    # Get the date column (first column)
    date_col = close.columns[0]
    stock_cols = close.columns[1:]

    # Get dates for mapping indices to dates
    dates = close.select(date_col).to_series().to_list()

    # Apply resample if needed
    if resample != 'D':
        position = _resample_position(position, dates, resample, resample_offset)

    # Ensure position has same stock columns
    position_stock_cols = [c for c in position.columns if c in stock_cols]
    if not position_stock_cols:
        raise ValueError("Position and prices must have common stock columns")

    # Select only common stocks and reorder
    close_data = close.select(position_stock_cols)
    original_prices_data = original_prices.select(position_stock_cols)

    # Calculate rebalance indices if not provided
    if rebalance_indices is None:
        # Get position dates
        pos_date_col = position.columns[0]
        position_dates = position.select(pos_date_col).to_series().to_list()

        rebalance_indices = []
        for pd in position_dates:
            try:
                idx = dates.index(pd)
                rebalance_indices.append(idx)
            except ValueError:
                # Date not found, skip
                pass

        if not rebalance_indices:
            raise ValueError("No matching dates between prices and position")

        # When rebalance_indices auto-calculated, position already has correct rows
        position_data = position.select(position_stock_cols)
    else:
        # When rebalance_indices provided, filter position to only those rows
        # This handles the case where position has all dates (forward-filled)
        # Rust expects weights indexed by rebalance event number, not time index
        position_data = position.select(position_stock_cols)[rebalance_indices]

    # Cast to float if needed
    position_data = position_data.cast(pl.Float64)

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
        finlab_mode=True,  # Use finlab mode for creturn calculation (matches Finlab perfectly)
    )

    # Run backtest with trades tracking
    # close_data: adjusted prices for return calculation
    # original_prices_data: original prices for trade records
    result = _backtest_with_trades(
        close_data,
        original_prices_data,
        position_data,
        rebalance_indices,
        config,
    )

    # Create Report object
    return Report(
        creturn=result.creturn,
        trades=result.trades,
        dates=dates,
        stock_columns=position_stock_cols,
        position=position,
        fee_ratio=fee_ratio,
        tax_ratio=tax_ratio,
    )
