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


from datetime import date, timedelta


def _parse_resample_freq(resample: str) -> tuple[str, int | None]:
    """Parse pandas-style resample frequency to polars interval format.

    Args:
        resample: Pandas-style frequency string like 'D', 'W', 'W-FRI', 'M', 'Q', 'Y'

    Returns:
        Tuple of (polars_interval, weekday) where weekday is 1-7 (Mon-Sun) for weekly anchors.

    Raises:
        ValueError: If the frequency is not recognized.
    """
    resample = resample.upper()

    # Daily
    if resample == 'D':
        return ('1d', None)

    # Weekly with anchor (W-MON, W-FRI, etc.)
    if resample.startswith('W-'):
        day_map = {'MON': 1, 'TUE': 2, 'WED': 3, 'THU': 4, 'FRI': 5, 'SAT': 6, 'SUN': 7}
        anchor = resample[2:]
        if anchor not in day_map:
            raise ValueError(f"Invalid weekly anchor: {anchor}")
        return ('1w', day_map[anchor])

    # Weekly (default Sunday end, like pandas)
    if resample == 'W':
        return ('1w', 7)  # Sunday

    # Monthly (end of month)
    if resample in ('M', 'ME', 'BM', 'SM', 'CBM'):
        return ('1mo', None)

    # Monthly (start of month)
    if resample == 'MS':
        return ('1mo_start', None)

    # Quarterly (end of quarter)
    if resample in ('Q', 'QE', 'BQ'):
        return ('3mo', None)

    # Quarterly (start of quarter)
    if resample == 'QS':
        return ('3mo_start', None)

    # Yearly (end of year)
    if resample in ('A', 'Y', 'YE', 'BY'):
        return ('1y', None)

    # Yearly (start of year)
    if resample in ('AS', 'YS'):
        return ('1y_start', None)

    raise ValueError(f"Invalid resample frequency: {resample}")


def _parse_offset(offset_str: str) -> timedelta:
    """Parse pandas-style offset string to timedelta.

    Args:
        offset_str: Offset string like '1D', '-1D', '2W', '1M' (only D/W supported for offset)

    Returns:
        timedelta object

    Raises:
        ValueError: If the offset format is not recognized.
    """
    import re

    if not offset_str:
        return timedelta(0)

    # Parse format: optional sign, number, unit
    match = re.match(r'^(-)?(\d+)([DWHMST])$', offset_str.upper())
    if not match:
        raise ValueError(f"Invalid offset format: {offset_str}")

    sign = -1 if match.group(1) else 1
    value = int(match.group(2))
    unit = match.group(3)

    if unit == 'D':
        return timedelta(days=sign * value)
    elif unit == 'W':
        return timedelta(weeks=sign * value)
    elif unit == 'H':
        return timedelta(hours=sign * value)
    elif unit == 'M':
        return timedelta(minutes=sign * value)
    elif unit == 'S':
        return timedelta(seconds=sign * value)
    else:
        raise ValueError(f"Unsupported offset unit: {unit}")


def _get_period_end_dates(
    start_date: date,
    end_date: date,
    freq: str,
    weekday: int | None = None,
) -> list[date]:
    """Generate period-end dates between start and end dates using polars.

    Args:
        start_date: Start date
        end_date: End date
        freq: Polars-style frequency ('1w', '1mo', '3mo', '1y', etc.)
        weekday: For weekly frequency, which day ends the week (1=Mon, 7=Sun)

    Returns:
        List of period-end dates
    """
    result_dates = []

    if freq == '1w':
        # Weekly: find all specified weekdays between start and end
        # weekday is 1-7 (Mon-Sun)
        current = start_date
        target_weekday = weekday if weekday else 7  # Default Sunday

        # Find first occurrence of target weekday
        days_ahead = target_weekday - current.isoweekday()
        if days_ahead < 0:
            days_ahead += 7
        current = current + timedelta(days=days_ahead)

        while current <= end_date:
            result_dates.append(current)
            current = current + timedelta(weeks=1)

    elif freq == '1mo':
        # Monthly end: last day of each month
        current = start_date
        while current <= end_date:
            # Find last day of current month
            if current.month == 12:
                next_month_start = date(current.year + 1, 1, 1)
            else:
                next_month_start = date(current.year, current.month + 1, 1)
            month_end = next_month_start - timedelta(days=1)

            if month_end >= start_date:
                result_dates.append(month_end)

            # Move to next month
            current = next_month_start

    elif freq == '1mo_start':
        # Monthly start: first day of each month
        current = date(start_date.year, start_date.month, 1)
        while current <= end_date:
            if current >= start_date:
                result_dates.append(current)
            # Move to next month
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)

    elif freq == '3mo':
        # Quarterly end: last day of March, June, September, December
        quarter_end_months = [3, 6, 9, 12]
        current = start_date
        while current <= end_date:
            # Find next quarter end
            for qm in quarter_end_months:
                if current.month <= qm:
                    # Calculate last day of quarter month
                    if qm == 12:
                        q_end = date(current.year, 12, 31)
                    else:
                        next_month_start = date(current.year, qm + 1, 1)
                        q_end = next_month_start - timedelta(days=1)

                    if q_end >= start_date and q_end <= end_date:
                        if q_end not in result_dates:
                            result_dates.append(q_end)
                    break
            # Move to next quarter
            if current.month >= 10:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, ((current.month - 1) // 3 + 1) * 3 + 1, 1)

    elif freq == '3mo_start':
        # Quarterly start: first day of January, April, July, October
        quarter_start_months = [1, 4, 7, 10]
        current = start_date
        while current <= end_date:
            for qm in quarter_start_months:
                q_start = date(current.year, qm, 1)
                if q_start >= start_date and q_start <= end_date:
                    if q_start not in result_dates:
                        result_dates.append(q_start)
            # Move to next year
            current = date(current.year + 1, 1, 1)

    elif freq == '1y':
        # Yearly end: December 31st
        current_year = start_date.year
        while True:
            year_end = date(current_year, 12, 31)
            if year_end > end_date:
                break
            if year_end >= start_date:
                result_dates.append(year_end)
            current_year += 1

    elif freq == '1y_start':
        # Yearly start: January 1st
        current_year = start_date.year
        while True:
            year_start = date(current_year, 1, 1)
            if year_start > end_date:
                break
            if year_start >= start_date:
                result_dates.append(year_start)
            current_year += 1

    else:
        raise ValueError(f"Unsupported frequency: {freq}")

    return sorted(set(result_dates))

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
    "BacktestNamespace",
    # Main API (long format)
    "backtest",
    "backtest_with_report",
    # Wide format API (for finlab compatibility)
    "backtest_wide",
    "backtest_with_report_wide",
    # Low-level functions
    "backtest_signals",
    "backtest_weights",
    # Statistics expressions
    "daily_returns",
    "cumulative_returns",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "drawdown_series",
    "portfolio_return",
    "equal_weights",
    # Helper functions
    "_parse_resample_freq",
    "_parse_offset",
    "_get_period_end_dates",
    "_filter_changed_positions",
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
    if resample == 'D':
        return position

    date_col = position.columns[0]
    stock_cols = position.columns[1:]

    # Parse resample frequency
    try:
        freq, weekday = _parse_resample_freq(resample)
    except ValueError as e:
        raise ValueError(f"Invalid resample frequency: {resample}. Error: {e}")

    # Convert price_dates to date objects for comparison
    def to_date(d) -> date:
        if isinstance(d, date):
            return d
        elif isinstance(d, str):
            return date.fromisoformat(d)
        else:
            # Assume datetime-like
            return d.date() if hasattr(d, 'date') else date.fromisoformat(str(d)[:10])

    all_dates = [to_date(d) for d in price_dates]

    # Create all_dates DataFrame for joining
    all_dates_df = pl.DataFrame({date_col: [str(d) for d in all_dates]})

    # Forward fill position to all trading dates
    pos_filled = (
        all_dates_df
        .join(position, on=date_col, how="left")
        .with_columns([
            pl.col(col).forward_fill() for col in stock_cols
        ])
    )

    # Generate rebalance dates
    start_date = all_dates[0]
    end_date = all_dates[-1]

    # Extend end_date by one period to include upcoming rebalance date (Finlab behavior)
    if freq == '1w':
        extended_end = end_date + timedelta(weeks=1)
    elif freq == '1mo' or freq == '1mo_start':
        # Add approximately one month
        if end_date.month == 12:
            extended_end = date(end_date.year + 1, 1, end_date.day)
        else:
            try:
                extended_end = date(end_date.year, end_date.month + 1, end_date.day)
            except ValueError:
                # Handle day overflow (e.g., Jan 31 -> Feb 28)
                extended_end = date(end_date.year, end_date.month + 2, 1) - timedelta(days=1)
    elif freq == '3mo' or freq == '3mo_start':
        # Add approximately one quarter
        new_month = end_date.month + 3
        new_year = end_date.year + (new_month - 1) // 12
        new_month = ((new_month - 1) % 12) + 1
        try:
            extended_end = date(new_year, new_month, end_date.day)
        except ValueError:
            extended_end = date(new_year, new_month + 1, 1) - timedelta(days=1)
    elif freq == '1y' or freq == '1y_start':
        extended_end = date(end_date.year + 1, end_date.month, end_date.day)
    else:
        extended_end = end_date

    # Get period-end dates
    rebalance_dates = _get_period_end_dates(start_date, extended_end, freq, weekday)

    # Apply resample_offset if provided
    if resample_offset is not None:
        try:
            offset = _parse_offset(resample_offset)
            rebalance_dates = [d + offset for d in rebalance_dates]
        except ValueError as e:
            raise ValueError(f"Invalid resample_offset: {resample_offset}. Error: {e}")

    # Filter to valid trading dates and find the last trading day <= each rebalance date
    all_dates_set = set(all_dates)
    selected_dates = []

    for rebal_date in rebalance_dates:
        # Find the last trading day that is <= rebalance date
        valid_dates = [d for d in all_dates if d <= rebal_date]
        if valid_dates:
            last_trading_day = max(valid_dates)
            if last_trading_day not in selected_dates:
                selected_dates.append(last_trading_day)

    if not selected_dates:
        # If no valid dates found, return original position
        return position

    # Sort selected dates
    selected_dates = sorted(selected_dates)

    # Convert selected_dates to strings for filtering
    selected_date_strs = [str(d) for d in selected_dates]

    # Filter pos_filled to selected dates
    pos_at_dates = pos_filled.filter(pl.col(date_col).is_in(selected_date_strs))

    # Remove duplicates and ensure sorted
    pos_at_dates = (
        pos_at_dates
        .unique(subset=[date_col], keep="last")
        .sort(date_col)
    )

    return pos_at_dates


def _filter_changed_positions(position: pl.DataFrame) -> pl.DataFrame:
    """Filter position DataFrame to only rows where position changed.

    This implements Finlab's resample=None behavior:
    - Only rebalance when portfolio composition changes
    - Always include the first row if it has any non-null values

    Finlab logic (backtest.py lines 574-580):
        change = (position.diff().abs().sum(axis=1) != 0) | (
            (position.index == position.index[0]) & position.iloc[0].notna().any())
        position = position.loc[change]

    Args:
        position: Position DataFrame with date column and stock columns.

    Returns:
        Filtered position DataFrame with only changed rows.
    """
    if position.height <= 1:
        return position

    date_col = position.columns[0]
    stock_cols = position.columns[1:]

    # Cast boolean columns to float for diff calculation
    # Finlab does: position.astype(float).fillna(0)
    cast_exprs = []
    for col in stock_cols:
        dtype = position[col].dtype
        if dtype == pl.Boolean:
            cast_exprs.append(pl.col(col).cast(pl.Float64).alias(col))
        elif dtype not in [pl.Float32, pl.Float64]:
            cast_exprs.append(pl.col(col).cast(pl.Float64).alias(col))

    if cast_exprs:
        position_float = position.with_columns(cast_exprs)
    else:
        position_float = position

    # Calculate diff for each stock column
    # diff().abs().sum(axis=1) != 0 means position changed
    diff_exprs = [
        pl.col(col).diff().abs().fill_null(0.0).alias(f"_diff_{col}")
        for col in stock_cols
    ]

    # Add diff columns
    with_diff = position_float.with_columns(diff_exprs)

    # Sum of absolute diffs across all stocks
    diff_cols = [f"_diff_{col}" for col in stock_cols]
    with_diff = with_diff.with_columns(
        pl.sum_horizontal(diff_cols).alias("_diff_sum")
    )

    # First row should always be included if it has any non-null values
    # Check if first row has any non-zero/non-null values
    first_row_has_values = False
    if position.height > 0:
        first_row = position.row(0)
        for i, col in enumerate(position.columns):
            if col != date_col:
                val = first_row[i]
                if val is not None and val != 0:
                    first_row_has_values = True
                    break

    # Create mask: diff_sum != 0 OR first row
    with_diff = with_diff.with_row_index("_row_idx")
    if first_row_has_values:
        mask = (pl.col("_diff_sum") != 0) | (pl.col("_row_idx") == 0)
    else:
        mask = pl.col("_diff_sum") != 0

    # Get the row indices to keep
    kept_indices = with_diff.filter(mask).get_column("_row_idx").to_list()

    # Filter original position using the indices (preserves original types)
    result = position.with_row_index("_row_idx").filter(
        pl.col("_row_idx").is_in(kept_indices)
    ).drop("_row_idx")

    return result


def backtest_wide(
    prices: pl.DataFrame,
    position: pl.DataFrame,
    resample: str | None = 'D',
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
    """Run portfolio backtest simulation with wide format data (T+1 execution, Finlab-compatible).

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
                  - None: Only rebalance when position changes (Finlab default)
                  - 'D': Daily (rebalance every day)
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
    if resample is None:
        # resample=None: Only rebalance when position changes (Finlab behavior)
        position = _filter_changed_positions(position)
    elif resample != 'D':
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
        for pos_d in position_dates:
            try:
                idx = price_dates.index(pos_d)
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
        first_signal_index: int = 0,
    ):
        """Initialize Report.

        Args:
            creturn: List of cumulative returns
            trades: List of TradeRecord objects from Rust
            dates: List of dates from price data
            stock_columns: List of stock column names
            position: Position weights DataFrame (can be long or wide format)
            fee_ratio: Transaction fee ratio
            tax_ratio: Transaction tax ratio
            first_signal_index: Index of first date with signals (for Finlab compatibility)
        """
        self._creturn_list = creturn
        self._trades_raw = trades
        self._dates = dates
        self._stock_columns = stock_columns
        self._position = position
        self.fee_ratio = fee_ratio
        self.tax_ratio = tax_ratio
        self._first_signal_index = first_signal_index

    @property
    def creturn(self) -> pl.DataFrame:
        """Cumulative return DataFrame with date column.

        Like Finlab, starts from the first date with signals.
        """
        return pl.DataFrame({
            "date": self._dates[self._first_signal_index:],
            "creturn": self._creturn_list[self._first_signal_index:],
        })

    @property
    def position(self) -> pl.DataFrame:
        """Position weights DataFrame (can be long or wide format)."""
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
            # entry_index can be None for pending entries (signals not yet executed)
            entry_date = self._dates[t.entry_index] if t.entry_index is not None and t.entry_index < len(self._dates) else None
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


def backtest_with_report_wide(
    close: pl.DataFrame,
    position: pl.DataFrame,
    resample: str | None = 'D',
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
    touched_exit: bool = False,
) -> Report:
    """Run backtest with trades tracking on wide format data, returning a Report object.

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
                  - None: Only rebalance when position changes (Finlab default)
                  - 'D': Daily (rebalance every day)
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
        touched_exit: If True, use OHLC prices for intraday stop detection.
                     When high/low prices touch stop_loss/take_profit thresholds,
                     exit at the touched price rather than waiting for close.
                     NOTE: NOT YET IMPLEMENTED - will raise NotImplementedError.

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
    # Check for touched_exit requirements
    if touched_exit:
        if open is None or high is None or low is None:
            raise ValueError(
                "touched_exit=True requires open, high, and low price DataFrames. "
                "Please provide all OHLC prices for intraday stop detection."
            )

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
    if resample is None:
        # resample=None: Only rebalance when position changes (Finlab behavior)
        position = _filter_changed_positions(position)
    elif resample != 'D':
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

        # Find first position date index
        first_idx = None
        for pos_d in position_dates:
            try:
                first_idx = dates.index(pos_d)
                break
            except ValueError:
                pass

        if first_idx is None:
            raise ValueError("No matching dates between prices and position")

        if resample == 'D':
            # resample='D' means daily rebalance (like Finlab)
            # Every day from first position date is a rebalance point
            rebalance_indices = list(range(first_idx, len(dates)))

            # Expand position to all dates by forward-filling using Polars
            # Create a DataFrame with all dates
            all_dates_df = pl.DataFrame({pos_date_col: dates})

            # Join and forward fill
            position_expanded = (
                all_dates_df
                .join(position, on=pos_date_col, how="left")
                .select([pos_date_col] + position_stock_cols)
                .with_columns([
                    pl.col(col).forward_fill() for col in position_stock_cols
                ])
            )

            # Take only from first_idx onwards
            position_expanded = position_expanded.slice(first_idx)

            # Select only stock columns for position_data
            position_data = position_expanded.select(position_stock_cols)
        else:
            # Non-daily resample: only rebalance on position dates
            rebalance_indices = []
            unmatched_position_indices = []  # Track position indices without matching price dates
            for i, pos_d in enumerate(position_dates):
                try:
                    idx = dates.index(pos_d)
                    rebalance_indices.append(idx)
                except ValueError:
                    # Date not found - track it for pending entries handling
                    unmatched_position_indices.append(i)

            if not rebalance_indices:
                raise ValueError("No matching dates between prices and position")

            # Handle unmatched position dates (Finlab behavior: add as pending entries)
            # If there are position dates after all matched dates (e.g., next month-end),
            # add the last price date as a rebalance point so the signal becomes pending
            if unmatched_position_indices:
                # Check if unmatched dates are AFTER all matched dates
                last_matched_pos_idx = max(
                    i for i, pos_d in enumerate(position_dates)
                    if pos_d in [dates[idx] for idx in rebalance_indices]
                ) if rebalance_indices else -1

                for unmatched_idx in unmatched_position_indices:
                    if unmatched_idx > last_matched_pos_idx:
                        # This is a future position date - add last price date as rebalance point
                        # This will make the signal pending at end of simulation
                        last_price_idx = len(dates) - 1
                        if last_price_idx not in rebalance_indices:
                            rebalance_indices.append(last_price_idx)

            # When rebalance_indices auto-calculated, position already has correct rows
            position_data = position.select(position_stock_cols)
    else:
        # When rebalance_indices provided, filter position to only those rows
        # This handles the case where position has all dates (forward-filled)
        # Rust expects weights indexed by rebalance event number, not time index
        position_data = position.select(position_stock_cols)[rebalance_indices]

    # Cast to float if needed
    position_data = position_data.cast(pl.Float64)

    # Find the first rebalance with any non-zero signals (like Finlab)
    # Finlab behavior:
    # - If first signal is on first day of data: include that day (creturn = 1.0)
    # - If first signal is after first day: start from T+1 execution day
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

    # Calculate first_signal_index based on Finlab behavior
    # Finlab behavior depends on fees:
    # - With fees: start from signal day (creturn = 1.0 on signal day)
    # - Without fees: start from T+1 execution day (creturn = 1.0 on execution day)
    if rebalance_indices:
        signal_day_index = rebalance_indices[first_signal_rebalance_idx]
        if signal_day_index == 0:
            # First signal is on first day of data - include that day
            first_signal_index = 0
        elif fee_ratio > 0 or tax_ratio > 0:
            # With fees: start from signal day (like Finlab)
            first_signal_index = signal_day_index
        else:
            # Without fees: start from T+1 execution day
            first_signal_index = signal_day_index + 1
    else:
        first_signal_index = 0

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
        touched_exit=touched_exit,
    )

    # Prepare OHLC data for touched_exit mode
    open_data = None
    high_data = None
    low_data = None
    if touched_exit and open is not None and high is not None and low is not None:
        open_data = open.select(position_stock_cols)
        high_data = high.select(position_stock_cols)
        low_data = low.select(position_stock_cols)

    # Run backtest with trades tracking
    # close_data: adjusted prices for return calculation
    # original_prices_data: original prices for trade records
    result = _backtest_with_trades(
        close_data,
        original_prices_data,
        position_data,
        rebalance_indices,
        config,
        open_data,
        high_data,
        low_data,
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
        first_signal_index=first_signal_index,
    )


# Register DataFrame namespace (df.bt) and import main API functions
# Import at the end to avoid circular imports
from polars_backtest.namespace import (
    BacktestNamespace,
    backtest,
    backtest_with_report,
)
