"""Wide format backtest API.

This module provides backtest functions for wide format DataFrames
(dates as rows, stocks as columns). For most use cases, prefer the
long format API in namespace.py (df.bt.backtest).
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from polars_backtest._polars_backtest import (
    BacktestConfig,
    TradeRecord,
)
from polars_backtest._polars_backtest import (
    backtest_wide as _backtest_wide_rust,
)
from polars_backtest._polars_backtest import (
    backtest_with_report_wide_impl as _backtest_with_report_wide_impl,
)


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
    if resample == "D":
        return ("1d", None)

    # Weekly with anchor (W-MON, W-FRI, etc.)
    if resample.startswith("W-"):
        day_map = {"MON": 1, "TUE": 2, "WED": 3, "THU": 4, "FRI": 5, "SAT": 6, "SUN": 7}
        anchor = resample[2:]
        if anchor not in day_map:
            raise ValueError(f"Invalid weekly anchor: {anchor}")
        return ("1w", day_map[anchor])

    # Weekly (default Sunday end, like pandas)
    if resample == "W":
        return ("1w", 7)  # Sunday

    # Monthly (end of month)
    if resample in ("M", "ME", "BM", "SM", "CBM"):
        return ("1mo", None)

    # Monthly (start of month)
    if resample == "MS":
        return ("1mo_start", None)

    # Quarterly (end of quarter)
    if resample in ("Q", "QE", "BQ"):
        return ("3mo", None)

    # Quarterly (start of quarter)
    if resample == "QS":
        return ("3mo_start", None)

    # Yearly (end of year)
    if resample in ("A", "Y", "YE", "BY"):
        return ("1y", None)

    # Yearly (start of year)
    if resample in ("AS", "YS"):
        return ("1y_start", None)

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
    match = re.match(r"^(-)?(\d+)([DWHMST])$", offset_str.upper())
    if not match:
        raise ValueError(f"Invalid offset format: {offset_str}")

    sign = -1 if match.group(1) else 1
    value = int(match.group(2))
    unit = match.group(3)

    if unit == "D":
        return timedelta(days=sign * value)
    elif unit == "W":
        return timedelta(weeks=sign * value)
    elif unit == "H":
        return timedelta(hours=sign * value)
    elif unit == "M":
        return timedelta(minutes=sign * value)
    elif unit == "S":
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

    if freq == "1w":
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

    elif freq == "1mo":
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

    elif freq == "1mo_start":
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

    elif freq == "3mo":
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

    elif freq == "3mo_start":
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

    elif freq == "1y":
        # Yearly end: December 31st
        current_year = start_date.year
        while True:
            year_end = date(current_year, 12, 31)
            if year_end > end_date:
                break
            if year_end >= start_date:
                result_dates.append(year_end)
            current_year += 1

    elif freq == "1y_start":
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
    if resample == "D":
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
            return d.date() if hasattr(d, "date") else date.fromisoformat(str(d)[:10])

    all_dates = [to_date(d) for d in price_dates]

    # Create all_dates DataFrame for joining
    all_dates_df = pl.DataFrame({date_col: [str(d) for d in all_dates]})

    # Forward fill position to all trading dates
    pos_filled = all_dates_df.join(position, on=date_col, how="left").with_columns(
        [pl.col(col).forward_fill() for col in stock_cols]
    )

    # Generate rebalance dates
    start_date = all_dates[0]
    end_date = all_dates[-1]

    # Extend end_date by one period to include upcoming rebalance date (Finlab behavior)
    if freq == "1w":
        extended_end = end_date + timedelta(weeks=1)
    elif freq == "1mo" or freq == "1mo_start":
        # Add approximately one month
        if end_date.month == 12:
            extended_end = date(end_date.year + 1, 1, end_date.day)
        else:
            try:
                extended_end = date(end_date.year, end_date.month + 1, end_date.day)
            except ValueError:
                # Handle day overflow (e.g., Jan 31 -> Feb 28)
                extended_end = date(end_date.year, end_date.month + 2, 1) - timedelta(days=1)
    elif freq == "3mo" or freq == "3mo_start":
        # Add approximately one quarter
        new_month = end_date.month + 3
        new_year = end_date.year + (new_month - 1) // 12
        new_month = ((new_month - 1) % 12) + 1
        try:
            extended_end = date(new_year, new_month, end_date.day)
        except ValueError:
            extended_end = date(new_year, new_month + 1, 1) - timedelta(days=1)
    elif freq == "1y" or freq == "1y_start":
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
    pos_at_dates = pos_at_dates.unique(subset=[date_col], keep="last").sort(date_col)

    return pos_at_dates


def _filter_changed_positions(position: pl.DataFrame) -> pl.DataFrame:
    """Filter position DataFrame to only rows where position changed.

    This implements Finlab's resample=None behavior:
    - Only rebalance when portfolio composition changes
    - Always include the first row if it has any non-null values

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
    diff_exprs = [
        pl.col(col).diff().abs().fill_null(0.0).alias(f"_diff_{col}") for col in stock_cols
    ]

    # Add diff columns
    with_diff = position_float.with_columns(diff_exprs)

    # Sum of absolute diffs across all stocks
    diff_cols = [f"_diff_{col}" for col in stock_cols]
    with_diff = with_diff.with_columns(pl.sum_horizontal(diff_cols).alias("_diff_sum"))

    # First row should always be included if it has any non-null values
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
    result = (
        position.with_row_index("_row_idx")
        .filter(pl.col("_row_idx").is_in(kept_indices))
        .drop("_row_idx")
    )

    return result


def backtest_wide(
    prices: pl.DataFrame,
    position: pl.DataFrame,
    resample: str | None = "D",
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
    """Run portfolio backtest simulation with wide format data.

    Args:
        prices: DataFrame with dates as index (first column) and stock prices as columns.
        position: DataFrame with rebalance dates as index and position signals/weights.
        resample: Rebalance frequency ('D', 'W', 'M', 'Q', 'Y', None).
        resample_offset: Optional offset for rebalance dates.
        rebalance_indices: List of row indices where rebalancing occurs.
        fee_ratio: Transaction fee ratio.
        tax_ratio: Transaction tax ratio.
        stop_loss: Stop loss threshold.
        take_profit: Take profit threshold.
        trail_stop: Trailing stop threshold.
        position_limit: Maximum weight per stock.
        retain_cost_when_rebalance: Retain costs when rebalancing.
        stop_trading_next_period: Stop trading after stop triggered.
        finlab_mode: Use Finlab-compatible calculation.

    Returns:
        DataFrame with columns: date, creturn
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
        position = _filter_changed_positions(position)
    elif resample != "D":
        position = _resample_position(position, price_dates, resample, resample_offset)

    # Select only common stocks and reorder
    prices_data = prices.select(position_stock_cols)
    position_data = position.select(position_stock_cols)

    # Determine if signals (bool) or weights (float)
    first_col_dtype = position_data.dtypes[0]
    is_bool = first_col_dtype == pl.Boolean

    # Calculate rebalance indices if not provided
    if rebalance_indices is None:
        pos_date_col = position.columns[0]
        position_dates = position.select(pos_date_col).to_series().to_list()

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

    # Run backtest (convert bool to equal weights if needed)
    if is_bool:
        position_data = position_data.cast(pl.Float64)
    else:
        position_data = position_data.cast(pl.Float64)

    creturn = _backtest_wide_rust(prices_data, position_data, rebalance_indices, config)

    # Build result DataFrame
    dates = prices.select(date_col).to_series()
    result = pl.DataFrame(
        {
            date_col: dates,
            "creturn": creturn,
        }
    )

    return result


class Report:
    """Backtest report with trades and statistics.

    Attributes:
        creturn: Polars DataFrame with cumulative returns
        position: Polars DataFrame with position weights
        trades: Polars DataFrame with trade records
        fee_ratio: Transaction fee ratio used
        tax_ratio: Transaction tax ratio used
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
        """Initialize Report."""
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
        """Cumulative return DataFrame with date column."""
        return pl.DataFrame(
            {
                "date": self._dates[self._first_signal_index :],
                "creturn": self._creturn_list[self._first_signal_index :],
            }
        )

    @property
    def position(self) -> pl.DataFrame:
        """Position weights DataFrame."""
        return self._position

    @property
    def trades(self) -> pl.DataFrame:
        """Trade records DataFrame."""
        if not self._trades_raw:
            return pl.DataFrame(
                {
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
                }
            )

        records = []
        for t in self._trades_raw:
            stock_id = (
                self._stock_columns[t.stock_id]
                if t.stock_id < len(self._stock_columns)
                else str(t.stock_id)
            )
            entry_date = (
                self._dates[t.entry_index]
                if t.entry_index is not None and t.entry_index < len(self._dates)
                else None
            )
            exit_date = (
                self._dates[t.exit_index]
                if t.exit_index is not None and t.exit_index < len(self._dates)
                else None
            )
            entry_sig_date = (
                self._dates[t.entry_sig_index] if t.entry_sig_index < len(self._dates) else None
            )
            exit_sig_date = (
                self._dates[t.exit_sig_index]
                if t.exit_sig_index is not None and t.exit_sig_index < len(self._dates)
                else None
            )
            period = t.holding_period()

            records.append(
                {
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
                }
            )

        return pl.DataFrame(records)

    def __repr__(self) -> str:
        return (
            f"Report(creturn_len={len(self._creturn_list)}, trades_count={len(self._trades_raw)})"
        )


def backtest_with_report_wide(
    close: pl.DataFrame,
    position: pl.DataFrame,
    resample: str | None = "D",
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
    """Run backtest with trades tracking on wide format data.

    Args:
        close: DataFrame with adjusted close prices.
        position: DataFrame with position weights.
        resample: Rebalance frequency.
        resample_offset: Optional offset for rebalance dates.
        trade_at_price: Price type for trading ('close', 'open', 'high', 'low').
        open: DataFrame with open prices.
        high: DataFrame with high prices.
        low: DataFrame with low prices.
        factor: DataFrame with adjustment factors.
        rebalance_indices: List of row indices where rebalancing occurs.
        fee_ratio: Transaction fee ratio.
        tax_ratio: Transaction tax ratio.
        stop_loss: Stop loss threshold.
        take_profit: Take profit threshold.
        trail_stop: Trailing stop threshold.
        position_limit: Maximum weight per stock.
        retain_cost_when_rebalance: Retain costs when rebalancing.
        stop_trading_next_period: Stop trading after stop triggered.
        touched_exit: Use OHLC for intraday stop detection.

    Returns:
        Report object with creturn, position, and trades
    """
    # Check for touched_exit requirements
    if touched_exit:
        if open is None or high is None or low is None:
            raise ValueError(
                "touched_exit=True requires open, high, and low price DataFrames."
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
            raise ValueError(f"Invalid trade_at_price: {trade_at_price}")
    else:
        trade_prices = trade_at_price

    # Calculate original prices for trade records
    if factor is not None:
        date_col = trade_prices.columns[0]
        stock_cols = trade_prices.columns[1:]
        factor_stock_cols = [c for c in factor.columns if c in stock_cols]

        original_data = {date_col: trade_prices[date_col]}
        for col in stock_cols:
            if col in factor_stock_cols:
                original_data[col] = trade_prices[col] / factor[col]
            else:
                original_data[col] = trade_prices[col]
        original_prices = pl.DataFrame(original_data)
    else:
        original_prices = trade_prices

    # Get the date column (first column)
    date_col = close.columns[0]
    stock_cols = close.columns[1:]

    # Get dates for mapping indices to dates
    dates = close.select(date_col).to_series().to_list()

    # Apply resample if needed
    if resample is None:
        position = _filter_changed_positions(position)
    elif resample != "D":
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
        pos_date_col = position.columns[0]
        position_dates = position.select(pos_date_col).to_series().to_list()

        first_idx = None
        for pos_d in position_dates:
            try:
                first_idx = dates.index(pos_d)
                break
            except ValueError:
                pass

        if first_idx is None:
            raise ValueError("No matching dates between prices and position")

        if resample == "D":
            rebalance_indices = list(range(first_idx, len(dates)))

            all_dates_df = pl.DataFrame({pos_date_col: dates})
            position_expanded = (
                all_dates_df.join(position, on=pos_date_col, how="left")
                .select([pos_date_col] + position_stock_cols)
                .with_columns([pl.col(col).forward_fill() for col in position_stock_cols])
            )
            position_expanded = position_expanded.slice(first_idx)
            position_data = position_expanded.select(position_stock_cols)
        else:
            rebalance_indices = []
            unmatched_position_indices = []
            for i, pos_d in enumerate(position_dates):
                try:
                    idx = dates.index(pos_d)
                    rebalance_indices.append(idx)
                except ValueError:
                    unmatched_position_indices.append(i)

            if not rebalance_indices:
                raise ValueError("No matching dates between prices and position")

            if unmatched_position_indices:
                last_matched_pos_idx = (
                    max(
                        i
                        for i, pos_d in enumerate(position_dates)
                        if pos_d in [dates[idx] for idx in rebalance_indices]
                    )
                    if rebalance_indices
                    else -1
                )

                for unmatched_idx in unmatched_position_indices:
                    if unmatched_idx > last_matched_pos_idx:
                        last_price_idx = len(dates) - 1
                        if last_price_idx not in rebalance_indices:
                            rebalance_indices.append(last_price_idx)

            position_data = position.select(position_stock_cols)
    else:
        position_data = position.select(position_stock_cols)[rebalance_indices]

    # Cast to float if needed
    position_data = position_data.cast(pl.Float64)

    # Find the first rebalance with any non-zero signals
    first_signal_rebalance_idx = 0
    for i in range(len(position_data)):
        row = position_data[i]
        has_signal = any(
            row[col][0] is not None and row[col][0] > 0 for col in position_data.columns
        )
        if has_signal:
            first_signal_rebalance_idx = i
            break

    # Calculate first_signal_index
    if rebalance_indices:
        signal_day_index = rebalance_indices[first_signal_rebalance_idx]
        if signal_day_index == 0:
            first_signal_index = 0
        elif fee_ratio > 0 or tax_ratio > 0:
            first_signal_index = signal_day_index
        else:
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
        finlab_mode=True,
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

    # Run backtest with report
    result = _backtest_with_report_wide_impl(
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
