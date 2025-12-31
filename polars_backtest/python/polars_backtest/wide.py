"""Wide format backtest API.

This module provides backtest functions for wide format DataFrames
(dates as rows, stocks as columns). For most use cases, prefer the
long format API in namespace.py (df.bt.backtest).
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import cached_property
from typing import Literal

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
from polars_backtest.utils import (
    _get_period_end_dates,
    _parse_offset,
    _parse_resample_freq,
)

# Metrics section types
MetricsSection = Literal["backtest", "profitability", "risk", "ratio", "winrate"]
ALL_SECTIONS: list[MetricsSection] = [
    "backtest", "profitability", "risk", "ratio", "winrate"
]

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
        # Additional backtest parameters for get_metrics()
        resample: str | None = None,
        trade_at: str = "close",
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trail_stop: float | None = None,
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
        # Store backtest config
        self.resample = resample
        self.trade_at = trade_at
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trail_stop = trail_stop

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
                    "mae": [],
                    "gmfe": [],
                    "bmfe": [],
                    "mdd": [],
                    "pdays": [],
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
                    "mae": t.mae,
                    "gmfe": t.gmfe,
                    "bmfe": t.bmfe,
                    "mdd": t.mdd,
                    "pdays": t.pdays,
                }
            )

        return pl.DataFrame(records)

    def __repr__(self) -> str:
        return (
            f"Report(creturn_len={len(self._creturn_list)}, trades_count={len(self._trades_raw)})"
        )

    @cached_property
    def daily_creturn(self) -> pl.DataFrame:
        """Daily resampled cumulative return DataFrame."""
        return (
            self.creturn
            .with_columns(pl.col("date").cast(pl.Date))
            .group_by("date")
            .agg(pl.col("creturn").last())
            .sort("date")
        )

    def get_stats(self, riskfree_rate: float = 0.02) -> pl.DataFrame:
        """Get backtest statistics as DataFrame.

        Args:
            riskfree_rate: Annual risk-free rate for Sharpe/Sortino calculations

        Returns:
            DataFrame with one row containing all statistics
        """
        daily = self.daily_creturn

        if daily.height < 2:
            return pl.DataFrame({"error": ["Insufficient data"]})

        nperiods = 252
        rf_periodic = (1 + riskfree_rate) ** (1 / nperiods) - 1

        # Add returns and drawdown columns
        daily_with_stats = daily.with_columns(
            # Daily returns
            (pl.col("creturn") / pl.col("creturn").shift(1) - 1)
            .fill_null(0.0)
            .alias("return"),
            # Drawdown
            (pl.col("creturn") / pl.col("creturn").cum_max() - 1).alias("drawdown"),
        )

        # Calculate avg_drawdown using per-period min (Finlab compatible)
        avg_dd = self._calc_avg_drawdown(daily_with_stats)

        # Build all stats
        return (
            daily_with_stats
            .select(
                # Dates
                pl.col("date").first().cast(pl.Utf8).alias("start"),
                pl.col("date").last().cast(pl.Utf8).alias("end"),
                pl.lit(riskfree_rate).alias("rf"),
                # Total return
                (pl.col("creturn").last() / pl.col("creturn").first() - 1).alias("total_return"),
                # CAGR
                (
                    (pl.col("creturn").last() / pl.col("creturn").first())
                    .pow(
                        1.0 / ((pl.col("date").last() - pl.col("date").first())
                               .dt.total_days() / 365.25)
                    )
                    - 1
                ).alias("cagr"),
                # Max drawdown
                pl.col("drawdown").min().alias("max_drawdown"),
                # Avg drawdown (per-period min average, Finlab compatible)
                pl.lit(avg_dd).alias("avg_drawdown"),
                # Daily mean (annualized raw return)
                (pl.col("return").mean() * nperiods).alias("daily_mean"),
                # Daily volatility (annualized)
                (pl.col("return").std(ddof=1) * (nperiods ** 0.5)).alias("daily_vol"),
                # Sharpe ratio
                (
                    (pl.col("return") - rf_periodic).mean()
                    / (pl.col("return") - rf_periodic).std(ddof=1)
                    * (nperiods ** 0.5)
                ).alias("daily_sharpe"),
                # Sortino ratio (Finlab compatible: min(excess, 0) then std)
                (
                    (pl.col("return") - rf_periodic).mean()
                    / pl.when(pl.col("return") < rf_periodic)
                    .then(pl.col("return") - rf_periodic)
                    .otherwise(0.0)
                    .std(ddof=1)
                    * (nperiods ** 0.5)
                ).alias("daily_sortino"),
                # Best/worst day
                pl.col("return").max().alias("best_day"),
                pl.col("return").min().alias("worst_day"),
                # Calmar ratio
                (
                    (
                        (pl.col("creturn").last() / pl.col("creturn").first())
                        .pow(
                            1.0 / ((pl.col("date").last() - pl.col("date").first())
                                   .dt.total_days() / 365.25)
                        )
                        - 1
                    )
                    / pl.col("drawdown").min().abs()
                ).alias("calmar"),
                # Win ratio from trades
                pl.lit(self._calc_win_ratio()).alias("win_ratio"),
            )
        )

    def _calc_win_ratio(self) -> float:
        """Calculate win ratio from trades."""
        trades = self.trades.drop_nulls(subset=["return"])
        if trades.height == 0:
            return 0.0
        return trades.select(
            (pl.col("return").filter(pl.col("return") > 0).count().cast(pl.Float64)
             / pl.col("return").count().cast(pl.Float64))
        ).item()

    def _calc_avg_drawdown(self, daily: pl.DataFrame) -> float:
        """Calculate average drawdown (mean of per-period minimum drawdowns).

        Finlab definition: For each drawdown period (consecutive days where dd < 0),
        find the minimum drawdown value, then take the mean of all these minimums.
        """
        # Add drawdown period ID: increment when transitioning from >=0 to <0
        dd_with_period = daily.with_columns(
            # Detect period start: previous row was >=0 and current is <0
            pl.when(
                (pl.col("drawdown") < 0) &
                (pl.col("drawdown").shift(1).fill_null(0.0) >= 0)
            )
            .then(1)
            .otherwise(0)
            .cum_sum()
            .alias("dd_period")
        )

        # Filter to only drawdown rows and get min per period
        period_mins = (
            dd_with_period
            .filter(pl.col("drawdown") < 0)
            .group_by("dd_period")
            .agg(pl.col("drawdown").min().alias("period_min"))
        )

        if period_mins.height == 0:
            return 0.0

        return period_mins["period_min"].mean()

    def get_drawdown_details(self, top_n: int | None = 5) -> pl.DataFrame:
        """Get drawdown period details.

        Returns DataFrame with columns: start, end, length, drawdown
        Each row represents a drawdown period (consecutive days below peak).

        Args:
            top_n: Return only the top N largest drawdowns (default 5).
                   Set to None to return all drawdown periods.
        """
        # Add drawdown column
        daily = self.daily_creturn.with_columns(
            (pl.col("creturn") / pl.col("creturn").cum_max() - 1).alias("drawdown")
        )

        # Add drawdown period ID: increment when transitioning from >=0 to <0
        # Also mark recovery days (first day back to >= 0) as part of the period
        dd_with_period = daily.with_columns(
            # Period starts when: drawdown < 0 and previous day >= 0
            pl.when(
                (pl.col("drawdown") < 0) &
                (pl.col("drawdown").shift(1).fill_null(0.0) >= 0)
            )
            .then(1)
            .otherwise(0)
            .cum_sum()
            .alias("dd_period_raw")
        ).with_columns(
            # Recovery day: drawdown >= 0 and previous day < 0
            # Assign recovery day to the previous period
            pl.when(
                (pl.col("drawdown") >= 0) &
                (pl.col("drawdown").shift(1).fill_null(0.0) < 0)
            )
            .then(pl.col("dd_period_raw").shift(1))
            .otherwise(
                pl.when(pl.col("drawdown") < 0)
                .then(pl.col("dd_period_raw"))
                .otherwise(None)
            )
            .alias("dd_period")
        )

        # Get period details: start (first dd<0), end (recovery day), min drawdown
        period_details = (
            dd_with_period
            .filter(pl.col("dd_period").is_not_null())
            .group_by("dd_period")
            .agg(
                # Start: first day with drawdown < 0
                pl.col("date").filter(pl.col("drawdown") < 0).first().alias("start"),
                # End: last day in period (recovery day if exists)
                pl.col("date").last().alias("end"),
                pl.col("drawdown").min().alias("drawdown"),
            )
            .with_columns(
                (pl.col("end") - pl.col("start")).dt.total_days().alias("length")
            )
            .select("start", "end", "length", "drawdown")
            .sort("drawdown")  # Sort by drawdown (most negative first)
        )

        if top_n is not None:
            return period_details.head(top_n)

        return period_details

    def get_monthly_stats(self, riskfree_rate: float = 0.02) -> pl.DataFrame:
        """Get monthly statistics as DataFrame."""
        nperiods = 12
        rf_periodic = (1 + riskfree_rate) ** (1 / nperiods) - 1

        return (
            self.daily_creturn
            .with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
            .group_by("month")
            .agg(pl.col("creturn").last())
            .sort("month")
            .with_columns(
                (pl.col("creturn") / pl.col("creturn").shift(1) - 1)
                .fill_null(0.0)
                .alias("return")
            )
            .select(
                (pl.col("return").mean() * nperiods).alias("monthly_mean"),
                (pl.col("return").std(ddof=1) * (nperiods ** 0.5)).alias("monthly_vol"),
                (
                    (pl.col("return") - rf_periodic).mean()
                    / (pl.col("return") - rf_periodic).std(ddof=1)
                    * (nperiods ** 0.5)
                ).alias("monthly_sharpe"),
                (
                    (pl.col("return") - rf_periodic).mean()
                    / pl.when(pl.col("return") < rf_periodic)
                    .then(pl.col("return") - rf_periodic)
                    .otherwise(0.0)
                    .std(ddof=1)
                    * (nperiods ** 0.5)
                ).alias("monthly_sortino"),
                pl.col("return").max().alias("best_month"),
                pl.col("return").min().alias("worst_month"),
            )
        )

    def get_return_table(self) -> pl.DataFrame:
        """Get monthly return table as DataFrame."""
        return (
            self.daily_creturn
            .with_columns(
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.month().alias("month"),
            )
            .group_by(["year", "month"])
            .agg(pl.col("creturn").last().alias("month_end"))
            .sort(["year", "month"])
            .with_columns(
                (pl.col("month_end") / pl.col("month_end").shift(1) - 1)
                .fill_null(0.0)
                .alias("monthly_return")
            )
            .pivot(on="month", index="year", values="monthly_return")
            .sort("year")
        )

    @staticmethod
    def get_ndays_return(creturn_df: pl.DataFrame, n: int) -> pl.DataFrame:
        """Get N-day return as DataFrame."""
        return creturn_df.select(
            (pl.col("creturn").last() / pl.col("creturn").get(pl.len() - 1 - n) - 1)
            .alias("nday_return")
        )

    def get_metrics(
        self,
        sections: list[MetricsSection] = ALL_SECTIONS,
        riskfree_rate: float = 0.02,
    ) -> pl.DataFrame:
        """Get structured metrics as single-row DataFrame.

        Args:
            sections: List of sections to include. Options: 'backtest', 'profitability',
                     'risk', 'ratio', 'winrate'. Defaults to all sections.
            riskfree_rate: Annual risk-free rate for Sharpe/Sortino calculations.

        Returns:
            Single-row DataFrame with each metric as a column.
            Includes paper returns for open positions (like Finlab).
        """
        # Validate sections
        invalid = [s for s in sections if s not in ALL_SECTIONS]
        if invalid:
            raise ValueError(f"Invalid sections: {invalid}. Valid: {ALL_SECTIONS}")

        daily = self.daily_creturn

        # Prepare daily with returns
        daily_with_return = daily.with_columns(
            (pl.col("creturn") / pl.col("creturn").shift(1) - 1)
            .fill_null(0.0)
            .alias("return"),
            (pl.col("creturn") / pl.col("creturn").cum_max() - 1).alias("drawdown"),
        )

        monthly_with_return = (
            daily
            .with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
            .group_by("month")
            .agg(pl.col("creturn").last())
            .sort("month")
            .with_columns(
                (pl.col("creturn") / pl.col("creturn").shift(1) - 1)
                .fill_null(0.0)
                .alias("return")
            )
        )

        # Drawdown details for avgDrawdownDays
        dd_details = self.get_drawdown_details(top_n=None)
        avg_dd_days = dd_details["length"].mean() if dd_details.height > 0 else 0.0

        # Position statistics - filter by joining with daily date range
        position_df = self.position
        date_col = position_df.columns[0]
        stock_cols = [c for c in position_df.columns if c != date_col]

        # Use semi join to filter position to creturn date range
        date_range = daily.select(pl.col("date").cast(pl.Utf8).alias(date_col))
        position_filtered = position_df.join(date_range, on=date_col, how="semi")

        n_stocks_df = position_filtered.with_columns(
            pl.sum_horizontal([
                pl.col(c).cast(pl.Float64).abs().gt(0).cast(pl.Int32)
                for c in stock_cols
            ]).alias("n_stocks")
        )

        # Trade statistics with paper returns
        all_trades = self.trades
        closed_trades = all_trades.filter(
            pl.col("return").is_not_null() & pl.col("return").is_not_nan()
        )
        open_trades = all_trades.filter(
            pl.col("entry_date").is_not_null() &
            pl.col("exit_date").is_null() &
            pl.col("trade_price@entry_date").is_not_null() &
            pl.col("trade_price@entry_date").is_not_nan()
        )

        # Calculate paper returns using join instead of loop
        if open_trades.height > 0:
            last_creturn = daily.select(pl.col("creturn").last()).item()
            paper_df = (
                open_trades
                .select(pl.col("entry_date"))
                .join(
                    daily.select(
                        pl.col("date").cast(pl.Utf8).alias("entry_date"),
                        pl.col("creturn").alias("entry_creturn"),
                    ),
                    on="entry_date",
                    how="left",
                )
                .with_columns(
                    (pl.lit(last_creturn) / pl.col("entry_creturn") - 1).alias("return")
                )
                .select("return")
                .drop_nulls()
            )
            trades_with_paper = pl.concat([
                closed_trades.select("return"),
                paper_df,
            ])
        else:
            trades_with_paper = closed_trades.select("return")

        # Precompute values
        nperiods = 252
        rf_periodic = (1 + riskfree_rate) ** (1 / nperiods) - 1

        # Build select expressions based on sections
        exprs: list[pl.Expr] = []

        if "backtest" in sections:
            exprs.extend([
                pl.col("date").first().cast(pl.Utf8).alias("startDate"),
                pl.col("date").last().cast(pl.Utf8).alias("endDate"),
                pl.lit(self.fee_ratio).alias("feeRatio"),
                pl.lit(self.tax_ratio).alias("taxRatio"),
                pl.lit("daily").alias("freq"),
                pl.lit(self.trade_at).alias("tradeAt"),
                pl.lit(self.stop_loss).alias("stopLoss"),
                pl.lit(self.take_profit).alias("takeProfit"),
                pl.lit(self.trail_stop).alias("trailStop"),
            ])

        if "profitability" in sections:
            exprs.extend([
                (
                    (pl.col("creturn").last() / pl.col("creturn").first())
                    .pow(1.0 / ((pl.col("date").last() - pl.col("date").first())
                                .dt.total_days() / 365.25))
                    - 1
                ).alias("annualReturn"),
                pl.lit(n_stocks_df["n_stocks"].mean()).alias("avgNStock"),
                pl.lit(n_stocks_df["n_stocks"].max()).alias("maxNStock"),
            ])

        if "risk" in sections:
            var_5 = monthly_with_return["return"].quantile(0.05)
            cvar_5 = (
                monthly_with_return.filter(pl.col("return") <= var_5)["return"].mean()
                if var_5 is not None else None
            )
            exprs.extend([
                pl.col("drawdown").min().alias("maxDrawdown"),
                pl.lit(self._calc_avg_drawdown(daily_with_return)).alias("avgDrawdown"),
                pl.lit(avg_dd_days).alias("avgDrawdownDays"),
                pl.lit(var_5).alias("valueAtRisk"),
                pl.lit(cvar_5).alias("cvalueAtRisk"),
            ])

        if "ratio" in sections:
            p95 = daily_with_return["return"].quantile(0.95)
            p05 = daily_with_return["return"].quantile(0.05)
            tail_ratio = abs(p95 / p05) if p95 and p05 and p05 != 0 else float("inf")

            pos_sum = trades_with_paper.filter(pl.col("return") > 0)["return"].sum()
            neg_sum = trades_with_paper.filter(pl.col("return") < 0)["return"].sum()
            profit_factor = abs(pos_sum / neg_sum) if neg_sum != 0 else float("inf")

            exprs.extend([
                (
                    (pl.col("return") - rf_periodic).mean()
                    / (pl.col("return") - rf_periodic).std(ddof=1)
                    * (nperiods ** 0.5)
                ).alias("sharpeRatio"),
                (
                    (pl.col("return") - rf_periodic).mean()
                    / pl.when(pl.col("return") < rf_periodic)
                    .then(pl.col("return") - rf_periodic)
                    .otherwise(0.0)
                    .std(ddof=1)
                    * (nperiods ** 0.5)
                ).alias("sortinoRatio"),
                (
                    (
                        (pl.col("creturn").last() / pl.col("creturn").first())
                        .pow(1.0 / ((pl.col("date").last() - pl.col("date").first())
                                    .dt.total_days() / 365.25))
                        - 1
                    )
                    / pl.col("drawdown").min().abs()
                ).alias("calmarRatio"),
                (pl.col("return").std(ddof=1) * (nperiods ** 0.5)).alias("volatility"),
                pl.lit(profit_factor).alias("profitFactor"),
                pl.lit(tail_ratio).alias("tailRatio"),
            ])

        if "winrate" in sections:
            expectancy = trades_with_paper["return"].mean()
            mae = closed_trades["mae"].mean() if "mae" in closed_trades.columns else None
            mfe = closed_trades["gmfe"].mean() if "gmfe" in closed_trades.columns else None

            exprs.extend([
                pl.lit(self._calc_win_ratio()).alias("winRate"),
                pl.lit(expectancy).alias("expectancy"),
                pl.lit(mae).alias("mae"),
                pl.lit(mfe).alias("mfe"),
            ])

        return daily_with_return.select(exprs)

    # =========================================================================
    # Stage 4: Position Information
    # =========================================================================

    @cached_property
    def weights(self) -> pl.DataFrame:
        """Current position weights (last row of position DataFrame).

        Returns:
            Single-row DataFrame with stock_id and weight columns.
        """
        position = self._position
        date_col = position.columns[0]
        stock_cols = [c for c in position.columns if c != date_col]

        last_row = position.tail(1)
        last_date = last_row[date_col][0]

        # Unpivot to get stock_id -> weight pairs
        return (
            last_row
            .unpivot(
                index=date_col,
                on=stock_cols,
                variable_name="stock_id",
                value_name="weight",
            )
            .filter(pl.col("weight").abs() > 0)
            .select("stock_id", "weight")
            .with_columns(pl.lit(last_date).alias("date"))
        )

    @cached_property
    def next_weights(self) -> pl.DataFrame | None:
        """Next period weights (not available in wide format backtest).

        Returns:
            None - would require future position data.
        """
        return None

    @cached_property
    def current_trades(self) -> pl.DataFrame:
        """Active trades (positions without exit or exiting on last date).

        Returns:
            DataFrame with current/active trade records.
        """
        trades = self.trades
        if trades.height == 0:
            return trades

        # Get last date from creturn
        last_date = self.creturn.select(pl.col("date").last()).item()
        last_date_str = str(last_date)[:10] if last_date else None

        # Filter: no exit_date OR exit_date is today
        return trades.filter(
            pl.col("exit_date").is_null() |
            (pl.col("exit_date").cast(pl.Utf8).str.slice(0, 10) == last_date_str)
        )

    @cached_property
    def actions(self) -> pl.DataFrame:
        """Trade actions for current positions (enter/exit/hold).

        Returns:
            DataFrame with stock_id and action columns.
        """
        trades = self.trades
        if trades.height == 0:
            return pl.DataFrame({"stock_id": [], "action": []})

        last_date = self.creturn.select(pl.col("date").last()).item()
        last_date_str = str(last_date)[:10] if last_date else None

        # Determine action for each trade
        return trades.select(
            pl.col("stock_id"),
            pl.when(
                pl.col("entry_date").cast(pl.Utf8).str.slice(0, 10) == last_date_str
            ).then(pl.lit("enter"))
            .when(
                pl.col("exit_date").cast(pl.Utf8).str.slice(0, 10) == last_date_str
            ).then(pl.lit("exit"))
            .when(
                pl.col("exit_date").is_null()
            ).then(pl.lit("hold"))
            .otherwise(pl.lit("closed"))
            .alias("action"),
        ).filter(pl.col("action") != "closed")

    def position_info(self) -> pl.DataFrame:
        """Get position information for API/dashboard.

        Returns:
            DataFrame with current position details including:
            - stock_id, weight, entry_date, return, action
        """
        current = self.current_trades
        weights = self.weights

        if current.height == 0:
            return pl.DataFrame({
                "stock_id": [],
                "weight": [],
                "entry_date": [],
                "exit_date": [],
                "return": [],
                "action": [],
            })

        # Join with weights to get current weight
        result = (
            current
            .join(weights.select("stock_id", "weight"), on="stock_id", how="left")
            .join(self.actions, on="stock_id", how="left")
            .select(
                "stock_id",
                pl.col("weight").fill_null(0.0),
                "entry_date",
                "exit_date",
                pl.col("return").fill_null(0.0),
                pl.col("action").fill_null("hold"),
            )
        )
        return result

    def position_info2(self) -> dict:
        """Get detailed position information for dashboard.

        Returns:
            Dict with positions list and positionConfig.
        """
        pos_info = self.position_info()
        positions = pos_info.to_dicts() if pos_info.height > 0 else []

        # Convert dates to ISO strings
        for p in positions:
            if p.get("entry_date"):
                p["entry_date"] = str(p["entry_date"])[:10]
            if p.get("exit_date"):
                p["exit_date"] = str(p["exit_date"])[:10]

        # Get last date from creturn
        last_date = self.creturn.select(pl.col("date").last()).item()
        last_date_str = str(last_date)[:10] if last_date else None

        # Get weights date
        weights_date = None
        if self.weights.height > 0 and "date" in self.weights.columns:
            weights_date = str(self.weights["date"][0])[:10]

        return {
            "positions": positions,
            "positionConfig": {
                "feeRatio": self.fee_ratio,
                "taxRatio": self.tax_ratio,
                "resample": self.resample,
                "tradeAt": self.trade_at,
                "stopLoss": self.stop_loss,
                "takeProfit": self.take_profit,
                "trailStop": self.trail_stop,
                "currentRebalanceDate": weights_date,
                "lastDate": last_date_str,
            },
        }

    def is_rebalance_due(self) -> bool:
        """Check if rebalance is due based on position changes.

        Returns:
            True if the last position differs from the previous position.
        """
        position = self._position
        if position.height < 2:
            return False

        date_col = position.columns[0]
        stock_cols = [c for c in position.columns if c != date_col]

        # Compare last two rows
        last_two = position.tail(2).select(stock_cols)
        return any(
            last_two[c][1] != last_two[c][0]
            for c in stock_cols
        )

    def is_stop_triggered(self) -> bool:
        """Check if any trade was triggered by stop loss or take profit.

        Returns:
            True if any current trade hit SL/TP.
        """
        current = self.current_trades
        if current.height == 0:
            return False

        # Check if any trade has exit with return matching SL/TP thresholds
        # SL: return <= -stop_loss, TP: return >= take_profit
        if self.stop_loss is not None and self.stop_loss < 1.0:
            sl_triggered = current.filter(
                pl.col("return").is_not_null() &
                (pl.col("return") <= -self.stop_loss)
            ).height > 0
            if sl_triggered:
                return True

        if self.take_profit is not None and self.take_profit < float("inf"):
            tp_triggered = current.filter(
                pl.col("return").is_not_null() &
                (pl.col("return") >= self.take_profit)
            ).height > 0
            if tp_triggered:
                return True

        return False


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

    # Determine trade_at string
    if isinstance(trade_at_price, str):
        trade_at_str = trade_at_price
    else:
        trade_at_str = "custom"

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
        resample=resample,
        trade_at=trade_at_str,
        stop_loss=stop_loss if stop_loss < 1.0 else None,
        take_profit=take_profit if take_profit < float("inf") else None,
        trail_stop=trail_stop if trail_stop < float("inf") else None,
    )
