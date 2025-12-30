"""Utility functions for polars_backtest."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl


def long_to_wide(
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


def parse_resample_freq(resample: str) -> tuple[str, int | None]:
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


def parse_offset(offset_str: str) -> timedelta:
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


def get_period_end_dates(
    start_date: date,
    end_date: date,
    freq: str,
    weekday: int | None = None,
) -> list[date]:
    """Generate period-end dates between start and end dates.

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


# Keep underscore aliases for backward compatibility
_parse_resample_freq = parse_resample_freq
_parse_offset = parse_offset
_get_period_end_dates = get_period_end_dates
