# polars-backtest API Reference

Grounded in `polars_backtest/python/polars_backtest/namespace.py`, `wide.py`,
`__init__.py`, and `src/report.rs`. All names and defaults below are copied from
the source signatures.

## Entry points

```python
import polars as pl
import polars_backtest as pl_bt

# Long format (primary API) — namespace and function forms are identical
result = df.bt.backtest(...)                    # -> pl.DataFrame [date, creturn]
result = pl_bt.backtest(df, ...)                # same

report = df.bt.backtest_with_report(...)        # -> BacktestReport
report = pl_bt.backtest_with_report(df, ...)    # same

# Wide format (Finlab-compatible, secondary API)
result = pl_bt.backtest_wide(prices, position, ...)               # -> pl.DataFrame
report = pl_bt.backtest_with_report_wide(close, position, ...)    # -> Report (Python class)
```

`pl_bt.backtest(df, ...)` simply delegates to `df.bt.backtest(...)`; use either.

## `backtest()` / `backtest_with_report()` — full signature

Long format. Every parameter, with exact defaults:

### Column mapping (str or `pl.Expr`)

| Parameter | Default | Notes |
|---|---|---|
| `trade_at_price` | `"close"` | Price used for position valuation and trade execution. Use dividend/split-adjusted prices for correct returns. Accepts an expression, e.g. `pl.col("adj_close")`. |
| `position` | `"weight"` | Float weights or Boolean signals. Nulls filled with `0.0` (float) / `False` (bool) before the run. Bool columns are cast to `Float64` and converted to equal weights by the core. |
| `date` | `"date"` | `pl.Date` recommended; string dates work (parsed by the core). Data may be unsorted — the wrapper checks `is_sorted()` and the core sorts if needed, but pre-sorting by date is faster. |
| `symbol` | `"symbol"` | String ticker. |
| `open` / `high` / `low` | `"open"` / `"high"` / `"low"` | Only resolved and validated when `touched_exit=True`; otherwise ignored. |
| `factor` | `"factor"` | str only (no Expr). Adjustment factor: `raw_price = adj_price / factor`. If the column does not exist, factor silently defaults to 1.0. Raw prices feed `entry_raw_price` / `exit_raw_price` in trades and the liquidity metrics. |

### Rebalancing

| Parameter | Default | Notes |
|---|---|---|
| `resample` | `"D"` | One of `None`, `"D"`, `"W"`, `"W-MON"`…`"W-SUN"`, `"M"`, `"ME"`, `"MS"`, `"Q"`, `"QE"`, `"QS"`, `"Y"`, `"YE"`, `"A"`. Anything else raises `ValueError`. `None` = rebalance only when the position column changes. |
| `resample_offset` | `None` | Calendar-day delay of rebalance dates. Formats parsed by the Rust core: `"1d"`, `"2d"`, `"1D"`, `"1W"`, `"2W"` (weeks × 7 days). **Positive offsets only; unparseable strings are silently treated as no offset** (`ResampleOffset::from_str` returns `None`). |

### Costs

| Parameter | Default | Notes |
|---|---|---|
| `fee_ratio` | `0.001425` | Charged on both buy and sell notional (Taiwan brokerage default 0.1425%). |
| `tax_ratio` | `0.003` | Charged on sell notional only (Taiwan securities tax 0.3%). |

### Risk management

| Parameter | Default | Notes |
|---|---|---|
| `stop_loss` | `1.0` (disabled) | **Positive fraction.** `0.1` = exit when the position loses 10% from entry. `1.0` disables (threshold `1-stop_loss=0` is unreachable). Never pass a negative number — it stops positions out immediately. |
| `take_profit` | `inf` (disabled) | Positive fraction. `0.2` = exit at +20% from entry. |
| `trail_stop` | `inf` (disabled) | Positive fraction. `0.08` = exit when the position's cumulative return falls 0.08 below its running peak since entry. |
| `touched_exit` | `False` | Detect stops intraday with OHLC and exit same day (T+0) at open or at the stop level. Requires open/high/low columns. |
| `stop_trading_next_period` | `True` | After a stop fires, the stock is excluded from the next rebalance (its weight is zeroed and the rest rescaled). `False` allows immediate re-entry. |
| `position_limit` | `1.0` (no limit) | Cap on each stock's weight, applied after normalization. Float weights: a final clamp — excess is not redistributed (becomes cash). Boolean signals: equal weights are capped and iteratively re-normalized. |

### Calculation mode

| Parameter | Default | Notes |
|---|---|---|
| `retain_cost_when_rebalance` | `False` | `True`: positions continuing through a rebalance in the same direction keep their entry cost basis (cr/maxcr are not reset), so stop_loss/take_profit measure from original entry. `False`: cost resets at every rebalance. |

### Report-only parameters (`backtest_with_report` only)

| Parameter | Default | Notes |
|---|---|---|
| `benchmark` | `None` | `str`: a symbol value present in your data (e.g. `"0050"`) — its `trade_at_price` series is normalized to a creturn starting at 1.0. `pl.DataFrame`: must have `date` and `creturn` columns. Enables `alpha`, `beta`, `m12WinRate` in `get_metrics()`. |
| `limit_up` | `"limit_up"` | Column with raw limit-up price. If present, enables `buyHigh` metric. |
| `limit_down` | `"limit_down"` | Column with raw limit-down price. If present, enables `sellLow` metric. |

| `trading_value` | `"trading_value"` | Column with raw daily trading value (e.g. `close_raw * volume`, TWD). If present, enables the `capacity` metric. |

## BacktestReport

Returned by `backtest_with_report()`. Implemented in Rust (`src/report.rs`).

### Properties

| Property | Type | Description |
|---|---|---|
| `report.creturn` | DataFrame | `date`, `creturn` (starts at 1.0). |
| `report.trades` | DataFrame | Full trade records — see table below. |
| `report.stats` | DataFrame | Shortcut for `get_stats(riskfree_rate=0.02)`. |
| `report.fee_ratio` / `report.tax_ratio` | float | Config echo. |
| `report.stop_loss` / `report.take_profit` / `report.trail_stop` | float \| None | `None` when disabled. |
| `report.trade_at` | str | Always `"close"` currently. |
| `report.resample` | str \| None | Resample frequency used. |
| `report.benchmark` | DataFrame \| None | Get/set. Setter validates `date` + `creturn` columns. |

### Methods

| Method | Returns | Description |
|---|---|---|
| `get_stats(riskfree_rate=0.02)` | 1-row DataFrame | `start, end, rf, total_return, cagr, max_drawdown, avg_drawdown, daily_mean, daily_vol, daily_sharpe, daily_sortino, best_day, worst_day, calmar, win_ratio`. |
| `get_monthly_stats(riskfree_rate=0.02)` | 1-row DataFrame | `monthly_mean, monthly_vol, monthly_sharpe, monthly_sortino, best_month, worst_month`. |
| `get_return_table()` | DataFrame | Monthly returns pivoted year × month (columns 1–12). |
| `get_metrics(sections=None, riskfree_rate=0.02)` | 1-row DataFrame | Sections below. Invalid section name raises `ValueError`. |
| `daily_creturn()` | DataFrame | Daily-resampled creturn. |
| `current_trades()` | DataFrame | Open positions + pending entries + trades whose `exit_sig_date` equals the last rebalance date. |
| `actions()` | DataFrame | `symbol, action ("enter"/"exit"/"hold"), weight, weight_date, next_weight, next_weight_date`. Closed trades excluded. |
| `weights()` | DataFrame | `symbol, weight, date` — currently held positions only (pending exits excluded), normalized, sum ≤ 1. |
| `next_weights()` | DataFrame | `symbol, weight, date` — next-period targets (hold + enter; pending exits excluded), normalized, sum ≤ 1. |
| `is_stop_triggered()` | bool | True if any *current* trade's return breaches `stop_loss` or `take_profit`. Note: does not check `trail_stop`. |

### `get_metrics` sections

| Section | Columns |
|---|---|
| `backtest` | `startDate, endDate, feeRatio, taxRatio, freq, tradeAt, stopLoss, takeProfit, trailStop` |
| `profitability` | `annualReturn, avgNStock, maxNStock, alpha, beta` |
| `risk` | `maxDrawdown, avgDrawdown, avgDrawdownDays, valueAtRisk, cvalueAtRisk` (VaR/CVaR = 5th percentile of monthly returns) |
| `ratio` | `sharpeRatio, sortinoRatio, calmarRatio, volatility, profitFactor, tailRatio` |
| `winrate` | `winRate, m12WinRate, expectancy, mae, mfe` |
| `liquidity` | `buyHigh, sellLow, capacity` (null unless the corresponding columns were in the input) |

`alpha`, `beta`, `m12WinRate` are null unless a benchmark is set (at call time or
via `report.benchmark = df`).

### `report.trades` columns

| Column | Type | Description |
|---|---|---|
| `stock_id` | str | Symbol. |
| `entry_date` | Date | Execution date (T+1 after signal). Null = pending entry. |
| `exit_date` | Date | Execution date. Null = still open or pending exit. |
| `entry_sig_date` | Date | Date the entry signal was generated. |
| `exit_sig_date` | Date | Date the exit signal was generated. Null = no exit signal yet. |
| `position` | f64 | Weight at entry. |
| `period` | i32 | Trading days held. |
| `return` | f64 | Trade return net of costs (0.05 = +5%). |
| `entry_price` / `exit_price` | f64 | Execution prices in `trade_at_price` terms (adjusted). |
| `entry_raw_price` / `exit_raw_price` | f64 | Unadjusted = price / factor. Used for limit-up/down checks. |
| `mae` | f64 | Max adverse excursion during trade (≤ 0). |
| `gmfe` | f64 | Gross max favorable excursion. |
| `bmfe` | f64 | MFE recorded before the MAE trough. |
| `mdd` | f64 | Max drawdown within the trade. |
| `pdays` | i32 | Profitable days count. |

Pending-trade filters:

```python
pending_entries = report.trades.filter(pl.col("entry_date").is_null())
open_positions  = report.trades.filter(
    pl.col("entry_date").is_not_null()
    & pl.col("exit_date").is_null()
    & pl.col("exit_sig_date").is_null()
)
pending_exits   = report.trades.filter(
    pl.col("entry_date").is_not_null()
    & pl.col("exit_date").is_null()
    & pl.col("exit_sig_date").is_not_null()
)
```

## Wide-format API (Finlab-style matrices)

For users holding wide DataFrames (first column = date, one column per symbol).
Long format is preferred; the wide API exists mainly for Finlab parity testing.

```python
result = pl_bt.backtest_wide(
    prices,            # wide DataFrame: date + one column per stock
    position,          # wide DataFrame of weights/bool signals (rebalance rows)
    resample="D", resample_offset=None, rebalance_indices=None,
    fee_ratio=0.001425, tax_ratio=0.003,
    stop_loss=1.0, take_profit=float("inf"), trail_stop=float("inf"),
    position_limit=1.0, retain_cost_when_rebalance=False,
    stop_trading_next_period=True, finlab_mode=False,
)

report = pl_bt.backtest_with_report_wide(
    close, position,
    resample="D", resample_offset=None,
    trade_at_price="close",        # 'close' | 'open' | 'high' | 'low' | a wide DataFrame
    open=None, high=None, low=None, factor=None,   # wide DataFrames
    rebalance_indices=None,
    fee_ratio=0.001425, tax_ratio=0.003,
    stop_loss=1.0, take_profit=float("inf"), trail_stop=float("inf"),
    position_limit=1.0, retain_cost_when_rebalance=False,
    stop_trading_next_period=True, touched_exit=False,
)  # -> polars_backtest.Report (Python class in wide.py)
```

The wide `Report` mirrors `BacktestReport` (`creturn`, `trades`, `get_stats`,
`get_monthly_stats`, `get_return_table`, `get_metrics`, `weights`, `next_weights`,
`actions`, `current_trades`, `is_stop_triggered`) and adds `position`,
`get_drawdown_details(top_n=5)`, `position_info()`, `position_info2()`,
`is_rebalance_due()`.

Helper: `polars_backtest.utils.long_to_wide(df, value_col, date_col="date", symbol_col="symbol")`.

## Statistics expressions (Polars plugin functions)

All registered as plugin expressions; usable in `select`/`with_columns`:

```python
from polars_backtest import (
    daily_returns,        # price series -> daily returns (first value null)
    cumulative_returns,   # daily returns -> creturn starting at 1.0
    sharpe_ratio,         # returns -> annualized Sharpe (rf=0, 252 days)
    sortino_ratio,        # returns -> annualized Sortino (rf=0, 252 days)
    max_drawdown,         # creturn -> max drawdown (negative scalar)
    drawdown_series,      # creturn -> per-row drawdown (negative values)
    portfolio_return,     # (weights, returns) -> weighted return scalar
    equal_weights,        # boolean signals -> equal weights summing to 1.0
)

per_symbol = df.with_columns(ret=daily_returns("close").over("symbol"))
df.select(sharpe=sharpe_ratio("ret"), mdd=max_drawdown("creturn"))
```

Note the fixed conventions: rf=0 and 252 periods/year for the expression-level
Sharpe/Sortino (unlike `report.get_stats`, which takes `riskfree_rate`).

## Also exported

- `BacktestConfig` — Rust config object (rarely constructed manually; the keyword
  arguments above build it for you).
- `BacktestReport` — the report class (for isinstance checks / typing).
- `polars_backtest.DataFrame` — typing alias so `df.bt` type-checks.
