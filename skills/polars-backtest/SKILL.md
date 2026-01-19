---
name: polars-backtest
description: Help users backtest trading strategies with polars-backtest library. Use when user asks about backtesting, portfolio simulation, trading strategy analysis, or working with polars-backtest.
---

# Polars Backtest Usage Skill

Help users use polars-backtest to backtest their trading strategies efficiently.

## Installation

```bash
pip install polars-backtest
# or
uv add polars-backtest
```

## Data Format

**Long format**: one row per (date, symbol) pair.

```python
import polars as pl
import polars_backtest as pl_bt

df = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "symbol": ["2330", "2317", "2330", "2317"],
    "close": [100.0, 50.0, 102.0, 51.0],
    "weight": [0.6, 0.4, 0.6, 0.4],
})
```

## Basic Usage

```python
# DataFrame namespace
result = df.bt.backtest(trade_at_price="close", position="weight")

# Function API
result = pl_bt.backtest(df, trade_at_price="close", position="weight")

# With report (includes trades, stats)
report = df.bt.backtest_with_report(position="weight", resample="M")
```

## Complete Parameter Reference

### Column Mapping Parameters

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `trade_at_price` | `"close"` | str \| Expr | Price column for trade execution. Use adjusted close for accurate returns. |
| `position` | `"weight"` | str \| Expr | Position weight column. Can be float weights or boolean signals (True=buy). Null values are filled with 0. |
| `date` | `"date"` | str \| Expr | Date column. Must be sortable. |
| `symbol` | `"symbol"` | str \| Expr | Stock symbol/ticker column. |
| `open` | `"open"` | str \| Expr | Open price column. Required when `touched_exit=True`. |
| `high` | `"high"` | str \| Expr | High price column. Required when `touched_exit=True`. |
| `low` | `"low"` | str \| Expr | Low price column. Required when `touched_exit=True`. |
| `factor` | `"factor"` | str | Adjustment factor column. Used to calculate raw price: `raw_price = adj_price / factor`. If column doesn't exist, defaults to 1.0. Useful for dividend/split adjusted prices. |

### Rebalancing Parameters

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `resample` | `"D"` | str \| None | Rebalance frequency. Options: `"D"` (daily), `"W"` (weekly), `"W-FRI"` (weekly on Friday), `"M"` (monthly), `"Q"` (quarterly), `"Y"` (yearly), `None` (only when position changes). |
| `resample_offset` | `None` | str \| None | Delay rebalance execution. Examples: `"1d"` (1 day delay), `"2d"`, `"1W"`. Useful for simulating delayed signal execution. |

### Cost Parameters

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `fee_ratio` | `0.001425` | float | Transaction fee rate (both buy and sell). Taiwan stock default is 0.1425%. |
| `tax_ratio` | `0.003` | float | Transaction tax rate (sell only). Taiwan stock default is 0.3%. |

### Risk Management Parameters

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `stop_loss` | `1.0` | float | Stop loss threshold. Exit when loss reaches this ratio. `1.0` = disabled (100% loss). Example: `-0.1` exits at 10% loss. |
| `take_profit` | `inf` | float | Take profit threshold. Exit when profit reaches this ratio. `inf` = disabled. Example: `0.2` exits at 20% profit. |
| `trail_stop` | `inf` | float | Trailing stop threshold. Exit when price drops this much from peak. `inf` = disabled. Example: `0.05` exits when 5% below peak. |
| `position_limit` | `1.0` | float | Maximum weight per single stock. `1.0` = no limit. Example: `0.1` caps each stock at 10% of portfolio. |
| `touched_exit` | `False` | bool | Use intraday OHLC for stop detection. When True, checks if stop/take profit was touched during the day using high/low prices, not just close. Requires open/high/low columns. |
| `stop_trading_next_period` | `True` | bool | When stop is triggered, skip trading in the next period. Prevents immediate re-entry after stop. |

### Calculation Mode Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `finlab_mode` | `False` (backtest) / `True` (backtest_with_report) | Use Finlab-compatible calculation. When True, boolean signals are converted to equal weights. Affects weight normalization behavior. |
| `retain_cost_when_rebalance` | `False` | When rebalancing, retain the cost basis instead of resetting. Affects return calculation for partially sold positions. |

### Benchmark Parameters (backtest_with_report only)

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `benchmark` | `None` | str \| DataFrame \| None | Benchmark for alpha/beta calculation. **str**: symbol value in your data (e.g., `"0050"`), uses that symbol's price. **DataFrame**: must have `date` and `creturn` columns (cumulative return starting at 1.0). |

### Liquidity Metrics Parameters (backtest_with_report only)

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `limit_up` | `"limit_up"` | str | Column name for limit-up price. Used to calculate `buyHigh` metric (ratio of entries at limit-up). |
| `limit_down` | `"limit_down"` | str | Column name for limit-down price. Used to calculate `sellLow` metric (ratio of exits at limit-down). |
| `trading_value` | `"trading_value"` | str | Column for trading value (e.g., `close * volume`). Used to calculate `capacity` metric. |

## BacktestReport Object

```python
report = df.bt.backtest_with_report(position="weight")

# Properties
report.creturn      # DataFrame with date, creturn columns
report.trades       # DataFrame with trade records (see Trades DataFrame section)
report.stats        # Statistics DataFrame (shortcut for get_stats())
report.fee_ratio    # Fee ratio used
report.tax_ratio    # Tax ratio used
report.stop_loss    # Stop loss threshold (None if disabled)
report.take_profit  # Take profit threshold (None if disabled)
report.trail_stop   # Trail stop threshold (None if disabled)
report.trade_at     # Trade timing (e.g., 'close')
report.resample     # Resample frequency
report.benchmark    # Benchmark DataFrame (can be set after creation)

# Set benchmark after creation
report.benchmark = benchmark_df
```

## Trades DataFrame

The `report.trades` property returns a DataFrame with detailed trade records:

| Column | Type | Description |
|--------|------|-------------|
| `stock_id` | str | Stock symbol/ticker |
| `entry_date` | Date | Trade entry execution date. `null` for pending entry trades. |
| `exit_date` | Date | Trade exit execution date. `null` for open/pending positions. |
| `entry_sig_date` | Date | Date when entry signal was generated. |
| `exit_sig_date` | Date | Date when exit signal was generated. `null` if no exit signal yet. |
| `position` | f64 | Position weight at entry. |
| `period` | i32 | Number of trading days held. |
| `return` | f64 | Trade return ratio (e.g., 0.05 = 5% profit). |
| `entry_price` | f64 | Entry execution price (adjusted price, for return calculation). |
| `exit_price` | f64 | Exit execution price (adjusted price). `null` for open positions. |
| `entry_raw_price` | f64 | Entry raw price (unadjusted = entry_price / factor, for liquidity metrics). |
| `exit_raw_price` | f64 | Exit raw price (unadjusted = exit_price / factor). `null` for open positions. |
| `mae` | f64 | Maximum Adverse Excursion - worst drawdown during trade (negative value). |
| `gmfe` | f64 | Gross Maximum Favorable Excursion - best unrealized gain during trade. |
| `bmfe` | f64 | Before-MAE MFE - MFE at the time when MAE occurred. |
| `mdd` | f64 | Maximum drawdown during the trade. |
| `pdays` | i32 | Number of profitable days during the trade. |

### Pending Trades

Pending trades represent signals that exist but haven't been executed yet (T+1 execution model):

- **Pending entry**: `entry_date = null`, `entry_sig_date = latest signal date`
  - Stock has a buy signal but trade hasn't executed yet
- **Pending exit**: `exit_date = null`, `exit_sig_date = latest signal date`
  - Stock has a sell signal but trade hasn't closed yet

```python
# Get pending entry trades
pending_entries = report.trades.filter(pl.col("entry_date").is_null())

# Get pending exit trades (open positions with exit signal)
pending_exits = report.trades.filter(
    pl.col("entry_date").is_not_null() &
    pl.col("exit_date").is_null() &
    pl.col("exit_sig_date").is_not_null()
)

# Get open positions (no exit signal yet)
open_positions = report.trades.filter(
    pl.col("entry_date").is_not_null() &
    pl.col("exit_date").is_null() &
    pl.col("exit_sig_date").is_null()
)
```

## BacktestReport Methods

### get_stats(riskfree_rate=0.02)
Get basic statistics as single-row DataFrame.

```python
report.get_stats()  # or report.stats
```

Columns: `start`, `end`, `rf`, `total_return`, `cagr`, `max_drawdown`, `avg_drawdown`, `daily_mean`, `daily_vol`, `daily_sharpe`, `daily_sortino`, `best_day`, `worst_day`, `calmar`, `win_ratio`

### get_monthly_stats(riskfree_rate=0.02)
Get monthly statistics.

```python
report.get_monthly_stats()
```

Columns: `monthly_mean`, `monthly_vol`, `monthly_sharpe`, `monthly_sortino`, `best_month`, `worst_month`

### get_return_table()
Get monthly return table pivoted by year x month.

```python
report.get_return_table()
```

Returns DataFrame with year as rows and months (1-12) as columns.

### get_metrics(sections=None, riskfree_rate=0.02)
Get structured metrics as single-row DataFrame.

```python
metrics = report.get_metrics()  # All sections
metrics = report.get_metrics(sections=["profitability", "risk"])
```

### current_trades()
Get active trades (positions without exit or exiting on last date).

```python
report.current_trades()
```

### actions()
Get trade actions for current positions with weights. Finlab-compatible logic.

```python
report.actions()
# Returns DataFrame with columns: symbol, action, weight, next_weight, weight_date, next_weight_date
```

#### Output Columns

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | str | Stock symbol/ticker |
| `action` | str | Action type: `"enter"`, `"exit"`, or `"hold"` |
| `weight` | f64 | Current position weight (0 for enter, value for hold/exit) |
| `next_weight` | f64 | Next period target weight (0 for exit, value for hold/enter) |
| `weight_date` | Date | Date of current weights (signal date) |
| `next_weight_date` | Date | Date of next weights (next rebalance date) |

#### Action Types Summary

| action | weight | next_weight | 說明 |
|--------|--------|-------------|------|
| `enter` | 0 | >0 | 待進場，尚未持有 |
| `exit` | >0 | 0 | 待出場，將離開投資組合 |
| `hold` | >0 | >0 | 持續持有 |

#### Weight Properties

- `sum(weight)` ≤ 1.0 — 當前投資組合的總權重
- `sum(next_weight)` ≤ 1.0 — 下期目標投資組合的總權重
- **hold + enter 的權重不會超過 1**，因為：
  - `weight` 只計算當前持有的股票（hold + exit）
  - `next_weight` 是下期目標權重（hold + enter，已正規化）

#### Usage Example

```python
report = df.bt.backtest_with_report(position="weight", resample="M")

# Get actions with weights and dates
actions = report.actions()
print(actions)
# shape: (115, 6)
# ┌────────┬────────┬──────────┬─────────────┬─────────────┬──────────────────┐
# │ symbol ┆ action ┆ weight   ┆ weight_date ┆ next_weight ┆ next_weight_date │
# │ str    ┆ str    ┆ f64      ┆ date        ┆ f64         ┆ date             │
# ╞════════╪════════╪══════════╪═════════════╪═════════════╪══════════════════╡
# │ 2330   ┆ hold   ┆ 0.023809 ┆ 2026-01-15  ┆ 0.017543    ┆ 2026-02-15       │
# │ 2317   ┆ exit   ┆ 0.023809 ┆ 2026-01-15  ┆ 0.0         ┆ null             │
# │ 2454   ┆ enter  ┆ 0.0      ┆ null        ┆ 0.017543    ┆ 2026-02-15       │
# └────────┴────────┴──────────┴─────────────┴─────────────┴──────────────────┘

# Filter by action type
entering = actions.filter(pl.col("action") == "enter")
exiting = actions.filter(pl.col("action") == "exit")
holding = actions.filter(pl.col("action") == "hold")

# Verify weight sums
print(f"Current portfolio weight: {actions['weight'].sum():.4f}")      # ≤ 1.0
print(f"Next portfolio weight: {actions['next_weight'].sum():.4f}")    # ≤ 1.0
```

Note: Closed trades (both `entry_date` and `exit_date` set) are excluded from actions.

### weights()
Get current position weights (normalized). Finlab-compatible.

```python
report.weights()
# Returns DataFrame with columns: symbol, weight, date
```

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | str | Stock symbol/ticker |
| `weight` | f64 | Normalized weight (sum ≤ 1.0) |
| `date` | Date | Date of current weights (signal date) |

Returns weights for currently held positions only (stocks with `entry_date` set, `exit_date` null).

### next_weights()
Get next period target weights (normalized). Finlab-compatible.

```python
report.next_weights()
# Returns DataFrame with columns: symbol, weight, date
```

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | str | Stock symbol/ticker |
| `weight` | f64 | Normalized weight (sum ≤ 1.0) |
| `date` | Date | Date of next weights (next rebalance date) |

Returns target weights for the next rebalancing period. Includes:
- Hold stocks: continuing positions
- Enter stocks: pending entry positions

Excludes stocks with pending exit signals.

### is_stop_triggered()
Check if any trade was triggered by stop loss or take profit.

```python
if report.is_stop_triggered():
    print("Stop was triggered")
```

### daily_creturn()
Get daily resampled cumulative return DataFrame.

```python
report.daily_creturn()
```

## get_metrics Sections

| Section | Metrics | Description |
|---------|---------|-------------|
| **backtest** | `startDate`, `endDate`, `feeRatio`, `taxRatio`, `freq`, `tradeAt`, `stopLoss`, `takeProfit`, `trailStop` | Backtest configuration |
| **profitability** | `annualReturn`, `avgNStock`, `maxNStock`, `alpha`, `beta` | Returns and benchmark comparison |
| **risk** | `maxDrawdown`, `avgDrawdown`, `avgDrawdownDays`, `valueAtRisk`, `cvalueAtRisk` | Risk metrics |
| **ratio** | `sharpeRatio`, `sortinoRatio`, `calmarRatio`, `volatility`, `profitFactor`, `tailRatio` | Risk-adjusted ratios |
| **winrate** | `winRate`, `expectancy`, `mae`, `mfe`, `m12WinRate` | Win rate and trade analysis |
| **liquidity** | `buyHigh`, `sellLow`, `capacity` | Liquidity metrics (requires columns) |

Note: `alpha`, `beta`, `m12WinRate` require benchmark to be set.

## Statistics Expressions

Use these expressions for custom calculations:

```python
from polars_backtest import daily_returns, cumulative_returns, sharpe_ratio, max_drawdown

df.with_columns(
    ret=daily_returns("close"),
    creturn=cumulative_returns("ret"),
)

df.select(
    sharpe=sharpe_ratio("ret"),
    mdd=max_drawdown("creturn"),
)
```

## Usage Examples

### Momentum Strategy with Monthly Rebalancing
```python
df = df.with_columns(
    pl.when(pl.col("close") >= pl.col("close").rolling_max(60).over("symbol"))
    .then(1.0)
    .otherwise(0.0)
    .alias("weight")
)
report = df.bt.backtest_with_report(position="weight", resample="M")
```

### With Risk Management
```python
report = df.bt.backtest_with_report(
    position="weight",
    stop_loss=-0.1,        # -10% stop loss
    take_profit=0.2,       # +20% take profit
    trail_stop=0.05,       # 5% trailing stop
    touched_exit=True,     # Use intraday OHLC for detection
)
```

### With Benchmark Comparison
```python
# Using a symbol in your data as benchmark
report = df.bt.backtest_with_report(
    position="weight",
    benchmark="0050",      # ETF ticker
)

# Or set after creation
report.benchmark = benchmark_df

# Get metrics with alpha, beta, m12WinRate
metrics = report.get_metrics(sections=["profitability", "winrate"])
```

### With Factor Column (Adjusted Prices)
```python
# When using adjusted prices, factor converts back to raw price
# raw_price = adj_close / factor
df = df.with_columns(
    (pl.col("close_raw") / pl.col("close_adj")).alias("factor")
)
report = df.bt.backtest_with_report(
    trade_at_price="close_adj",
    factor="factor",
)
```

### With Liquidity Metrics
```python
df = df.with_columns([
    pl.col("limit_up_price").alias("limit_up"),
    pl.col("limit_down_price").alias("limit_down"),
    (pl.col("close") * pl.col("volume")).alias("trading_value"),
])
report = df.bt.backtest_with_report(position="weight")
metrics = report.get_metrics(sections=["liquidity"])
# Returns: buyHigh, sellLow, capacity
```

### Using Polars Expressions
```python
# Pass expressions directly instead of column names
result = df.bt.backtest(
    trade_at_price=pl.col("adj_close"),
    position=pl.col("signal").cast(pl.Float64),
    resample="M",
)
```

### Delayed Rebalancing
```python
# Rebalance monthly, but execute 2 days after signal
report = df.bt.backtest_with_report(
    position="weight",
    resample="M",
    resample_offset="2d",
)
```

### Analyzing Current Positions
```python
report = df.bt.backtest_with_report(position="weight")

# Get current active trades
current = report.current_trades()

# Get recommended actions
actions = report.actions()  # enter/exit/hold per stock

# Get current and next weights (Finlab compatible)
weights = report.weights()        # Currently held positions (sum ≤ 1)
next_wts = report.next_weights()  # Target portfolio for next period (sum ≤ 1)

# Note: hold + enter positions are in next_weights
# This avoids double-counting weights when comparing current vs next portfolio
```

## Resample Options

| Value | Description |
|-------|-------------|
| `None` | Only rebalance when position changes |
| `'D'` | Daily |
| `'W'` | Weekly (Sunday) |
| `'W-FRI'` | Weekly (Friday) |
| `'M'` | Monthly |
| `'Q'` | Quarterly |
| `'Y'` | Yearly |

## Tips

1. **Weight normalization**: Weights are automatically normalized to sum to 1.0 per date
2. **T+1 execution**: Trades execute at next day's price (realistic simulation)
3. **Boolean signals**: Pass boolean column as position, library converts True to equal weights
4. **Null handling**: Null values in position column are filled with 0.0
5. **resample=None**: Only rebalance when position changes (reduces turnover)
6. **Position limit**: Use `position_limit=0.1` to cap each stock at 10%
