# Recipes

Realistic, tested patterns. All examples use the real API; column construction
follows production usage patterns (long format, expression-based, per-date and
per-symbol windowing).

## 1. Data preparation (adjusted prices + factor)

Build the canonical backtest frame once, reuse for every strategy:

```python
import polars as pl
import polars_backtest as pl_bt

df = (
    raw_ohlcv
    .unique(subset=["symbol", "date"])            # dedupe defensively
    .sort("symbol", "date")
    .with_columns(
        pl.col("close").alias("close_raw"),
        (pl.col("close") / pl.col("prev_ref_price") - 1).alias("ret"),
    )
    .with_columns((pl.col("ret") + 1).cum_prod().over("symbol").alias("close"))  # adjusted
    .with_columns((pl.col("close") / pl.col("close_raw")).alias("factor"))
    .filter(pl.col("close").is_not_null())
    .sort("date", "symbol")                        # engine wants date-sorted
)
```

Checklist: `date` as `pl.Date`, one row per (date, symbol), adjusted `close`,
`factor` column, sorted by date. Drop symbols with too little history for your
indicators: `.filter(pl.len().over("symbol") >= 50)`.

## 2. Momentum with monthly rebalance

```python
df = df.with_columns(
    (pl.col("close") >= pl.col("close").rolling_max(300).over("symbol"))
    .alias("weight")                       # Boolean -> equal weights among True
)
report = df.bt.backtest_with_report(position="weight", resample="M", factor="factor")
print(report.stats)
```

## 3. Top-N ranking strategy (cross-sectional)

Rank per date; keep the top N. Mask excluded names to **null before ranking**
so they leave the cross-section instead of ranking at the bottom:

```python
n = 20
df = (
    df.with_columns(
        (pl.col("close") / pl.col("close").shift(60).over("symbol") - 1).alias("score")
    )
    # liquidity universe first, null out the rest (see recipe 4)
    .with_columns(
        pl.when(liquid).then(pl.col("score")).otherwise(None).alias("score")
    )
    .with_columns(
        (pl.col("score").rank(descending=True, method="ordinal").over("date") <= n)
        .fill_null(False)
        .alias("weight")                   # Boolean -> equal weight top-N
    )
)
report = df.bt.backtest_with_report(position="weight", resample="M")
```

Score-proportional weights instead of equal weight: emit floats that sum ≤ 1
per date (`score / score.sum().over("date")` after masking losers to 0).

## 4. Liquidity / universe masks

Boolean masks compose with `&`. Standard liquidity proxy: raw trading value.

```python
trading_value = pl.col("close_raw") * pl.col("volume")

liquid = (
    trading_value.rank(method="ordinal", descending=True).over("date") <= 500
)                                          # top-500 by trading value
min_price = pl.col("close_raw") > 10.0

df = df.with_columns(
    (pl.col("signal") & liquid & min_price).alias("weight")
)
```

Provide a trading-value column to unlock the `capacity` metric (the column
name is configurable via the `trading_value` parameter), and add
`limit_up`/`limit_down` raw-price columns to get `buyHigh`/`sellLow`:

```python
df = df.with_columns(trading_value.alias("txn_value"))
report = df.bt.backtest_with_report(
    position="weight", resample="M", trading_value="txn_value"
)
report.get_metrics(sections=["liquidity"])   # buyHigh, sellLow, capacity
```

Restrict the trading window without losing indicator warm-up rows:

```python
df = df.with_columns(
    pl.when(pl.col("date") >= pl.date(2018, 1, 1))
    .then(pl.col("weight")).otherwise(0.0).alias("weight")
)
```

## 5. Stop loss / take profit / trailing stop

```python
report = df.bt.backtest_with_report(
    position="weight",
    resample="M",
    stop_loss=0.1,       # exit at -10% (POSITIVE number)
    take_profit=0.3,     # exit at +30%
    trail_stop=0.15,     # exit 15% below peak since entry
)
```

Intraday detection with same-day exit at the stop level:

```python
report = df.bt.backtest_with_report(
    position="weight", resample="M",
    stop_loss=0.1, touched_exit=True,      # requires open/high/low columns
)
```

Make stops measure from the original entry across rebalances:

```python
report = df.bt.backtest_with_report(
    position="weight", resample="M",
    stop_loss=0.15, retain_cost_when_rebalance=True,
)
```

Inspect stop behavior: `report.trades.filter(pl.col("return") <= -0.09)`,
`report.is_stop_triggered()`.

## 6. Long/short evaluation

Run the legs separately and compare:

```python
q_hi = pl.col("score") >= pl.col("score").quantile(0.8).over("date")
q_lo = pl.col("score") <= pl.col("score").quantile(0.2).over("date")

long_rpt  = df.with_columns(q_hi.alias("weight")).bt.backtest_with_report(
    position="weight", resample="M")
short_rpt = df.with_columns(
    pl.when(q_lo).then(-1.0).otherwise(0.0).alias("weight")
).bt.backtest_with_report(position="weight", resample="M")
ls_rpt    = df.with_columns(
    pl.when(q_hi).then(1.0).when(q_lo).then(-1.0).otherwise(0.0).alias("weight")
).bt.backtest_with_report(position="weight", resample="M")
```

Negative float weights open short positions; normalization uses Σ|w|.

## 7. Comparing strategies / robustness sweeps

Collect one stats row per variant and sort:

```python
rows = []
for resample in ["W", "M", None]:
    for offset in [None, "5d", "10d", "15d"]:
        if resample != "M" and offset:
            continue
        try:
            r = df.bt.backtest_with_report(
                position="weight", resample=resample, resample_offset=offset)
        except Exception as e:            # thin universes can raise
            print(f"{resample}/{offset}: {e}")
            continue
        rows.append(r.stats.with_columns(
            pl.lit(str(resample)).alias("resample"),
            pl.lit(str(offset)).alias("offset"),
        ))

summary = pl.concat(rows, how="diagonal").sort("calmar", descending=True)
```

An offset sweep on monthly rebalancing measures start-date sensitivity — a
robust strategy shouldn't live or die on which day of the month it trades.
Remember offsets are calendar days, non-negative only.

## 8. Benchmark, alpha/beta

```python
# Symbol present in your data (e.g. Taiwan 0050 ETF)
report = df.bt.backtest_with_report(position="weight", resample="M", benchmark="0050")
report.get_metrics(sections=["profitability", "winrate"])   # alpha, beta, m12WinRate

# Or attach later
report.benchmark = bm_df                  # needs 'date' and 'creturn' columns
```

## 9. Live trading: what to buy/sell tomorrow

The last date's signals are pending trades. Get actionable orders:

```python
report = df.bt.backtest_with_report(position="weight", resample="M")

actions = report.actions()   # symbol, action(enter/exit/hold), weight, next_weight, dates
to_buy  = actions.filter(pl.col("action") == "enter")
to_sell = actions.filter(pl.col("action") == "exit")

current = report.weights()        # held now, sum <= 1
target  = report.next_weights()   # next period target, sum <= 1
```

Order sizing: `shares ≈ capital * next_weight / raw_price`.

## 10. Cost sensitivity

Always sanity-check gross vs net:

```python
gross = df.bt.backtest(position="weight", resample="M", fee_ratio=0.0, tax_ratio=0.0)
net   = df.bt.backtest(position="weight", resample="M")   # TW defaults
```

If gross is good but net is flat, reduce turnover: `resample="M"` or
`resample=None`, stickier signals, or hysteresis (enter top 10%, exit only
below top 30%).

## 11. Quick equity-curve-only runs

For optimization loops where you only need the curve, `backtest()` (no report)
is cheaper than `backtest_with_report()`:

```python
creturn = df.bt.backtest(position="weight", resample="M")   # DataFrame [date, creturn]
final = creturn["creturn"][-1]
```

Combine with the plugin expressions for custom scoring:

```python
from polars_backtest import daily_returns, sharpe_ratio, max_drawdown
score = creturn.with_columns(ret=daily_returns("creturn")).select(
    sharpe=sharpe_ratio("ret"), mdd=max_drawdown("creturn"))
```
