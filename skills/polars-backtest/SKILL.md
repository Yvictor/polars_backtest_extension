---
name: polars-backtest
description: Backtest trading strategies with the polars-backtest library (Rust-powered Polars extension, Finlab-compatible). Use when the user asks about backtesting, portfolio simulation, trading strategy evaluation, stop loss/take profit simulation, rebalancing, migrating from finlab, or anything involving polars_backtest / df.bt. 使用 polars-backtest 回測交易策略。當使用者提到回測、投資組合模擬、策略績效評估、停損停利、再平衡、finlab 遷移或 polars_backtest 時使用。
---

# polars-backtest

High-performance portfolio backtesting for Polars long-format DataFrames.
Rust core, Finlab-compatible semantics (parity ~2e-16), T+1 execution.

```bash
pip install polars-backtest    # or: uv add polars-backtest
```

## Reference files — read when needed

| File | When to read |
|---|---|
| [references/api.md](references/api.md) | Full parameter reference, BacktestReport surface, trades schema, wide-format API, plugin expressions |
| [references/execution-semantics.md](references/execution-semantics.md) | Exact timing/price/fee/stop mechanics — read before answering "when does it trade / at what price / how are fees applied" |
| [references/finlab-migration.md](references/finlab-migration.md) | Porting finlab `sim()` code; param and Report mapping; unsupported params |
| [references/recipes.md](references/recipes.md) | Data prep, top-N ranking, masks, long/short, sweeps, live-trading actions |

## Data requirements

**Long format**: one row per (date, symbol). Required columns (names remappable
via parameters, values can also be `pl.Expr`):

| Column | Dtype | Notes |
|---|---|---|
| `date` | `pl.Date` preferred (ISO strings accepted) | Sort by date; unsorted input is auto-sorted but slower |
| `symbol` | `str` | |
| `close` (= `trade_at_price`) | `Float64` | Use **adjusted** prices; NaN/≤0 treated as invalid |
| `weight` (= `position`) | `Boolean` or `Float64` | Bool → equal weights among True. Float → Finlab normalization. Nulls auto-filled with False/0.0 |

Optional columns picked up when present / when features need them:

- `factor` — adjustment factor, `raw_price = adj_price / factor`. Missing column
  silently means factor=1.0.
- `open`, `high`, `low` — required only when `touched_exit=True`.
- `limit_up`, `limit_down`, `trading_value` — enable liquidity metrics
  (`buyHigh`, `sellLow`, `capacity`). Column names are configurable via the
  `limit_up`/`limit_down`/`trading_value` parameters of `backtest_with_report`.

## Quick start

```python
import polars as pl
import polars_backtest as pl_bt   # registers the df.bt namespace

# Boolean signal -> equal-weight portfolio, monthly rebalance
df = df.with_columns(
    (pl.col("close") >= pl.col("close").rolling_max(60).over("symbol"))
    .alias("weight")
)

report = df.bt.backtest_with_report(position="weight", resample="M")
report.stats          # 1-row DataFrame: total_return, cagr, max_drawdown, daily_sharpe, ...
report.creturn        # DataFrame [date, creturn]
report.trades         # per-trade records with mae/gmfe/bmfe/mdd/pdays

# Equity curve only (faster, for optimization loops)
curve = df.bt.backtest(position="weight", resample="M")   # DataFrame [date, creturn]

# Function form is identical: pl_bt.backtest(df, ...) / pl_bt.backtest_with_report(df, ...)
```

## Parameters at a glance

Exact signature — see [references/api.md](references/api.md) for full semantics.

| Parameter | Default | Meaning |
|---|---|---|
| `trade_at_price` | `"close"` | Price column/Expr for valuation and execution |
| `position` | `"weight"` | Weight column/Expr (bool signals or float weights) |
| `date`, `symbol` | `"date"`, `"symbol"` | Column names/Exprs |
| `open`, `high`, `low` | `"open"…` | Used only with `touched_exit=True` |
| `factor` | `"factor"` | str only; raw = adj / factor; missing → 1.0 |
| `resample` | `"D"` | `None`, `D`, `W`, `W-MON…W-SUN`, `M`, `ME`, `Q`, `QE`, `Y`, `YE`, `A`. `MS`/`QS` raise `ValueError` (no start-of-period support). `None` = trade only on position change |
| `resample_offset` | `None` | Delay rebalance by calendar days: `"1d"`, `"2d"`, `"1W"`. **Non-negative only; bad strings silently ignored** |
| `fee_ratio` | `0.001425` | Fee on buy AND sell notional (TW default) |
| `tax_ratio` | `0.003` | Tax on sell side only (TW default) |
| `stop_loss` | `1.0` = off | **Positive fraction**: `0.1` exits at −10% |
| `take_profit` | `inf` = off | `0.2` exits at +20% |
| `trail_stop` | `inf` = off | `0.08` exits 0.08 below peak cr since entry |
| `touched_exit` | `False` | Intraday OHLC stop detection, same-day exit at the stop level |
| `stop_trading_next_period` | `True` | Stopped stock excluded from the next rebalance |
| `position_limit` | `1.0` | Max weight per stock (clip; float-weight excess → cash) |
| `retain_cost_when_rebalance` | `False` | `True`: stops keep measuring from original entry across rebalances |
| `benchmark` (report only) | `None` | Symbol string in your data, or DataFrame with `date`+`creturn` → enables alpha/beta/m12WinRate |
| `limit_up`, `limit_down` (report only) | `"limit_up"`, `"limit_down"` | Raw limit-price columns for liquidity metrics |

## Execution model (the 20-second version)

Full mechanics: [references/execution-semantics.md](references/execution-semantics.md).

- **T+1**: a signal on date T trades on the next trading day at that day's
  `trade_at_price`. No same-bar fills. Last-date signals become *pending trades*
  (null `entry_date`/`exit_date` in `report.trades`).
- **Delta trading**: rebalances trade only `target_value − current_value` per
  stock. Fees: `fee_ratio` on the traded amount both ways, `+ tax_ratio` when
  reducing/closing. Full round trip ≈ 0.585% at TW defaults.
- **Stops**: evaluated on cumulative return since entry at each close;
  triggered stocks exit **next day at next day's price** (gaps can overshoot
  the threshold). With `touched_exit=True`, OHLC is checked intraday and the
  exit happens **same day at the touched level** (priority open > high > low).
- **Weights per date**: bool → equal weight (sum=1). Float → Σ|w|>1 scaled down
  to 1; **Σ|w|<1 kept as-is, remainder in cash**. Negative weights = shorts.
- **creturn** starts at 1.0 from the first signal date.

## Report essentials

Full surface: [references/api.md](references/api.md).

```python
report.get_stats(riskfree_rate=0.02)     # = report.stats
report.get_monthly_stats()
report.get_return_table()                # year x month returns
report.get_metrics(sections=["backtest", "profitability", "risk",
                             "ratio", "winrate", "liquidity"])
report.actions()        # enter/exit/hold + weight/next_weight  (live trading)
report.weights()        # current holdings, normalized, sum <= 1
report.next_weights()   # next-period targets, sum <= 1
report.current_trades() # open + pending + just-exited trades
report.is_stop_triggered()
report.benchmark = bm_df                 # attach later; needs date + creturn
```

## Common pitfalls

1. **stop_loss sign**: pass `stop_loss=0.1` for a 10% stop. The threshold
   formula is `1 - stop_loss`, so a negative value (e.g. `-0.1`) makes the
   threshold > 1 and stops every position out immediately.
2. **Look-ahead bias**: any feature must use only data available at signal
   time. Use `.shift(k).over("symbol")` for time series and never rank on
   same-day future info. The engine's T+1 handles execution lag, not your
   feature construction.
3. **Unadjusted prices**: raw close across ex-dividend/split dates fabricates
   losses. Feed adjusted prices to `trade_at_price` and supply `factor` so raw
   prices are still available for trade records and liquidity metrics.
4. **Weights that don't sum to 1**: float weights with Σ|w| < 1 are NOT scaled
   up — the rest sits in cash. If you want full investment, normalize per date:
   `w / w.abs().sum().over("date")`, or just use a Boolean column.
5. **Nulls in position**: auto-filled with 0/False, which is usually right for
   rolling-window warmups. But for *ranking*, mask excluded stocks to **null
   before ranking** so they leave the cross-section, then `fill_null(False)` on
   the final signal.
6. **Silent no-ops**: a misspelled `resample_offset` (e.g. `"-1d"`, `"5days"`)
   is ignored without error; a missing `factor` column means factor=1.0 without
   error. Verify both when results look suspicious.
7. **Duplicate (date, symbol) rows**: dedupe with
   `.unique(subset=["symbol", "date"])` before backtesting.
7b. **Sparse weights + `resample="D"` liquidate**: a symbol missing its row (or
   null/NaN weight) on a day is sold the next day — weights are NOT
   forward-filled. Emit weights on every held date or use a coarser resample.
8. **Nulls in date/symbol raise**: fill or drop them first. Null prices are
   treated as missing data (converted to NaN internally) — no error, the
   position is simply carried without a return that day.
9. **Empty universe**: if masking leaves zero weights everywhere,
   `backtest_with_report` returns an empty report (height-0 creturn/trades);
   thin universes in sweeps may raise — wrap sweep iterations in try/except.
10. **Costs eat the alpha**: check gross first
    (`fee_ratio=0, tax_ratio=0`), then reduce turnover (`resample="M"` or
    `None`) rather than dropping the cost assumptions.

## Finlab migration

polars-backtest reproduces Finlab `sim()` bit-for-bit in parity tests. Wide
pandas position → long Boolean/float column; `rolling` → `.over("symbol")`;
cross-sectional ops → `.over("date")`. Same parameter names for
fee/tax/stops/resample; `mae_mfe_window*`, `fast_mode`, and upload/notification
params are unsupported. Full mapping and verification recipe:
[references/finlab-migration.md](references/finlab-migration.md).

## Wide-format API

`pl_bt.backtest_wide(prices, position, ...)` and
`pl_bt.backtest_with_report_wide(close, position, ...)` accept Finlab-style wide
DataFrames (first column date, one column per symbol). Prefer long format —
wide exists mainly for Finlab parity. Details in
[references/api.md](references/api.md).
