# Migrating from Finlab `sim()`

Grounded in `FINLAB_PARAMS_COMPARISON.md` and `backtest_flow.md`. The engine
replicates Finlab's calculation (max diff ~2.2e-16 in parity tests), so a
correct migration should reproduce Finlab's creturn almost exactly.

## Shape change: wide → long

Finlab uses wide pandas DataFrames (index=date, columns=stock ids). polars-backtest's
primary API is **long format**: one row per (date, symbol).

```python
# Finlab
close = data.get("price:收盤價")                       # wide, pandas
position = close >= close.rolling(300).max()          # wide bool
report = backtest.sim(position, resample="M")

# polars-backtest — same strategy in long format
df = df.with_columns(
    (pl.col("close") >= pl.col("close").rolling_max(300).over("symbol"))
    .alias("weight")                                   # bool column = signals
)
report = df.bt.backtest_with_report(position="weight", resample="M")
```

Rules of thumb:

- Wide `rolling(...)` → long `rolling_*(...).over("symbol")`.
- Wide `rank(axis=1)` → long `.rank().over("date")`.
- Wide `shift(1)` → long `.shift(1).over("symbol")`.
- Bool position DataFrame → Boolean column (equal-weight conversion is identical).
- Float weights keep Finlab semantics: Σ|w| > 1 normalized down, Σ|w| < 1 kept
  (remainder in cash).

If you already hold wide polars DataFrames, `pl_bt.backtest_wide(prices, position, ...)`
and `pl_bt.backtest_with_report_wide(close, position, ...)` accept them directly
(first column = date, other columns = symbols). Long format is faster and preferred.

## Parameter mapping

The long-format engine always uses finlab-style accounting (the former
`finlab_mode` parameter was a no-op on this path and has been removed).

| Finlab `sim()` | polars-backtest | Notes |
|---|---|---|
| `position` (wide df) | `position=` column/Expr in long df | bool → equal weights; float → Finlab normalization |
| `resample` | `resample` | Same strings: `None`, `'D'`, `'W'`, `'W-FRI'`, `'M'`, `'Q'`, `'Y'`. `'MS'`/`'QS'` raise `ValueError` in the long format. `None` = trade on position change |
| `resample_offset` | `resample_offset` | **Non-negative offsets only** (`"1d"`, `"1W"`). Finlab's `'-1D'` raises `ValueError` in the long format (use the wide API for negative offsets) |
| `trade_at_price` | `trade_at_price` | Here it's a *column name/Expr* (long) instead of `'close'/'open'` string on wide data |
| `fee_ratio` (0.001425) | `fee_ratio` (0.001425) | identical |
| `tax_ratio` (0.003) | `tax_ratio` (0.003) | identical |
| `stop_loss` (None) | `stop_loss` (1.0 = off) | Same positive-fraction convention (0.1 = −10%) |
| `take_profit` (None) | `take_profit` (inf = off) | identical convention |
| `trail_stop` (None) | `trail_stop` (inf = off) | drop from peak: `maxcr − trail_stop` |
| `touched_exit` (False) | `touched_exit` (False) | Requires `open`/`high`/`low` columns in the long df |
| `position_limit` (1.0) | `position_limit` (1.0) | Same clip behavior |
| `retain_cost_when_rebalance` (False) | same name | identical |
| `stop_trading_next_period` (True) | same name | identical |
| — | `factor` | polars-backtest extra: adjustment factor column, `raw = adj / factor` |
| — | `benchmark` | polars-backtest extra: symbol string or (date, creturn) DataFrame |

## Unsupported Finlab parameters

No equivalent — remove them when migrating:

- `mae_mfe_window`, `mae_mfe_window_step` (windowed MAE/MFE analysis; the
  per-trade `mae`/`gmfe`/`bmfe`/`mdd`/`pdays` columns ARE provided in
  `report.trades`)
- `fast_mode`
- Service/metadata params: `name`, `upload`, `notification_enable`,
  `line_access_token`, `live_performance_start`, `market`

## Report object mapping

| Finlab Report | polars-backtest BacktestReport | Notes |
|---|---|---|
| `report.creturn` (pd.Series) | `report.creturn` (pl.DataFrame: date, creturn) | |
| `report.trades` (pd.DataFrame) | `report.trades` (pl.DataFrame) | Same columns incl. `entry_sig_date`, `exit_sig_date`, `mae`, `gmfe`, `bmfe`, `mdd`, `pdays` |
| `report.get_stats()` | `report.get_stats()` / `report.stats` | polars 1-row DataFrame |
| `report.get_metrics()` | `report.get_metrics(sections=[...])` | camelCase metric names (`annualReturn`, `maxDrawdown`, …) |
| `report.position_info()` | `report.actions()` + `report.weights()` + `report.next_weights()` | Long report splits it; wide-format `Report` also has `position_info()` |
| `report.current_trades` | `report.current_trades()` | method, not property |
| live signals for orders | `report.actions()` → enter/exit/hold with `weight`/`next_weight` | pending trades have null `entry_date`/`exit_date` |
| benchmark via market | `benchmark="0050"` param or `report.benchmark = df` | df needs `date` + `creturn` |

## Data prep for parity

Finlab feeds `sim()` adjusted prices internally. To match:

1. Build an adjusted close per symbol (cumulative product of returns) and keep
   the raw close.
2. `factor = adj_close / raw_close`; pass `factor="factor"`.
3. Cast `date` to `pl.Date`, sort by date, dedupe (symbol, date).

```python
df = (
    raw.unique(subset=["symbol", "date"])
       .sort("symbol", "date")
       .with_columns((pl.col("close") / pl.col("ref_price") - 1).alias("ret"),
                     pl.col("close").alias("close_raw"))
       .with_columns((pl.col("ret") + 1).cum_prod().over("symbol").alias("close"))
       .with_columns((pl.col("close") / pl.col("close_raw")).alias("factor"))
       .sort("date", "symbol")
)
```

## Verifying a migration

Compare curves directly:

```python
finlab_creturn = finlab_report.creturn                     # pandas Series
pb = report.creturn                                        # polars DataFrame
merged = pb.join(
    pl.from_pandas(finlab_creturn.rename("finlab").reset_index()
                   .rename(columns={"index": "date"})).with_columns(pl.col("date").cast(pl.Date)),
    on="date", how="inner",
)
max_diff = (merged["creturn"] - merged["finlab"]).abs().max()
```

Expect ~1e-15 differences. Larger gaps usually mean: different adjusted-price
construction, unsynchronized universes (missing symbols/dates), or a stop
parameter passed with the wrong sign.
