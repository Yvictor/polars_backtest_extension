# Execution Semantics

How the Rust core (`btcore`) actually simulates trading. Grounded in
`btcore/src/simulation/long.rs`, `stops.rs`, `weights.rs`, `position.rs`,
`config.rs`, and `backtest_flow.md`. Use this to reason precisely about when
money moves and at what price.

## The daily loop

For each trading day `t > 0`:

1. **Mark to market** — every open position's cumulative return is updated:
   `cr *= price[t] / price[t-1]`, `maxcr = max(maxcr, cr)`.
2. **Touched exit** (only if `touched_exit=True`) — check OHLC ratios; if a stop
   level was touched intraday, exit **today (T+0)** at the touched level.
3. **Execute pending stop exits** — stops detected on day t-1 exit **today (T+1)
   at today's `trade_at_price`** (not at the stop level).
4. **Detect stops** — compare each position's `cr` at close against thresholds;
   triggered stocks are queued for exit tomorrow.
5. **Execute rebalance** — if day t-1 was a rebalance boundary, the target
   weights computed from t-1's position column are traded **today at today's
   `trade_at_price`** (T+1 execution).
6. **Record balance** — `equity = cash + Σ position values`. The creturn series
   is this equity normalized; recording starts at the first signal.

Consequences:

- A signal in the `position` column on date T is traded on the **next trading
  day** at that day's `trade_at_price`. There is no same-bar fill.
- The last date's signals produce **pending trades** (`entry_date`/`exit_date`
  null in `report.trades`) — that is what `actions()` / `next_weights()` expose
  for live trading.
- Fees are deducted from cash/position value at trade time; there is no
  separate "cost drag" term. The equity curve on rebalance days already nets
  costs.

## Fees and tax

Rebalancing trades **only the delta** per stock: `amount = target_value −
current_value` (positions are marked to market first).

```rust
// btcore/src/simulation/long.rs
let cost = if is_entry {
    amount.abs() * config.fee_ratio                      // increasing exposure
} else {
    amount.abs() * (config.fee_ratio + config.tax_ratio) // reducing / closing
};
```

- `fee_ratio` (default 0.001425): charged on **both** buy and sell notional.
- `tax_ratio` (default 0.003): charged only on the reducing/closing side
  (sell for longs). Full exits and stop exits pay `fee_ratio + tax_ratio` on
  the whole exited value.
- Per-trade `return` in `report.trades` is computed with the same convention:
  entry cost `entry_price*shares*fee_ratio`, exit cost
  `exit_price*shares*(fee_ratio+tax_ratio)`.

Round-trip cost for a full position ≈ `2*fee_ratio + tax_ratio` ≈ 0.585% with
Taiwan defaults. Daily rebalancing of a high-turnover signal is expensive —
check with `fee_ratio=0, tax_ratio=0` to see gross alpha.

## Stops: stop_loss / take_profit / trail_stop

All three are **positive magnitudes** (Finlab convention):

```python
stop_loss=0.1     # exit when position is down 10% from entry
take_profit=0.2   # exit when position is up 20% from entry
trail_stop=0.08   # exit when cr falls 0.08 below its running peak (maxcr)
```

Disabled defaults: `stop_loss=1.0`, `take_profit=inf`, `trail_stop=inf`
(`stop_loss=1.0` gives threshold `1-1.0=0`, unreachable). Never pass a negative
`stop_loss`: the threshold `1 - stop_loss` exceeds 1 and every position stops
out immediately.

Trigger logic (long positions), evaluated on `cr` since `stop_entry_price`:

```
min_r = max(1 - stop_loss, maxcr - trail_stop)
max_r = 1 + take_profit
take profit if cr >= max_r
stop / trail if cr <  min_r
```

Short positions mirror this: `max_r = min(1 + stop_loss, maxcr + trail_stop)`,
`min_r = 1 - take_profit`.

Key details:

- `stop_entry_price` is the **original entry** and is only carried across
  rebalances when `retain_cost_when_rebalance=True`; otherwise cr/maxcr reset
  to 1.0 at every rebalance, so stops measure from the latest rebalance, not
  the original entry.
- Default (non-touched) stop exits are detected at close of day T and executed
  at day T+1's price. The realized loss can therefore exceed `stop_loss` if the
  price gaps down overnight.
- `cr_at_close` is computed as `cr * close / price` (multiply-then-divide) to
  reproduce Finlab's float rounding bit-for-bit at exact threshold boundaries.

### touched_exit=True

Requires `open`, `high`, `low` columns. Changes both detection and timing:

```
open_r = cr/r * (open/prev);  high_r = cr/r * (high/prev);  low_r = cr/r * (low/prev)
priority: open > high > low
open_r >= max_r or open_r <= min_r  -> exit at open        (gap through the level)
high_r >= max_r                     -> exit at max_r level  (take profit touched)
low_r  <= min_r                     -> exit at min_r level  (stop touched)
```

- Exit is **same day (T+0)** at the touched level (position scaled by
  `exit_ratio`), not next day's close. Reported `exit_price = close * exit_ratio`.
- The regular T+1 pending-stop path is disabled while touched_exit is on.

### stop_trading_next_period (default True)

After a stop fires, the symbol goes on a one-period blocklist: it is zeroed out
of the next rebalance's weights and the remaining weights are scaled up to keep
the same total investment. The blocklist clears after that rebalance. Set
`False` to allow immediate re-entry at the next rebalance.

## retain_cost_when_rebalance (default False)

Applies to positions that **continue in the same direction** through a rebalance:

- `True`: the new position snapshot keeps `stop_entry_price`, `max_price`,
  `cr`, `maxcr` — stops keep measuring from the original entry and its peak.
  Use with stop parameters when you want "10% from where I first bought".
- `False`: cr/maxcr reset to 1.0 and stop entry resets to the rebalance price —
  stops measure per rebalance period.

Positions that flip direction or newly enter always reset.

## Weight normalization

Input `position` column dispatches on dtype:

- **Boolean signals**: `True` names get equal weight `1/n_true`, fully invested
  (sum = 1.0). All-False dates = 100% cash. Weights are capped at
  `position_limit` and iteratively re-normalized.
- **Float weights** (Finlab rule): divisor = `max(Σ|w|, 1.0)` per date.
  - `Σ|w| > 1` → scaled down to sum 1.
  - `Σ|w| < 1` → **left as-is**; the shortfall is held as cash. Weights are NOT
    scaled up. `weight=0.5` on one stock means 50% invested, 50% cash.
  - Negative weights = short positions; the abs-sum is what's normalized.
  - After normalization each weight is clamped to `±position_limit` with **no
    re-normalization** — capped excess becomes cash.
- Weights are relative *within a date*; nulls were already filled with 0/False
  by the Python wrapper.

## Resample (rebalance schedule)

- `resample="D"`: rebalance to the target weights every trading day.
  **Footgun (sparse data):** on each day the *entire* holding set is replaced by
  that day's rows. A symbol with no row / null / NaN weight today is treated as
  weight 0 and **fully sold on the next trading day** — there is no
  forward-fill of yesterday's weights. Long frames that only carry weights on
  rebalance days will be liquidated the day after every signal under `"D"`;
  emit a weight for every (date, symbol) you want to keep holding, or use
  `resample=None` / a coarser frequency.
- `"W"`, `"W-MON"`…`"W-SUN"`: weekly at the given weekday boundary (plain `W` =
  Sunday). `"M"`/`"ME"` month end, `"Q"`/`"QE"` quarter end,
  `"Y"`/`"YE"`/`"A"` yearly. `"MS"`/`"QS"` raise `ValueError` — the engine has
  no start-of-period boundaries.
- `resample=None`: **rebalance only when the normalized weights change**
  (Finlab's position-change mode). Best for event-driven signals; minimizes
  turnover.
- Boundaries are detected on the transition between consecutive trading days,
  so a boundary that lands on a holiday fires on the next trading day; multiple
  boundaries inside a long gap are all caught (deduplicated).
- Signals are read from the last trading day at/before the boundary and traded
  T+1 as usual.

### resample_offset

Delays each rebalance by N **calendar** days: signal weights are taken from the
last trading day ≤ (boundary + offset), then executed T+1.

- Accepted formats: `"1d"`, `"2d"`, `"1D"`, `"1W"`, `"2W"` (weeks → 7 days).
- **Only non-negative offsets are supported.** `"-1D"` or any unparseable
  string raises `ValueError` (the wide-format API supports negative offsets).

## Prices, factor, and raw prices

- `trade_at_price` should be an **adjusted** price series (dividends/splits
  folded in) or returns will be wrong across ex-dividend dates.
- Provide `factor = adj_price / raw_price` so the engine can reconstruct raw
  prices: `raw = adj / factor`. Raw prices populate
  `entry_raw_price`/`exit_raw_price` in trades and drive the `buyHigh`/`sellLow`
  limit-price checks and `capacity`. Missing factor column → factor = 1.0
  (raw = adjusted), silently.
- Prices that are NaN or ≤ 0 are treated as invalid and skipped (position
  carried at last valid price); pre-filter obviously bad rows anyway.

## Precision vs Finlab

The engine replicates Finlab's cumulative `cr *= r` float accumulation and its
`cr * close / price` formula, achieving max diff ~2.2e-16 vs Finlab in
comparison tests (`tests/test_wide_vs_finlab.py`). At exact stop boundaries a
1-bit difference can flip a trigger; this is inherent to the float convention,
not a bug.
