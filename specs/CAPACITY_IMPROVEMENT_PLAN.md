# Capacity (胃納量) / Liquidity Metrics — Analysis & Improvement Plan

Source: deep-dive research 2026-07-04 (verified against installed finlab 1.5.5).

## Current implementation

- `get_metrics(sections=["liquidity"])` emits `buyHigh`, `sellLow` (from
  `report.rs::calc_liquidity_metrics`, lines ~1474-1572) and `capacity`
  (`calc_capacity`, ~1580-1650). Missing input columns → null metrics, no error.
- `limit_up`/`limit_down` are **raw price levels**; comparison against
  `entry_raw_price`/`exit_raw_price` (= adj/factor) is exact float `>=`/`<=`.
- capacity (faithful finlab port, verified against `finlab/core/metrics.pyx`):

  ```
  amf(trade) = (TV@entry × 0.05 / |position| + TV@exit × 0.05 / |position|) / 2
  capacity   = quantile(amf over closed trades, 0.10, linear)   # unit = unit of trading_value (TWD 元)
  ```

  `percentage_of_volume=0.05` hardcoded; open trades and trades with missing TV excluded.

## Finlab reference semantics (differences to know)

- finlab's `buy_high`/`sell_low` do **not** use limit prices — they use
  `pct_change vs adjusted prev close > 0.95 × limit range` (7% pre-2015/6/1,
  10% after), are **direction-aware** (long entry at top / short cover at top,
  etc.), evaluate **both legs**, and take `max(entry_leg, exit_leg)`.
  For long-only, polars' entry-only buyHigh / exit-only sellLow coincides.
- finlab returns `0` when data missing; polars returns null (better).
- finlab also reports low-volume/low-turnover/警示/處置/全額交割 ratios
  (`etl:is_flagged_stock`) — no dalpha data source for the flags yet.

## Confirmed bugs

- **B1 (FIXED 2026-07-04)**: `trading_value` param not forwarded in namespace
  while `.pyi` advertised it → forwarded now; tests in `test_input_validation.py`.
- **B2 (FIXED 2026-07-04, Stage B)**: buyHigh/sellLow are now direction-aware
  with two legs and max-aggregation.
- **B3 (FIXED 2026-07-04, Stage B)**: comparisons use 1e-6 relative tolerance.
- **B4 (FIXED 2026-07-04)**: no dedup of `(date,symbol)` in the extracted
  limit/TV frames → `.unique(keep=first)` added in `lib.rs`.

## Formula-inherited flaws (also present in finlab — improvement targets)

a. Same-day TV includes the strategy's own volume; on limit-locked days TV≈0
   yet the fill is assumed to happen.
b. `(entry+exit)/2` averaging overstates — binding constraint is `min(entry, exit)`.
c. Per-trade quantile ignores same-day concurrent entries binding jointly.
d. Weight drift / intermediate rebalances between entry and exit not modeled.
e. No 處置股 haircut (batch auctions cut participatable volume).

## Plan

### Stage A — plumbing (no semantic change) — DONE 2026-07-04
1. Forward `trading_value` param. ✅
2. Dedup extracted frames on `(date,symbol)`. ✅
3. Docs updated (skill references, FINLAB_PARAMS_COMPARISON.md). ✅
   report.rs `get_metrics` docstring mentions "liquidity". ✅

### Stage B — finlab-faithful metric corrections — DONE 2026-07-04
(changes buyHigh/sellLow for shorts only; tests in tests/test_capacity_methods.py)
```
adverse_up(px, lim) = px >= lim * (1 - 1e-6)
adverse_dn(px, lim) = px <= lim * (1 + 1e-6)
buyHigh = max( mean(pos>0 & adverse_up(entry_raw, limit_up@entry)),
               mean(pos<0 & adverse_up(exit_raw,  limit_up@exit)) )
sellLow = max( mean(pos<0 & adverse_dn(entry_raw, limit_down@entry)),
               mean(pos>0 & adverse_dn(exit_raw,  limit_down@exit)) )
```
Long-only results unchanged except epsilon. Keep default `capacity` formula
byte-compatible with finlab.

### Stage C — better capacity (opt-in report method) — DONE 2026-07-04
`report.capacity(percentage_of_volume=0.05, quantile=0.1, window=20,
method="finlab" | "min_leg" | "adv")`:
- `min_leg`: `pov × min(TV@entry, TV@exit) / |position|` → fixes flaw (b).
- `adv`: rolling median of TV over `window` days ending at the **signal** date
  → removes own-impact + limit-lock zero-volume artifacts (a); production-like.
- `report.capacity_by_date()` diagnostic: per rebalance date,
  `NAV_max(d) = min over entering trades of cap_entry` — addresses (c) without
  changing the scalar.
- All outputs in the unit of `trading_value` (TWD 元) — document.

### dalpha data wiring
| backtest column | source | note |
|---|---|---|
| `trading_value` | `txn_value` (TwStkPriceDaily.txnValue) | **千元 — multiply by 1000 before passing** (verified 2026-07-05 vs close×volume); capacity expects TWD 元 |
| `limit_up`/`limit_down` | **TwStkRefPrice.upPrice/dnPrice** (same-day, exact incl. ex-div) | extend `load_ref_price`; keep raw, keep `factor` column in input |
| alt | `oNextUp`/`oNextDn` shifted +1 trading day | fallback |
| 處置股 flag | not in catalog | future data source |

### Test plan (fast suite, hand-computable)
1. Forwarding (done). 2. Capacity value fixture: trades with
   TV=(1M,2M),|pos|=.5 → amf=150k and TV=1M,|pos|=.25 → 200k;
   `quantile(0.1)`=155k (`min_leg`: 110k). 3. buyHigh with `factor=1.03` +
   epsilon at exact limit. 4. sellLow via stop_loss on a −11% day.
5. Shorts after Stage B. 6. duplicate rows → same result post-dedup;
   missing exit-day TV → excluded, capacity finite.
7. Slow-suite parity vs finlab `LiquidityAnalysis` (exact for method="finlab").
