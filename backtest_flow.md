# Backtest Flow: Finlab vs Polars Backtest

Comparison between:
- **Finlab**: `backtest.py` + `backtest_core.pyx` (Python/Cython)
- **Polars Backtest**: `namespace.py` + `btcore/` (Python/Rust)

## Architecture

### Finlab
```
User → sim() → arguments() → backtest_() → get_trade_stocks() → Report
         │          │            │                  │
         │          │            ├─ Main loop       ├─ trades
         │          │            ├─ rebalance()     └─ stock_operations
         │          │            └─ set_position()
         │          │
         │          └─ Price arrays (numpy)
         │             Position arrays (numpy)
         │             Resample dates (numpy)
         │
         └─ Resample handling (pandas)
            Market price fetching
            Line notification
```

### Polars Backtest
```
User → df.bt.backtest() / pl_bt.backtest()
         │
         ├─ namespace.py
         │    ├─ Column resolution (Expr support)
         │    ├─ BacktestConfig
         │    └─ Resample validation
         │
         ├─ polars-arrow → arrow-rs (FFI, zero-copy)
         │
         └─ btcore::backtest_long_arrow
              ├─ Main loop
              ├─ detect_stops_finlab()
              ├─ detect_touched_exit()
              └─ execute_rebalance()
```

---

## Main Loop Flow

### Each Day (t > 0):

1. **Update positions**: `cr *= r` where `r = price[t] / prev_price`
2. **Touched exit** (if enabled): Check OHLC for intraday stops → immediate exit (T+0)
3. **Detect stops**: Check cr vs thresholds → queue for T+1 exit
4. **Execute pending stops** (yesterday's T+1)
5. **Execute rebalance** (if signal from T-1)
6. **Record balance**

### Key Formula

```
cr[t] = cr[t-1] * (price[t] / price[t-1])
maxcr = max(maxcr, cr)

# Stop thresholds (long position)
max_r = 1 + take_profit
min_r = max(1 - stop_loss, maxcr - trail_stop)

# Trigger
if cr >= max_r → take_profit
if cr < min_r → stop_loss / trail_stop
```

---

## Touched Exit (OHLC)

```
# Calculate OHLC ratios relative to cumulative return
open_r = cr / r * (open / prev_price)
high_r = cr / r * (high / prev_price)
low_r  = cr / r * (low / prev_price)

# Priority: open > high > low
if open_r >= max_r or open_r <= min_r → exit at open
elif high_r >= max_r → exit at max_r (take profit)
elif low_r <= min_r → exit at min_r (stop loss)
```

---

## Rebalance

```
balance = cash + Σ(position_values)
ratio = balance / Σ|target_weights|

for each target:
    target_value = weight * ratio
    amount = target_value - current_value

    if is_entry:
        cost = |amount| * fee_ratio
    else:
        cost = |amount| * (fee_ratio + tax_ratio)
```

### retain_cost_when_rebalance
- `True`: Keep cr/maxcr for continuing positions (same direction)
- `False`: Reset cr=1, maxcr=1 for all positions

### stop_trading_next_period
- `True`: Zero out stopped stocks, rescale remaining weights
- `False`: Allow stopped stocks to re-enter on rebalance

---

## Code Organization

### Finlab
```
finlab/
├── backtest.py           # sim(), resample, market
└── core/
    ├── backtest_core.pyx # Core loop, rebalance
    └── mae_mfe.pyx       # Trade tracking
```

### Polars Backtest
```
polars_backtest/
├── python/polars_backtest/
│   ├── namespace.py      # df.bt API, backtest(), backtest_with_report()
│   ├── wide.py           # Wide format API (Finlab compatible)
│   └── utils.py          # Resample helpers
├── src/
│   ├── lib.rs            # PyO3 bindings, Arrow FFI
│   └── ffi_convert.rs    # polars-arrow → arrow-rs
└── btcore/src/
    ├── simulation/
    │   ├── wide.rs       # Wide format backtest
    │   └── long.rs       # Long format backtest (Arrow)
    ├── stops.rs          # Stop detection
    ├── rebalance.rs      # Rebalance execution
    └── tracker.rs        # Trade tracking trait
```

---

## Feature Comparison

| Feature | Finlab | Polars Backtest |
|---------|--------|-----------------|
| T+1 execution | ✅ | ✅ |
| stop_loss / take_profit / trail_stop | ✅ | ✅ |
| touched_exit (OHLC) | ✅ | ✅ |
| retain_cost_when_rebalance | ✅ | ✅ |
| stop_trading_next_period | ✅ | ✅ |
| position_limit | ✅ | ✅ |
| Short positions | ✅ | ✅ |
| resample (D/W/M/Q/Y) | ✅ | ✅ |
| resample_offset | ✅ | ✅ |
| MAE/MFE analysis | ✅ | ❌ |
| Cloud upload | ✅ | ❌ |

---

## Precision Notes

Finlab uses cumulative multiplication (`cr *= r`) which accumulates floating point differently:
- At exact stop boundaries (e.g., stop_loss=0.05), 1-bit precision differences can cause different trigger behavior
- Our implementation replicates Finlab's `cr_at_close = cr * close / price` formula to match
