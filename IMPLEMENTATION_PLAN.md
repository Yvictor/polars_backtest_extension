# Polars Native Backtest Implementation

## Architecture

```
User API (Long Format)
    ↓
pl_bt.backtest() / df.bt.backtest()
    ↓
Rust (partition_by, zero-copy)
    ↓
btcore (pure Rust, no polars dependency)
```

## API

```python
import polars as pl
import polars_backtest as pl_bt

# Long format input
df = pl.DataFrame({
    "date": [...],
    "symbol": [...],
    "close": [...],
    "weight": [...],
})

# Function API
result = pl_bt.backtest(df, trade_at_price="close", position="weight")
report = pl_bt.backtest_with_report(df, trade_at_price="close", position="weight")

# DataFrame namespace
result = df.bt.backtest(trade_at_price="close", position="weight")
report = df.bt.backtest_with_report(trade_at_price="close", position="weight")
```

---

## Completed Stages

### Stage 1-3: Long Format API ✓
- `df.bt` namespace
- Long → Wide conversion in Python
- Validated against Finlab

### Stage 4: Native Long Format in Rust ✓
- `partition_by` instead of pivot (1.5x faster)
- Zero-copy with `cont_slice()`
- All tests pass (81 fast + 86 slow)

**Performance (real data ~10M rows):**
| Method | Time |
|--------|------|
| Wide format | ~0.77s |
| Long format (partition_by) | ~1.75s |

---

## Remaining Work

### Stage 5: Unify TradeTracker

**Status**: 5.1 Complete, 5.2 Pending

**5.1 ✓**: Unified `TradeTracker` trait with associated types
```rust
pub trait TradeTracker {
    type Key: Clone + Eq + Hash;
    type Date: Copy;
    type Record;
    // ...
}
```

**5.2**: Eliminate duplicate code in `long.rs` (~400 lines)
- [ ] Unify `execute_rebalance` / `execute_rebalance_with_tracker`
- [ ] Unify `execute_pending_stops` / `execute_pending_stops_with_tracker`
- [ ] Create `simulate_backtest_long<T: TradeTracker>` generic function

### Known Issue: test_trades_match
- Wide: 6430 trades vs Long: 5729 trades (10.9% diff)
- Investigate after Stage 5.2

---

## Test Structure

```
polars_backtest/tests/
├── test_namespace.py        # 16 tests
├── test_wide.py             # 25 tests
├── test_resample_polars.py  # 28 tests
├── test_utils.py            # 1 test
├── test_wide_vs_finlab.py   # Gold standard (slow)
└── test_long_vs_wide.py     # Long vs Wide (slow)
```

```bash
just test        # Fast tests (81)
just test-slow   # Slow tests (86)
just test-wide   # Wide vs Finlab
just test-long   # Long vs Wide
```
