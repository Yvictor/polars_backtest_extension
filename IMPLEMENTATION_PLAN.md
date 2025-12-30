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
