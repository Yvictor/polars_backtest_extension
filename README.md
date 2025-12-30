# Polars Backtest Extension

High-performance portfolio backtesting extension for Polars. Rust-powered with zero-copy Arrow FFI.

## Quick Start

```python
import polars as pl
import polars_backtest as pl_bt

# Long format data: one row per (date, symbol)
df = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "symbol": ["AAPL", "GOOGL", "AAPL", "GOOGL"],
    "close": [100.0, 50.0, 102.0, 51.0],
    "weight": [0.6, 0.4, 0.6, 0.4],
})

# Run backtest
result = df.bt.backtest(trade_at_price="close", position="weight")
```

## Performance

| Data Size | Time |
|-----------|------|
| 5,000 rows | ~3.5ms |
| 100,000 rows | ~60ms |
| 500,000 rows | ~300ms |
| 2,000,000 rows | ~1.2s |

```bash
just bench  # Run benchmarks
```

## Features

- **Rust Core** - Pure Rust implementation with zero-copy Arrow FFI
- **Native Polars** - Works with long format DataFrames, supports Polars expressions
- **T+1 Execution** - Realistic trading simulation
- **Risk Management** - Stop loss, take profit, trailing stop, touched exit (OHLC)
- **Flexible Rebalancing** - Daily, weekly, monthly, or on position change

---

## Usage

### Basic Backtest

```python
import polars_backtest as pl_bt

# Function API
result = pl_bt.backtest(df, trade_at_price="close", position="weight")

# DataFrame namespace
result = df.bt.backtest(trade_at_price="close", position="weight")
```

### With Expressions

```python
result = df.bt.backtest(
    trade_at_price="close",
    position=pl.col("signal").cast(pl.Float64),
    resample="M",
)
```

### Full Report with Trades

```python
report = pl_bt.backtest_with_report(
    df,
    trade_at_price="close",
    position="weight",
    resample="M",
    fee_ratio=0.001425,
    tax_ratio=0.003,
    stop_loss=0.1,
    take_profit=0.2,
)

print(report.creturn)   # DataFrame with date, creturn
print(report.trades)    # DataFrame with trade records
```

### Resample Options

| Value | Description |
|-------|-------------|
| `None` | Only rebalance when position changes |
| `'D'` | Daily |
| `'W'` | Weekly (last trading day) |
| `'W-FRI'` | Weekly on Friday |
| `'M'` | Monthly (last trading day) |
| `'Q'` | Quarterly |
| `'Y'` | Yearly |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trade_at_price` | `"close"` | Price column for execution |
| `position` | `"weight"` | Position/weight column |
| `resample` | `"D"` | Rebalance frequency |
| `fee_ratio` | `0.001425` | Transaction fee |
| `tax_ratio` | `0.003` | Transaction tax |
| `stop_loss` | `1.0` | Stop loss threshold (1.0 = disabled) |
| `take_profit` | `inf` | Take profit threshold |
| `trail_stop` | `inf` | Trailing stop threshold |
| `touched_exit` | `False` | Use OHLC for intraday stop detection |

---

## Development

### Setup

```bash
just sync   # Install dependencies
just build  # Build extension
```

### Test

```bash
just test       # Fast tests (81)
just test-slow  # Slow tests (86)
just test-all   # All tests
just bench      # Benchmarks
```

### Workflow

```bash
just check      # Rust check
just test-rust  # Rust tests
just build      # Build extension
just test       # Python tests
just ci         # Full CI
```

---

## Project Structure

```
polars_backtest_extension/
├── btcore/                 # Pure Rust core
│   └── src/simulation/
│       ├── wide.rs         # Wide format backtest
│       └── long.rs         # Long format (Arrow FFI)
└── polars_backtest/
    ├── python/             # Python API
    │   └── polars_backtest/
    │       ├── namespace.py  # df.bt namespace
    │       └── wide.py       # Wide format API
    ├── src/                # PyO3 bindings
    │   └── lib.rs
    ├── tests/
    └── benchmarks/
```

## License

MIT
