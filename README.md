# polars-backtest

Blazingly fast portfolio backtesting for Polars

- Blazingly fast, written in Rust with Arrow
- Native Polars integration with `df.bt.backtest()` namespace
- T+1 execution, stop loss, take profit, trailing stop
- Touched exit with intraday OHLC detection

## Installation

```bash
pip install polars-backtest
# or
uv add polars-backtest
```

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

**300-day breakout strategy** (~2000 stocks, 17 years daily data, 12M rows):

```python
# Finlab
position = close >= close.rolling(300).max()
report = backtest.sim(position, resample="M")

# polars_backtest
df = df.with_columns(
    (pl.col("close") >= pl.col("close").rolling_max(300).over("symbol"))
    .alias("weight")
)
report = df.bt.backtest_with_report(position="weight", resample="M")
```

| | Finlab | polars_backtest |
|---|--------|-----------------|
| Time | 3.7s | 244ms |
| Speedup | 1x | **15x faster** |

```bash
just bench  # Run benchmarks
```

## Features

- **Rust Core** - Pure Rust implementation with Arrow
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
| `trail_stop` | `inf` | Trailing stop: exit when `maxcr - cr >= trail_stop` |
| `touched_exit` | `False` | Use OHLC for intraday stop detection |

---

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
