# polars_backtest

High-performance portfolio backtesting extension for Polars, implemented in Rust.

## Installation

```bash
uv add polars_backtest
# or
pip install polars_backtest
```

## Quick Start

```python
import polars as pl
import polars_backtest as pl_bt

# Long format: one row per (date, symbol)
df = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "symbol": ["2330", "2317", "2330", "2317"],
    "close": [100.0, 50.0, 102.0, 51.0],
    "weight": [0.6, 0.4, 0.6, 0.4],
})

# Function API
result = pl_bt.backtest(df, trade_at_price="close", position="weight")

# Or DataFrame namespace
result = df.bt.backtest(trade_at_price="close", position="weight")
```

### With Report

```python
report = pl_bt.backtest_with_report(
    df,
    trade_at_price="close",
    position="weight",
    resample="M",
    fee_ratio=0.001425,
    tax_ratio=0.003,
    stop_loss=0.1,
)

print(report.creturn)   # Cumulative returns
print(report.trades)    # Trade records
print(report.position)  # Position history
```

### Statistics Expressions

```python
from polars_backtest import daily_returns, cumulative_returns, sharpe_ratio, max_drawdown

df.with_columns(
    ret=daily_returns("close"),
    creturn=cumulative_returns("ret"),
)

df.select(
    sharpe=sharpe_ratio("ret"),
    mdd=max_drawdown("creturn"),
)
```

## Features

- **Zero-copy Arrow FFI** - Direct memory sharing between Polars and Rust
- **T+1 execution** - Realistic trading simulation
- **Stop loss / Take profit / Trailing stop** - Risk management
- **Resample support** - D, W, W-FRI, M, Q, Y

## Resample Options

| Value | Description |
|-------|-------------|
| `None` | Only rebalance when position changes |
| `'D'` | Daily |
| `'W'` | Weekly (Sunday) |
| `'W-FRI'` | Weekly (Friday) |
| `'M'` | Monthly |
| `'Q'` | Quarterly |
| `'Y'` | Yearly |

## Development

```bash
# Setup
uv sync

# Build
uv run maturin develop --release

# Test
uv run pytest tests/ -v                       # Fast tests (default)
uv run pytest tests/ -v -m slow               # Slow tests (finlab)
uv run pytest tests/test_wide_vs_finlab.py -v # Wide vs Finlab
uv run pytest tests/test_long_vs_wide.py -v   # Long vs Wide
```

## License

MIT
