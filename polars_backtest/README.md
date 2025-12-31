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
report = pl_bt.backtest_with_report(df, trade_at_price="adj_close", resample="M")
report
```

```
BacktestReport(
  creturn_len=4219,
  trades_count=6381,
  total_return=8761.03%,
  cagr=29.85%,
  max_drawdown=-35.21%,
  sharpe=1.13,
  win_ratio=46.33%
)
```

```python
report.get_stats()  # or report.stats
```

```
shape: (1, 15)
┌────────────┬────────────┬──────┬──────────────┬──────────┬──────────────┬──────────────┐
│ start      ┆ end        ┆ rf   ┆ total_return ┆ cagr     ┆ max_drawdown ┆ avg_drawdown │
│ ---        ┆ ---        ┆ ---  ┆ ---          ┆ ---      ┆ ---          ┆ ---          │
│ date       ┆ date       ┆ f64  ┆ f64          ┆ f64      ┆ f64          ┆ f64          │
╞════════════╪════════════╪══════╪══════════════╪══════════╪══════════════╪══════════════╡
│ 2008-10-31 ┆ 2025-12-31 ┆ 0.02 ┆ 87.610293    ┆ 0.298538 ┆ -0.352092    ┆ -0.042957    │
└────────────┴────────────┴──────┴──────────────┴──────────┴──────────────┴──────────────┘
┌────────────┬───────────┬──────────────┬───────────────┬──────────┬───────────┬─────────┬───────────┐
│ daily_mean ┆ daily_vol ┆ daily_sharpe ┆ daily_sortino ┆ best_day ┆ worst_day ┆ calmar  ┆ win_ratio │
│ ---        ┆ ---       ┆ ---          ┆ ---           ┆ ---      ┆ ---       ┆ ---     ┆ ---       │
│ f64        ┆ f64       ┆ f64          ┆ f64           ┆ f64      ┆ f64       ┆ f64     ┆ f64       │
╞════════════╪═══════════╪══════════════╪═══════════════╪══════════╪═══════════╪═════════╪═══════════╡
│ 0.300815   ┆ 0.249645  ┆ 1.131947     ┆ 1.834553      ┆ 0.195416 ┆ -0.160707 ┆ 0.84784 ┆ 0.463303  │
└────────────┴───────────┴──────────────┴───────────────┴──────────┴───────────┴─────────┴───────────┘
```

```python
report.creturn   # Cumulative returns DataFrame
report.trades    # Trade records with MAE/MFE metrics
report.stats     # Statistics (same as get_stats())
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

PolyForm Noncommercial 1.0.0

For commercial use, please contact the author to obtain a commercial license.
