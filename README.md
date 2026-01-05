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

### Benchmark Comparison

Compare strategy performance against a benchmark to get alpha, beta, and rolling win rate:

```python
# Method 1: Use a symbol from your data as benchmark
report = df.bt.backtest_with_report(
    position="weight",
    benchmark="0050",  # Symbol value (e.g., ETF ticker)
)

# Method 2: Provide a benchmark DataFrame
benchmark_df = pl.DataFrame({
    "date": [...],
    "creturn": [...],  # Cumulative return starting at 1.0
})
report = df.bt.backtest_with_report(position="weight", benchmark=benchmark_df)

# Method 3: Set benchmark after creation
report = df.bt.backtest_with_report(position="weight")
report.benchmark = benchmark_df

# get_metrics includes benchmark metrics when benchmark is set
metrics = report.get_metrics()
# Includes: alpha, beta, m12WinRate (12-month rolling win rate vs benchmark)
```

### Liquidity Metrics

Get liquidity metrics by providing optional columns in your DataFrame:

```python
df = pl.DataFrame({
    "date": dates,
    "symbol": symbols,
    "close": prices,
    "weight": weights,
    "limit_up": limit_up_prices,      # For buyHigh metric
    "limit_down": limit_down_prices,  # For sellLow metric
    "trading_value": trading_values,  # For capacity metric (e.g., close_raw * volume)
})

report = df.bt.backtest_with_report(position="weight")
metrics = report.get_metrics(sections=["liquidity"])
# Includes: buyHigh, sellLow, capacity
```

| Metric | Required Column | Description |
|--------|-----------------|-------------|
| `buyHigh` | `limit_up` | Ratio of entries at limit-up price |
| `sellLow` | `limit_down` | Ratio of exits at limit-down price |
| `capacity` | `trading_value` | Strategy capacity (10th percentile of accepted money flow) |

Metrics return `null` if the required column is not present.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trade_at_price` | `"close"` | Price column for execution |
| `position` | `"weight"` | Position/weight column |
| `benchmark` | `None` | Benchmark: symbol str or DataFrame with (date, creturn) |
| `resample` | `"D"` | Rebalance frequency |
| `resample_offset` | `None` | Delay rebalance by N days (e.g., `"1d"`, `"2d"`, `"1W"`) |
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

PolyForm Noncommercial 1.0.0

For commercial use, please contact the author to obtain a commercial license.
