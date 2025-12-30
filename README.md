# Polars Backtest Extension

High-performance portfolio backtesting extension for Polars, implemented in Rust with Python bindings. Designed to match Finlab's backtest behavior exactly.

## Features

- **Zero-copy Arrow FFI** - Direct memory sharing between Polars and Rust
- **Long format support** - Native Polars DataFrame operations
- **Finlab compatible** - Matches Finlab's `backtest.sim()` exactly
- **T+1 execution** - Realistic trading simulation
- **Stop loss / Take profit / Trailing stop** - Risk management
- **Resample support** - Daily, Weekly, Monthly rebalancing

## Project Structure

```
polars_backtest_extension/
├── Cargo.toml                    # Rust workspace config
├── btcore/                       # Rust core library (no Python deps)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs                # Module exports
│   │   ├── config.rs             # BacktestConfig struct
│   │   ├── position.rs           # Position tracking (cost, value, etc)
│   │   ├── portfolio.rs          # PortfolioState for wide format
│   │   ├── simulation/
│   │   │   ├── mod.rs
│   │   │   ├── wide.rs           # Wide format backtest (2D array)
│   │   │   └── long.rs           # Long format backtest (Arrow, zero-copy)
│   │   ├── rebalance.rs          # Portfolio rebalancing logic
│   │   ├── returns.rs            # Return calculations
│   │   ├── stats.rs              # Statistics (CAGR, Sharpe, max drawdown)
│   │   ├── stops.rs              # Stop loss/take profit/trailing stop
│   │   ├── tracker.rs            # Trade tracking and results
│   │   ├── trades.rs             # Trade record structures
│   │   └── weights.rs            # Weight/position handling
│   └── tests/
│       └── integration_test.rs
├── polars_backtest/              # Python extension (PyO3/Maturin)
│   ├── Cargo.toml
│   ├── pyproject.toml
│   ├── src/
│   │   ├── lib.rs                # PyO3 bindings (main API)
│   │   ├── ffi_convert.rs        # FFI conversion (zero-copy Arrow)
│   │   └── expressions.rs        # Polars expression registration
│   ├── python/
│   │   └── polars_backtest/
│   │       ├── __init__.py       # Package exports
│   │       ├── namespace.py      # df.bt namespace API
│   │       ├── wide.py           # Wide format API (Finlab compatible)
│   │       └── utils.py          # Utility functions (resample, offset)
│   └── tests/
│       ├── test_namespace.py     # df.bt namespace tests
│       ├── test_wide.py          # Wide format API tests
│       ├── test_resample_polars.py  # Resample functionality tests
│       ├── test_utils.py         # Utility function tests
│       ├── test_wide_vs_finlab.py   # Wide vs Finlab.sim (slow)
│       └── test_long_vs_wide.py     # Long vs Wide format (slow)
├── benchmarks/
│   └── bench_backtest.py         # Performance benchmarks
└── devnb/                        # Development notebooks
```

---

## Rust Development

> Working directory: `polars_backtest_extension/` (project root)

### Build

```bash
cargo build                    # Debug build
cargo build --release          # Release build
cargo build -p btcore          # btcore only
```

### Test

```bash
cargo test                     # All Rust tests
cargo test -p btcore           # btcore tests only
cargo test -p btcore -- --nocapture  # With output
cargo test -p btcore --test integration_test  # Integration tests
```

### Check & Lint

```bash
cargo check                    # Check compilation errors
cargo clippy                   # Lint
cargo fmt                      # Format code
cargo fmt -- --check           # Check format only
```

---

## Python Development

> Working directory: `polars_backtest_extension/polars_backtest/`

### Setup

```bash
cd polars_backtest
uv sync
```

### Build Extension

```bash
uv run maturin develop              # Debug build
uv run maturin develop --release    # Release build (recommended)
```

### Test

```bash
uv run pytest tests/ -v                       # Fast tests only (default)
uv run pytest tests/ -v -m slow               # Slow tests only (finlab data)
uv run pytest tests/ -v -m ''                 # All tests (fast + slow)
uv run pytest tests/test_wide_vs_finlab.py -v # Wide vs Finlab comparison
uv run pytest tests/test_long_vs_wide.py -v   # Long vs Wide comparison
```

### Run Scripts

```bash
uv run python your_script.py
```

---

## Development Workflow

```bash
# 1. Edit Rust code
#    - btcore/src/simulation/*.rs (core logic)
#    - polars_backtest/src/lib.rs (Python bindings)

# 2. Check & test Rust
cargo check
cargo test -p btcore

# 3. Rebuild Python extension
cd polars_backtest
uv run maturin develop --release

# 4. Test Python
uv run pytest tests/ -v
```

---

## API Usage

### Long Format (Recommended)

Long format DataFrames work natively with Polars operations.

```python
import polars as pl
import polars_backtest as pb

# Long format data: one row per (date, symbol)
df = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "symbol": ["2330", "2317", "2330", "2317"],
    "close": [100.0, 50.0, 102.0, 51.0],
    "weight": [0.6, 0.4, 0.6, 0.4],
})

# Function API
result = pb.backtest(df, trade_at_price="close", position="weight")

# Or use DataFrame namespace
result = df.bt.backtest(trade_at_price="close", position="weight")
```

### With Expressions

Supports Polars expressions for computed columns:

```python
result = df.bt.backtest(
    trade_at_price="close",
    position=pl.col("signal").cast(pl.Float64),  # Expr supported
    resample="M",
)
```

### Backtest with Report

```python
report = pb.backtest_with_report(
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
print(report.position)  # DataFrame with positions
```

### Wide Format (Finlab Compatible)

For Finlab-style wide format DataFrames:

```python
from polars_backtest import backtest_wide, backtest_with_report_wide

# Wide format: dates as rows, symbols as columns
prices = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "2330": [100.0, 102.0, 105.0],
    "2317": [50.0, 51.0, 52.0],
})

position = pl.DataFrame({
    "date": ["2024-01-01"],
    "2330": [0.6],
    "2317": [0.4],
})

result = backtest_wide(prices, position)

# With full report
report = backtest_with_report_wide(
    close=adj_close,
    position=position,
    factor=factor,             # adj_price / original_price
    resample="M",
)
```

### Statistics Expressions

```python
from polars_backtest import daily_returns, cumulative_returns, sharpe_ratio, max_drawdown

df.with_columns(
    daily_ret=daily_returns("close"),
    cum_ret=cumulative_returns("daily_ret"),
)

df.select(
    sharpe=sharpe_ratio("daily_ret"),
    mdd=max_drawdown("cum_ret"),
)
```

### Resample Options

| Value | Description |
|-------|-------------|
| `None` | Only rebalance when position changes |
| `'D'` | Daily (rebalance every day) |
| `'W'` | Weekly (last trading day of week) |
| `'W-FRI'` | Weekly on Friday |
| `'M'` | Monthly (last trading day of month) |
| `'Q'` | Quarterly |
| `'Y'` | Yearly |

---

## Environment Variables

```bash
# .env
FINLAB_API_TOKEN=your_token_here
```

### Profiling

```bash
POLARS_BACKTEST_PROFILE=1 python your_script.py
```

## License

MIT
