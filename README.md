# Polars Backtest Extension

High-performance portfolio backtesting extension for Polars, implemented in Rust with Python bindings. Designed to match Finlab's backtest behavior exactly.

## Project Structure

```
polars_backtest_extension/
├── Cargo.toml               # Workspace configuration
├── btcore/                  # Rust core library
│   ├── Cargo.toml
│   ├── src/
│   │   └── simulation.rs
│   └── tests/
│       └── integration_test.rs
├── polars_backtest/         # Python extension
│   ├── Cargo.toml
│   ├── pyproject.toml
│   ├── src/
│   │   └── lib.rs           # PyO3 bindings
│   └── python/
│       └── polars_backtest/
│           └── __init__.py
└── tests/
    └── python/
        ├── test_finlab_comparison.py
        ├── test_resample.py
        └── test_trades_tracking.py
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
uv run pytest ../tests/ -v                                      # All tests
uv run pytest ../tests/python/test_finlab_comparison.py -v      # Specific file
uv run pytest ../tests/test_vs_finlab.py -v                     # Finlab comparison
uv run pytest ../tests/ --tb=short                              # Short traceback
```

### Run Scripts

```bash
uv run python your_script.py
```

---

## Development Workflow

```bash
# 1. Edit Rust code
#    - btcore/src/simulation.rs (core logic)
#    - polars_backtest/src/lib.rs (Python bindings)

# 2. Check & test Rust
cargo check
cargo test -p btcore

# 3. Rebuild Python extension
cd polars_backtest
uv run maturin develop --release

# 4. Test Python
uv run pytest ../tests/ -v
```

---

## API Usage

### Basic Backtest

```python
import polars as pl
from polars_backtest import backtest

close = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "2330": [100.0, 102.0, 105.0],
})

position = pl.DataFrame({
    "date": ["2024-01-01"],
    "2330": [1.0],
})

result = backtest(close, position)
```

### Backtest with Report (Finlab-compatible)

```python
from polars_backtest import backtest_with_report

report = backtest_with_report(
    close=adj_close,
    position=position,
    factor=factor,             # adj_price / original_price
    resample='D',              # 'D', 'W', 'M', etc.
    fee_ratio=0.001425,
    tax_ratio=0.003,
    stop_loss=0.1,
    take_profit=0.2,
    trail_stop=0.05,
)

print(report.creturn)   # Cumulative return
print(report.trades)    # Trade records
print(report.position)  # Position
```

### Resample Options

| Value | Description |
|-------|-------------|
| `'D'` | Daily (rebalance every day) |
| `'W'` | Weekly (last trading day of week) |
| `'M'` | Monthly (last trading day of month) |

---

## Finlab Compatibility

Matches Finlab's `backtest.sim()` behavior:

- T+1 Execution
- Daily Rebalance (resample='D')
- Fee/Tax Handling
- Stop Loss / Take Profit / Trailing Stop

| Test Case | Difference |
|-----------|------------|
| Single stock | 0.0000000000% |
| Multi-stock | ~0.0009% |

---

## Environment Variables

```bash
# .env
FINLAB_API_TOKEN=your_token_here
```

## License

MIT
