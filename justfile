# polars_backtest development commands

# Default: show available commands
default:
    @just --list

# =============================================================================
# Rust
# =============================================================================

# Check Rust code
check:
    cargo check

# Run Rust tests
test-rust:
    cargo test -p btcore

# Lint Rust code
clippy:
    cargo clippy

# Format Rust code
fmt-rust:
    cargo fmt

# =============================================================================
# Python (run from polars_backtest/)
# =============================================================================

# Build extension (release)
build:
    cd polars_backtest && uv run maturin develop --release

# Build extension (debug)
build-debug:
    cd polars_backtest && uv run maturin develop

# Run fast tests (default)
test:
    cd polars_backtest && uv run pytest tests/ -v

# Run slow tests (finlab data required)
test-slow:
    cd polars_backtest && uv run pytest tests/ -v -m slow

# Run all tests
test-all:
    cd polars_backtest && uv run pytest tests/ -v -m ''

# Run Wide vs Finlab comparison
test-wide:
    cd polars_backtest && uv run pytest tests/test_wide_vs_finlab.py -v -m slow

# Run Long vs Wide comparison
test-long:
    cd polars_backtest && uv run pytest tests/test_long_vs_wide.py -v -m slow

# Run specific test file
test-file FILE:
    cd polars_backtest && uv run pytest tests/{{FILE}} -v

# Run benchmarks
bench:
    cd polars_backtest && uv run python benchmarks/bench_backtest.py

# Lint Python code
lint:
    cd polars_backtest && uv run ruff check python/

# Format Python code
fmt:
    cd polars_backtest && uv run ruff format python/

# Sync dependencies
sync:
    cd polars_backtest && uv sync

# =============================================================================
# Combined
# =============================================================================

# Full check: Rust + Python tests
ci: check test-rust build test

# Clean all build artifacts
clean:
    cargo clean
    rm -rf polars_backtest/.pytest_cache/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
