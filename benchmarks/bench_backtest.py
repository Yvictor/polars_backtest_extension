"""Benchmark for backtest performance.

Compares:
- Rust backtest (long format with internal pivot)
- Python backtest with Rust path (resample=D)
- Python backtest with Python pivot path (resample=W-FRI forces Python)
"""

import time
import sys
import os
import polars as pl
import numpy as np
from datetime import date, timedelta

# Suppress polars-ops debug messages
os.environ["POLARS_VERBOSE"] = "0"


def generate_long_format_data(n_dates: int, n_symbols: int, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic long format price data."""
    np.random.seed(seed)

    # Generate dates
    start_date = date(2020, 1, 1)
    dates = [str(start_date + timedelta(days=i)) for i in range(n_dates)]

    # Generate symbols
    symbols = [f"SYM{i:04d}" for i in range(n_symbols)]

    # Generate data
    rows = []
    for d in dates:
        for s in symbols:
            base_price = 100.0 + hash(s) % 100
            price = base_price * (1 + np.random.randn() * 0.02)
            weight = np.random.random()
            rows.append({
                "date": d,
                "symbol": s,
                "close": price,
                "weight": weight if weight > 0.3 else 0.0,  # Sparse weights
            })

    return pl.DataFrame(rows)


def benchmark_rust_pivot(df: pl.DataFrame, n_runs: int = 5) -> dict:
    """Benchmark Rust backtest with pivot."""
    from polars_backtest._polars_backtest import backtest as rust_backtest, BacktestConfig

    config = BacktestConfig()

    # Warmup
    rust_backtest(df, "date", "symbol", "close", "weight", "D", config)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = rust_backtest(df, "date", "symbol", "close", "weight", "D", config)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "creturn_len": len(result.creturn),
    }


def benchmark_rust_partitioned(df: pl.DataFrame, n_runs: int = 5) -> dict:
    """Benchmark Rust backtest with partition_by (no pivot)."""
    from polars_backtest._polars_backtest import backtest_partitioned, BacktestConfig

    config = BacktestConfig()

    # Warmup
    backtest_partitioned(df, "date", "symbol", "close", "weight", "D", config)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = backtest_partitioned(df, "date", "symbol", "close", "weight", "D", config)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "creturn_len": len(result.creturn),
    }


def benchmark_python_rust_path(df: pl.DataFrame, n_runs: int = 5) -> dict:
    """Benchmark Python API using Rust path (resample=D)."""
    # Warmup
    df.bt.backtest(price="close", weight="weight", resample="D")

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = df.bt.backtest(price="close", weight="weight", resample="D")
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "creturn_len": len(result),
    }


def benchmark_python_pivot_path(df: pl.DataFrame, n_runs: int = 5) -> dict:
    """Benchmark Python API using Python pivot path (forced via resample_offset)."""
    # Force Python path by using resample_offset (not supported by Rust)
    # Warmup
    df.bt.backtest(price="close", weight="weight", resample="D", resample_offset="0D")

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = df.bt.backtest(price="close", weight="weight", resample="D", resample_offset="0D")
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "creturn_len": len(result),
    }


def run_benchmarks():
    """Run benchmarks with different data sizes."""
    sizes = [
        ("Small", 100, 50),      # 5,000 rows
        ("Medium", 500, 200),    # 100,000 rows
        ("Large", 1000, 500),    # 500,000 rows
    ]

    print("=" * 70)
    print("Backtest Performance Benchmark")
    print("=" * 70)
    print("\nComparing:")
    print("  1. Rust pivot:      backtest() with Polars pivot")
    print("  2. Rust partition:  backtest_partitioned() with partition_by")
    print("  3. Python pivot:    df.bt.backtest() with Python fallback")

    for name, n_dates, n_symbols in sizes:
        print(f"\n{'='*60}")
        print(f"{name}: {n_dates} dates x {n_symbols} symbols = {n_dates * n_symbols:,} rows")
        print("-" * 60)

        df = generate_long_format_data(n_dates, n_symbols)

        # 1. Rust with pivot
        rust_pivot = benchmark_rust_pivot(df)
        print(f"1. Rust pivot:     {rust_pivot['mean']*1000:8.2f}ms ± {rust_pivot['std']*1000:.2f}ms")

        # 2. Rust with partition_by
        rust_part = benchmark_rust_partitioned(df)
        print(f"2. Rust partition: {rust_part['mean']*1000:8.2f}ms ± {rust_part['std']*1000:.2f}ms")

        # 3. Python pivot path
        py_pivot = benchmark_python_pivot_path(df)
        print(f"3. Python pivot:   {py_pivot['mean']*1000:8.2f}ms ± {py_pivot['std']*1000:.2f}ms")

        # Find fastest
        results = [
            ("Rust pivot", rust_pivot['mean']),
            ("Rust partition", rust_part['mean']),
            ("Python pivot", py_pivot['mean']),
        ]
        fastest = min(results, key=lambda x: x[1])

        print(f"\nFastest: {fastest[0]}")
        print(f"  vs Rust pivot:     {rust_pivot['mean'] / fastest[1]:.2f}x")
        print(f"  vs Rust partition: {rust_part['mean'] / fastest[1]:.2f}x")
        print(f"  vs Python pivot:   {py_pivot['mean'] / fastest[1]:.2f}x")


if __name__ == "__main__":
    run_benchmarks()
