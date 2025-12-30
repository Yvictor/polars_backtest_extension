"""Benchmark for backtest performance.

Usage:
    cd polars_backtest
    uv run python benchmarks/bench_backtest.py
"""

import os
import time
from datetime import date, timedelta

import numpy as np
import polars as pl

os.environ["POLARS_VERBOSE"] = "0"


def generate_long_format_data(n_dates: int, n_symbols: int, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic long format price data."""
    np.random.seed(seed)

    start_date = date(2020, 1, 1)
    dates = [str(start_date + timedelta(days=i)) for i in range(n_dates)]
    symbols = [f"SYM{i:04d}" for i in range(n_symbols)]

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
                "weight": weight if weight > 0.3 else 0.0,
            })

    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())


def benchmark(func, df: pl.DataFrame, n_runs: int = 5) -> dict:
    """Run benchmark."""
    # Warmup
    func(df)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(df)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def run_benchmarks():
    """Run benchmarks with different data sizes."""
    import polars_backtest  # noqa: F401 - registers df.bt namespace

    sizes = [
        ("Small", 100, 50),       # 5,000 rows
        ("Medium", 500, 200),     # 100,000 rows
        ("Large", 1000, 500),     # 500,000 rows
        ("XLarge", 2000, 1000),   # 2,000,000 rows
    ]

    print("=" * 60)
    print("Backtest Performance Benchmark")
    print("=" * 60)

    for name, n_dates, n_symbols in sizes:
        n_rows = n_dates * n_symbols
        print(f"\n{name}: {n_dates} dates x {n_symbols} symbols = {n_rows:,} rows")
        print("-" * 50)

        df = generate_long_format_data(n_dates, n_symbols)

        # backtest (creturn only)
        result = benchmark(
            lambda d: d.bt.backtest(trade_at_price="close", position="weight"),
            df
        )
        print(f"  backtest:             {result['mean']*1000:8.2f}ms ± {result['std']*1000:.2f}ms")

        # backtest_with_report (creturn + trades)
        result = benchmark(
            lambda d: d.bt.backtest_with_report(trade_at_price="close", position="weight"),
            df
        )
        print(f"  backtest_with_report: {result['mean']*1000:8.2f}ms ± {result['std']*1000:.2f}ms")

        # With resample
        result = benchmark(
            lambda d: d.bt.backtest(trade_at_price="close", position="weight", resample="M"),
            df
        )
        print(f"  backtest (resample=M):{result['mean']*1000:8.2f}ms ± {result['std']*1000:.2f}ms")


if __name__ == "__main__":
    run_benchmarks()
