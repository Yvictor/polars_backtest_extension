# Polars Rust Extension Specification

## Overview

使用 **uv + maturin** 建立 Polars 回測擴展，採用 **Cargo Workspace** 架構，將純 Rust 核心與 Python 擴展分離。

## Repository

```
git@github.com:Yvictor/polars_backtest_extension.git
```

## Naming

| 用途 | 名稱 | 發布平台 |
|------|------|----------|
| Rust Core | `btcore` | crates.io |
| Python Extension | `polars_backtest` | PyPI |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Cargo Workspace                         │
│       github.com/Yvictor/polars_backtest_extension          │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         │                                         │
         ▼                                         ▼
┌─────────────────────┐              ┌─────────────────────────┐
│      btcore         │              │    polars_backtest      │
│   (Pure Rust)       │◄─────────────│    (Python Extension)   │
│                     │  path dep    │                         │
│ • Portfolio sim     │              │ • Polars expressions    │
│ • Return calc       │              │ • PyO3 bindings         │
│ • Statistics        │              │ • pyo3-polars           │
│ • Trade tracking    │              │                         │
│                     │              │ crate-type = ["cdylib"] │
│ crate-type = ["lib"]│              └─────────────────────────┘
│ → crates.io         │                       │
└─────────────────────┘                       │
                                              ▼
                               ┌─────────────────────────────┐
                               │      Python Package         │
                               │  import polars_backtest     │
                               │  → PyPI                     │
                               └─────────────────────────────┘
```

## Directory Structure

```
polars_backtest_extension/
├── Cargo.toml                    # Workspace root
├── Cargo.lock
├── README.md
├── LICENSE
│
├── btcore/                       # Pure Rust library
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs               # Public API
│       ├── portfolio.rs         # Portfolio management
│       ├── returns.rs           # Return calculations
│       ├── rebalance.rs         # Rebalancing logic
│       ├── trades.rs            # Trade tracking
│       └── stats.rs             # Statistics
│
├── polars_backtest/              # Python extension
│   ├── Cargo.toml
│   ├── pyproject.toml           # uv + maturin config
│   ├── src/
│   │   └── lib.rs               # PyO3 + pyo3-polars
│   └── python/
│       └── polars_backtest/
│           ├── __init__.py
│           └── py.typed
│
└── tests/
    ├── rust/
    └── python/
        └── test_against_finlab.py
```

## Workspace Configuration

### Cargo.toml (Root)

```toml
[workspace]
resolver = "2"
members = [
    "btcore",
    "polars_backtest",
]

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "MIT"
repository = "https://github.com/Yvictor/polars_backtest_extension"

[workspace.dependencies]
chrono = { version = "0.4", default-features = false, features = ["std"] }
thiserror = "2"
rayon = "1.10"
serde = { version = "1", features = ["derive"] }

# Polars (only for extension)
polars = { version = "0.45", default-features = false }
pyo3 = { version = "0.22" }
pyo3-polars = "0.18"
```

### btcore/Cargo.toml

```toml
[package]
name = "btcore"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
description = "High-performance portfolio backtesting engine in pure Rust"
keywords = ["backtest", "trading", "finance", "portfolio", "quantitative"]
categories = ["finance", "simulation"]

[lib]
crate-type = ["lib"]

[dependencies]
chrono.workspace = true
thiserror.workspace = true
rayon.workspace = true
serde.workspace = true

[dev-dependencies]
criterion = "0.5"

[[bench]]
name = "benchmark"
harness = false
```

### polars_backtest/Cargo.toml

```toml
[package]
name = "polars_backtest"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
description = "Polars expressions for portfolio backtesting"

[lib]
name = "polars_backtest"
crate-type = ["cdylib"]

[dependencies]
btcore = { path = "../btcore" }
polars = { workspace = true, features = ["dtype-date", "dtype-datetime", "rolling_window"] }
pyo3 = { workspace = true, features = ["extension-module"] }
pyo3-polars.workspace = true
serde.workspace = true
```

### polars_backtest/pyproject.toml

```toml
[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[project]
name = "polars-backtest"
version = "0.1.0"
description = "High-performance backtesting with Polars expressions"
requires-python = ">=3.9"
license = "MIT"
dependencies = ["polars>=1.0.0"]

[project.urls]
Repository = "https://github.com/Yvictor/polars_backtest_extension"

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.maturin]
python-source = "python"
module-name = "polars_backtest._internal"
features = ["pyo3/extension-module"]
strip = true

[tool.uv]
dev-dependencies = ["pytest>=8.0", "ruff>=0.8"]

[tool.uv.cache-keys]
files = [
    "pyproject.toml",
    "Cargo.toml",
    "src/**/*.rs",
    "../btcore/src/**/*.rs",
]
```

## Development Workflow with uv + maturin

### Setup

```bash
# Clone repository
git clone git@github.com:Yvictor/polars_backtest_extension.git
cd polars_backtest_extension


# Install maturin
uv add --dev maturin
```

### Daily Development

```bash
# Pure Rust development
cargo build -p btcore
cargo test -p btcore

# Python extension development
cd polars_backtest

# Debug build (fast iteration)
uvx maturin develop --uv

# Release build (benchmarking)
uvx maturin develop --release --uv

# Run tests via uv (auto-rebuild via cache-keys)
cd ..
uv run pytest tests/python/
```

### Build & Publish

```bash
# Publish Rust crate to crates.io
cargo publish -p btcore

# Build Python wheel
cd polars_backtest
uvx maturin build --release

# Publish to PyPI
uvx maturin publish
```

## Pure Rust Core (btcore)

### btcore/src/lib.rs

```rust
//! btcore: High-performance portfolio backtesting engine
//!
//! Pure Rust implementation without Python dependencies.

pub mod portfolio;
pub mod returns;
pub mod rebalance;
pub mod trades;
pub mod stats;

pub use returns::{daily_returns, cumulative_returns, portfolio_return};
pub use stats::{BacktestStats, sharpe_ratio, max_drawdown, calc_cagr};
pub use trades::TradeRecord;

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub fee_ratio: f64,
    pub tax_ratio: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub position_limit: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            fee_ratio: 0.001425,
            tax_ratio: 0.003,
            stop_loss: None,
            take_profit: None,
            position_limit: 1.0,
        }
    }
}
```

### btcore/src/returns.rs

```rust
/// Calculate daily returns from prices
pub fn daily_returns(prices: &[f64]) -> Vec<Option<f64>> {
    if prices.is_empty() {
        return vec![];
    }

    let mut returns = Vec::with_capacity(prices.len());
    returns.push(None);

    for window in prices.windows(2) {
        let (prev, curr) = (window[0], window[1]);
        if prev > 0.0 && prev.is_finite() && curr.is_finite() {
            returns.push(Some(curr / prev - 1.0));
        } else {
            returns.push(None);
        }
    }

    returns
}

/// Calculate cumulative returns (starting at 1.0)
pub fn cumulative_returns(returns: &[Option<f64>]) -> Vec<f64> {
    let mut cum = 1.0;
    returns
        .iter()
        .map(|r| {
            if let Some(ret) = r {
                cum *= 1.0 + ret;
            }
            cum
        })
        .collect()
}

/// Calculate weighted portfolio return
pub fn portfolio_return(weights: &[f64], returns: &[f64]) -> f64 {
    weights.iter().zip(returns).map(|(w, r)| w * r).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daily_returns() {
        let prices = vec![100.0, 105.0, 103.0, 110.0];
        let returns = daily_returns(&prices);

        assert!(returns[0].is_none());
        assert!((returns[1].unwrap() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_cumulative_returns() {
        let returns = vec![None, Some(0.01), Some(0.02), Some(-0.01)];
        let creturn = cumulative_returns(&returns);

        assert!((creturn[0] - 1.0).abs() < 1e-10);
        assert!((creturn[3] - 1.01 * 1.02 * 0.99).abs() < 1e-10);
    }
}
```

### btcore/src/stats.rs

```rust
#[derive(Debug, Clone, Default)]
pub struct BacktestStats {
    pub total_return: f64,
    pub cagr: f64,
    pub max_drawdown: f64,
    pub daily_sharpe: f64,
    pub daily_sortino: f64,
    pub calmar: f64,
    pub win_ratio: f64,
}

pub fn sharpe_ratio(returns: &[f64], rf: f64, annualize: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;
    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = variance.sqrt();

    if std > 0.0 {
        (mean - rf / annualize) / std * annualize.sqrt()
    } else {
        0.0
    }
}

pub fn max_drawdown(creturn: &[f64]) -> f64 {
    let mut peak = 0.0f64;
    let mut max_dd = 0.0f64;

    for &val in creturn {
        peak = peak.max(val);
        if peak > 0.0 {
            let dd = (val - peak) / peak;
            max_dd = max_dd.min(dd);
        }
    }

    max_dd
}

pub fn calc_cagr(start: f64, end: f64, years: f64) -> f64 {
    if start <= 0.0 || end <= 0.0 || years <= 0.0 {
        return 0.0;
    }
    (end / start).powf(1.0 / years) - 1.0
}
```

## Python Extension (polars_backtest)

### polars_backtest/src/lib.rs

```rust
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use polars::prelude::*;

use btcore::{
    daily_returns as core_daily_returns,
    cumulative_returns as core_cumulative_returns,
    sharpe_ratio as core_sharpe,
    max_drawdown as core_max_dd,
};

fn float64_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new("result".into(), DataType::Float64))
}

#[polars_expr(output_type_func=float64_output)]
fn daily_returns(inputs: &[Series]) -> PolarsResult<Series> {
    let prices = inputs[0].f64()?;
    let price_vec: Vec<f64> = prices.into_iter().flatten().collect();
    let returns = core_daily_returns(&price_vec);

    let result: Float64Chunked = returns.into_iter().collect();
    Ok(result.into_series())
}

#[polars_expr(output_type_func=float64_output)]
fn cumulative_returns(inputs: &[Series]) -> PolarsResult<Series> {
    let returns = inputs[0].f64()?;
    let returns_vec: Vec<Option<f64>> = returns.into_iter().collect();
    let creturn = core_cumulative_returns(&returns_vec);

    Ok(Float64Chunked::from_vec("creturn".into(), creturn).into_series())
}

#[polars_expr(output_type_func=float64_output)]
fn sharpe_ratio(inputs: &[Series]) -> PolarsResult<Series> {
    let returns = inputs[0].f64()?;
    let rf = inputs[1].f64()?.get(0).unwrap_or(0.0);
    let annualize = inputs[2].f64()?.get(0).unwrap_or(252.0);

    let returns_vec: Vec<f64> = returns.into_iter().flatten().collect();
    let sharpe = core_sharpe(&returns_vec, rf, annualize);

    Ok(Series::new("sharpe".into(), vec![sharpe]))
}

#[polars_expr(output_type_func=float64_output)]
fn max_drawdown(inputs: &[Series]) -> PolarsResult<Series> {
    let creturn = inputs[0].f64()?;
    let creturn_vec: Vec<f64> = creturn.into_iter().flatten().collect();
    let max_dd = core_max_dd(&creturn_vec);

    Ok(Series::new("max_drawdown".into(), vec![max_dd]))
}

#[polars_expr(output_type_func=float64_output)]
fn equal_weight(inputs: &[Series]) -> PolarsResult<Series> {
    let signals = inputs[0].bool()?;
    let count = signals.into_iter().filter(|x| x.unwrap_or(false)).count();
    let weight = if count > 0 { 1.0 / count as f64 } else { 0.0 };

    let result: Float64Chunked = signals
        .into_iter()
        .map(|s| Some(if s.unwrap_or(false) { weight } else { 0.0 }))
        .collect();

    Ok(result.into_series())
}

#[pymodule]
fn _internal(_py: Python, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
```

### polars_backtest/python/polars_backtest/__init__.py

```python
"""Polars expressions for portfolio backtesting, powered by btcore (Rust)."""

from pathlib import Path
import polars as pl
from polars.plugins import register_plugin_function
from polars.type_aliases import IntoExpr

__version__ = "0.1.0"
LIB = Path(__file__).parent / "_internal"


def daily_returns(expr: IntoExpr) -> pl.Expr:
    """Calculate daily returns from prices."""
    return register_plugin_function(
        plugin_path=LIB,
        function_name="daily_returns",
        args=[expr],
        is_elementwise=False,
    )


def cumulative_returns(expr: IntoExpr) -> pl.Expr:
    """Calculate cumulative returns from daily returns."""
    return register_plugin_function(
        plugin_path=LIB,
        function_name="cumulative_returns",
        args=[expr],
        is_elementwise=False,
    )


def sharpe_ratio(expr: IntoExpr, rf: float = 0.0, annualize: float = 252.0) -> pl.Expr:
    """Calculate annualized Sharpe ratio."""
    return register_plugin_function(
        plugin_path=LIB,
        function_name="sharpe_ratio",
        args=[expr, pl.lit(rf), pl.lit(annualize)],
        is_elementwise=False,
    )


def max_drawdown(expr: IntoExpr) -> pl.Expr:
    """Calculate maximum drawdown from cumulative returns."""
    return register_plugin_function(
        plugin_path=LIB,
        function_name="max_drawdown",
        args=[expr],
        is_elementwise=False,
    )


def equal_weight(expr: IntoExpr) -> pl.Expr:
    """Assign equal weights to True signals (sum = 1.0)."""
    return register_plugin_function(
        plugin_path=LIB,
        function_name="equal_weight",
        args=[expr],
        is_elementwise=True,
    )


__all__ = [
    "daily_returns",
    "cumulative_returns",
    "sharpe_ratio",
    "max_drawdown",
    "equal_weight",
]
```

## References

- [uv and maturin - quanttype](https://quanttype.net/posts/2025-09-12-uv-and-maturin.html)
- [Creating projects | uv](https://docs.astral.sh/uv/concepts/projects/init/)
- [Local Development - Maturin](https://www.maturin.rs/local_development.html)
- [Cargo workspaces - maturin #291](https://github.com/PyO3/maturin/issues/291)
