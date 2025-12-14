# polars_backtest

High-performance portfolio backtesting extension for Polars.

## Installation

```bash
pip install polars_backtest
```

## Usage

```python
import polars as pl
from polars_backtest import daily_returns, cumulative_returns, sharpe_ratio

# Calculate daily returns
df = df.with_columns(
    ret=daily_returns("close")
)

# Calculate cumulative returns
df = df.with_columns(
    creturn=cumulative_returns("ret")
)

# Calculate Sharpe ratio
stats = df.select(
    sharpe=sharpe_ratio("ret")
)
```

## Features

- Daily returns calculation
- Cumulative returns
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Drawdown series
- Portfolio return calculation
- Equal weight allocation

## License

MIT
