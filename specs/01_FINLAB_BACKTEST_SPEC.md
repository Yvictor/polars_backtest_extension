# Finlab Backtest Engine Specification

## Overview

This document describes the internal workings of `finlab.backtest.sim()` function, reverse-engineered from source code analysis.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     backtest.sim()                          │
│                    (backtest.py)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Input Validation & Preprocessing                        │
│     - Position DataFrame validation                         │
│     - Resample date generation                              │
│     - Price data alignment                                  │
│                                                             │
│  2. Core Engine (Cython)                                    │
│     backtest_core.backtest_()                               │
│     - Portfolio simulation                                  │
│     - Stop loss / Take profit handling                      │
│     - Trade tracking                                        │
│                                                             │
│  3. Post-processing                                         │
│     - Trade record generation                               │
│     - MAE/MFE calculation                                   │
│     - Report object construction                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Input Specifications

### Position Input

| Attribute | Specification |
|-----------|---------------|
| Type | `pd.DataFrame` |
| Index | `DatetimeIndex` (signal dates) |
| Columns | Stock symbols (str) |
| Values | `bool` or `float` (True/1.0 = hold, False/0.0 = no position) |

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `position` | DataFrame | Required | Buy/sell signals |
| `resample` | str/None | None | Rebalance frequency ('D', 'W', 'M', 'Q') |
| `trade_at_price` | str | 'close' | Price type for execution |
| `fee_ratio` | float | 0.001425 | Transaction fee |
| `tax_ratio` | float | 0.003 | Transaction tax |
| `stop_loss` | float | None | Stop loss threshold |
| `take_profit` | float | None | Take profit threshold |
| `trail_stop` | float | None | Trailing stop threshold |
| `position_limit` | float | 1 | Max weight per stock |

## Resample Logic

### Resample Date Generation

```python
# For resample='M' (monthly)
alldates = pd.date_range(
    position.index[0],
    present_data_date + datetime.timedelta(days=360),
    freq='M'
)

# Filter dates within position range
dates = [d for d in alldates
         if position.index[0] <= d <= present_data_date]
```

### Signal Timing

```
Signal Date (T)     →  Trade Execution (T+1)
月底產生信號         →  下月第一個交易日進場
```

## Core Engine Interface

### backtest_() Function Signature

```python
from finlab.core.backtest_core import backtest_

creturn_value = backtest_(
    price_values,           # np.ndarray: Trading prices
    close_values,           # np.ndarray: Close prices
    high_values,            # np.ndarray: High prices
    low_values,             # np.ndarray: Low prices
    open_values,            # np.ndarray: Open prices
    price_index,            # np.ndarray: Date index (int64)
    price_columns,          # np.ndarray: Stock symbols
    position_values,        # np.ndarray: Position weights
    position_index,         # np.ndarray: Position dates (int64)
    position_columns,       # np.ndarray: Position symbols
    resample_dates,         # np.ndarray: Rebalance dates (int64)
    encryption=encryption,  # str: API key encryption
    fee_ratio=0.001425,
    tax_ratio=0.003,
    stop_loss=1.0,          # 1.0 = disabled
    take_profit=np.inf,     # inf = disabled
    trail_stop=np.inf,
    touched_exit=False,
    position_limit=1.0,
    retain_cost_when_rebalance=False,
    stop_trading_next_period=True,
    mae_mfe_window=0,
    mae_mfe_window_step=1,
    periodically_rebalance=True
)
```

## Weight Calculation

### Equal Weight Distribution

```python
# Original position (bool or float signals)
total_weight = position.abs().sum(axis=1).clip(1, None)

# Normalized position (equal weight)
position = position.astype(float).div(
    total_weight.where(total_weight != 0, np.nan), axis=0
).fillna(0).clip(-abs(position_limit), abs(position_limit))

# Result: sum of weights = 1.0
# Each stock weight = 1 / number_of_holdings
```

### Example

If holding 4 stocks:
- Each stock weight = 0.25 (25%)
- Total weight = 1.0 (100%)

## Return Calculation

### Daily Portfolio Return

```python
# r[t] = weighted average of individual stock returns
portfolio_return[t] = sum(weight[i] * stock_return[i,t] for i in stocks)

# stock_return[i,t] = (price[i,t] - price[i,t-1]) / price[i,t-1]
```

### Cumulative Return

```python
creturn[t] = creturn[t-1] * (1 + portfolio_return[t])
# Starting value: creturn[0] = 1.0
```

## Trade Record Generation

### get_trade_stocks() Output

| Field | Type | Description |
|-------|------|-------------|
| `stock_id` | str | Stock symbol + name |
| `entry_date` | datetime | Actual entry date |
| `exit_date` | datetime | Actual exit date |
| `entry_sig_date` | datetime | Signal date for entry |
| `exit_sig_date` | datetime | Signal date for exit |
| `position` | float | Position weight |
| `period` | int | Holding period (days) |
| `return` | float | Trade return |

### Trade Return Calculation

```python
# Basic return (no fees)
trade_return = exit_price / entry_price - 1

# With fees
trade_return = (1 - fee_ratio) * (exit_price / entry_price) * (1 - tax_ratio - fee_ratio) - 1
```

## Statistics Calculation

### Key Statistics (from ffn_core.py)

| Metric | Formula |
|--------|---------|
| `total_return` | `creturn[-1] / creturn[0] - 1` |
| `cagr` | `(creturn[-1] / creturn[0]) ^ (1 / years) - 1` |
| `max_drawdown` | `min((creturn - cummax) / cummax)` |
| `daily_sharpe` | `(mean_return - rf) / std_return * sqrt(252)` |
| `daily_sortino` | `(mean_return - rf) / downside_std * sqrt(252)` |
| `calmar` | `cagr / abs(max_drawdown)` |

### Sharpe Ratio Calculation

```python
# Daily returns
daily_returns = creturn.pct_change()

# Sharpe ratio (annualized)
daily_sharpe = (daily_returns.mean() - rf) / daily_returns.std() * np.sqrt(252)
```

### Sortino Ratio Calculation

```python
# Only consider downside volatility
downside_returns = daily_returns[daily_returns < rf]
downside_std = downside_returns.std()

daily_sortino = (daily_returns.mean() - rf) / downside_std * np.sqrt(252)
```

## Report Object

### Report Class Structure

```python
class Report:
    # Core attributes
    creturn: pd.Series           # Cumulative return curve
    daily_creturn: pd.Series     # creturn * 100
    position: pd.DataFrame       # Rebalanced position weights
    trades: pd.DataFrame         # Trade records

    # Parameters
    fee_ratio: float
    tax_ratio: float
    stop_loss: float
    take_profit: float
    trail_stop: float

    # Methods
    def get_stats(self) -> dict
    def display(self)
    def to_pickle(filepath)
```

### get_stats() Output

Returns a dict with ~50 metrics:

```python
{
    'total_return': 1.462213,
    'cagr': 0.095841,
    'max_drawdown': -0.296544,
    'daily_sharpe': 0.572957,
    'daily_sortino': 0.789308,
    'calmar': 0.323181,
    'win_ratio': 0.471134,
    # ... many more
}
```

## Performance Considerations

### Cython Optimization

The core backtest loop is implemented in Cython (`backtest_core.cpython-*.so`) for performance:

- Compiled to native code
- NumPy array operations
- Minimal Python overhead
- Handles ~2000 stocks × 2500 days efficiently

### Memory Layout

```
Prices: [dates × stocks] matrix (float64)
Position: [rebalance_dates × stocks] matrix (float64)
```

All data is converted to NumPy arrays before passing to Cython core for optimal performance.
