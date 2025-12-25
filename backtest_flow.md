# Backtest Flow Comparison: Finlab vs Polars Backtest

## Overview

This document compares the backtest simulation flow between:
1. **Finlab**: `backtest.py` (orchestration) + `backtest_core.pyx` (Cython core)
2. **Polars Backtest**: `__init__.py` (orchestration) + `simulation.rs` (Rust core)

---

## High-Level Architecture

### Finlab

```
backtest.sim()  →  arguments()  →  backtest_()  →  get_trade_stocks()  →  Report
     │                 │              │                   │
     ├─ resample       ├─ price       ├─ daily loop       ├─ trades
     ├─ position       ├─ close       ├─ rebalance        └─ operations
     └─ config         └─ position    └─ stop/tp
```

### Polars Backtest

```
backtest_with_report()  →  _resample_position()  →  _backtest_with_trades()  →  Report
        │                         │                         │
        ├─ resample               ├─ price dates            ├─ simulate_backtest()
        ├─ position               └─ position dates         ├─ TradeTracker
        └─ config                                           └─ trades
```

---

## Detailed Flow Comparison

### 1. Entry Point & Pre-processing

#### Finlab (`backtest.py:sim()`)

```python
# 1. Input validation
if not isinstance(position.index, pd.DatetimeIndex):
    raise TypeError(...)

# 2. Get trading prices
price = market.get_trading_price(trade_at_price, adj=True)
close, high, low, open_ = price, price, price, price

# 3. Resample handling
if resample is str:
    alldates = pd.date_range(position.index[0], end, freq=resample, tz=tz)
    dates = [d for d in alldates if position.index[0] <= d <= present_data_date]
elif resample is None:
    # Only rebalance when portfolio changes
    change = position.diff().abs().sum(axis=1) != 0
    position = position.loc[change]

# 4. Build arguments array
args = arguments(price, close, high, low, open_, position, dates)
```

#### Polars Backtest (`__init__.py:backtest_with_report()`)

```python
# 1. Resolve trade_at_price
if trade_at_price == "close":
    trade_prices = close
elif trade_at_price == "open":
    trade_prices = open

# 2. Calculate original prices (factor adjustment)
if factor is not None:
    original_prices = trade_prices / factor

# 3. Resample handling
if resample != 'D':
    position = _resample_position(position, dates, resample, resample_offset)

# 4. Calculate rebalance indices
rebalance_indices = [dates.index(pos_d) for pos_d in position_dates]
```

**Differences:**
| Aspect | Finlab | Polars Backtest |
|--------|--------|-----------------|
| Resample | Uses pandas date_range directly | Uses `_resample_position()` helper |
| resample=None | Only rebalance on change | Not supported (use 'D') |
| Price adjustment | Uses Market class | Uses factor DataFrame |

---

### 2. Core Simulation Loop

#### Finlab (`backtest_core.pyx:backtest_()`)

```cython
# Main loop order (lines 270-534)
for d, date in enumerate(price_index):

    # 2.1 Skip dates before position starts
    if date < pos_index[0]:
        creturn[d] = 1
        continue

    # 2.2 Update positions with daily return (lines 284-401)
    balance = cash
    for sid in positions:
        # Handle NaN prices
        if isnan(previous_price[sidprice]):
            previous_price[sidprice] = price_values[d, sidprice]

        # Calculate return ratio
        r = price_values[d, sidprice] / previous_price[sidprice]
        if isnan(r):
            r = 1

        # Update position value
        pos[sid] *= r

        # Update cr/maxcr for stop tracking
        cr[sid] *= r
        maxcr[sid] = max(maxcr[sid], cr[sid])

        # Stop loss/take profit detection
        if touched_exit:
            # Intraday stop detection using high/low
            ...
        else:
            # Close-based stop detection
            cr_at_close = cr[sid] * close_values[d, sidprice] / price_values[d, sidprice]
            if cr_at_close >= max_r:
                exit_stocks_temp.push_back(sid)
            elif cr_at_close < min_r:
                exit_stocks_temp.push_back(-sid)

        balance += pos[sid]

    # 2.3 Record balance BEFORE rebalance
    creturn[d] = balance

    # 2.4 Process yesterday's exits (lines 432-451)
    for sid in exit_stocks:
        if not will_be_set_by_rebalance:
            cash = set_position(pos, sid, 0, cash, ...)
            exited_stocks.push_back(sid)
            cr[sid] = 1
            maxcr[sid] = 1

    # 2.5 Transfer today's stops to exit_stocks (lines 456-459)
    exit_stocks = exit_stocks_temp.copy()
    exit_stocks_temp.clear()

    # 2.6 Rebalance (lines 464-491)
    if should_rebalance:
        # Reset cr/maxcr
        if retain_cost_when_rebalance:
            for sid, pv in enumerate(pos_values[pos_id]):
                if new_position_or_direction_change:
                    cr[sid] = 1
                    maxcr[sid] = 1
        else:
            cr.fill(1)
            maxcr.fill(1)

        # Zero out stopped stocks
        if stop_trading_next_period:
            for sid in exited_stocks:
                pos_values[pos_id, abs(sid)] = 0

        # Execute rebalance
        cash = rebalance(pos, pos_values[pos_id], cash, ...)

    # 2.7 Calculate final balance with close prices (lines 496-511)
    balance = cash
    for pos in positions:
        balance += pos.value * close_price / trade_price
    creturn[d] = balance
```

#### Polars Backtest (`simulation.rs:simulate_backtest()`)

```rust
// Main loop (lines 547-843)
for t in 0..n_times {
    if config.finlab_mode {
        // ====== FINLAB MODE ======
        if t > 0 {
            // Step 1: Update cr/maxcr for all positions
            update_max_prices(&mut portfolio, &close_prices[t]);

            // Detect stops for T+1 execution
            let mut today_stops = detect_stops_finlab(&portfolio, &close_prices[t], config);

            // Step 2: Execute pending stop exits (yesterday's detection)
            if !pending_stop_exits.is_empty() {
                // Check will_be_set_by_rebalance
                for stock_id in exits_to_process {
                    let market_value = pos.last_market_value;
                    let sell_value = market_value - market_value.abs() * (fee + tax);
                    portfolio.cash += sell_value;

                    if stop_trading_next_period {
                        stopped_stocks[stock_id] = true;
                    }
                }
                pending_stop_exits.clear();
            }

            // Transfer today's stops to pending
            pending_stop_exits.extend(today_stops);

            // Step 3: Execute rebalance
            if let Some(target_weights) = pending_weights.take() {
                // Zero out stopped stocks
                if stop_trading_next_period {
                    for (i, stopped) in stopped_stocks.iter().enumerate() {
                        if *stopped { target_weights[i] = 0.0; }
                    }
                    // Re-normalize remaining weights
                    ...
                }

                execute_finlab_rebalance(&mut portfolio, &target_weights, ...);
                stopped_stocks = vec![false; n_assets];
            }
        }

        // Check for new rebalance signal
        if rebalance_indices.contains(&t) {
            pending_weights = Some(normalize_weights_finlab(&weights[weight_idx], ...));
            weight_idx += 1;
        }

        // Record balance
        creturn.push(portfolio.balance_finlab(&close_prices[t]));
    }
}
```

**Key Processing Order Comparison:**

| Step | Finlab Order | Polars Backtest Order |
|------|-------------|----------------------|
| 1 | Update positions (pos *= r) | Update cr/maxcr (`update_max_prices`) |
| 2 | Update cr/maxcr | Detect today's stops |
| 3 | Detect today's stops | Execute pending stops (yesterday's) |
| 4 | Record balance (before rebalance) | Transfer today's stops to pending |
| 5 | Execute pending stops (yesterday's) | Execute pending rebalance |
| 6 | Transfer today's stops | Check new rebalance signal |
| 7 | Execute rebalance | Record balance |
| 8 | Calculate final balance | Update prev_prices |

---

### 3. Stop Loss / Take Profit Detection

#### Finlab (`backtest_core.pyx`, lines 326-393)

```cython
# Calculate thresholds
entry_pos = pos[sid] / cr[sid]
if entry_pos > 0:  # Long position
    max_r = 1 + take_profit_abs
    min_r = max(1 - stop_loss_abs, maxcr[sid] - trail_stop_abs)
else:  # Short position
    max_r = min(1 + stop_loss_abs, maxcr[sid] + trail_stop_abs)
    min_r = 1 - take_profit_abs

# Detection using cr_at_close
cr_at_close = cr[sid] * close_values[d, sidprice] / price_values[d, sidprice]
if cr_at_close >= max_r:
    exit_stocks_temp.push_back(sid)   # Take profit
elif cr_at_close < min_r:
    exit_stocks_temp.push_back(-sid)  # Stop loss
```

#### Polars Backtest (`simulation.rs:detect_stops_finlab()`)

```rust
// Use cumulative cr from Position (updated by update_max_prices)
let cr = pos.cr;

// Finlab's cr_at_close formula (even when close == price)
let cr_at_close = cr * current_price / current_price;

// Use cumulative maxcr from Position
let maxcr = pos.maxcr;

// Check take profit: cr_at_close >= 1 + take_profit
if config.take_profit < f64::INFINITY && cr_at_close >= 1.0 + config.take_profit {
    return Some(stock_id);
}

// Calculate min_r using Finlab formula
let stop_threshold = 1.0 - config.stop_loss;
let trail_threshold = if config.trail_stop < f64::INFINITY {
    maxcr - config.trail_stop
} else {
    f64::NEG_INFINITY
};
let min_r = stop_threshold.max(trail_threshold);

// Check stop loss / trail stop: cr_at_close < min_r
if cr_at_close < min_r {
    return Some(stock_id);
}
```

**Differences:**
| Aspect | Finlab | Polars Backtest |
|--------|--------|-----------------|
| cr_at_close | Uses `close/price` ratio | Simplified `price/price` (same in finlab_mode) |
| Long/Short | Different formulas | Only long positions supported |
| touched_exit | Full OHLC support | Not yet implemented |

---

### 4. Rebalance Execution

#### Finlab (`backtest_core.pyx:rebalance()`, lines 107-137)

```cython
cdef double rebalance(...):
    # Calculate total balance
    balance = cash
    for pos in positions:
        balance += pos.value

    # Calculate ratio for weight scaling
    ratio = balance / max(abs(newp).sum(), 1)

    # Execute positions using set_position
    for sid, v in enumerate(newp):
        v2 = v * ratio
        # Apply position limit
        if abs(v2) > balance * position_limit:
            v2 = balance * position_limit * sign(v2)

        cash = set_position(pos, sid, v2, cash, ...)

    return cash
```

#### Finlab (`set_position()`, lines 54-105)

```cython
cdef double set_position(pos, sid, position, cash, fee_ratio, tax_ratio, ...):
    # Record entry/exit
    if not prev_has_position and next_has_position:
        record_entry(sid, position/balance, entry_transaction=1)
    elif prev_has_position and not next_has_position:
        record_exit(sid, exit_transaction=1)

    # Fast exit path
    if position == 0:
        if exist:
            cash += pos[sid] - abs(pos[sid]) * (fee_ratio + tax_ratio)
            pos[sid] = 0
        return cash

    # Calculate trade amount
    amount = position - pos[sid]
    is_entry = (position >= 0 and amount > 0) or (position <= 0 and amount < 0)
    cost = abs(amount) * fee_ratio if is_entry else abs(amount) * (fee_ratio + tax_ratio)

    if buy:
        cash -= amount
        pos[sid] += amount - cost
    else:
        cash += amount - cost
        pos[sid] -= amount

    return cash
```

#### Polars Backtest (`simulation.rs:execute_finlab_rebalance()`)

```rust
fn execute_finlab_rebalance(...) {
    // Step 1: Update all positions to market value using last_market_value
    for (stock_id, pos) in portfolio.positions.iter_mut() {
        pos.value = pos.last_market_value;
        pos.entry_price = close_price;
    }

    // Step 2: Calculate current market value (balance)
    let balance = portfolio.total_cost_basis();

    // Step 3: Calculate ratio for weight scaling
    let ratio = balance / total_target_weight.max(1.0);

    // Step 4: Process each stock using Finlab's set_position logic
    let old_positions = portfolio.positions.clone();
    portfolio.positions.clear();
    let mut cash = portfolio.cash;

    for (stock_id, target_weight) in target_weights.iter().enumerate() {
        let target_value = target_weight * ratio;
        let current_value = old_positions.get(&stock_id).copied().unwrap_or(0.0);
        let amount = target_value - current_value;

        // Apply fee logic
        let is_entry = (target_value >= 0.0 && amount > 0.0) ||
                       (target_value <= 0.0 && amount < 0.0);
        let cost = if is_entry {
            amount.abs() * config.fee_ratio
        } else {
            amount.abs() * (config.fee_ratio + config.tax_ratio)
        };

        // Update cash and position
        if is_buy {
            cash -= amount;
            new_position_value = current_value + amount - cost;
        } else {
            cash += sell_amount - cost;
            new_position_value = current_value - sell_amount;
        }

        // Handle cr/maxcr preservation
        if config.retain_cost_when_rebalance && is_continuing {
            // Preserve old stop tracking
            ...
        } else {
            // Reset all
            cr = 1.0; maxcr = 1.0;
        }
    }

    portfolio.cash = cash;
}
```

**Differences:**
| Aspect | Finlab | Polars Backtest |
|--------|--------|-----------------|
| Position storage | C++ map<int, double> | Rust HashMap<usize, Position> |
| Balance calc | Sum of position values | `total_cost_basis()` method |
| Fee handling | Inline in set_position | Separate logic in rebalance |

---

### 5. Trade Record Generation

#### Finlab (`backtest_core.pyx:get_trade_stocks()`)

```cython
cpdef get_trade_stocks(pos_columns, price_index, touched_exit):
    ret = [
        [pos_columns[t[0]],  # stock_id
         price_index[t[1]],  # entry_date
         price_index[t[2]] if t[2] != -1 else -1,  # exit_date
         price_index[t[1]-1],  # entry_sig_date
         (price_index[t[2]-1] if not touched_exit else price_index[t[2]]) if t[2] != -1 else -1,
         mae_mfe.trade_positions[i],  # position
         t[2] - t[1] if t[2] != -1 else len(price_index) - t[1],  # period
         t[1],  # entry_index
         t[2],  # exit_index
        ] for i, t in enumerate(mae_mfe.trades)]

    return ret, stock_operations
```

#### Polars Backtest (`simulation.rs:RealTracker`)

```rust
impl TradeTracker for RealTracker {
    fn open_trade(&mut self, stock_id, entry_index, signal_index, entry_price, weight) {
        self.open_trades.insert(stock_id, OpenTrade {
            stock_id, entry_index, entry_sig_index: signal_index, weight, entry_price,
        });
    }

    fn close_trade(&mut self, stock_id, exit_index, exit_sig_index, exit_price, ...) {
        if let Some(open_trade) = self.open_trades.remove(&stock_id) {
            let trade = TradeRecord {
                stock_id,
                entry_index: Some(open_trade.entry_index),
                exit_index: Some(exit_index),
                entry_sig_index: open_trade.entry_sig_index,
                exit_sig_index,
                position_weight: open_trade.weight,
                entry_price: open_trade.entry_price,
                exit_price: Some(exit_price),
                trade_return: None,
            };
            let trade_return = trade.calculate_return(fee_ratio, tax_ratio);
            self.completed_trades.push(TradeRecord { trade_return, ..trade });
        }
    }

    fn add_pending_entry(&mut self, stock_id, signal_index, weight) {
        // For signals not yet executed (Finlab: entry_date=NaT)
        self.completed_trades.push(TradeRecord {
            stock_id,
            entry_index: None,
            exit_index: None,
            ...
        });
    }
}
```

**Differences:**
| Aspect | Finlab | Polars Backtest |
|--------|--------|-----------------|
| Trade tracking | mae_mfe module (Cython) | RealTracker (Rust trait) |
| Zero-cost mode | N/A | NoopTracker for performance |
| Pending entries | Explicit in get_trade_stocks | `add_pending_entry()` method |
| Signal date calc | `entry_index - 1` | Explicit signal_index parameter |

---

### 6. Balance Calculation

#### Finlab

```cython
# During loop (line 408)
creturn[d] = balance  # balance = cash + sum(pos values) BEFORE rebalance

# After rebalance (lines 496-511)
balance = cash
for pos in positions:
    trade_price = price_values[d, pos2price[sid]]
    close_price = close_values[d, pos2price[sid]]
    if isnan(trade_price) or isnan(close_price):
        balance += pos.value
    else:
        balance += pos.value * close_price / trade_price
creturn[d] = balance
```

#### Polars Backtest

```rust
// balance_finlab method
fn balance_finlab(&self, prices: &[f64]) -> f64 {
    let pos_value: f64 = self.positions.iter()
        .map(|(&stock_id, p)| {
            // Use last_market_value (updated via pos *= r)
            p.last_market_value
        })
        .sum();
    self.cash + pos_value
}
```

**Differences:**
| Aspect | Finlab | Polars Backtest |
|--------|--------|-----------------|
| Before rebalance | Uses position values | Uses last_market_value |
| After rebalance | Uses close/trade ratio | Same as before (simplified) |
| NaN handling | Falls back to pos.value | Uses last valid market value |

---

## Feature Comparison Matrix

| Feature | Finlab | Polars Backtest | Notes |
|---------|--------|-----------------|-------|
| T+1 execution | ✅ | ✅ | Same behavior |
| resample='D' | ✅ | ✅ | Daily rebalance |
| resample='W'/'M'/'Q' | ✅ | ✅ | Periodic rebalance |
| resample=None | ✅ | ✅ | Only on change |
| resample_offset | ✅ | ✅ | Date offset |
| stop_loss | ✅ | ✅ | Close-based |
| take_profit | ✅ | ✅ | Close-based |
| trail_stop | ✅ | ✅ | Using maxcr |
| touched_exit | ✅ | 🔶 WIP | OHLC intraday (7/8 tests pass) |
| retain_cost_when_rebalance | ✅ | ✅ | Preserve cr/maxcr |
| stop_trading_next_period | ✅ | ✅ | Skip stopped stocks |
| position_limit | ✅ | ✅ | Max weight |
| Short positions | ✅ | ✅ | Negative weights |
| mae_mfe analysis | ✅ | ❌ | Trade analysis |
| Trade records | ✅ | ✅ | Different format |
| Factor adjustment | Via Market | Via factor DataFrame | |

---

## Known Differences & Edge Cases

### 1. Floating Point Precision

Finlab uses cumulative multiplication (`cr *= r`) which accumulates floating point errors differently than our Rust implementation. This can cause stop triggers to differ at exact threshold boundaries.

**Example:**
```python
# Finlab: cr = 0.9499999999999998 after many multiplications
# Polars: cr = 0.95 exactly
# At stop_loss=0.05, Finlab triggers stop, Polars doesn't
```

### 2. cr_at_close Calculation

Finlab always computes `cr_at_close = cr * close / price`, even when close == price. This multiply-divide operation affects floating point precision:

```python
# cr = 0.9499999999999998
# cr_at_close = cr * price / price = 0.95  (precision change!)
```

Our implementation replicates this behavior in `detect_stops_finlab()`.

### 3. Stopped Stock Re-normalization

When `stop_trading_next_period=True`, Finlab zeros out stopped stocks and the rebalance ratio automatically scales up remaining weights. We replicate this with explicit re-normalization.

### 4. Pending Entries

Finlab adds pending entries for stocks with buy signals on the last day. We handle this in `add_pending_entry()` for the TradeTracker.

### 5. Touched Exit Implementation

The `touched_exit` feature uses OHLC prices for intraday stop detection. Key implementation details:

**NaN Handling Fix:**
- Finlab uses a per-position `previous_price` that tracks the last valid price
- When a day has NaN price, the ratio `r = close / previous_price` becomes NaN, which Finlab sets to 1
- Our implementation uses `pos.previous_price` in `detect_touched_exit()` which maintains the last valid price
- The `update_previous_prices()` function is called AFTER touched_exit detection to ensure correct timing

**Remaining Issue (trail_stop=0.1):**
- One test fails due to floating point precision at exact threshold boundaries
- Example: `low_r = 0.928571428571429` vs `min_r = 0.928571428571428`
- The difference is ~1e-16 (within double precision error)
- This causes `low_r <= min_r` to evaluate differently between Python/Cython and Rust
- Other trail_stop values (e.g., 0.15) pass because they don't hit exact boundary conditions

---

## Code Organization

### Finlab
```
finlab/
├── backtest.py          # High-level orchestration, resample, config
├── core/
│   ├── backtest_core.pyx  # Cython core simulation
│   ├── mae_mfe.pyx        # Trade tracking, MAE/MFE
│   └── report.py          # Report class
└── market.py              # Market abstraction for prices
```

### Polars Backtest
```
polars_backtest/
├── python/
│   └── polars_backtest/
│       └── __init__.py    # High-level orchestration, Report class
└── btcore/
    └── src/
        ├── simulation.rs   # Core simulation, TradeTracker
        ├── weights.rs      # Weight normalization
        └── lib.rs          # Python bindings
```

---

## Performance Notes

1. **NoopTracker**: Polars Backtest uses zero-cost abstraction for trade tracking when not needed
2. **Rust vs Cython**: Both compile to native code with similar performance characteristics
3. **Memory**: Rust uses HashMap, Finlab uses C++ map (both O(log n) or O(1) access)
4. **Parallelization**: Neither implementation parallelizes the main loop (sequential by nature)
