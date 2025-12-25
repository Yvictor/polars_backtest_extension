# Backtest Flow Comparison: Finlab vs Polars Backtest

## Overview

This document compares the backtest simulation flow between:
1. **Finlab**: `backtest.py` + `backtest_core.pyx` (Python/Cython)
2. **Polars Backtest**: `__init__.py` + `simulation.rs` (Python/Rust)

---

## High-Level Architecture

### Finlab Architecture
```
User → sim() → arguments() → backtest_() → get_trade_stocks() → Report
         │          │            │                  │
         │          │            ├─ Main loop       ├─ trades
         │          │            ├─ rebalance()     └─ stock_operations
         │          │            └─ set_position()
         │          │
         │          └─ Price arrays (numpy)
         │             Position arrays (numpy)
         │             Resample dates (numpy)
         │
         └─ Resample handling (pandas)
            Market price fetching
            Line notification
```

### Polars Backtest Architecture
```
User → backtest_with_report() → _backtest_with_trades() → Report
              │                         │
              │                         ├─ simulate_backtest<T>()
              │                         │       ├─ update_max_prices()
              │                         │       ├─ detect_stops_finlab()
              │                         │       ├─ detect_touched_exit()
              │                         │       └─ execute_finlab_rebalance()
              │                         │
              │                         └─ TradeTracker trait
              │                               ├─ NoopTracker (zero-cost)
              │                               └─ RealTracker (full tracking)
              │
              └─ _resample_position() (pandas-based)
                 _filter_changed_positions()
```

---

## Step-by-Step Flow Comparison

### Phase 1: Entry Point & Pre-processing

#### Finlab (`backtest.py` sim() lines 247-601)

```python
def sim(position, resample=None, ...):
    # 1. Market Resolution (line 404-408)
    market = config.get_market() or get_market_by_name(market)

    # 2. Price Data Resolution (line 424-462)
    price = market.get_trading_price(trade_at_price, adj=True)
    if touched_exit:
        high = market.get_price('high', adj=True)
        low = market.get_price('low', adj=True)
        open_ = market.get_price('open', adj=True)

    # 3. Resample Handling (line 507-580)
    if isinstance(resample, str):
        alldates = pd.date_range(position.index[0], end, freq=resample)
        dates = [d for d in alldates if d <= present_data_date]
    elif resample is None:
        # Only rebalance when portfolio changes (line 574-580)
        change = (position.diff().abs().sum(axis=1) != 0) | \
                 ((position.index == position.index[0]) & position.iloc[0].notna().any())
        position = position.loc[change]

    # 4. Build arguments array (line 601-602)
    args = arguments(price, close, high, low, open_, position, dates)
    # Returns: [price_values, close_values, high_values, low_values, open_values,
    #           price_index, price_columns, pos_values, pos_index, pos_columns, resample_dates]

    # 5. Call Cython core
    creturn_value = backtest_(*args, encryption=..., fee_ratio=..., ...)
```

#### Polars Backtest (`__init__.py` backtest_with_report() lines 713-1063)

```python
def backtest_with_report(close, position, resample='D', ...):
    # 1. Resolve trade_at_price (line 832-851)
    if trade_at_price == "close":
        trade_prices = close
    elif trade_at_price == "open":
        trade_prices = open

    # 2. Calculate original prices via factor (line 855-871)
    if factor is not None:
        original_prices = trade_prices / factor

    # 3. Resample Handling (line 881-885)
    if resample is None:
        position = _filter_changed_positions(position)  # line 338-420
    elif resample != 'D':
        position = _resample_position(position, dates, resample, resample_offset)

    # 4. Calculate rebalance indices (line 897-977)
    rebalance_indices = []
    for pos_d in position_dates:
        idx = dates.index(pos_d)
        rebalance_indices.append(idx)

    # 5. Create config and call Rust (line 1017-1051)
    config = BacktestConfig(fee_ratio=..., finlab_mode=True, ...)
    result = _backtest_with_trades(close_data, original_prices_data, position_data,
                                   rebalance_indices, config, open_data, high_data, low_data)
```

### 🔴 Key Differences: Entry Point

| Aspect | Finlab | Polars Backtest |
|--------|--------|-----------------|
| Market abstraction | Built-in `Market` class | User provides DataFrames directly |
| Price types | 5 options: close, open, open_close_avg, high_low_avg, price_avg | close, open, high, low, or custom DataFrame |
| Date handling | `pd.DatetimeIndex` | Polars Series (dates as strings) |
| Resample | Pandas `date_range` + manual filtering | Custom `_resample_position()` helper |
| Original prices | Via `Market.get_trading_price(adj=False)` | Via `factor` DataFrame |

---

### Phase 2: Main Simulation Loop

#### Finlab (`backtest_core.pyx` backtest_() lines 270-534)

```cython
for d, date in enumerate(price_index):

    # Skip dates before position starts
    if date < pos_index[0]:
        creturn[d] = 1
        continue

    # ======== STEP A: Update positions with daily return ========
    balance = cash
    it = pos.begin()
    while it != pos.end():
        sid = dereference(it).first
        sidprice = pos2price[sid]

        # A1. Handle NaN in previous_price
        if isnan(previous_price[sidprice]):
            previous_price[sidprice] = price_values[d, sidprice]

        # A2. Calculate return ratio
        r = price_values[d, sidprice] / previous_price[sidprice]
        if isnan(r):
            r = 1

        # A3. Update position value (Finlab's core formula)
        pos[sid] *= r

        # A4. Update cr and maxcr for stop tracking
        cr[sid] *= r
        maxcr[sid] = max(maxcr[sid], cr[sid])

        # ======== STEP B: Stop Detection ========
        entry_pos = pos[sid] / cr[sid]

        if entry_pos > 0:  # Long position
            max_r = 1 + take_profit_abs
            min_r = max(1 - stop_loss_abs, maxcr[sid] - trail_stop_abs)
        else:  # Short position
            max_r = min(1 + stop_loss_abs, maxcr[sid] + trail_stop_abs)
            min_r = 1 - take_profit_abs

        if touched_exit:
            # B1. Calculate OHLC ratios (lines 342-344)
            high_r = cr[sid] / r * (high_values[d, sidprice] / previous_price[sidprice])
            low_r = cr[sid] / r * (low_values[d, sidprice] / previous_price[sidprice])
            open_r = cr[sid] / r * (open_values[d, sidprice] / previous_price[sidprice])

            # B2. Check touch conditions
            touch_open = open_r >= max_r or open_r <= min_r
            touch_high = high_r >= max_r
            touch_low = low_r <= min_r

            # B3. Priority: open > high > low, immediate exit (T+0)
            if touch_open:
                pos[sid] *= open_r / r
            elif touch_high:
                pos[sid] = entry_pos * max_r
            elif touch_low:
                pos[sid] = entry_pos * min_r

            if touch_low or touch_high or touch_open:
                # Immediate exit with fees
                cash = set_position(pos, sid, 0, cash, fee_ratio, tax_ratio, ...)
                exited_stocks.push_back(sid if touch_high else -sid)

        else:  # Close-based detection (lines 384-393)
            cr_at_close = cr[sid] * close_values[d, sidprice] / price_values[d, sidprice]
            if cr_at_close >= max_r:
                exit_stocks_temp.push_back(sid)    # Take profit
            elif cr_at_close < min_r:
                exit_stocks_temp.push_back(-sid)   # Stop loss

        # A5. Update previous_price and accumulate balance
        if not isnan(price_values[d, sidprice]):
            previous_price[sidprice] = price_values[d, sidprice]
        balance += pos[sid]
        postincrement(it)

    # ======== STEP C: Record balance BEFORE rebalance ========
    creturn[d] = balance

    # ======== STEP D: Process yesterday's stop exits (T+1) ========
    for sid in exit_stocks:
        abssid = abs(sid)
        # Check: will this stock be set by rebalance anyway?
        will_be_set_by_rebalance = should_rebalance and not stop_trading_next_period \
                                   and pos_values[pos_id, abssid] != 0

        if pos.find(abssid) != pos.end() and pos[abssid] != 0 and not will_be_set_by_rebalance:
            cash = set_position(pos, abssid, 0, cash, fee_ratio, tax_ratio, ...)
            exited_stocks.push_back(sid)
            cr[abssid] = 1
            maxcr[abssid] = 1
            # Remove from today's detection to prevent duplicates (lines 444-449)

    exit_stocks.clear()

    # ======== STEP E: Transfer today's stops to pending ========
    exit_stocks = exit_stocks_temp.copy()
    exit_stocks_temp.clear()

    # ======== STEP F: Rebalance if needed ========
    if should_rebalance:
        # F1. Reset cr/maxcr based on retain_cost_when_rebalance (lines 468-478)
        if retain_cost_when_rebalance:
            for sid, pv in enumerate(pos_values[pos_id]):
                # Only reset for NEW positions or DIRECTION CHANGE
                if pv != 0 and (pos.find(sid) == pos.end() or pos[sid] * pv <= 0):
                    cr[sid] = 1
                    maxcr[sid] = 1
        else:
            cr.fill(1)
            maxcr.fill(1)

        # F2. Zero out stopped stocks (lines 481-483)
        if stop_trading_next_period:
            for sid in exited_stocks:
                pos_values[pos_id, abs(sid)] = 0

        # F3. Execute rebalance
        cash = rebalance(pos, pos_values[pos_id], cash, fee_ratio, tax_ratio, position_limit)
        exited_stocks.clear()

    # ======== STEP G: Calculate final balance with close prices ========
    balance = cash
    for sid, val in pos.items():
        trade_price = price_values[d, pos2price[sid]]
        close_price = close_values[d, pos2price[sid]]
        if isnan(trade_price) or isnan(close_price):
            balance += val
        else:
            balance += val * close_price / trade_price
    creturn[d] = balance
```

#### Polars Backtest (`simulation.rs` simulate_backtest() lines 526-933)

```rust
for t in 0..n_times {
    if config.finlab_mode {
        // ====== FINLAB MODE ======
        if t > 0 {
            // ======== STEP 1: Update cr for all positions ========
            // Called BEFORE touched_exit to ensure cr reflects today's return
            update_max_prices(&mut portfolio, &close_prices[t]);
            // This does: pos.cr *= r; pos.last_market_value *= r; pos.maxcr = max(maxcr, cr)

            // ======== STEP 2: Touched exit (T+0) ========
            if config.touched_exit {
                if let Some(ref ohlc_data) = ohlc {
                    let touched_exits = detect_touched_exit(
                        &portfolio, &ohlc_data.open[t], &ohlc_data.high[t],
                        &ohlc_data.low[t], &close_prices[t], &prev_prices, config
                    );

                    for touched in &touched_exits {
                        if let Some(pos) = portfolio.positions.remove(&touched.stock_id) {
                            // Adjust to touched price
                            let exit_value = pos.last_market_value * touched.exit_ratio;
                            let sell_value = exit_value - exit_value.abs() * (fee + tax);
                            portfolio.cash += sell_value;

                            if config.stop_trading_next_period {
                                stopped_stocks[touched.stock_id] = true;
                            }

                            // Track trade (exit_sig_index = t for touched_exit)
                            tracker.close_trade(touched.stock_id, t, Some(t), ...);
                        }
                    }
                }
            }

            // Update previous_prices AFTER touched_exit detection
            update_previous_prices(&mut portfolio, &close_prices[t]);

            // ======== STEP 3: Detect stops for T+1 ========
            let mut today_stops = if config.touched_exit {
                Vec::new()  // Already handled above
            } else {
                detect_stops_finlab(&portfolio, &close_prices[t], config)
            };

            // ======== STEP 4: Execute pending stop exits (yesterday's T+1) ========
            if !pending_stop_exits.is_empty() {
                let exits_to_process: Vec<usize> = pending_stop_exits
                    .iter()
                    .filter(|&&stock_id| {
                        // Check will_be_set_by_rebalance (Finlab line 434)
                        if let Some(ref weights) = pending_weights {
                            let has_nonzero_weight = stock_id < weights.len()
                                                     && weights[stock_id].abs() > 1e-10;
                            if config.stop_trading_next_period {
                                true  // Always process
                            } else {
                                !has_nonzero_weight  // Skip if rebalance will handle it
                            }
                        } else {
                            true  // No pending rebalance
                        }
                    })
                    .collect();

                for stock_id in exits_to_process {
                    if let Some(pos) = portfolio.positions.remove(&stock_id) {
                        // Sell using market value (Finlab: pos[sid] already updated by pos *= r)
                        let market_value = pos.last_market_value;
                        let sell_value = market_value - market_value.abs() * (fee + tax);
                        portfolio.cash += sell_value;

                        if config.stop_trading_next_period {
                            stopped_stocks[stock_id] = true;
                        }

                        // Remove from today's stops to prevent duplicates
                        today_stops.retain(|&x| x != stock_id);

                        tracker.close_trade(stock_id, t, None, ...);
                    }
                }
                pending_stop_exits.clear();
            }

            // Transfer today's stops to pending (Finlab: exit_stocks_temp → exit_stocks)
            pending_stop_exits.extend(today_stops);

            // ======== STEP 5: Execute rebalance ========
            if let Some(mut target_weights) = pending_weights.take() {
                let signal_index = pending_signal_index.take().unwrap_or(t - 1);

                // Zero out stopped stocks and re-normalize (Finlab lines 481-483)
                if config.stop_trading_next_period {
                    let original_sum: f64 = target_weights.iter().map(|w| w.abs()).sum();
                    for (i, stopped) in stopped_stocks.iter().enumerate() {
                        if *stopped && i < target_weights.len() {
                            target_weights[i] = 0.0;
                        }
                    }
                    // Re-normalize remaining weights
                    let remaining_sum: f64 = target_weights.iter().map(|w| w.abs()).sum();
                    if remaining_sum > 0.0 && remaining_sum < original_sum {
                        let scale_factor = original_sum / remaining_sum;
                        for w in target_weights.iter_mut() {
                            *w *= scale_factor;
                        }
                    }
                }

                // Close all trades during rebalance (Finlab behavior)
                for stock_id in portfolio.positions.keys() {
                    tracker.close_trade(*stock_id, t, Some(signal_index), ...);
                }

                execute_finlab_rebalance(&mut portfolio, &target_weights, &close_prices[t], config);
                active_weights = target_weights.clone();

                // Open new trades
                for (stock_id, &target_weight) in target_weights.iter().enumerate() {
                    if target_weight != 0.0 && portfolio.positions.contains_key(&stock_id) {
                        tracker.open_trade(stock_id, t, signal_index, ...);
                    }
                }

                stopped_stocks = vec![false; n_assets];
            }

            update_entry_prices_after_nan(&mut portfolio, &close_prices[t], &prev_prices);
        }

        // ======== STEP 6: Check for new rebalance signal ========
        if rebalance_indices.contains(&t) && weight_idx < weights.len() {
            let target_weights = normalize_weights_finlab(&weights[weight_idx], ...);
            pending_weights = Some(target_weights);
            pending_signal_index = Some(t);
            weight_idx += 1;
        }

        // ======== STEP 7: Record balance ========
        creturn.push(portfolio.balance_finlab(&close_prices[t]));
    }

    prev_prices = close_prices[t].clone();
}
```

### 🔴 Key Differences: Main Loop

| Step | Finlab Order | Polars Backtest Order |
|------|-------------|----------------------|
| 1 | Update pos *= r, cr *= r, maxcr | `update_max_prices()` (cr *= r) |
| 2 | Check touched_exit (inline) | `detect_touched_exit()` |
| 3 | Update previous_price | `update_previous_prices()` |
| 4 | Store to exit_stocks_temp | `detect_stops_finlab()` |
| 5 | Record balance (BEFORE rebalance) | Execute pending stops |
| 6 | Process exit_stocks (yesterday's) | Execute pending rebalance |
| 7 | Transfer temp → stocks | Check new rebalance |
| 8 | Execute rebalance | Record balance |
| 9 | Calculate final balance (AFTER) | |

**Important**: Finlab records balance twice - once before and once after rebalance. Polars Backtest records once at the end.

---

### Phase 3: Stop Loss / Take Profit Detection

#### Finlab Close-Based (`backtest_core.pyx` lines 384-393)

```cython
# Uses cumulative return ratio
cr_at_close = cr[sid] * close_values[d, sidprice] / price_values[d, sidprice]

# For long positions:
max_r = 1 + take_profit_abs
min_r = max(1 - stop_loss_abs, maxcr[sid] - trail_stop_abs)

# Trigger conditions (note: < not <=)
if cr_at_close >= max_r:      # Take profit
    exit_stocks_temp.push_back(sid)
elif cr_at_close < min_r:     # Stop loss or trail stop
    exit_stocks_temp.push_back(-sid)
```

#### Polars Backtest (`simulation.rs` detect_stops_finlab() lines 1178-1272)

```rust
fn detect_stops_finlab(portfolio: &PortfolioState, prices: &[f64], config: &BacktestConfig) -> Vec<usize> {
    // Use cumulative cr from Position (already updated by update_max_prices)
    let cr = pos.cr;

    // Finlab's cr_at_close formula - multiply-divide affects floating point!
    let cr_at_close = cr * current_price / current_price;

    // Use cumulative maxcr
    let maxcr = pos.maxcr;

    if is_long {
        // Take profit: cr_at_close >= 1 + take_profit
        if config.take_profit < f64::INFINITY && cr_at_close >= 1.0 + config.take_profit {
            return Some(stock_id);
        }

        // Stop loss / trail stop: cr_at_close < min_r
        let stop_threshold = 1.0 - config.stop_loss;
        let trail_threshold = maxcr - config.trail_stop;
        let min_r = stop_threshold.max(trail_threshold);

        if cr_at_close < min_r {
            return Some(stock_id);
        }
    }
}
```

### ✅ Equivalent: Stop detection formulas match exactly

---

### Phase 4: Touched Exit (Intraday Stop Detection)

#### Finlab (`backtest_core.pyx` lines 339-380)

```cython
if touched_exit:
    # Calculate OHLC ratios (relative to cumulative return)
    high_r = cr[sid] / r * (high_values[d, sidprice] / previous_price[sidprice])
    low_r  = cr[sid] / r * (low_values[d, sidprice] / previous_price[sidprice])
    open_r = cr[sid] / r * (open_values[d, sidprice] / previous_price[sidprice])

    # Check touch conditions
    touch_low = low_r <= min_r
    touch_high = high_r >= max_r
    touch_open = open_r >= max_r or open_r <= min_r

    # Priority: open > high > low
    if touch_open:
        pos[sid] *= open_r / r
    elif touch_high:
        pos[sid] = entry_pos * max_r
    elif touch_low:
        pos[sid] = entry_pos * min_r

    # Immediate exit (T+0)
    if touch_low or touch_high or touch_open:
        org_cash = cash
        cash = set_position(pos, sid, 0, cash, fee_ratio, tax_ratio, ...)
        balance += cash - org_cash

        if (touch_low and pos[sid] > 0) or (touch_high and pos[sid] < 0):
            exited_stocks.push_back(-sid)  # Stop loss
        else:
            exited_stocks.push_back(sid)   # Take profit
```

#### Polars Backtest (`simulation.rs` detect_touched_exit() lines 1304-1453)

```rust
fn detect_touched_exit(...) -> Vec<TouchedExitResult> {
    // Get cr from position (already updated by update_max_prices: cr *= r)
    let cr_new = pos.cr;
    let prev_price = pos.previous_price;  // Per-position tracking

    // Calculate r = close / prev_price (same as Finlab line 305)
    let r = close_price / prev_price;

    // Calculate OHLC ratios using EXACT same formula as Finlab
    // cr_new = cr_old * r, so cr_old = cr_new / r
    let open_r = cr_new / r * (open_price / prev_price);
    let high_r = cr_new / r * (high_price / prev_price);
    let low_r = cr_new / r * (low_price / prev_price);

    // Check touch conditions
    let touch_open = open_r >= max_r || open_r <= min_r;
    let touch_high = high_r >= max_r;
    let touch_low = low_r <= min_r;

    // Calculate exit_ratio for adjusting position value
    // Priority: open > high > low
    if touch_open {
        // pos[sid] *= open_r / r
        exit_ratio = open_r / r;
    } else if touch_high {
        // pos[sid] = entry_pos * max_r = pos / cr * max_r
        exit_ratio = max_r / cr_new;
    } else if touch_low {
        // pos[sid] = entry_pos * min_r = pos / cr * min_r
        exit_ratio = min_r / cr_new;
    }

    TouchedExitResult { stock_id, exit_ratio, is_take_profit }
}
```

### ✅ Equivalent: Touched exit formulas match exactly

---

### Phase 5: Rebalance Execution

#### Finlab (`backtest_core.pyx` rebalance() lines 107-137)

```cython
cdef double rebalance(map[int, double] &p, np.ndarray newp, double cash,
                      double fee_ratio, double tax_ratio, double position_limit):
    # Calculate total balance
    cdef double balance = cash
    for val in p.values():
        balance += val

    # Calculate ratio for weight scaling
    cdef double ratio = balance / max(abs(newp).sum(), 1)
    if isnan(ratio):
        ratio = 1

    # Process each target position
    for sid, v in enumerate(newp):
        v2 = v * ratio

        # Apply position limit
        if abs(v2) > balance * position_limit:
            sign = (v2 > 0) * 2 - 1
            v2 = balance * position_limit * sign

        cash = set_position(p, sid, v2, cash, fee_ratio, tax_ratio, ...)

    return cash
```

#### Finlab (`backtest_core.pyx` set_position() lines 54-105)

```cython
cdef double set_position(map[int, double] &p, int sid, double position, double cash,
                         double fee_ratio, double tax_ratio, ...):
    # Fast exit path
    if position == 0:
        if exist:
            cash += p[sid] - abs(p[sid]) * (fee_ratio + tax_ratio)
            p[sid] = 0
        return cash

    # Calculate trade amount (difference method)
    cdef double amount = position - p[sid]
    buy = amount > 0
    is_entry = (position >= 0 and amount > 0) or (position <= 0 and amount < 0)

    # Fee calculation: entry pays fee only, exit pays fee + tax
    cdef double cost = abs(amount) * fee_ratio if is_entry else abs(amount) * (fee_ratio + tax_ratio)

    if buy:
        cash -= amount
        p[sid] += amount - cost
    else:
        cash += amount - cost
        p[sid] -= amount

    return cash
```

#### Polars Backtest (`simulation.rs` execute_finlab_rebalance() lines 1581-1825)

```rust
fn execute_finlab_rebalance(portfolio, target_weights, prices, config) {
    // Step 1: Update all positions to market value
    for (stock_id, pos) in portfolio.positions.iter_mut() {
        pos.value = pos.last_market_value;  // Use cumulative pos *= r
        pos.entry_price = close_price;
    }

    // Step 2: Calculate balance
    let balance = portfolio.total_cost_basis();  // cash + sum(values)

    // Step 3: Calculate ratio
    let total_target_weight: f64 = target_weights.iter().map(|w| w.abs()).sum();
    let ratio = balance / total_target_weight.max(1.0);

    // Step 4: Store old values and rebuild positions
    let old_positions = portfolio.positions.clone();
    portfolio.positions.clear();
    let mut cash = portfolio.cash;

    for (stock_id, &target_weight) in target_weights.iter().enumerate() {
        let target_value = target_weight * ratio;
        let current_value = old_positions.get(&stock_id).copied().unwrap_or(0.0);
        let amount = target_value - current_value;

        // Fee logic matching set_position
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
            new_value = current_value + amount - cost;
        } else {
            cash += sell_amount - cost;
            new_value = current_value - sell_amount;
        }

        // Handle cr/maxcr based on retain_cost_when_rebalance
        let is_continuing = old_value.abs() > 1e-10 && old_value * target_weight > 0.0;
        if config.retain_cost_when_rebalance && is_continuing {
            // Preserve old stop tracking values
            (stop_entry, max_price, cr_val, maxcr_val, prev_price) = old values
        } else {
            // Reset all (Finlab: cr.fill(1); maxcr.fill(1))
            (stop_entry, max_price, cr_val, maxcr_val, prev_price) = (price, price, 1.0, 1.0, price)
        }

        portfolio.positions.insert(stock_id, Position { ... });
    }

    portfolio.cash = cash;
}
```

### 🔴 Key Differences: Rebalance

| Aspect | Finlab | Polars Backtest |
|--------|--------|-----------------|
| Position storage | C++ `map<int, double>` modified in-place | Rust HashMap rebuilt |
| stop_trading | Modifies `pos_values` array before rebalance | Scales up remaining weights |
| Position value | Single `double` | `Position` struct with cr, maxcr, etc. |
| Fee calculation | In `set_position()` | Inline in rebalance |

---

### Phase 6: Balance Calculation

#### Finlab (`backtest_core.pyx` lines 496-511)

```cython
# After rebalance, calculate final balance
balance = cash
for sid, val in pos.items():
    trade_price = price_values[d, pos2price[sid]]
    close_price = close_values[d, pos2price[sid]]

    if isnan(trade_price) or isnan(close_price):
        balance += val  # Use position value directly
    else:
        balance += val * close_price / trade_price

creturn[d] = balance
```

#### Polars Backtest (`simulation.rs` balance_finlab() lines 400-431)

```rust
fn balance_finlab(&self, prices: &[f64]) -> f64 {
    // When close == trade_price (both adj_close), simplifies to:
    // cash + Σ(last_market_value)
    // last_market_value has been updated via cumulative pos *= r
    let pos_value: f64 = self.positions.iter()
        .map(|(&stock_id, p)| p.last_market_value)
        .sum();
    self.cash + pos_value
}
```

### ✅ Equivalent: Balance formula matches (when close == trade_price)

---

## Feature Comparison Matrix

| Feature | Finlab | Polars Backtest | Notes |
|---------|--------|-----------------|-------|
| **Core** |
| T+1 execution | ✅ | ✅ | Same behavior |
| Finlab mode | N/A | ✅ | Exact replication |
| **Resampling** |
| resample='D' | ✅ | ✅ | Daily rebalance |
| resample='W'/'M'/'Q' | ✅ | ✅ | Periodic rebalance |
| resample=None | ✅ | ✅ | Only on position change |
| resample_offset | ✅ | ✅ | Date offset |
| **Stop/Exit** |
| stop_loss | ✅ | ✅ | Using cr formula |
| take_profit | ✅ | ✅ | Using cr formula |
| trail_stop | ✅ | ✅ | Using maxcr formula |
| touched_exit | ✅ | ✅ | OHLC intraday |
| **Options** |
| retain_cost_when_rebalance | ✅ | ✅ | Preserve cr/maxcr |
| stop_trading_next_period | ✅ | ✅ | Skip stopped stocks |
| position_limit | ✅ | ✅ | Max weight |
| Short positions | ✅ | ✅ | Negative weights |
| **Output** |
| Trade records | ✅ | ✅ | Different structure |
| MAE/MFE analysis | ✅ | ❌ | Not implemented |
| Factor adjustment | Via Market | Via factor DataFrame | |
| **Other** |
| Cloud upload | ✅ | ❌ | |
| Line notification | ✅ | ❌ | |
| fast_mode | ✅ | ❌ | |

---

## Known Differences & Edge Cases

### 1. Floating Point Precision

Finlab uses cumulative multiplication (`cr *= r`) which accumulates floating point errors differently than Rust:

```python
# Finlab: cr = 0.9499999999999998 after many multiplications
# Polars: cr = 0.95 exactly
# At stop_loss=0.05, Finlab triggers stop, Polars doesn't
```

### 2. cr_at_close Calculation

Finlab always computes `cr_at_close = cr * close / price`, even when close == price:

```python
# cr = 0.9499999999999998
# cr_at_close = cr * price / price = 0.95  (precision change!)
```

Our implementation replicates this in `detect_stops_finlab()`.

### 3. Trail Stop Floating Point Edge Cases

For trail_stop=0.05 and 0.1, floating point precision at exact boundaries causes differences:
- `low_r = 0.92857142857142860315`
- `min_r = 0.92857142857142849213`
- Difference: ~1e-16

This causes `low_r <= min_r` to evaluate differently.

### 4. Stopped Stock Handling

Finlab zeros out `pos_values` in-place, letting `rebalance()` naturally scale up. We explicitly re-normalize remaining weights.

### 5. Balance Recording

Finlab records balance twice (before and after rebalance). We record once at end.

---

## Code Organization

### Finlab
```
finlab/
├── backtest.py              # High-level: resample, market, config
├── core/
│   ├── backtest_core.pyx    # Core loop, rebalance, set_position
│   ├── mae_mfe.pyx          # Trade tracking, MAE/MFE calculation
│   └── report.py            # Report class
└── market.py                # Market abstraction for prices
```

### Polars Backtest
```
polars_backtest/
├── python/
│   └── polars_backtest/
│       └── __init__.py      # High-level: resample, Report class
└── btcore/
    └── src/
        ├── simulation.rs    # Core loop, rebalance, stop detection
        ├── weights.rs       # Weight normalization
        └── lib.rs           # Python bindings (PyO3)
```

---

## Performance Characteristics

| Aspect | Finlab | Polars Backtest |
|--------|--------|-----------------|
| Language | Cython → C | Rust → native |
| Position storage | C++ `map<int, double>` | Rust `HashMap<usize, Position>` |
| Trade tracking | Global module | Zero-cost trait (`NoopTracker`) |
| Memory | Shared numpy arrays | Vec copies |
| Main loop | Sequential | Sequential |
