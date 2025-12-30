# BTCore Refactoring Plan

## Goal Architecture

```
btcore/src/
├── lib.rs                 # Public API (unchanged)
├── config.rs              # BacktestConfig (unchanged)
├── position.rs            # Position struct (unchanged)
├── tracker.rs             # TradeTracker trait (unchanged)
├── trades.rs              # Trade records (unchanged)
│
├── core/                  # NEW: Shared core logic
│   ├── mod.rs
│   ├── updates.rs         # cr *= r, maxcr, previous_price
│   ├── stops.rs           # detect_stops (unified)
│   ├── rebalance.rs       # execute_rebalance (unified)
│   ├── fees.rs            # Fee calculation
│   └── constants.rs       # ZERO_THRESHOLD, MAX_ITERATIONS
│
├── simulation/
│   ├── mod.rs
│   ├── wide.rs            # Wide format (simplified)
│   └── long.rs            # Long format (simplified)
│
└── (delete old stops.rs, rebalance.rs, weights.rs after extraction)
```

---

## Phase 1: Remove Dead Code

**Status**: Not Started

**File**: `btcore/src/simulation/wide.rs`

Delete these `#[allow(dead_code)]` functions:
- [ ] `execute_pending_stops()` (~L582)
- [ ] `check_stops()` (~L602)
- [ ] `execute_pending_stops_finlab()` (~L625)
- [ ] `check_stops_finlab()` (~L659)
- [ ] `enter_new_positions()` (~L964)
- [ ] `exit_and_adjust_positions()` (~L1037)
- [ ] `rebalance_to_target_weights()` (~L1145)

**Estimate**: -400 lines, zero risk

**Test**: `cargo test -p btcore`

---

## Phase 2: Unify Constants

**Status**: Not Started

**Create** `btcore/src/core/constants.rs`:
```rust
pub const ZERO_THRESHOLD: f64 = 1e-10;
pub const MAX_REBALANCE_ITERATIONS: usize = 100;
```

**Modify**: Replace all `1e-10` and `max_iterations = 100` with constants

**Estimate**: +20, -50 lines

**Test**: `cargo test -p btcore`

---

## Phase 3: Extract Shared Logic

**Status**: Not Started

### 3.1 Position Update Logic

**Current**:
- `wide.rs`: `portfolio.update_max_prices()`
- `long.rs`: `update_positions()`

**Unified** in `core/updates.rs`:
```rust
pub fn update_position_returns<K>(
    positions: &mut HashMap<K, Position>,
    prices: impl Fn(&K) -> Option<f64>,
) {
    for (key, pos) in positions.iter_mut() {
        if let Some(price) = prices(key) {
            let r = price / pos.previous_price;
            pos.cr *= r;
            pos.last_market_value *= r;
            pos.maxcr = pos.maxcr.max(pos.cr);
            pos.max_price = pos.max_price.max(price);
        }
    }
}
```

### 3.2 Stop Detection

**Current**:
- `wide.rs`: `detect_stops_finlab()`
- `long.rs`: `detect_stops_unified()`

**Unified** in `core/stops.rs`:
```rust
pub fn detect_stops<K: Clone>(
    positions: &HashMap<K, Position>,
    get_price: impl Fn(&K) -> f64,
    config: &BacktestConfig,
) -> Vec<K>
```

### 3.3 Rebalance Execution

**Current**:
- `wide.rs`: `execute_finlab_rebalance()` (253 lines)
- `long.rs`: `execute_rebalance_impl()` (95 lines)

**Issue**: Logic differs! wide.rs is more complete

**Unified** in `core/rebalance.rs`: Use wide.rs complete logic

**Estimate**: +200, -400 lines

**Test**: `cargo test -p btcore && just test && just test-slow`

---

## Phase 4: Split simulate_backtest()

**Status**: Not Started

**Current**: 407 lines, finlab_mode and standard_mode mixed

**Split**:
```rust
// simulation/wide.rs
pub fn simulate_backtest_finlab<T: TradeTracker>(...) -> Vec<f64>
pub fn simulate_backtest_standard<T: TradeTracker>(...) -> Vec<f64>

pub fn simulate_backtest<T: TradeTracker>(...) -> Vec<f64> {
    if config.finlab_mode {
        simulate_backtest_finlab(...)
    } else {
        simulate_backtest_standard(...)
    }
}
```

**Estimate**: +50, -20 lines

**Test**: `cargo test -p btcore && just test`

---

## Phase 5: Position Documentation

**Status**: Not Started

**Current**: 8 fields with unclear semantics

**Add** to `position.rs`:
```rust
/// Position in a backtest portfolio.
///
/// # Finlab Mode Field Semantics
/// - `value`: cost_basis (entry amount - entry fee), constant after entry
/// - `entry_price`: price at position entry, reset on rebalance
/// - `stop_entry_price`: price for stop calculation, NEVER reset
/// - `last_market_value`: updated daily via `cr *= r`
/// - `cr`: cumulative return ratio = close / entry_price (via multiplication)
/// - `maxcr`: maximum cr seen (for trailing stop)
///
/// # Invariant
/// actual_market_value = value * close / entry_price = last_market_value
pub struct Position { ... }
```

**Estimate**: +50 lines

---

## Summary

| Phase | Content | Risk | Lines Change |
|-------|---------|------|--------------|
| 1 | Remove dead code | Low | -400 |
| 2 | Unify constants | Low | +20, -50 |
| 3 | Extract shared logic | Medium | +200, -400 |
| 4 | Split simulate | Medium | +50, -20 |
| 5 | Documentation | Low | +50 |
| **Total** | | | **-550** |

---

## Test Strategy

After each phase:
1. `cargo test -p btcore` - All Rust tests pass
2. `just test` - Python tests pass
3. `just test-slow` - Finlab comparison tests pass

**Critical**: Phase 3 must verify wide and long produce identical outputs
