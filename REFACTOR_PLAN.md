# BTCore Refactoring Plan

## Current State Analysis

### Module Structure
```
btcore/src/
├── lib.rs           # Public API exports
├── config.rs        # BacktestConfig (11 flat fields)
├── position.rs      # Position struct (8 fields, ambiguous docs)
├── portfolio.rs     # PortfolioState + weight utils
├── tracker.rs       # TradeTracker trait + 2 implementations
├── stops.rs         # Stop detection (detect_stops, detect_stops_finlab, detect_touched_exit)
├── weights.rs       # Weight normalization
├── rebalance.rs     # DEAD MODULE - never imported
├── trades.rs        # Trade records
├── returns.rs       # Return calculation
├── stats.rs         # Statistics
├── simulation/
│   ├── mod.rs       # Module exports
│   ├── wide.rs      # Wide format (1552 lines after cleanup)
│   └── long.rs      # Long format (1871 lines)
```

### Key Issues Identified

| Issue | Location | Impact | Priority |
|-------|----------|--------|----------|
| Magic number `1e-10` used 34+ times | throughout | Maintainability | High |
| Dead module never imported | rebalance.rs | Clutter | Low |
| Duplicate position update logic | wide.rs:504, long.rs:1115 | DRY | Medium |
| Finlab/Standard mixed in 400-line function | wide.rs:36-443 | Testability | High |
| 7 HashMaps for state snapshot | wide.rs:636-672 | Performance | Medium |
| Inconsistent NaN handling | 6 different patterns | Correctness | Medium |
| Position struct ambiguous documentation | position.rs:11-44 | Clarity | Medium |
| Duplicate TradeTracker implementations | tracker.rs:146-188 | DRY | Low |

---

## Phase 1: Quick Wins (Low Risk, High Value)

**Status**: In Progress

### 1.1 Remove Dead Code from wide.rs ✓

Deleted 6 `#[allow(dead_code)]` functions:
- [x] `execute_pending_stops()`
- [x] `check_stops()`
- [x] `execute_pending_stops_finlab()`
- [x] `check_stops_finlab()`
- [x] `enter_new_positions()`
- [x] `exit_and_adjust_positions()`

**Note**: `rebalance_to_target_weights()` is NOT dead - used by standard mode at line 854.

**Result**: -276 lines

### 1.2 Extract FLOAT_EPSILON Constant

**Current**: 34+ uses of `1e-10` scattered across files

**Create** in `lib.rs`:
```rust
/// Threshold for floating-point comparisons (Finlab compatibility)
pub const FLOAT_EPSILON: f64 = 1e-10;
```

**Replace** all `1e-10` with `FLOAT_EPSILON`:
- position.rs (2 uses)
- portfolio.rs (3 uses)
- stops.rs (4 uses)
- wide.rs (15+ uses)
- long.rs (10+ uses)

**Test**: `cargo test -p btcore`

### 1.3 Add Price Validation Helper

**Current**: 6 different NaN/validity check patterns

**Add** to `lib.rs` or `position.rs`:
```rust
/// Check if price is valid for calculations (positive, non-NaN)
#[inline]
pub fn is_valid_price(p: f64) -> bool {
    p > 0.0 && !p.is_nan()
}
```

**Replace** patterns like:
```rust
// Before
if close_price > 0.0 && !close_price.is_nan()

// After
if is_valid_price(close_price)
```

---

## Phase 2: Structural Improvements (Medium Risk)

**Status**: Not Started

### 2.1 Extract PositionSnapshot Struct

**Current**: 7 separate HashMaps in execute_finlab_rebalance() (wide.rs:636-672)
```rust
let old_positions: HashMap<usize, f64> = ...;
let old_market_values: HashMap<usize, f64> = ...;
let old_stop_entries: HashMap<usize, f64> = ...;
let old_max_prices: HashMap<usize, f64> = ...;
let old_cr: HashMap<usize, f64> = ...;
let old_maxcr: HashMap<usize, f64> = ...;
let old_previous_prices: HashMap<usize, f64> = ...;
```

**Refactor** to single struct:
```rust
struct PositionSnapshot {
    cost_basis: f64,
    market_value: f64,
    stop_entry_price: f64,
    max_price: f64,
    cr: f64,
    maxcr: f64,
    previous_price: f64,
}

let old_positions: HashMap<usize, PositionSnapshot> = portfolio
    .positions
    .iter()
    .map(|(&k, v)| (k, PositionSnapshot::from(v)))
    .collect();
```

**Benefit**: Single allocation, cleaner code, easier to extend

### 2.2 Consolidate Position Update Logic

**Current**:
- `update_position_values()` in wide.rs:504-528
- `update_positions()` in long.rs:1115-1140
- `Position::update_with_return()` in position.rs:79-99

**Refactor**: Remove helper functions, use Position method directly:
```rust
// In PortfolioState
pub fn update_positions_with_prices(&mut self, prices: &[f64]) {
    for (stock_id, pos) in self.positions.iter_mut() {
        if *stock_id < prices.len() {
            pos.update_with_return(prices[*stock_id]);
        }
    }
}
```

### 2.3 Improve Position Documentation

**Add** clear invariants:
```rust
/// Position in a backtest portfolio.
///
/// # Field Semantics (Finlab Mode)
/// - `value`: Cost basis at entry (constant after entry, excludes fee)
/// - `entry_price`: Price for market value calculation (may reset on rebalance)
/// - `stop_entry_price`: Price for stop detection (NEVER reset - tracks original entry)
/// - `last_market_value`: Current position value = value * close / entry_price
/// - `cr`: Cumulative return ratio (updated via cr *= daily_return)
/// - `maxcr`: Maximum cr seen (for trailing stop)
/// - `max_price`: Maximum price seen (for trailing stop)
/// - `previous_price`: Last valid price (for daily return calculation)
///
/// # Update Order (Critical for Finlab compatibility)
/// 1. r = close / previous_price
/// 2. cr *= r
/// 3. last_market_value *= r
/// 4. maxcr = max(maxcr, cr)
/// 5. max_price = max(max_price, close)
/// 6. previous_price = close
```

---

## Phase 3: Major Refactoring (Higher Risk)

**Status**: Not Started

### 3.1 Split simulate_backtest() by Mode

**Current**: Single 400-line function with `if config.finlab_mode { ... } else { ... }`

**Split** into:
```rust
pub fn simulate_backtest<T: TradeTracker>(...) -> Vec<f64> {
    if config.finlab_mode {
        simulate_backtest_finlab(...)
    } else {
        simulate_backtest_standard(...)
    }
}

fn simulate_backtest_finlab<T: TradeTracker>(...) -> Vec<f64> {
    // ~200 lines - Finlab-specific logic
}

fn simulate_backtest_standard<T: TradeTracker>(...) -> Vec<f64> {
    // ~200 lines - Standard mode logic
}
```

**Benefit**: Easier to test each mode independently, clearer code paths

### 3.2 Optional: Strategy Pattern

**Alternative design** for future extensibility:
```rust
trait BacktestMode {
    fn update_positions(&mut self, portfolio: &mut PortfolioState, prices: &[f64]);
    fn detect_stops(&self, portfolio: &PortfolioState, prices: &[f64]) -> Vec<usize>;
    fn execute_rebalance(&mut self, portfolio: &mut PortfolioState, ...);
    fn calculate_balance(&self, portfolio: &PortfolioState, prices: &[f64]) -> f64;
}

struct FinlabMode;
struct StandardMode;
```

**Benefit**: Easy to add new modes, better testability
**Risk**: More complex, may be over-engineering for current needs

---

## Phase 4: Cleanup (Low Priority)

**Status**: Not Started

### 4.1 Remove or Integrate rebalance.rs

**Current**: Module exports `find_rebalance_indices()` but is never imported in lib.rs

**Options**:
1. Delete the file entirely (if logic is duplicated elsewhere)
2. Import and use in lib.rs (if needed for Python API)

### 4.2 Simplify portfolio.balance_finlab()

**Current** (redundant code at lines 57-61):
```rust
if close_price > 0.0 && !close_price.is_nan() {
    p.last_market_value  // Returns this
} else {
    p.last_market_value  // Or this - SAME VALUE!
}
```

**Fix**: Just return `p.last_market_value` directly

### 4.3 Generic TradeTracker Implementation

**Current**: IndexTracker and SymbolTracker have ~50 lines of duplicate code

**Refactor** to:
```rust
struct TradeTrackerImpl<K: Clone + Eq + Hash, D: Copy> {
    open_trades: HashMap<K, OpenTradeInfo<D>>,
    completed_trades: Vec<TradeRecord>,
}

type IndexTracker = TradeTrackerImpl<usize, usize>;
type SymbolTracker = TradeTrackerImpl<String, i32>;
```

---

## Summary

| Phase | Content | Risk | Lines Change | Status |
|-------|---------|------|--------------|--------|
| 1.1 | Remove dead code from wide.rs | Low | -276 | Done |
| 1.2 | Extract FLOAT_EPSILON | Low | ~-30 | Not Started |
| 1.3 | Add is_valid_price() | Low | ~-20 | Not Started |
| 2.1 | PositionSnapshot struct | Medium | ~-30 | Not Started |
| 2.2 | Consolidate update logic | Medium | ~-50 | Not Started |
| 2.3 | Position documentation | Low | +30 | Not Started |
| 3.1 | Split simulate_backtest | Medium | +50 | Not Started |
| 4.1 | Remove rebalance.rs | Low | ~-100 | Not Started |
| 4.2 | Simplify balance_finlab | Low | ~-5 | Not Started |
| **Total** | | | **~-430** | |

---

## Test Strategy

After each change:
1. `cargo test -p btcore` - Rust unit tests (91 tests)
2. `cargo test -p btcore --test integration_test` - Integration tests (19 tests)
3. `just test` - Python tests
4. `just test-slow` - Finlab comparison tests

**Critical**: Finlab comparison tests must pass to ensure backward compatibility.
