# Polars Native Backtest Refactoring Plan

## Goal
將回測 API 從 wide format + 函數呼叫 改為 long format + DataFrame namespace extension。

---

## Core Philosophy

### 終極目標：全 Long Format 架構
- **Long format** 是整個系統的終極目標，從 API 到 Rust core 都應該是 long format
- 這是更自然、更 Polars-native 的資料格式
- 終極架構：`Long format API → Long format Rust core`

### 現狀：Wide Format 作為過渡實作
- 目前的 Rust core 使用 wide format，這是為了先對齊 Finlab 答案而做的暫時方案
- Python 層做 long → wide 轉換，呼叫 wide format Rust backend
- 這個過渡方案讓我們能先有一個完全正確的版本

### test_vs_finlab.py 是 Gold Standard
- `test_vs_finlab.py` 使用真實資料，與 Finlab 完整驗證過
- **這是唯一的正確性來源**
- 未來無論 Rust core 怎麼重構（改成 long format），都必須通過這個測試
- `backtest_wide()` API 必須保留，供 `test_vs_finlab.py` 使用

### 演進路徑
```
現狀：
  使用者 API (Long Format)
      ↓
  Python 轉換層 (long → wide pivot)
      ↓
  Wide Format Rust Core (已驗證 ✓)

終極目標：
  使用者 API (Long Format)
      ↓
  Long Format Rust Core (新實作，對齊 test_vs_finlab.py)
```

---

## Target API

```python
import polars as pl
import polars_backtest as pl_bt

# Long format input
df = pl.DataFrame({
    "date": [...],
    "symbol": [...],
    "open": [...],
    "high": [...],
    "low": [...],
    "close": [...],
    "weight": [...],  # 策略權重 (0-1 做多, 負數做空)
})

# API 1: Standalone function
result = pl_bt.backtest(df, price="close", weight="weight", resample="M")
report = pl_bt.backtest_with_report(df, price="close", weight="weight", resample="M")

# API 2: DataFrame namespace (auto-registered when import polars_backtest)
result = df.bt.backtest(price="close", weight="weight", resample="M")
report = df.bt.backtest_with_report(price="close", weight="weight", resample="M")

# 返回: DataFrame with creturn per date
# date | creturn

# Report 物件
# - report.creturn: DataFrame
# - report.trades: DataFrame
# - report.position: DataFrame
```

---

## Stage 1: Python Namespace Skeleton
**Goal**: 建立 `df.bt` namespace 基本結構
**Status**: Complete

### Tasks
1. 在 `polars_backtest/__init__.py` 加入 namespace registration
2. 建立 `BacktestNamespace` class
3. 實作空的 `backtest()` 和 `backtest_with_trades()` 方法
4. 驗證 `df.bt.backtest()` 可被呼叫

### Success Criteria
- `df.bt.backtest()` 可執行（即使只返回空結果）
- Type hints 正確

---

## Stage 2: Long to Wide Conversion (Python Layer)
**Goal**: 在 Python 層實現 long → wide 轉換，保持 Rust core 不變
**Status**: Complete

### Tasks
1. 實作 `_long_to_wide_prices()` - 將 long format 的 OHLC 轉為 wide format
2. 實作 `_long_to_wide_weights()` - 將 long format 的 weight 轉為 wide format
3. 處理 resample 邏輯（reuse 現有的 `_resample_position`）
4. 呼叫現有的 Rust backend

### Success Criteria
- `df.bt.backtest()` 產生正確的 creturn
- 與 `backtest_with_report()` 結果一致

---

## Stage 2.5: Standalone Function API
**Goal**: 新增 `pl_bt.backtest(df, ...)` 函數式介面
**Status**: Complete

### Tasks
1. 重新命名舊的 wide format `backtest()` 為 `backtest_wide()`
2. 重新命名舊的 wide format `backtest_with_report()` 為 `backtest_with_report_wide()`
3. Export namespace.py 的 `backtest()` 和 `backtest_with_report()` 作為主要 API
4. 更新 `__all__` exports

### Success Criteria
- `pl_bt.backtest(df, price="close", weight="weight")` 可用
- `pl_bt.backtest_with_report(df, ...)` 可用
- 舊 API 透過 `backtest_wide()` 仍可使用
- 測試通過

---

## Stage 3: Finlab Validation Layer
**Goal**: 確保 wide format API 持續作為 gold standard 驗證
**Status**: Complete

### Tasks
1. ✅ `test_vs_finlab.py` 使用 `backtest_wide()` 驗證（已完成）
2. ✅ 保留 wide format API 供驗證使用（不 deprecate）
3. ✅ 新 long format API 內部透過 wide format 確保正確性

### Success Criteria
- `test_vs_finlab.py` 全部通過（37 tests）
- `test_namespace.py` 全部通過（17 tests）

---

## Stage 4: Native Long Format in Rust (終極目標)
**Goal**: Rust core 直接處理 long format，消除 Python 層轉換
**Status**: Not Started

### Why This Matters
- 消除 long → wide → long 的轉換開銷
- 更自然的 Polars-native 資料流
- 這是整個重構的終極目標

---

### API 設計

**API 命名規則:**
- `backtest()` - 預設 API，接受 long format
- `backtest_wide()` - wide format API，用於 Finlab 驗證

**使用者 API 不變:**
```python
result = df.bt.backtest(price="close", weight="weight")
result = pl_bt.backtest(df, price="close", weight="weight")
```

**內部實作變更:**
```
現狀:
  backtest() → Python pivot → backtest_wide() → Rust

Stage 4:
  backtest() → Rust 直接處理
```

---

### 測試策略

**核心原則: 透過現有測試驗證正確性**

```
test_vs_finlab.py: backtest_wide() == Finlab ✓ (Gold Standard)
test_namespace.py: backtest() 結果正確 ✓

Stage 4 驗證:
  1. backtest_wide() 不變 → test_vs_finlab.py 繼續通過
  2. backtest() 內部改用 Rust → test_namespace.py 確認結果一致
```

**參數組合覆蓋:**
- `test_namespace.py` 已測試的參數組合要繼續通過
- 確保覆蓋:
  - resample: D, W, M, None
  - fees: 各種費率
  - stop_loss, take_profit, trail_stop
  - touched_exit (需 OHLC)
  - short positions (負權重)

**效能測試:**
```python
def test_performance():
    """Rust 應比 Python pivot 快"""
    large_df = generate_large_dataset(n_dates=5000, n_symbols=2000)
    t_before = timeit(lambda: backtest_via_pivot(large_df))
    t_after = timeit(lambda: backtest_rust_native(large_df))
    assert t_after < t_before
```

---

### 架構設計

#### Zero-Copy 設計原則

**現有問題 (有 copy):**
```rust
// df_to_f64_2d() 逐 row 存取，每個值都 copy
for row_idx in 0..n_rows {
    for col in df.get_columns() {
        let val = col.get(row_idx);  // O(1) but copies
        row.push(f64_val);
    }
}
// 結果: Vec<Vec<f64>> - 全部資料都被 copy 了
```

**Zero-Copy 方式:**
```rust
// 直接存取 Arrow buffer 底層 slice
let prices = df.column("close")?.f64()?;
let prices_slice: &[f64] = prices.cont_slice()?;  // zero-copy!

let symbols = df.column("symbol")?.str()?;
// 用 Polars 的 group_by 在 Rust 端處理
```

**關鍵條件:**
- `cont_slice()` 需要單一 chunk 且無 null
- 必要時先 `rechunk()` 整合
- 回測邏輯改為直接操作 slice/ChunkedArray

#### 新增 Rust PyO3 函數

```rust
// polars_backtest/src/lib.rs
use pyo3_polars::PyExpr;

#[pyfunction]
fn backtest(
    df: PyDataFrame,
    date: PyExpr,              // IntoExpr
    symbol: PyExpr,            // IntoExpr
    price: PyExpr,             // IntoExpr
    weight: PyExpr,            // IntoExpr
    rebalance_rule: Option<&str>,
    config: Option<PyBacktestConfig>,
) -> PyResult<PyBacktestResult> {
    let df = df.0;

    // Zero-copy: 直接取得 column slices
    let dates = df.column("date")?.date()?;
    let symbols = df.column("symbol")?.str()?;
    let prices = df.column("close")?.f64()?.rechunk();
    let weights = df.column("weight")?.f64()?.rechunk();

    // 用 slice 直接處理，不用 Vec<Vec<f64>>
    let prices_slice = prices.cont_slice()?;
    let weights_slice = weights.cont_slice()?;

    // 呼叫新的 long format 回測邏輯
    run_backtest_columnar(dates, symbols, prices_slice, weights_slice, &config)
}
```

#### btcore 新增模組

```rust
// btcore/src/columnar.rs

/// Columnar 回測 - 直接操作 slice，不轉換格式
pub fn run_backtest_columnar(
    dates: &DateChunked,
    symbols: &StringChunked,
    prices: &[f64],          // zero-copy slice
    weights: &[f64],         // zero-copy slice
    config: &BacktestConfig,
) -> BacktestResult {
    // 方案 A: 用 Polars group_by 在 Rust 處理
    // 方案 B: 建立 symbol → indices 的 mapping，避免 pivot
}
```

---

### 實作階段

#### Stage 4.1: Rust 骨架 (Pivot in Rust)
**Goal**: 建立 Rust 接收 long format 的入口，內部先 pivot

**Tasks:**
1. 在 `polars_backtest/src/lib.rs` 新增 `backtest()` (PyO3)
2. 使用 PyExpr 接收欄位表達式
3. 內部用 Polars Rust API 做 pivot，複用現有 wide 邏輯
4. 跑 `test_namespace.py` 確認正確

**技術細節:**
```rust
// 階段 4.1: 在 Rust 做 pivot
fn backtest(df: PyDataFrame, ...) -> PyResult<PyBacktestResult> {
    let df = df.0;

    // Polars Rust pivot (仍有 copy，但在 Rust 層)
    let wide_prices = df.pivot(...)?;
    let wide_weights = df.pivot(...)?;

    // 複用現有邏輯
    run_backtest(&wide_prices_2d, &wide_weights_2d, ...)
}
```

#### Stage 4.2: Python 切換
**Goal**: namespace.py 改用 Rust 函數

**Tasks:**
1. `backtest()` 改呼叫 Rust `backtest()`
2. 移除 Python pivot 邏輯 (`_long_to_wide()`)
3. 跑所有測試確認

#### Stage 4.3: Zero-Copy Columnar 處理
**Goal**: 重寫 Rust 內部邏輯，zero-copy 直接處理 long format

**Tasks:**
1. 新增 `btcore/src/columnar.rs`
2. 實作 `run_backtest_columnar()` 直接操作 slices
3. 用 `cont_slice()` 達成 zero-copy
4. 不再 pivot，用 symbol indices mapping 處理
5. 保持結果與 wide format 完全一致

**技術細節:**
```rust
// 階段 4.3: Zero-copy columnar 處理
fn backtest(df: PyDataFrame, ...) -> PyResult<PyBacktestResult> {
    let df = df.0.rechunk();  // 確保單一 chunk

    // Zero-copy slice 存取
    let prices = df.column("close")?.f64()?;
    let prices_slice = prices.cont_slice()?;  // &[f64]

    // 建立 symbol → row indices mapping
    let symbol_groups = build_symbol_index_map(&df)?;

    // 直接在 columnar 資料上回測
    run_backtest_columnar(symbol_groups, prices_slice, weights_slice, config)
}
```

---

### 檔案變更

**Rust:**
- `polars_backtest/src/lib.rs` - 新增 `backtest()` PyO3 函數
- `btcore/src/columnar.rs` (新) - Zero-copy columnar 回測邏輯
- `btcore/src/lib.rs` - Export columnar 模組
- `btcore/Cargo.toml` - 可能需要 polars features (for ChunkedArray types)

**Python:**
- `polars_backtest/namespace.py` - 改用 Rust 函數

---

### Success Criteria

1. **正確性**: `test_namespace.py` 全部通過
2. **相容性**: `test_vs_finlab.py` 繼續通過
3. **效能**: Rust 處理 < Python pivot + Rust wide
4. **Zero-Copy**: Stage 4.3 達成 `cont_slice()` zero-copy 存取
5. **API**: 使用者介面不變

---

## Architecture Comparison

### 原始狀態 (Wide Format Only)
```
Python:                          Rust:
position (wide) ──┐
                  ├─► convert ──► Vec<Vec<f64>> ──► backtest ──► Vec<f64>
adj_close (wide) ─┘

驗證: test_vs_finlab.py ✓ (Gold Standard)
```

### 現狀: Stage 1-3 (Long API + Wide Core)
```
Python:                                      Rust:
df (long) ──► pivot ──► position (wide) ─┐
                        adj_close (wide) ─┼─► backtest ──► Vec<f64>
                                          │
                        (reuse validated) ┘

驗證: test_vs_finlab.py ✓ + test_namespace.py ✓
```

### Stage 4.1-4.2: Rust Pivot
```
Python:                     Rust:
df (long) ──► PyDataFrame ──► Polars pivot ──► wide format ──► existing logic ──► result
                             (copy in Rust)

驗證: test_namespace.py ✓
```

### Stage 4.3: Zero-Copy Columnar (終極目標)
```
Python:                     Rust:
df (long) ──► PyDataFrame ──► cont_slice() ──► columnar backtest ──► result
                             (zero-copy!)

驗證: test_namespace.py ✓ + 效能提升
```

---

## File Changes

### Stage 1-2
- `polars_backtest/__init__.py` - Add namespace registration
- `polars_backtest/namespace.py` (new) - BacktestNamespace class

### Stage 3
- `tests/test_vs_finlab.py` - Add namespace API tests
- `tests/test_namespace.py` (new) - Namespace-specific tests

---

## Notes

1. **Backward Compatibility**: 保留 `backtest_with_report()` 函數，標記為 deprecated
2. **Column Naming**: 使用字串參數 (`price="close"`) 而非 Expression，更簡單
3. **Resample**: 沿用現有邏輯，只是包裝在 namespace 方法中
