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
**Status**: Complete ✓ (Stage 4.1-4.3 All Done)

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

#### 是否需要 Pivot？—— 不需要！

**關鍵發現：用 `partition_by` 取代 pivot**

Polars Rust API 提供 `partition_by`：
```rust
// 按 date 分割 → 每個 partition 包含該日期所有 symbols
let partitions: Vec<DataFrame> = df.partition_by(["date"], true)?;

for date_df in partitions {
    // date_df 包含這一天所有 stocks
    let symbols = date_df.column("symbol")?.str()?;
    let prices = date_df.column("close")?.f64()?.cont_slice()?;  // zero-copy!
    let weights = date_df.column("weight")?.f64()?.cont_slice()?;

    // 處理這一天的所有 stocks
    process_day(&mut portfolio, symbols, prices, weights, config)?;
}
```

**優勢：**
- 無需 pivot（long format 直接處理）
- zero-copy（`cont_slice()` 直接存取 Arrow buffer）
- 保持 time-major 迭代（與現有演算法一致）

#### 架構分層

```
polars_backtest crate (有 polars 依賴):
  ┌─────────────────────────────────────────┐
  │  PyO3 入口: backtest(df, ...)           │
  │  ↓                                      │
  │  df.partition_by(["date"])              │
  │  ↓                                      │
  │  for date_df in partitions:             │
  │      prices = date_df["close"].slice()  │  ← zero-copy
  │      weights = date_df["weight"].slice()│  ← zero-copy
  │      btcore::process_day(...)           │
  └─────────────────────────────────────────┘

btcore crate (純 Rust，無 polars 依賴):
  ┌─────────────────────────────────────────┐
  │  fn process_day(                        │
  │      portfolio: &mut Portfolio,         │
  │      symbols: &[&str],                  │
  │      prices: &[f64],                    │  ← 接收 slices
  │      weights: &[f64],                   │
  │      config: &Config,                   │
  │  )                                      │
  └─────────────────────────────────────────┘
```

#### Zero-Copy 實現

**現有問題 (有 copy):**
```rust
// df_to_f64_2d() 逐 row 存取，每個值都 copy
for row_idx in 0..n_rows {
    for col in df.get_columns() {
        row.push(col.get(row_idx));  // copy
    }
}
```

**新方式 (zero-copy):**
```rust
// partition_by + cont_slice = zero-copy
let partitions = df.partition_by(["date"], true)?;
for date_df in &partitions {
    let prices = date_df.column("close")?.f64()?.rechunk();
    let prices_slice = prices.cont_slice()?;  // &[f64], zero-copy!
    // ...
}
```

**關鍵條件:**
- `cont_slice()` 需要單一 chunk 且無 null
- 必要時先 `rechunk()` 整合
- btcore 保持純 Rust，只接收 slices

#### 新增 Rust PyO3 函數

```rust
// polars_backtest/src/lib.rs

#[pyfunction]
fn backtest(
    df: PyDataFrame,
    date_col: &str,            // "date"
    symbol_col: &str,          // "symbol"
    price_col: &str,           // "close"
    weight_col: &str,          // "weight"
    rebalance_dates: Option<Vec<String>>,  // None = 每日 rebalance
    config: Option<PyBacktestConfig>,
) -> PyResult<PyBacktestResult> {
    let df = df.0;
    let cfg = config.map(|c| c.inner).unwrap_or_default();

    // partition_by date → 每個 partition 是一天的資料
    let partitions = df
        .sort([date_col], SortMultipleOptions::default())?
        .partition_by([date_col], true)?;

    // 建立 symbol → index mapping (for Portfolio tracking)
    let all_symbols = df.column(symbol_col)?.str()?.unique()?;
    let symbol_to_idx: HashMap<&str, usize> = all_symbols
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(i, s)| (s, i))
        .collect();

    let mut portfolio = PortfolioState::new();
    let mut creturn = Vec::with_capacity(partitions.len());

    for date_df in &partitions {
        let symbols = date_df.column(symbol_col)?.str()?;
        let prices = date_df.column(price_col)?.f64()?.rechunk();
        let weights = date_df.column(weight_col)?.f64()?.rechunk();

        let prices_slice = prices.cont_slice()?;
        let weights_slice = weights.cont_slice()?;

        // 呼叫 btcore 處理這一天
        btcore::process_day(
            &mut portfolio,
            &symbol_to_idx,
            symbols,
            prices_slice,
            weights_slice,
            &cfg,
        );

        creturn.push(portfolio.balance());
    }

    Ok(PyBacktestResult { creturn, trades: vec![] })
}
```

#### btcore 新增 API

```rust
// btcore/src/day_processing.rs (新檔案)

/// 處理單日的回測邏輯
/// 接收 slices，不依賴 Polars 類型
pub fn process_day(
    portfolio: &mut PortfolioState,
    symbol_to_idx: &HashMap<&str, usize>,
    symbols: impl Iterator<Item = Option<&str>>,  // 當日所有 symbols
    prices: &[f64],                               // 當日價格 (與 symbols 對應)
    weights: &[f64],                              // 當日權重 (與 symbols 對應)
    config: &BacktestConfig,
) {
    // 1. 更新持倉價值
    // 2. 檢測 stops
    // 3. 執行 rebalance (如果需要)
    // 4. 更新 portfolio state
}
```

**優勢:**
- btcore 不依賴 polars（保持純 Rust）
- 接收 slices，達成 zero-copy
- 單日處理邏輯可重用於不同資料來源

---

### 實作階段

#### Stage 4.1: Rust Pivot + 現有邏輯
**Goal**: Rust 層實作 pivot，消除 Python pivot 開銷
**Status**: Complete ✓

**實作方式:**
- 使用 `polars-ops::pivot::pivot()` 在 Rust 層做 long → wide 轉換
- 新增 `backtest()` PyO3 函數，接受 long format DataFrame
- 呼叫現有 `run_backtest()` 確保正確性

**完成內容:**
1. ✅ 在 `polars_backtest/src/lib.rs` 新增 `backtest()` (PyO3)
2. ✅ 新增 `long_to_wide()` helper 使用 Polars pivot
3. ✅ 新增 `compute_rebalance_indices()` 計算 resample indices
4. ✅ `test_namespace.py` 17 tests 全部通過
5. ✅ Rust backtest 結果與 Python 實作完全一致

**技術細節:**
```rust
fn backtest(df: PyDataFrame, ...) -> PyResult<PyBacktestResult> {
    // 使用 polars-ops pivot 轉換
    let wide_prices = long_to_wide(&df, date_col, symbol_col, price_col)?;
    let wide_weights = long_to_wide(&df, date_col, symbol_col, weight_col)?;

    // 複用現有 wide format 邏輯
    run_backtest(&wide_prices, &wide_weights, ...)
}
```

#### Stage 4.2: Python 切換
**Goal**: namespace.py 改用 Rust 函數
**Status**: Complete ✓

**完成內容:**
1. ✅ `BacktestNamespace.backtest()` 在基本情況下使用 Rust `backtest()`
2. ✅ 支援的 resample: None, "D", "W", "M"
3. ✅ 不支援的情況自動 fallback 到 Python 實作:
   - 複雜 resample patterns (W-FRI, MS, Q, etc.)
   - resample_offset
   - Boolean signals (需要 equal weight 轉換)
4. ✅ `test_namespace.py` 17 tests 全部通過

**技術細節:**
```python
# Check if we can use Rust path
use_rust = (
    resample in (None, "D", "W", "M")
    and resample_offset is None
    and not is_bool_signal
)

if use_rust:
    result = _rust_backtest(self._df, ...)
else:
    # Fallback to Python path
    ...
```

#### Stage 4.3: Zero-Copy 處理（多方案 Benchmark）
**Goal**: 完全移除 wide format 轉換，達成 zero-copy
**Status**: Complete ✓

**完成內容:**
1. ✅ 優化 `df_to_f64_2d()` 使用 columnar access
2. ✅ 實作 `backtest_partitioned()` 使用 partition_by 取代 pivot
3. ✅ 建立 `benchmarks/bench_backtest.py` 效能測試
4. ✅ 所有測試通過 (17 + 37 = 54 tests)
5. ✅ 切換 namespace.py 使用 partition_by 版本

**Benchmark 結果:**
| Size | Rust pivot | Rust partition | Python pivot |
|------|------------|----------------|--------------|
| Small (5K) | 7.97ms | **5.84ms** | 16.59ms |
| Medium (100K) | 49.24ms | **39.76ms** | 53.86ms |
| Large (500K) | 210.21ms | **137.65ms** | 145.97ms |

**結論: partition_by 比 pivot 快 1.5x (大資料)**

---

**方案 A: partition_by**
```rust
let partitions = df.partition_by(["date"], true)?;
for date_df in &partitions {
    let prices = date_df.column("close")?.f64()?.cont_slice()?;
    btcore::process_day(&mut portfolio, prices, weights);
}
```
- 優點：API 簡潔，Polars 原生支援
- 缺點：Vec\<DataFrame\> 分配開銷

---

**方案 B: Sort + Boundary Detection**
```rust
let df = df.sort(["date"], Default::default())?;
let dates = df.column("date")?.date()?.cont_slice()?;
let prices = df.column("close")?.f64()?.cont_slice()?;

// 單次遍歷找邊界
let boundaries = find_date_boundaries(dates);  // Vec<(start, end)>

for (start, end) in boundaries {
    let day_prices = &prices[start..end];  // 純 slice，零分配
    btcore::process_day(&mut portfolio, day_prices, ...);
}
```
- 優點：最小記憶體分配，純 slice 操作
- 缺點：需要自己維護邊界邏輯

---

**方案 C: group_by + apply**
```rust
df.group_by(["date"])?.apply(|group_df| {
    // 在 closure 中處理每個 group
})?;
```
- 優點：可並行處理
- 缺點：回測需要 sequential state，可能不適用

---

**方案 D: Polars Lazy + Streaming**
```rust
let lf = df.lazy()
    .group_by(["date"])
    .agg([...])
    .collect()?;
```
- 優點：Polars 優化器自動優化
- 缺點：回測的 stateful 邏輯難以表達

---

**Benchmark 計畫:**
```rust
// benches/backtest_benchmark.rs
fn benchmark_partition_by(c: &mut Criterion) { ... }
fn benchmark_sort_boundary(c: &mut Criterion) { ... }
fn benchmark_group_apply(c: &mut Criterion) { ... }
```

測試資料：
- Small: 1000 dates × 100 symbols
- Medium: 5000 dates × 500 symbols
- Large: 10000 dates × 2000 symbols

---

### 檔案變更

**Rust:**
- `polars_backtest/src/lib.rs` - 新增 `backtest()` PyO3 函數
- `polars_backtest/src/strategies.rs` (新) - 多種處理策略實作
- `polars_backtest/benches/backtest_benchmark.rs` (新) - Criterion benchmark
- `btcore/src/day_processing.rs` (新) - 單日處理邏輯，接收 slices
- `btcore/src/lib.rs` - Export day_processing 模組
- btcore 不新增 polars 依賴（保持純 Rust）

**Python:**
- `polars_backtest/namespace.py` - 改用 Rust `backtest()` 函數

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
