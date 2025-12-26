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

### Tasks
1. 設計 long format 的 Rust 資料結構
2. 修改 `btcore` 支援 (date, symbol, price, weight) 格式
3. 使用 polars-arrow 直接操作 columnar data
4. 新實作必須通過 `test_vs_finlab.py` 驗證

### Success Criteria
- 所有 `test_vs_finlab.py` 測試通過（對齊現有正確結果）
- 效能優於現有 Python 轉換方案

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

### 終極目標: Stage 4 (Full Long Format)
```
Python:                     Rust:
df (long) ──► Arrow IPC ──► process long format directly ──► result

驗證: 必須通過 test_vs_finlab.py (對齊現有正確結果)
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
