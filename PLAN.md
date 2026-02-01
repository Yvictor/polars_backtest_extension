# polars_backtest Long Format 支援 Minute/Tick Level 實作計畫

## 目標

將 polars_backtest 的 Long Format API 從只支援日級 (daily) 擴展到支援任意時間粒度 (hourly/minute/tick)。

---

## 設計哲學

### Date 和 Datetime 本質上是一樣的

- **Date**: 精度到天的時間戳
- **Datetime**: 精度到微秒/毫秒的時間戳

兩者都可以用 `i64` 統一表示（毫秒時間戳），只是：
- Date 輸入 → 轉換為 `days * 86400000` 毫秒
- Datetime 輸入 → 直接使用毫秒時間戳

### TradeTracker trait 已經支援泛型 Date

```rust
pub trait TradeTracker {
    type Date: Copy;  // ← 已經是泛型！
    ...
}
```

這意味著我們可以：
- 現有 Long format: `Date = i32` (天數)
- 新增 Datetime format: `Date = i64` (毫秒)

---

## 修正後的設計方案

### 核心策略：統一用 i64 毫秒時間戳

```
輸入: Date 或 Datetime
         ↓
    FFI 層轉換（統一為 i64 毫秒）
         ↓
    ┌─────────────────────────────────┐
    │  timestamps: Vec<i64> (毫秒)     │  ← 統一表示
    │  input_mode: DateMode           │  ← 記錄原始類型
    └─────────────────────────────────┘
         ↓
    backtest_impl (使用 i64)
         ↓
    輸出: (Vec<i64>, Vec<f64>)
         ↓
    根據 input_mode 轉換回 Date 或 Datetime
```

### 關鍵設計點

1. **統一內部表示**：所有時間都用 `i64` 毫秒
2. **保留輸入類型資訊**：記錄是 Date 還是 Datetime
3. **輸出類型對應輸入**：Date 輸入 → Date 輸出，Datetime 輸入 → Datetime 輸出
4. **TradeRecord 適配**：
   - 使用 `i64` 存時間戳
   - Python 層根據 mode 轉換為 Date 或 Datetime

### 改動範圍

| 項目 | 現狀 | 目標 | 風險 |
|------|------|------|------|
| TradeRecord.entry_date | `Option<i32>` | `Option<i64>` | 中 |
| backtest_impl get_date | `Fn(usize) -> i32` | `Fn(usize) -> i64` | 中 |
| BacktestResult.dates | `Vec<i32>` | `Vec<i64>` | 中 |
| 邊界檢測函數 | `_i32` 版本 | 統一 `_ms` 版本 | 低 |
| FFI 層 | 只支援 Date | 支援 Date + Datetime | 低 |

---

## 關鍵修改檔案

| 檔案 | 風險 | 修改內容 |
|------|------|---------|
| `btcore/src/simulation/long.rs` | 中 | ResampleFreq、邊界檢測、backtest_impl 增加 timestamp |
| `polars_backtest/src/ffi_convert.rs` | 低 | 新增 `polars_i64_to_arrow` |
| `polars_backtest/src/lib.rs` | 中 | 支援 Datetime 類型，傳遞 timestamp |
| `polars_backtest/python/polars_backtest/namespace.py` | 低 | 新增 resample 值驗證 |

---

## 實作完成狀態

### Stage 1: Rust 核心類型修改 ✅ 已完成

- [x] TradeRecord: 所有日期字段 i32 → i64
- [x] BacktestResult.dates: Vec<i32> → Vec<i64>
- [x] StockOperations: weight_date, next_weight_date → i64
- [x] RecordBuilder for TradeRecord: `type Date = i64`
- [x] NoopSymbolTracker 類型別名更新

### Stage 2: Rust 新功能 ✅ 已完成

- [x] ResampleFreq 新增 `Interval(u32)` 變體
- [x] 實作 `parse_interval` 解析 "H", "5T", "30S"
- [x] 新增 `crossed_interval_boundary(prev_ms, curr_ms, interval_secs)`
- [x] 保留現有 `_i32` 函數供日曆邊界使用

### Stage 3: FFI 層 ✅ 已完成

- [x] 新增 `polars_i64_to_arrow` in ffi_convert.rs
- [x] 新增 `DateMode` enum
- [x] 修改日期處理邏輯（統一轉為 i64 毫秒）
- [x] 修改輸出轉換（根據 DateMode 轉回原始類型）
- [x] 修改 `trades_to_dataframe` 支援 DateMode
- [x] 支援 String 類型自動轉換為 Date

### Stage 4: Python API ✅ 已完成

- [x] 擴展 `_validate_resample` 支援 "H", "nT", "nS"
- [x] 新增 `_INTERVAL_PATTERN` 正則驗證

### Stage 5: 測試 ✅ 已完成

- [x] 101 Rust 測試通過
- [x] 120 Python 測試通過
- [x] Date 輸入 → Date 輸出
- [x] Datetime 輸入 → Datetime 輸出
- [x] 日級 creturn 計算結果不變

---

## 新增的 resample 格式

| 格式 | 說明 | 範例 |
|------|------|------|
| `H` | 每小時 | `resample="H"` |
| `nH` | 每 n 小時 | `resample="4H"` |
| `nT` | 每 n 分鐘 | `resample="5T"` |
| `nmin` | 每 n 分鐘 | `resample="5min"` |
| `nS` | 每 n 秒 | `resample="30S"` |

---

## 使用範例

```python
import polars as pl
from datetime import datetime, timedelta

# 生成分鐘數據
data = [
    {"datetime": datetime(2024, 1, 15, 9, 0) + timedelta(minutes=i),
     "symbol": "AAPL", "close": 100.0 + i * 0.01, "weight": 1.0}
    for i in range(60)
]
df = pl.DataFrame(data).with_columns(
    pl.col("datetime").cast(pl.Datetime("us"))
)

# 測試各種 resample
for resample in ["D", "H", "5T", "1T"]:
    result = df.bt.backtest(
        date="datetime",
        symbol="symbol",
        trade_at_price="close",
        position="weight",
        resample=resample,
    )
    print(f"{resample}: {result.height} rows")
```

---

## 設計優點

1. **一致性**：Date 和 Datetime 統一用 i64 毫秒表示，邏輯一致
2. **透明性**：輸入什麼類型，輸出就是什麼類型
3. **擴展性**：支援任意時間間隔，從秒級到年級
4. **效能**：i64 在現代 CPU 上與 i32 效率相當
5. **複用**：日曆邊界檢測（月末/週末等）完全複用現有 `_i32` 函數
