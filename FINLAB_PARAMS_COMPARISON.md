# Finlab backtest.sim() Parameters Comparison 參數對比分析

## Complete Parameter List 完整參數列表對比

Based on `finlab.backtest.sim()` function signature analysis:

### ✅ Supported Parameters 已支援的參數 (17/23)

| Parameter | Finlab Default | Our Implementation | Notes 備註 |
|------|--------------|-----------|------|
| `position` | (required) | ✓ | Required, position signals/weights 必需參數，持倉信號/權重 |
| `resample` | None | ✓ | Rebalance frequency (D/W/M/Q/Y) 重新平衡頻率 |
| `trade_at_price` | 'close' | ✓ | Trading execution price type 交易執行價格類型 |
| `position_limit` | 1.0 | ✓ | Maximum weight per stock 單個股票最大權重 |
| `fee_ratio` | 0.001425 | ✓ | Transaction fee ratio 交易手續費率 |
| `tax_ratio` | 0.003 | ✓ | Transaction tax ratio 交易稅率 |
| `stop_loss` | None | ✓ | Stop loss threshold 停損閾值<br/>**Verified**: Matches finlab with max diff 2.22e-16. 已驗證與 finlab 一致 |
| `take_profit` | None | ✓ | Take profit threshold 停利閾值<br/>**Verified**: Matches finlab with max diff 2.22e-16. 已驗證與 finlab 一致 |
| `trail_stop` | None | ✓ | Trailing stop 移動停損<br/>**Verified**: Matches finlab with max diff 2.22e-16. Formula: `(max_price - current_price) / entry_price`. 已驗證與 finlab 一致，公式：`(max_price - current_price) / entry_price` |
| `retain_cost_when_rebalance` | False | ✓ | Retain entry prices on rebalance 重新平衡時保留進場價格<br/>Controls whether entry prices (for stop loss calculation) are reset on rebalance. 控制重新平衡時是否重置進場價格（影響停損計算） |
| `stop_trading_next_period` | True | ✓ | Stop trading after stop loss/profit trigger 觸發停損/停利後下期停止交易<br/>When true, stock cannot re-enter after stop loss trigger. 當為 true 時，觸發停損後該股票不能重新進場 |

**Additional Parameters We Support (Not in Finlab) 我們額外支援的參數：**
| Parameter | Our Implementation | Purpose 用途 |
|------|-----------|------|
| `open` | ✓ | Open price DataFrame 開盤價 DataFrame |
| `high` | ✓ | High price DataFrame 最高價 DataFrame |
| `low` | ✓ | Low price DataFrame 最低價 DataFrame |
| `factor` | ✓ | Price adjustment factor 價格調整因子 |
| `rebalance_indices` | ✓ | Manually specify rebalance indices 手動指定重新平衡的索引 |

---

### ❌ Missing Parameters 缺失的參數 (6/23)

#### 🔴 Core Functionality 核心功能參數（需要支援）

| Parameter | Finlab Default | Priority 優先級 | Description 說明 |
|------|--------------|-------|------|
| `resample_offset` | None | **HIGH 高** | Rebalance offset, e.g. 'W-FRI' for Friday rebalance<br/>重新平衡偏移量，例如 'W-FRI' 表示每週五重新平衡 |
| `touched_exit` | False | **HIGH 高** | Use high/low prices for stop loss/profit detection<br/>使用 high/low 價格判斷是否觸發停損/停利 |
| `mae_mfe_window` | 0 | **MEDIUM 中** | MAE/MFE (Maximum Adverse/Favorable Excursion) window<br/>MAE/MFE 窗口大小 |
| `mae_mfe_window_step` | 1 | **MEDIUM 中** | MAE/MFE window step size<br/>MAE/MFE 窗口步長 |
| `fast_mode` | False | **MEDIUM 中** | Fast mode (may skip detailed calculations)<br/>快速模式（可能跳過某些詳細計算） |

#### 🟡 Metadata/Service Parameters 元數據/服務相關參數（低優先級）

| Parameter | Finlab Default | Priority 優先級 | Description 說明 |
|------|--------------|-------|------|
| `name` | '未命名' | LOW 低 | Backtest name (metadata)<br/>回測名稱（元數據） |
| `upload` | True | LOW 低 | Upload results to Finlab server<br/>是否上傳到 Finlab 伺服器 |
| `notification_enable` | False | LOW 低 | Enable notifications<br/>是否啟用通知 |
| `line_access_token` | '' | LOW 低 | LINE notification token<br/>LINE 通知 token |

#### 🟢 Live Trading Parameters 實時交易相關參數（可選）

| Parameter | Finlab Default | Priority 優先級 | Description 說明 |
|------|--------------|-------|------|
| `live_performance_start` | None | LOW 低 | Live trading start date<br/>實時交易開始日期 |
| `market` | None | LOW 低 | Market data source object<br/>市場資料源對象 |

---

## Parameter Details 參數詳細說明

### 1. `resample_offset` (Missing 缺失 - HIGH Priority 高優先級)

**Purpose 用途**: Control specific rebalance timing point 控制重新平衡的具體時間點

**Example 範例**:
```python
# Finlab usage
finlab.backtest.sim(
    position,
    resample='W',           # Weekly rebalancing 每週重新平衡
    resample_offset='W-FRI' # Rebalance on Friday 在週五重新平衡
)
```

**Implementation Notes 實作建議**:
- Need to support pandas offset syntax 需要支援 pandas 的 offset 語法
- 'W-MON', 'W-FRI', 'M-15' etc.
- Affects `resample` parameter behavior 影響 `resample` 參數的行為

---

### 2. `touched_exit` (Missing 缺失 - HIGH Priority 高優先級)

**Purpose 用途**: Use intraday high/low prices for stop loss/profit detection 使用日內高低價判斷是否觸發停損/停利

**Current Behavior 當前行為**:
- We only use close prices for stop loss/profit checks 我們只用收盤價檢查停損/停利
- Finlab's `touched_exit=True` uses high/low prices

**Example 範例**:
```python
# When touched_exit=True:
# - Check if low touches stop_loss threshold
# - Check if high touches take_profit threshold
# More realistic for actual trading scenarios
```

**Implementation Notes 實作建議**:
- Rust core already has high_prices/low_prices support
- Need to add parameter and logic switch
- Reference: `btcore/src/simulation.rs:1577-1580`

---

### 3. `mae_mfe_window` and `mae_mfe_window_step` (Missing 缺失 - MEDIUM Priority 中優先級)

**Purpose 用途**: Calculate MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable Excursion)

**Description 說明**:
- MAE: Maximum adverse movement during trade 交易過程中的最大不利偏移（最大虧損）
- MFE: Maximum favorable movement during trade 交易過程中的最大有利偏移（最大盈利）
- Used for analyzing trade quality and setting better stop loss/profit levels 用於分析交易質量和設定更好的停損/停利

**Example 範例**:
```python
finlab.backtest.sim(
    position,
    mae_mfe_window=20,      # Calculate MAE/MFE over 20-day window
    mae_mfe_window_step=5   # Calculate every 5 days
)
```

**Implementation Notes 實作建議**:
- Need to add MAE/MFE columns to trades output 需要在 trades 輸出中增加 MAE/MFE 欄位
- Need to track price movements during each trade 需要追蹤每筆交易期間的價格波動

---

### 4. `fast_mode` (Missing 缺失 - MEDIUM Priority 中優先級)

**Purpose 用途**: Fast mode, may skip detailed calculations for better performance 快速模式，可能跳過某些詳細計算以提升速度

**Implementation Notes 實作建議**:
- Need to investigate what Finlab's fast_mode skips 需要調查 Finlab 的 fast_mode 具體跳過了什麼
- Possibly includes 可能包括：
  - Skip trades recording 跳過 trades 記錄
  - Simplified MAE/MFE calculations 簡化 MAE/MFE 計算
  - Reduced intermediate result storage 減少中間結果儲存

---

### 5. Metadata/Service Parameters 元數據/服務相關參數 (LOW Priority 低優先級)

These parameters are mainly for Finlab service integration, not affecting backtest calculations:
這些參數主要用於與 Finlab 服務整合，不影響回測計算本身：

- `name`: Backtest name 回測名稱
- `upload`: Upload results 是否上傳結果
- `notification_enable`: Notification toggle 通知開關
- `line_access_token`: LINE notification token
- `live_performance_start`: Live trading start date 實時交易起始日期
- `market`: Finlab market data source

**Recommendation 建議**: Can add these parameters as no-op, or ignore them 可以添加這些參數但設為 no-op，或者直接忽略

---

## Implementation Priority Roadmap 實作優先級建議

### Phase 1: Core Functionality 核心功能補全 (HIGH Priority 高優先級)

1. ✅ **`resample_offset`** - Affects rebalance timing 影響重新平衡時間點
   - Need to integrate with existing resample logic 需要整合到現有的 resample 邏輯中
   - Support 'W-MON', 'W-FRI', 'M-15' etc. 支援 'W-MON', 'W-FRI', 'M-15' 等

2. ✅ **`touched_exit`** - Improve stop loss/profit realism 提升停損/停利的真實性
   - Rust core already has high/low prices support 已有 high/low prices 支援
   - Only need to add parameter and logic switch 只需添加參數和邏輯切換

### Phase 2: Analytics Enhancement 分析功能增強 (MEDIUM Priority 中優先級)

3. 🔄 **`mae_mfe_window`** and **`mae_mfe_window_step`**
   - Enhance trades analysis capability 增強 trades 分析能力
   - Need to modify trades output structure 需要修改 trades 輸出結構

4. 🔄 **`fast_mode`**
   - Performance optimization option 效能優化選項
   - Need to research Finlab's implementation 需要調研 Finlab 的具體實作

### Phase 3: Compatibility Parameters 兼容性參數 (LOW Priority 低優先級)

5. ⏸️ Metadata parameters (`name`, `upload`, `notification_enable`, `line_access_token`)
   - Mainly for service integration 主要用於服務整合
   - Can add as no-op 可以添加但設為 no-op

6. ⏸️ Live trading parameters (`live_performance_start`, `market`)
   - Beyond backtest scope 超出回測範圍
   - Can skip for now 可以暫時不支援

---

## ✅ Previously Critical - Now Fixed 之前的關鍵問題 - 現已修復

### 1. `retain_cost_when_rebalance` ✅ FIXED

**Status 狀態**: **IMPLEMENTED AND VERIFIED** 已實作並驗證

**Verification 驗證**:
- Tested against finlab with real stock data (2330, 2317, 2454, 2308, 3008)
- Max difference: 2.22e-16 (floating-point precision)
- Both `True` and `False` settings verified

**What it does 功能說明**:
- When `True`: Keep original entry price for continuing positions on rebalance
- 當 `True` 時：重新平衡時保留繼續持有股票的原始進場價格
- When `False` (default): Reset entry price on each rebalance
- 當 `False` 時（預設）：每次重新平衡時重設進場價格

---

### 2. `stop_trading_next_period` ✅ FIXED

**Status 狀態**: **IMPLEMENTED AND VERIFIED** 已實作並驗證

**Verification 驗證**:
- Tested against finlab with real stock data
- Max difference: 2.22e-16 (floating-point precision)
- Both `True` and `False` settings verified

**What it does 功能說明**:
- When `True` (default): After stop loss/take profit triggers, stock cannot re-enter in the next period
- 當 `True` 時（預設）：停損/停利觸發後，該股票在下一期被禁止重新進場
- When `False`: Stock can be re-entered immediately if signal is still active
- 當 `False` 時：如果信號仍然有效，股票可以立即重新進場

---

## Implementation Status: Stop Loss / Take Profit / Trailing Stop 實作狀態：停損/停利/移動停損

### ✅ Confirmed Implementation 已確認實作

**Parameters 參數**:
- `stop_loss` (default: 1.0 = disabled 預設值：1.0 = 停用)
- `take_profit` (default: f64::INFINITY = disabled 預設值：無限大 = 停用)
- `trail_stop` (default: f64::INFINITY = disabled 預設值：無限大 = 停用)

**Implementation Details 實作詳情**:

1. ✅ **Python API Layer Python API 層**:
   - Defined in `backtest()` function (`polars_backtest/__init__.py:292-294`) 在 `backtest()` 函式中定義
   - Defined in `backtest_with_report()` function (`polars_backtest/__init__.py:710-712`) 在 `backtest_with_report()` 函式中定義
   - Passed to `BacktestConfig` (`polars_backtest/__init__.py:408-413`, `883-885`) 傳遞給 `BacktestConfig`

2. ✅ **Rust Core Implementation Rust 核心實作**:
   - Defined in `BacktestConfig` struct (`btcore/src/simulation.rs:89-93`) 在 `BacktestConfig` 結構體中定義
   - Logic implemented in multiple places 邏輯實作於多處:
     - Lines 625-640: Check stop loss/take profit/trail stop during simulation
     - Lines 685-700: Similar checks for different simulation mode
     - Lines 1517-1524: Additional checks
     - Lines 2031-2046: Additional checks

3. ✅ **Rust Integration Tests Rust 整合測試**:
   - `test_stop_loss_exit()` in `btcore/tests/integration_test.rs:129-156`
     - Tests 10% stop loss trigger 測試 10% 停損觸發
   - `test_take_profit_exit()` in `btcore/tests/integration_test.rs:160-184`
     - Tests 20% take profit trigger 測試 20% 停利觸發
   - Trailing stop test exists around line 318 移動停損測試約在第 318 行

**How It Works 運作方式**:

- **Stop Loss 停損**:
  - Triggers when `return_since_entry <= -config.stop_loss`
  - Example: `stop_loss=0.10` means exit when loss ≥ 10% 例如：`stop_loss=0.10` 表示虧損 ≥ 10% 時退出

- **Take Profit 停利**:
  - Triggers when `return_since_entry >= config.take_profit`
  - Example: `take_profit=0.20` means exit when profit ≥ 20% 例如：`take_profit=0.20` 表示獲利 ≥ 20% 時退出

- **Trailing Stop 移動停損**:
  - Tracks maximum price seen (`pos.max_price`) 追蹤看到的最高價
  - **Finlab formula**: `drawdown = (max_price - current_price) / entry_price >= config.trail_stop`
  - Note: Uses `entry_price` as denominator, NOT `max_price` 注意：分母是 `entry_price`，不是 `max_price`
  - Example: `trail_stop=0.10` means exit when drawdown from peak ≥ 10% of entry price

### ✅ Python Tests - Now Available Python 測試 - 現已完成

**Current Status 當前狀態**:
- ✅ Unit tests in `polars_backtest/tests/test_backtest.py` (TestStopLossTakeProfit class)
- ✅ Finlab comparison tests in `verify_stop_loss.py`
- ✅ All 5 stop parameters verified against finlab with max diff 2.22e-16

**Tests included 包含的測試**:
1. `test_stop_loss_triggers_exit` - Stop loss T+1 execution
2. `test_stop_loss_no_trigger` - Stop loss below threshold
3. `test_take_profit_triggers_exit` - Take profit T+1 execution
4. `test_take_profit_no_trigger` - Take profit below threshold
5. `test_trail_stop_triggers_exit` - Trailing stop T+1 execution
6. `test_trail_stop_no_trigger` - Trailing stop below threshold
7. `test_stop_trading_next_period_true` - Block re-entry after stop
8. `test_stop_trading_next_period_false` - Allow re-entry after stop
9. `test_retain_cost_when_rebalance_false` - Reset entry price on rebalance
10. `test_retain_cost_when_rebalance_true` - Keep entry price on rebalance

---

## Testing Recommendations 測試建議

For each new parameter, we need to 針對每個新參數，需要：

1. Write unit tests to verify functionality 編寫單元測試驗證功能正確性
2. Compare with Finlab actual output (if possible) 與 Finlab 實際輸出對比（如果可能）
3. Update documentation for parameter usage and defaults 更新文件說明參數用途和預設值
4. Add parameter to `backtest_with_report()` 在 `backtest_with_report()` 中添加參數

---

## Summary 總結

### Parameter Status Overview 參數狀態總覽

| Status 狀態 | Count 數量 | Parameters 參數 |
|------------|-----------|----------------|
| ✅ Fully Working & Verified 完全運作並驗證 | 12 | position, resample, trade_at_price, position_limit, fee_ratio, tax_ratio, finlab_mode, stop_loss, take_profit, trail_stop, retain_cost_when_rebalance, stop_trading_next_period |
| ❌ Missing 缺失 | 11 | resample_offset, touched_exit, mae_mfe_*, fast_mode, name, upload, notification_enable, line_access_token, live_performance_start, market |

### Actual Coverage 實際覆蓋率

- **Fully Working & Verified 完全運作並驗證**: 12/23 parameters (52%)
- **Missing 缺失**: 11 parameters (mostly metadata/service parameters)

### ✅ Completed 已完成

1. ✅ **`stop_loss`** - Verified against finlab (max diff 2.22e-16)
2. ✅ **`take_profit`** - Verified against finlab (max diff 2.22e-16)
3. ✅ **`trail_stop`** - Verified against finlab (max diff 2.22e-16), fixed formula to use entry_price
4. ✅ **`retain_cost_when_rebalance`** - Verified against finlab (max diff 2.22e-16)
5. ✅ **`stop_trading_next_period`** - Verified against finlab (max diff 2.22e-16)
6. ✅ **Python unit tests** - 10 tests for all stop parameters with T+1 execution

### 🔄 Priority Recommendations 優先級建議

**Phase 1: Core Features 核心功能 (HIGH Priority 高優先級)**

1. 🔄 **`resample_offset`** - Complete rebalance functionality 完善重新平衡功能
2. 🔄 **`touched_exit`** - Improve stop loss/profit realism 提升停損/停利真實性

**Phase 2: Analytics 分析功能 (MEDIUM Priority 中優先級)**

3. 🔄 **`mae_mfe_*`** - Enhance trade analysis 增強交易分析
4. 🔄 **`fast_mode`** - Performance optimization 效能優化

**Phase 3: Compatibility 兼容性 (LOW Priority 低優先級)**

5. ⏸️ Metadata parameters (name, upload, notification_enable, line_access_token)
6. ⏸️ Live trading parameters (live_performance_start, market)
