# Finlab vs Polars Backtest 差異研究

## 目標
使 `df_full.filter(pl.col("creturn").round(6) != pl.col("creturn_pl").round(6))` 返回空結果

## 測試策略
```python
close = finlab_data.get('price:收盤價')
adj_close = finlab_data.get('etl:adj_close')
position = close >= close.rolling(300).max()
report = backtest.sim(position, resample='M')
```

## 當前狀態
- 最終 creturn 差異: ~1%
- 第一個差異出現在: 2010-09-01 (rebalance 日)

---

## 研究發現

### 1. Finlab 的基本 creturn 計算 (已驗證)

**結論**: 沒有複合效應，是簡單的 entry_value * (price / entry_price)

```python
# Entry on 2/1
entry_price = 1333.664
entry_value = 1 - fee_ratio = 0.998575

# 任意一天的 creturn
creturn[t] = entry_value * (price[t] / entry_price)
           = 0.998575 * (price[t] / 1333.664)
```

**驗證**:
| Date       | Finlab     | Simple Model | Diff       |
|------------|------------|--------------|------------|
| 2024-02-01 | 0.998575   | 0.998575     | 0          |
| 2024-02-02 | 1.009706   | 1.009706     | 0          |
| 2024-02-05 | 1.027197   | 1.027197     | 0          |
| 2024-02-15 | 1.108291   | 1.108291     | 0          |
| 2024-02-29 | 1.097160   | 1.097160     | 0          |

---

### 2. Finlab Rebalance 費用計算 (已驗證)

**結論**: 當 position 改變時，收取 sell + buy 費用

```python
# Position change: 2330 -> 2454 on 3/1
# Value before rebalance (at 3/1 prices)
value_before = entry_value * (price[3/1] / entry_price)
             = 0.998575 * (1463.2078 / 1333.664)
             = 1.095570

# Rebalance fee
total_fee = 2 * fee_ratio + tax_ratio = 0.00585

# Value after rebalance
value_after = value_before * (1 - total_fee)
            = 1.095570 * 0.99415
            = 1.089161

# Finlab creturn on 3/1 = 1.089161 ✓
```

---

### 3. Finlab 的 position resampling

當 resample='M' 時:
1. 產生月底日期作為 rebalance dates
2. `position.reindex(dates, method='ffill')`
3. T+1 執行: 月底 signal -> 下個月第一個交易日執行

**時間線範例**:
- 1/31: Signal = position on 1/31 (ffill from 1/30 if weekend)
- 2/1: Execute signal, buy stocks
- 2/2 - 2/29: Hold stocks (daily return = price change)
- 2/29: Signal = position on 2/29
- 3/1: Execute signal, sell old, buy new

---

### 4. backtest_core.pyx 關鍵邏輯分析

#### 4.1 每日更新 (lines 280-316)
```python
for sid in pos:
    sidprice = pos2price[sid]

    # 如果 previous_price 是 NaN，設為當前價格
    if isnan(previous_price[sidprice]):
        previous_price[sidprice] = price_values[d, sidprice]

    # 計算 r = current / previous
    r = price_values[d, sidprice] / previous_price[sidprice]
    if isnan(r):
        r = 1

    # 更新 pos
    pos[sid] *= r
```

#### 4.2 Balance 計算 (lines 466-481)
```python
balance = cash
for sid in pos:
    trade_price = price_values[d, pos2price[sid]]  # = adj_close
    close_price = close_values[d, pos2price[sid]]  # = adj_close

    if isnan(trade_price) or isnan(close_price):
        balance += pos[sid]
    else:
        balance += pos[sid] * close_price / trade_price
        # When trade_at_price='close': trade_price = close_price
        # So: balance += pos[sid] * 1.0 = pos[sid]

creturn[d] = balance
```

#### 4.3 Rebalance (lines 442-461)
```python
if should_rebalance:
    if not RETAIN_COST_WHEN_REBALANCE:
        for sid in range(pos_columns.size):
            cr[sid] = 1
            maxcr[sid] = 1

    cash = rebalance(pos, pos_values[pos_id], cash, fee_ratio, tax_ratio, position_limit)

    latest_rebalanced_pos = pos
    current_position_id = pos_id
    previous_price = price_values[d].copy()  # 更新 previous_price
```

---

### 5. 疑問與假設

#### 假設 1: pos *= r 應該產生複合效應，但實際 creturn 沒有

**觀察**:
- 根據代碼，pos *= r 每天執行
- r = price[today] / previous_price (previous_price 是 rebalance 日設定的)
- 這應該產生複合效應: pos[t+1] = pos[t] * r[t+1]

**但實際**:
- creturn 完全符合 simple model: entry_value * (price[t] / entry_price)
- 沒有複合效應

**可能解釋**:
1. balance 計算用 `pos[sid] * close_price / trade_price` 抵消了複合效應
2. 代碼有其他地方重置 pos
3. 我誤解了代碼邏輯

#### 假設 2: 差異來自多股票的權重計算

當持有多檔股票時:
- Finlab: 使用 `rebalance()` 函數處理
- 我們: 使用 `execute_finlab_rebalance()`

**待驗證**:
- 權重 normalization 方式
- 費用分配方式

---

## 實驗記錄

### 實驗 1: 單股票驗證
- **結果**: Simple model 完全匹配 Finlab
- **結論**: 單股票情況下，creturn = entry_value * (price / entry_price)

### 實驗 2: Position change 驗證
- **結果**: 費用 = value * (2*fee + tax) = 0.585%
- **結論**: Rebalance 時收取 sell + buy 費用

### 實驗 3: 完整策略對比
- **結果**: 1% 差異，從 2010-09-01 開始
- **待分析**: 需要找出差異的具體來源

---

## 重大發現 (2024-12 更新)

### 6. Finlab set_position 新版邏輯 (已確認)

**完整的 set_position 邏輯** (restored_backtest_core.pyx):
```python
# 賣出 fast path (position == 0)
if position == 0:
    if exist:
        cash += p[sid] - abs(p[sid]) * (fee_ratio + tax_ratio)
        p[sid] = 0
    return cash

# 新股票先設為 0
if not exist:
    p[sid] = 0

amount = position - p[sid]
buy = amount > 0
is_entry = (position >= 0 and amount > 0) or (position <= 0 and amount < 0)
cost = abs(amount) * fee_ratio if is_entry else abs(amount) * (fee_ratio + tax_ratio)

if buy:
    cash -= amount
    p[sid] += amount - cost    # ← 關鍵：持倉扣 cost
else:
    amount = -amount
    cash += amount - cost
    p[sid] -= amount
```

**驗證 (A -> B 換倉)**:
- 賣出 A: cash += 1.0 - 0.004425 = 0.995575
- 買入 B: cash -= 1.0, p['B'] = 1.0 - 0.001425 = 0.998575
- 最終: balance = -0.004425 + 0.998575 = 0.99415
- 費用率 = 0.585% ✓

### 7. NaN 價格處理 Bug 修復 (2024-12-23)

**問題**: `update_entry_prices_after_nan` 在 NaN 恢復時錯誤地更新 `entry_price`

**錯誤邏輯**:
```rust
// Case 2: 當前一天價格是 NaN 但今天有效
if prev_is_nan {
    pos.entry_price = curr_price;  // ← 錯誤！
}
```

**正確行為** (Finlab):
- `previous_price` 在 NaN 日不更新
- 恢復後用 NaN 前的價格計算報酬
- 這樣 NaN 期間的價格變化會在恢復時一次性反映

**修復後結果**:
- 兩股票 (2330+2317): 完全匹配 (差異 < 1e-14)
- 完整策略 (rolling 300 max): 從 ~3% 降到 ~0.24%

### 8. 剩餘差異分析

剩餘 0.24% 差異可能來自:
1. 浮點數精度累積誤差
2. 多股票等權重計算的細微差異
3. NaN 價格股票的 rebalance 處理

第一個差異出現在 2010-09-01:
- 差異: -0.000129 (-0.003%)
- 4609 股票在 8/31-9/3 連續 NaN
- rebalance 時需要賣出 4609，但價格是 NaN

---

## TODO

1. [x] 模擬 Finlab 的 `rebalance()` 函數，驗證權重和費用計算
2. [x] 分析 2010-09-01 差異的具體原因
3. [x] 驗證多股票情況下的行為
4. [x] 檢查 NaN 價格處理 - **已修復 Bug!**
5. [x] 檢查 previous_price 更新邏輯 - Finlab 在 NaN 日不更新
6. [x] 找出我們實現與 Finlab 的具體差異點 - 見上方分析

## 當前狀態 (2024-12-23)

- 單股票: **完全匹配** (差異 < 1e-13)
- 兩股票 (2330+2317): **完全匹配** (差異 < 1e-14)
- 完整策略 (rolling 300 max): **差異 ~0.24%** (從 ~3% 改善)

---

## 代碼參考

### Finlab files
- `/polars_backtest/.venv/lib/python3.13/site-packages/finlab/backtest.py`
- `restored_backtest_core.pyx` (Cython source)

### Our implementation
- `btcore/src/simulation.rs`
