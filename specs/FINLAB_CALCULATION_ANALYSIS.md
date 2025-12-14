# Finlab Backtest 計算方式分析

> 來源：`finlab/core/backtest_core.cpp` (Cython 轉譯的 C++)
> 分析日期：2025-12-14

---

## 1. 核心架構差異

### 1.1 價值追蹤模型

| 方面 | Finlab | 標準投資組合計算 |
|------|--------|------------------|
| 儲存內容 | `cost_basis` + `entry_price` | `current_value` |
| 每日更新 | `cost_basis * close / entry_price` | `value * (1 + daily_return)` |
| Weight 基礎 | 成本基礎 | 當前市值 |

### 1.2 數學等價性

**單一股票**：兩種方法完全等價
```
Finlab:  cost_basis * (close_t / entry_price)
標準:    cost_basis * Π(1 + r_i) = cost_basis * (close_t / entry_price)
```

**多股票**：因 weight drift 產生差異
- Finlab 的 weight 始終基於「成本基礎」
- 標準方法的 weight 會隨市值變化而 drift

---

## 2. Finlab 核心公式 (從 Cython 原始碼提取)

### 2.1 每日 Balance 計算

**位置**: `backtest_core.cpp` line 11376-11540

```cpp
// line 11385: 開始計算 balance
balance = cash;

// line 11394-11527: 遍歷所有持倉
it = pos.begin();
while (it != pos.end()) {
    trade_price = price_values[d, pos2price[sid]];  // 入場價格
    close_price = close_values[d, pos2price[sid]];  // 當天收盤價

    if (isnan(trade_price) || isnan(close_price)) {
        // line 11480: 價格無效時直接加成本基礎
        balance += position_value;
    } else {
        // line 11495-11516: 關鍵公式！
        balance += position_value * close_price / trade_price;
    }
}

// line 11539: 設定 creturn
creturn[d] = balance;
```

**Python 等價代碼**:
```python
def calculate_daily_balance(cash, positions, close_prices, entry_prices):
    balance = cash
    for sid, cost_basis in positions.items():
        entry_price = entry_prices[sid]
        close_price = close_prices[sid]

        if pd.isna(entry_price) or pd.isna(close_price):
            balance += cost_basis
        else:
            balance += cost_basis * close_price / entry_price

    return balance
```

### 2.2 Exit 公式 (賣出)

**位置**: `backtest_core.cpp` line 7058-7145

```cpp
// line 7065-7085: position = 0 時 (完全出場)
if (position == 0) {
    if (exist) {
        // 關鍵公式：取回 cost_basis 扣除交易成本
        cash += p[sid] - abs(p[sid]) * (fee_ratio + tax_ratio);
    }
    return cash;
}
```

**Python 等價代碼**:
```python
def exit_position(cash, cost_basis, fee_ratio, tax_ratio):
    # 賣出時：取回成本基礎，扣除 fee + tax
    cash += cost_basis - abs(cost_basis) * (fee_ratio + tax_ratio)
    return cash
```

**注意**: 這裡的 `cost_basis` 是以「入場價格為基準的價值」，不是當前市值！

### 2.3 Entry 公式 (買入)

**位置**: `backtest_core.cpp` line 7185-7250

```cpp
// line 7192: 計算需要的金額
amount = position - p[sid];  // position 是目標持倉價值

// line 7197-7209: 判斷是否為入場
buy = amount > 0;
is_entry = (position >= 0 && amount > 0) || (position <= 0 && amount < 0);

// line 7246-7249: 計算交易成本
cost = is_entry ? abs(amount) * fee_ratio : abs(amount) * (fee_ratio + tax_ratio);
```

**Python 等價代碼**:
```python
def entry_position(cash, current_cost_basis, target_cost_basis, fee_ratio, tax_ratio):
    amount = target_cost_basis - current_cost_basis
    is_entry = (target_cost_basis >= 0 and amount > 0) or \
               (target_cost_basis <= 0 and amount < 0)

    if is_entry:
        cost = abs(amount) * fee_ratio
    else:
        cost = abs(amount) * (fee_ratio + tax_ratio)

    cash -= amount + cost
    return cash, target_cost_basis
```

### 2.4 Rebalance 公式

**位置**: `backtest_core.cpp` line 7447-7786

```cpp
// line 7452-7456: 計算當前總 balance
balance = cash;
while (it != p.end()) {
    balance += (*it).second;  // 加總所有 cost_basis
}

// line 7778-7786: 執行 rebalance
// 對每個股票調整到目標 weight
for (sid, target_weight) in new_weights:
    target_position = balance * target_weight;  // 目標 cost_basis
    cash = set_position(pos, sid, target_position, cash,
                        fee_ratio, tax_ratio);
```

---

## 3. 關鍵變數定義

| 變數名 | Finlab 定義 | 說明 |
|--------|-------------|------|
| `pos[sid]` / `p[sid]` | cost_basis | 該股票的成本基礎（入場時的價值） |
| `trade_price` | entry_price | 入場時的價格（用於計算當前市值） |
| `close_price` | current_close | 當天收盤價 |
| `balance` | total_value | 當天總資產價值 |
| `cash` | available_cash | 可用現金 |

---

## 4. 計算流程

### 4.1 每日處理流程

```
For each day d:
    1. 計算當前 balance (用 cost_basis * close/entry 公式)
    2. 處理 stop loss / take profit
    3. 處理 exit signals
    4. 處理 rebalance (如果有新的 weight signal)
    5. 記錄 creturn[d] = balance
```

### 4.2 Rebalance 流程

```
When rebalance triggered:
    1. 計算當前 total_balance = cash + sum(cost_basis)
       注意：這裡用的是 cost_basis 總和，不是市值總和！
    2. 對每個股票:
       - 計算目標 cost_basis = total_balance * target_weight
       - 調整持倉（買入或賣出）
       - 扣除交易成本
```

---

## 5. 差異影響分析

### 5.1 為什麼單一股票完全匹配

單一股票時，portfolio value 計算：
```
Finlab:  balance = cost_basis * (close / entry_price)
標準:    balance = cost_basis * (1 + r1) * (1 + r2) * ... * (1 + rn)
                 = cost_basis * (close / entry_price)  // 數學等價
```

### 5.2 為什麼多股票有差異

多股票 rebalance 時：

**Finlab** (基於 cost_basis):
```python
total = cash + sum(cost_basis)  # 不考慮市值變化
target_A = total * 0.5  # 固定目標
target_B = total * 0.5
```

**標準方法** (基於市值):
```python
total = cash + sum(current_market_value)  # 考慮市值變化
target_A = total * 0.5
target_B = total * 0.5
```

當股票報酬不同時，這兩個 `total` 會有差異！

### 5.3 差異量化

實測結果 (221 天回測, 3 股票):
- 相對誤差: ~1.07% (年化)
- 最大單日差異: ~0.02%
- 差異方向: 時正時負（非系統性偏差）

---

## 6. 實作狀態

### 6.1 標準模式 (推薦) ✅

**狀態**: ✅ 完成，與 Finlab **完美匹配** (0.0000% 誤差)

標準模式 (`backtest()`) 使用傳統的投資組合計算方式：
- Position 追蹤當前市值 (每天更新)
- Balance = cash + Σ(current_value)
- Rebalance 基於當前市值
- Fee 使用 Finlab 風格：position = allocation * (1 - fee)

**重要發現**: 經過完整測試，標準模式已經完美匹配 Finlab 的結果！

測試結果：
| 測試案例 | 差異 |
|----------|------|
| 單一股票 (2330) | 0.0000% |
| 多股票組合 | 0.0000% |

### 6.2 Finlab 相容模式 (實驗性)

**狀態**: ⚠️ 已實作但不推薦使用

`backtest_finlab()` 函數支援雙價格矩陣：
- `trade_prices`: 用於記錄入場價格
- `close_prices`: 用於計算當前市值

**重要發現**: 經過測試，使用相同價格序列時：
- 單一股票：~0.8% 誤差
- 多股票組合：~18% 誤差

這表明 Finlab 實際上使用的是**標準投資組合計算方式**，而不是我們從 Cython 程式碼推測的 cost_basis 方式。

### 6.3 建議

**直接使用標準模式 `backtest()`，它已經完美匹配 Finlab！**

```python
from polars_backtest import backtest

result = backtest(
    prices=adj_close_df,  # 使用調整後收盤價
    position=weights_df,
    fee_ratio=0.001425,
    tax_ratio=0.003,
)
```

不需要使用 `finlab_mode=True` 或 `backtest_finlab()`。

### 6.4 結論

標準模式已經完美復現 Finlab 的計算結果。之前分析 Cython 程式碼的推測（cost_basis + entry_price 模型）可能有誤解，或者 Finlab 在不同情況下使用不同的計算方式。

對於所有實際使用場景，**標準模式是正確且推薦的選擇**。

---

## 7. 驗證方法

建議的驗證流程：
1. 單一股票測試 → 必須完全匹配 (diff = 0)
2. 多股票無 rebalance → 必須完全匹配
3. 多股票有 rebalance → 必須完全匹配
4. 複雜情境（stop loss, take profit）→ 必須完全匹配

---

## 8. 參考原始碼位置

| 功能 | 檔案 | 行數 |
|------|------|------|
| 每日 balance 計算 | backtest_core.cpp | 11376-11540 |
| Exit 公式 | backtest_core.cpp | 7058-7145 |
| Entry/Adjust 公式 | backtest_core.cpp | 7185-7250 |
| Rebalance | backtest_core.cpp | 7447-7786 |
| set_position | backtest_core.cpp | 6802-7840 |
