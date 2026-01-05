# Liquidity Metrics Implementation Plan

## Overview

Liquidity 指標用於評估策略的流動性風險，包括：
- 買賣在漲跌停的比例
- 警示/處置/全額交割股的比例
- 策略容量

## Finlab Liquidity 結構

```python
report.liquidity = {
    'buy_high': {'entry': 0.082, 'exit': 0.002},      # 買在漲停
    'sell_low': {'entry': 0.0, 'exit': 0.011},        # 賣在跌停
    'low_volume_stocks': {'entry': 0.086, 'exit': 0.157},  # 低成交量
    'low_turnover_stocks': {'entry': 0.012, 'exit': 0.022}, # 低周轉率
    '警示股': {'entry': 0.094, 'exit': 0.026},
    '處置股': {'entry': 0.026, 'exit': 0.014},
    '全額交割股': {'entry': 0.002, 'exit': 0.002},
}

report.get_metrics()['liquidity'] = {
    'capacity': 14430268.54,           # 策略容量 (金額)
    'disposalStockRatio': 0.026,       # 處置股比例 (entry)
    'warningStockRatio': 0.094,        # 警示股比例 (entry)
    'fullDeliveryStockRatio': 0.002,   # 全額交割股比例 (entry)
    'buyHigh': 0.082,                  # 買在漲停比例 (entry)
    'sellLow': 0.011,                  # 賣在跌停比例 (exit)
}
```

## 所需資料欄位

### 必要欄位
| 欄位 | 說明 | 來源 |
|------|------|------|
| `open` | 開盤價 | price:開盤價 |
| `high` | 最高價 | price:最高價 |
| `low` | 最低價 | price:最低價 |
| `close` | 收盤價 | price:收盤價 |
| `volume` | 成交量 | price:成交股數 |

### 選用欄位 (計算特定指標)
| 欄位 | 說明 | 來源 |
|------|------|------|
| `amount` | 成交金額 | price:成交金額 |
| `turnover` | 周轉率 | 計算或外部 |
| `disposal` | 處置股標記 | bool flag |
| `warning` | 警示股標記 | bool flag |
| `full_delivery` | 全額交割股標記 | bool flag |

---

## 指標計算方式

### 1. buyHigh (買在漲停比例)

**定義**: 進場時買在漲停價的交易比例

**計算方式**:
```
漲停判斷 = (open == high) AND (pct_change >= 0.095)
buyHigh = count(entry 當日漲停) / total_entries
```

**說明**:
- `open == high`: 開盤即鎖漲停，無法以更低價買入
- `pct_change >= 0.095`: 當日漲幅接近漲停 (10%)
- 這種情況下實際成交困難，策略報酬可能高估

**所需資料**: `open`, `high`, `close` (計算 pct_change)

---

### 2. sellLow (賣在跌停比例)

**定義**: 出場時賣在跌停價的交易比例

**計算方式**:
```
跌停判斷 = (open == low) AND (pct_change <= -0.095)
sellLow = count(exit 當日跌停) / total_exits
```

**說明**:
- `open == low`: 開盤即鎖跌停，無法以更高價賣出
- `pct_change <= -0.095`: 當日跌幅接近跌停 (-10%)
- 這種情況下實際無法出場，策略報酬可能高估

**所需資料**: `open`, `low`, `close` (計算 pct_change)

---

### 3. warningStockRatio (警示股比例)

**定義**: 進場時標的為警示股的交易比例

**計算方式**:
```
warningStockRatio = count(entry_sig_date 時為警示股) / total_entries
```

**說明**:
- 警示股有較高的交易風險和限制
- 需要外部提供警示股清單 (bool DataFrame)

**所需資料**: `warning` (bool flag, by date and symbol)

---

### 4. disposalStockRatio (處置股比例)

**定義**: 進場時標的為處置股的交易比例

**計算方式**:
```
disposalStockRatio = count(entry_sig_date 時為處置股) / total_entries
```

**說明**:
- 處置股有嚴格的交易限制 (如只能現股買賣)
- 需要外部提供處置股清單 (bool DataFrame)

**所需資料**: `disposal` (bool flag, by date and symbol)

---

### 5. fullDeliveryStockRatio (全額交割股比例)

**定義**: 進場時標的為全額交割股的交易比例

**計算方式**:
```
fullDeliveryStockRatio = count(entry_sig_date 時為全額交割股) / total_entries
```

**說明**:
- 全額交割股需全額付款，流動性極差
- 需要外部提供全額交割股清單 (bool DataFrame)

**所需資料**: `full_delivery` (bool flag, by date and symbol)

---

### 6. capacity (策略容量)

**定義**: 策略可容納的資金規模

**計算方式** (推測):
```
# 方法 1: 基於成交金額的最小值
每日可買入金額 = sum(持股的成交金額) * 參與比例
capacity = min(每日可買入金額) 或 percentile_5(每日可買入金額)

# 方法 2: 基於平均成交量
capacity = avg(持股成交金額 * 權重) * 安全係數
```

**說明**:
- 策略規模超過 capacity 會影響執行效果
- 實際計算方式需要確認 Finlab 的具體實作

**所需資料**: `amount` (成交金額), `volume` (成交量), `close` (收盤價)

---

## Implementation Plan

### Phase 3a: 基礎漲跌停指標 (不需額外資料)
- **buyHigh**: 使用 open, high, close 判斷
- **sellLow**: 使用 open, low, close 判斷

**實作位置**: `report.rs` 的 `get_metrics()` 加入 liquidity section

**資料需求**:
- trades DataFrame 需包含 `entry_date`, `exit_date`
- 需要在 Rust 層面 join OHLC 資料，或在 Python 層面預處理

### Phase 3b: 警示/處置/全額交割股指標
- **warningStockRatio**
- **disposalStockRatio**
- **fullDeliveryStockRatio**

**實作方式**:
- 使用者傳入 flag DataFrame (date, symbol, warning/disposal/full_delivery)
- 在 trades 上 join 這些 flags
- 計算比例

### Phase 3c: 策略容量
- **capacity**

**實作方式**:
- 需要每日成交金額資料
- 計算每個持倉期間的流動性
- 取保守估計值

---

## API Design

### Option 1: get_metrics() 參數擴展

```python
report.get_metrics(
    sections=['liquidity'],
    ohlc=df_ohlc,           # DataFrame with date, symbol, open, high, low, close
    stock_flags=df_flags,   # DataFrame with date, symbol, warning, disposal, full_delivery
    amount=df_amount,       # DataFrame with date, symbol, amount
)
```

### Option 2: 獨立 liquidity 方法

```python
report.get_liquidity(
    ohlc=df_ohlc,
    stock_flags=df_flags,
    amount=df_amount,
)
```

### Option 3: 預處理整合到 trades

在 backtest 時就加入相關欄位：
```python
df = df.with_columns([
    pl.col('open'),
    pl.col('high'),
    pl.col('low'),
    ...
])
report = df.bt.backtest_with_report(...)
# trades 自動包含漲跌停判斷
```

---

## Implementation Priority

1. **Phase 3a** (High): buyHigh, sellLow
   - 核心流動性指標
   - 不需額外資料 (假設 trades 已有 OHLC)

2. **Phase 3b** (Medium): 警示/處置/全額交割股
   - 需要外部 flag 資料
   - 可選實作

3. **Phase 3c** (Low): capacity
   - 計算複雜
   - 需要更多研究確認公式

---

## Notes

- Finlab 的 buy_high 可能使用更複雜的判斷邏輯 (不只是 open==high)
- 實際比例差異可能來自不同的漲跌停判斷方式
- 建議先實作基本版本，再根據實際需求調整
