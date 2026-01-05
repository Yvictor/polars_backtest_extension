# Liquidity Metrics Implementation Plan

## Overview

Liquidity 指標用於評估策略的流動性風險，主要包括：
- 買賣在漲跌停的比例 (buyHigh, sellLow)

## 設計原則

1. **資料驅動**: 根據 backtest 時傳入的 DataFrame 是否包含特定欄位來決定是否計算
2. **不自動計算漲跌停價**: 用戶需自行提供 `limit_up`, `limit_down` 欄位
3. **無需額外參數**: `get_metrics()` 不需要傳入額外參數
4. **不修改 btcore**: 使用 join 方式在 report 層計算，不傳給 btcore

## 所需資料欄位

| 欄位 | 說明 | 必要性 |
|------|------|--------|
| `limit_up` | 漲停價 | 可選 (計算 buyHigh) |
| `limit_down` | 跌停價 | 可選 (計算 sellLow) |

用戶在 backtest 時傳入的 DataFrame 若包含這些欄位，則會計算對應指標。

---

## 指標計算方式

### 1. buyHigh (買在漲停比例)

**定義**: 進場時交易價格 >= 漲停價的交易比例

**計算方式**:
```
buyHigh = count(entry_raw_price >= limit_up@entry_date) / total_entries
```

**判斷邏輯**:
- 使用 `entry_raw_price` (原始價格，非除權後價格) 與漲停價比較
- 若進場價格 >= 漲停價，代表買在漲停
- 這種情況下實際可能無法成交，策略報酬可能高估

**所需資料**: `limit_up` (漲停價，by date and symbol)

---

### 2. sellLow (賣在跌停比例)

**定義**: 出場時交易價格 <= 跌停價的交易比例

**計算方式**:
```
sellLow = count(exit_raw_price <= limit_down@exit_date) / total_exits
```

**判斷邏輯**:
- 使用 `exit_raw_price` (原始價格，非除權後價格) 與跌停價比較
- 若出場價格 <= 跌停價，代表賣在跌停
- 這種情況下實際可能無法出場，策略報酬可能高估

**所需資料**: `limit_down` (跌停價，by date and symbol)

---

## Implementation Plan (Join-based approach)

### 資料流程

```
1. 用戶準備 DataFrame，包含 limit_up, limit_down 欄位 (可選)

   df = pl.DataFrame({
       'date': [...],
       'symbol': [...],
       'close': [...],
       'weight': [...],
       'limit_up': [...],    # 漲停價 (原始價格)
       'limit_down': [...],  # 跌停價 (原始價格)
   })

2. backtest_with_report() 時：
   - 檢查 DataFrame 是否有 limit_up, limit_down 欄位
   - 若有，提取 (date, symbol, limit_up, limit_down) 並存入 Report
   - 不傳給 btcore，btcore 層不需要修改

3. get_metrics(sections=['liquidity']) 時：
   - 檢查 Report 是否有 limit_prices_df
   - 若有：
     a. Join trades with limit_prices on (entry_date, stock_id) -> get limit_up
     b. Join trades with limit_prices on (exit_date, stock_id) -> get limit_down
     c. 比較 entry_raw_price >= limit_up -> buyHigh
     d. 比較 exit_raw_price <= limit_down -> sellLow
   - 若無，這些指標為 null
```

### 實作步驟

#### Step 1: Rust - PyBacktestReport 加入 limit_prices_df

```rust
pub struct PyBacktestReport {
    pub(crate) creturn_df: DataFrame,
    pub(crate) trades_df: DataFrame,
    pub(crate) config: BacktestConfig,
    pub(crate) resample: Option<String>,
    pub(crate) benchmark_df: Option<DataFrame>,
    pub(crate) limit_prices_df: Option<DataFrame>,  // NEW: (date, symbol, limit_up, limit_down)
}
```

#### Step 2: Rust - backtest_with_report 提取 limit columns

```rust
// In backtest_with_report function:
// Check if limit_up/limit_down columns exist
let limit_prices_df = if df.column("limit_up").is_ok() || df.column("limit_down").is_ok() {
    // Extract (date, symbol, limit_up, limit_down)
    let mut cols = vec![col(date).alias("date"), col(symbol).alias("symbol")];
    if df.column("limit_up").is_ok() {
        cols.push(col("limit_up"));
    }
    if df.column("limit_down").is_ok() {
        cols.push(col("limit_down"));
    }
    Some(df.clone().lazy().select(cols).unique(None, Default::default()).collect()?)
} else {
    None
};
```

#### Step 3: Rust - get_metrics() 加入 liquidity section

```rust
// === LIQUIDITY SECTION ===
if sections_list.contains(&"liquidity") {
    let (buy_high, sell_low) = self.calc_liquidity_metrics()?;
    exprs.push(lit(buy_high).alias("buyHigh"));
    exprs.push(lit(sell_low).alias("sellLow"));
}
```

#### Step 4: Rust - calc_liquidity_metrics 實作

```rust
fn calc_liquidity_metrics(&self) -> PyResult<(f64, f64)> {
    let Some(limit_df) = &self.limit_prices_df else {
        return Ok((f64::NAN, f64::NAN));
    };

    let trades = &self.trades_df;

    // Join for buyHigh: trades.entry_date == limit.date AND trades.stock_id == limit.symbol
    let with_entry_limit = trades.clone().lazy()
        .join(
            limit_df.clone().lazy().select([
                col("date").alias("entry_date"),
                col("symbol").alias("stock_id"),
                col("limit_up"),
            ]),
            [col("entry_date"), col("stock_id")],
            [col("entry_date"), col("stock_id")],
            JoinArgs::new(JoinType::Left),
        )
        .with_column(
            col("entry_raw_price").gt_eq(col("limit_up")).alias("at_limit_up")
        )
        .collect()?;

    // Calculate buyHigh ratio
    let buy_high = with_entry_limit
        .lazy()
        .filter(col("limit_up").is_not_null())
        .select([
            col("at_limit_up").sum().alias("count_at_limit"),
            col("at_limit_up").count().alias("total"),
        ])
        .collect()?;
    // ... calculate ratio ...

    // Similar for sellLow with exit_date and limit_down
}
```

---

## API Usage Example

```python
import polars as pl
import polars_backtest as pl_bt

# 準備資料，包含漲跌停價
df = pl.DataFrame({
    'date': dates,
    'symbol': symbols,
    'close': close_prices,
    'weight': weights,
    'limit_up': limit_up_prices,     # 漲停價 (原始價格)
    'limit_down': limit_down_prices,  # 跌停價 (原始價格)
})

# backtest
report = df.bt.backtest_with_report(
    trade_at_price='close',
    position='weight',
)

# get_metrics 會自動包含 liquidity 指標
metrics = report.get_metrics(sections=['liquidity'])
print(metrics['buyHigh'])   # 買在漲停比例
print(metrics['sellLow'])   # 賣在跌停比例

# 若 DataFrame 沒有 limit_up/limit_down，這些指標為 null
df_no_limit = df.drop(['limit_up', 'limit_down'])
report2 = df_no_limit.bt.backtest_with_report(...)
metrics2 = report2.get_metrics(sections=['liquidity'])
print(metrics2['buyHigh'])  # null
print(metrics2['sellLow'])  # null
```

---

## 優點

1. **不修改 btcore**: btcore 層完全不需要改動，維持原有邏輯
2. **彈性**: 可以支援更多類似的 join-based 指標
3. **效能**: join 操作在 Polars 中很高效
4. **向後相容**: 沒有 limit columns 的 DataFrame 一樣可以正常 backtest

---

## Future Extensions (暫不實作)

以下指標暫時不實作，待需求確認：

- **warningStockRatio**: 警示股比例
- **disposalStockRatio**: 處置股比例
- **fullDeliveryStockRatio**: 全額交割股比例
- **capacity**: 策略容量

這些指標需要額外的資料欄位 (警示股/處置股 flag, 成交金額等)，
未來可以用相同的模式實作：DataFrame 有欄位就計算，沒有就 null。
