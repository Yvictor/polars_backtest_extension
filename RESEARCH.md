# Finlab vs Polars Backtest 研究筆記

## 當前狀態 ✅

**所有測試通過，完全匹配 Finlab！**

```
Final creturn: Finlab 66.597805 vs Polars 66.597805
Max relative diff: 0.0000%
Matches at round(6): True ✓
```

---

## 關鍵發現

### 1. creturn 計算（無複合效應）
```python
creturn[t] = entry_value * (price[t] / entry_price)
```

### 2. Rebalance 費用
- 換倉費 = 2 * fee_ratio + tax_ratio = 0.585%
- `value_after = value_before * (1 - total_fee)`

### 3. Stop Loss 計算
- Finlab 使用累積乘法: `cr *= (today/yesterday)`
- 必須追蹤 `cr`, `maxcr`, `previous_price` 以匹配浮點精度

### 4. NaN 價格處理
- NaN 日不更新 `previous_price`
- balance 使用 `last_market_value` 而非 `cost_basis`

### 5. Weight Normalization
- Stop 後剩餘權重必須重新縮放以維持 100% 投資
- `scale_factor = original_sum / remaining_sum`

---

## 已修復的 Bug

| Bug | 影響 | 修復 |
|-----|------|------|
| NaN 恢復時錯誤更新 entry_price | ~0.24% 累積差異 | 使用 last_market_value |
| Stop 後權重未重新正規化 | 3-4% 差異 | 加入 scale_factor |
| 累積 vs 直接 cr 計算 | 1 bit 浮點差異 | 改用累積乘法 |
| retain_cost_when_rebalance | 止損計算錯誤 | 保留舊 cr/maxcr |

---

## 代碼參考

- Finlab: `finlab/backtest.py`, `backtest_core.pyx`
- 我們: `btcore/src/simulation/wide.rs`, `btcore/src/stops.rs`
