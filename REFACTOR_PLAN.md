# Backtest Engine Refactor

## 目標結構 ✅

```
btcore/src/
├── lib.rs           # 對外 API
├── config.rs        # BacktestConfig
├── position.rs      # Position struct
├── portfolio.rs     # PortfolioState
├── tracker.rs       # TradeTracker trait
├── stops.rs         # 止損檢測
├── simulation/
│   ├── mod.rs       # 公共接口
│   ├── wide.rs      # Wide format 模擬
│   └── long.rs      # Long format 模擬
├── rebalance.rs     # Rebalance 執行
├── weights.rs       # 權重正規化
├── trades.rs        # TradeRecord
├── returns.rs       # 收益率計算
└── stats.rs         # 統計計算
```

---

## 已完成 ✅

| Stage | 內容 | 行數變化 |
|-------|------|----------|
| 1 | 抽取 Config, Position | simulation.rs -85 |
| 2 | 抽取 TradeTracker | simulation.rs -273 |
| 3 | 抽取止損檢測 | simulation.rs -341 |
| 4 | 重構 PortfolioState | simulation.rs -185 |
| 5 | 拆分 simulation/ | simulation.rs → wide.rs |
| 6 | Long Format 回測 | 新增 long.rs (643 行) |

**測試**: 88 單元 + 23 整合 = 111 通過

---

## 待完成

### Stage 7: Python 綁定
- [ ] `polars_backtest/src/lib.rs` 添加 `backtest_long()`
- [ ] 接收 numpy array (zero-copy)
- [ ] 更新 namespace.py
- [ ] 效能測試

### Stage 8: 清理
- [ ] 統一錯誤處理
- [ ] 補充文檔
- [ ] 無 clippy 警告

---

## Long Format 設計

```rust
pub fn backtest_long(
    date_indices: &[u32],    // 已排序
    symbol_ids: &[u32],
    prices: &[f64],
    weights: &[f64],
    n_dates: usize,
    rebalance_dates: &HashSet<u32>,
    config: &BacktestConfig,
) -> BacktestResult
```

**效能預估**:
| 場景 | Wide | Long | 提升 |
|------|------|------|------|
| 1000 stocks, 252 days | 252k cells | ~50k events | ~3x |
| 稀疏 10% | 252k | ~25k | ~5-10x |
