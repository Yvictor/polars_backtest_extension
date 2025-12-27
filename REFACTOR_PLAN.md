# Backtest Engine Refactor Plan

## Goal

1. **模組化 simulation.rs**：從 2835 行拆分成可維護的模組
2. **實現 Long Format 直接回測**：避免 pivot/partition_by 轉換開銷
3. **統一數據結構**：消除 simulation.rs 與其他模組的重複定義

---

## 現狀分析

### 代碼行數

| 文件 | 行數 | 職責 |
|------|------|------|
| simulation.rs | 2835 | 回測核心（過大！） |
| weights.rs | 334 | 權重正規化 |
| trades.rs | 299 | TradeRecord/TradeBook (未被 simulation 使用) |
| stats.rs | 299 | 統計計算 |
| rebalance.rs | 187 | 基本 rebalance (未被 simulation 使用) |
| returns.rs | 166 | 收益率計算 |
| portfolio.rs | 132 | PortfolioState (未被 simulation 使用) |
| lib.rs | 35 | 導出 |

### 問題

1. **simulation.rs 內部重複定義**：
   - 自己定義了 `Position`, `PortfolioState`（與 portfolio.rs 重複）
   - 自己定義了 `TradeRecord`, `TradeTracker`（與 trades.rs 概念重複）

2. **simulate_backtest() 過長**：~400 行，包含 finlab_mode 和 standard_mode 兩條分支

3. **輔助函數散亂**：止損檢測、價格更新、rebalance 執行都在同一文件

---

## 目標結構

```
btcore/src/
├── lib.rs                    # 對外 API 導出
├── config.rs                 # BacktestConfig (從 simulation.rs 抽取)
├── position.rs               # Position struct + 更新邏輯 (新建)
├── portfolio.rs              # PortfolioState (重構，統一)
├── tracker.rs                # TradeTracker trait + 實現 (從 simulation.rs 抽取)
├── stops.rs                  # 止損檢測邏輯 (新建)
├── simulation/
│   ├── mod.rs                # 公共接口
│   ├── wide.rs               # Wide format 模擬 (現有邏輯)
│   └── long.rs               # Long format 模擬 (新增！)
├── rebalance.rs              # Rebalance 執行 (擴充)
├── weights.rs                # 權重正規化 (保持)
├── trades.rs                 # TradeRecord 輸出格式 (保持)
├── returns.rs                # 收益率計算 (保持)
└── stats.rs                  # 統計計算 (保持)
```

---

## 階段計劃

### Stage 1: 抽取 Config 和 Position ✅ COMPLETED

**目標**：將獨立的數據結構抽取到單獨文件

**任務**：
- [x] 1.1 創建 `config.rs`：移動 `BacktestConfig`
- [x] 1.2 創建 `position.rs`：移動 `Position` struct
- [x] 1.3 更新 simulation.rs 使用新模組
- [x] 1.4 運行測試確保無破壞

**成功標準**：`cargo test` 全部通過 ✅

**實際改動**：
- 新增 `config.rs` (87 行，含測試)
- 新增 `position.rs` (162 行，含測試)
- simulation.rs 減少 ~85 行 (2835 → 2750)

---

### Stage 2: 抽取 TradeTracker ✅ COMPLETED

**目標**：將 trade tracking 邏輯獨立

**任務**：
- [x] 2.1 創建 `tracker.rs`：移動 `TradeTracker` trait + `NoopTracker` + `RealTracker`
- [x] 2.2 移動 `TradeRecord`（simulation 版本）到 tracker.rs
- [x] 2.3 更新 simulation.rs 引用
- [x] 2.4 運行測試

**成功標準**：`cargo test` 全部通過 ✅

**實際改動**：
- 新增 `tracker.rs` (377 行，含測試)
- simulation.rs 減少 ~273 行 (2752 → 2479)

---

### Stage 3: 抽取止損檢測 ✅ COMPLETED

**目標**：將止損邏輯獨立成模組

**任務**：
- [x] 3.1 創建 `stops.rs`
- [x] 3.2 移動 `detect_stops()`
- [x] 3.3 移動 `detect_stops_finlab()`
- [x] 3.4 移動 `detect_touched_exit()`
- [x] 3.5 移動 `TouchedExitResult`
- [x] 3.6 更新 simulation.rs 引用
- [x] 3.7 運行測試

**成功標準**：`cargo test` 全部通過 ✅

**實際改動**：
- 新增 `stops.rs` (561 行，含測試)
- simulation.rs 減少 ~341 行 (2479 → 2138)

---

### Stage 4: 重構 PortfolioState ✅ COMPLETED

**目標**：統一 PortfolioState，合併到 portfolio.rs

**任務**：
- [x] 4.1 擴展 `portfolio.rs` 支持 simulation 需要的功能
- [x] 4.2 移動 `update_max_prices()`, `update_previous_prices()`, `update_market_values()`
- [x] 4.3 移動 `balance()`, `balance_finlab()`, `total_cost_basis()`
- [x] 4.4 刪除 simulation.rs 中的重複定義
- [x] 4.5 運行測試

**成功標準**：`cargo test` 全部通過 ✅

**實際改動**：
- 重寫 `portfolio.rs` (133 → 367 行)
  - 新增 PortfolioState { cash, positions }
  - 新增 balance(), balance_finlab(), total_cost_basis()
  - 新增 update_max_prices(), update_previous_prices(), update_market_values()
  - 保留 normalize_weights(), apply_position_limit() 工具函數
- simulation.rs 減少 ~185 行 (2138 → 1953)

---

### Stage 5: 拆分 simulation 模組 ✅ COMPLETED

**目標**：將 simulation.rs 拆分成 simulation/ 目錄

**任務**：
- [x] 5.1 創建 `simulation/mod.rs`：公共 API
- [x] 5.2 創建 `simulation/wide.rs`：現有 `simulate_backtest()` 邏輯
- [x] 5.3 移動 `run_backtest()`, `run_backtest_with_trades()`
- [x] 5.4 移動輔助函數（rebalance 相關）
- [x] 5.5 更新 lib.rs 導出
- [x] 5.6 運行測試

**成功標準**：`cargo test` 全部通過，API 不變 ✅

**實際改動**：
- 新增 `simulation/mod.rs` (26 行，公共 API 導出)
- 新增 `simulation/wide.rs` (1534 行，從 simulation.rs 移入)
- 刪除 `simulation.rs` (1953 行)
- lib.rs 無需變動（自動指向 simulation/mod.rs）

---

### Stage 6: 實現 Long Format 回測 (核心目標) ✅ COMPLETED

**目標**：新增 long format 直接回測，避免 pivot/partition_by

**任務**：
- [x] 6.1 創建 `simulation/long.rs`
- [x] 6.2 定義 Long Format 輸入結構：
  ```rust
  pub struct LongFormatInput<'a> {
      pub date_indices: &'a [u32],   // 已按 date 排序
      pub symbol_ids: &'a [u32],
      pub prices: &'a [f64],
      pub weights: &'a [f64],
  }
  ```
- [x] 6.3 實現單次掃描回測核心：
  ```rust
  pub fn backtest_long(
      input: &LongFormatInput,
      n_dates: usize,
      n_symbols: usize,
      rebalance_mask: &[bool],
      config: &BacktestConfig,
  ) -> BacktestResult
  ```
- [x] 6.4 復用 Stage 1-4 抽取的模組（Position, stops, portfolio）
- [x] 6.5 添加單元測試（5 個 long format 測試）
- [x] 6.6 添加 vs wide format 的一致性測試（4 個測試）

**成功標準**：
- ✅ Long format 結果與 wide format 一致（誤差 < 1e-10）
- 效能提升待 Stage 7 Python 綁定後測量

**實際改動**：
- 新增 `simulation/long.rs` (643 行，含測試)
- 更新 `simulation/mod.rs` 導出 long module
- 新增 4 個整合測試 (test_long_vs_wide_*)
- 總測試：88 單元 + 23 整合 = 111 通過

---

### Stage 7: Python 綁定

**目標**：在 polars_backtest 暴露 long format API

**任務**：
- [ ] 7.1 在 `polars_backtest/src/lib.rs` 添加 `backtest_long()` 函數
- [ ] 7.2 接收四個 numpy array（zero-copy）
- [ ] 7.3 更新 Python namespace.py 使用新 API
- [ ] 7.4 添加 Python 測試
- [ ] 7.5 效能比較測試

**成功標準**：
- Python 測試通過
- 與 finlab 結果一致
- 效能提升可測量

---

### Stage 8: 清理與優化

**目標**：最終清理，統一 API

**任務**：
- [ ] 8.1 移除 simulation.rs（已被拆分）
- [ ] 8.2 統一錯誤處理
- [ ] 8.3 補充文檔
- [ ] 8.4 效能優化（如有需要）
- [ ] 8.5 更新 CLAUDE.md

**成功標準**：
- 代碼覆蓋率 > 80%
- 文檔完整
- 無 clippy 警告

---

## 數據結構設計

### Long Format 回測核心

```rust
// position.rs
pub struct Position {
    pub value: f64,              // 持倉價值
    pub entry_price: f64,        // 入場價格
    pub stop_entry_price: f64,   // 止損計算用入場價
    pub max_price: f64,          // 最高價（trailing stop）
    pub last_market_value: f64,  // 市值
    pub cr: f64,                 // 累積收益率
    pub maxcr: f64,              // 最大累積收益率
    pub previous_price: f64,     // 前一日價格
}

// simulation/long.rs
pub fn backtest_long(
    date_indices: &[u32],        // 已排序
    symbol_ids: &[u32],
    prices: &[f64],
    weights: &[f64],
    n_dates: usize,
    rebalance_dates: &HashSet<u32>,
    config: &BacktestConfig,
) -> BacktestResult {
    let mut portfolio = PortfolioState::new();
    let mut creturn = Vec::with_capacity(n_dates);

    let mut i = 0;
    for date_idx in 0..n_dates as u32 {
        // 收集當日數據（連續 rows，O(k) where k = stocks on this day）
        let day_start = i;
        while i < date_indices.len() && date_indices[i] == date_idx {
            i += 1;
        }
        let day_end = i;

        // 當日價格 HashMap
        let today_prices: HashMap<u32, f64> = ...;
        let today_weights: HashMap<u32, f64> = ...;

        // 更新持倉
        portfolio.update_positions(&today_prices);

        // 止損檢測
        let stops = detect_stops_sparse(&portfolio, &today_prices, config);

        // Rebalance
        if rebalance_dates.contains(&date_idx) {
            portfolio.rebalance_sparse(&today_weights, &today_prices, config);
        }

        // 記錄 creturn
        creturn.push(portfolio.balance_sparse(&today_prices));
    }

    BacktestResult { creturn, trades: vec![] }
}
```

---

## 效能預估

| 場景 | Wide Format | Long Format | 預估提升 |
|------|-------------|-------------|---------|
| 1000 stocks, 252 days | pivot + 遍歷 252k cells | 掃描 ~50k events | ~3x |
| 稀疏數據 (10% 活躍) | 仍遍歷 252k | 掃描 ~25k | ~5-10x |
| 記憶體 | O(n_dates × n_stocks) | O(n_events) | ~10x 減少 |

---

## 測試策略

1. **每個 Stage 後**：`cargo test` 全部通過
2. **Stage 6 後**：Long vs Wide 一致性測試
3. **Stage 7 後**：Python vs Finlab 一致性測試
4. **效能測試**：使用 `criterion` benchmark

---

## 風險與緩解

| 風險 | 緩解措施 |
|------|----------|
| 重構破壞現有功能 | 每階段運行完整測試套件 |
| Long format 與 Wide 結果不一致 | 添加 property-based testing |
| 效能未達預期 | 先實現正確性，再優化 |
| Python 綁定複雜 | 使用 PyO3 arrow 支持 zero-copy |

---

## 依賴關係

```
Stage 1 (Config, Position)
    ↓
Stage 2 (TradeTracker)
    ↓
Stage 3 (Stops) ←──────────┐
    ↓                      │
Stage 4 (Portfolio) ───────┤
    ↓                      │
Stage 5 (Split simulation) │
    ↓                      │
Stage 6 (Long Format) ─────┘ (復用 1-4 的模組)
    ↓
Stage 7 (Python)
    ↓
Stage 8 (Cleanup)
```

---

## 開始

從 **Stage 1** 開始，創建 `config.rs` 和 `position.rs`。
