# Report Class Implementation Plan

## Overview

Implement missing features from Finlab's Report class into polars_backtest's Report class.
Reference: `polars_backtest/restore/restored_report.pyx`

## Current Status

### Already Implemented
- `creturn` - Cumulative return DataFrame
- `position` - Position weights DataFrame
- `trades` - Trade records DataFrame (with MAE/MFE/GMFE/BMFE/MDD/pdays)
- `fee_ratio`, `tax_ratio` - Fee settings

### Not Needed
- `upload()` - Finlab cloud upload (not applicable)
- `market` object - Finlab-specific market abstraction
- `live_performance_start` - Live trading specific

---

## Stage 1: Core Statistics

**Goal**: Implement basic performance statistics calculation

**Success Criteria**:
- `get_stats()` returns dict with CAGR, Sharpe, max_drawdown, win_ratio, return_table
- Results match Finlab's calculation within acceptable tolerance

**Implementation**:
1. Add `daily_creturn` property (resample to daily)
2. Implement `get_stats(resample='1d', riskfree_rate=0.02)`:
   - CAGR calculation
   - Sharpe ratio (daily/monthly)
   - Max drawdown
   - Win ratio from trades
   - Return table (yearly/monthly breakdown)
3. Add `get_ndays_return(creturn, n)` static method

**Tests**:
- [ ] `get_stats()` returns expected keys
- [ ] CAGR calculation matches expected value
- [ ] Sharpe ratio calculation is correct
- [ ] Max drawdown calculation is correct
- [ ] Win ratio matches trades data

**Status**: ✅ Complete

**Implemented Methods**:
- `daily_creturn` (cached_property) - Daily resampled cumulative return
- `get_stats(riskfree_rate=0.02)` - Returns DataFrame with 15 statistics
- `get_monthly_stats(riskfree_rate=0.02)` - Returns DataFrame with monthly stats
- `get_return_table()` - Returns pivoted year/month return DataFrame
- `get_drawdown_details(top_n=5)` - Returns top N drawdown periods
- `get_ndays_return(creturn_df, n)` - Static method for N-day return

**Tested Metrics** (18 metrics, all match Finlab within 1% tolerance):
- ✓ cagr, total_return, max_drawdown, avg_drawdown, calmar
- ✓ daily_sharpe, daily_sortino, daily_mean, daily_vol, best_day, worst_day
- ✓ monthly_sharpe, monthly_sortino, monthly_mean, monthly_vol, best_month, worst_month
- ✓ win_ratio, return_table, drawdown_details

**Implementation Notes**:
- All calculations use pure Polars expressions (`df.with_columns(...).select(...)`)
- `avg_drawdown`: Mean of per-period minimum drawdowns (Finlab compatible)
- `sortino`: Uses `std(min(excess, 0), ddof=1)` formula (Finlab compatible)
- `drawdown_details`:
  - start = first day with drawdown < 0
  - end = recovery day (first day back to >= 0)
  - length = calendar days from start to end
  - All values match Finlab exactly (diff < 1e-14)

---

## Stage 2: Structured Metrics

**Goal**: Implement `get_metrics()` for structured metrics dictionary

**Success Criteria**:
- Returns nested dict with profitability, risk, ratio, winrate sections
- Compatible with dashboard/API consumption

**Finlab get_metrics() Structure**:
```python
{
    "backtest": {
        "startDate": str,           # 開始日期
        "endDate": str,             # 結束日期
        "feeRatio": float,          # 手續費率
        "taxRatio": float,          # 交易稅率
        "freq": str,                # 資料頻率 (daily, minute 等)
        "tradeAt": str,             # 交易時機 (open/close)
        "stopLoss": float | None,   # 停損設定
        "takeProfit": float | None, # 停利設定
        "trailStop": float | None,  # 追蹤停損設定
        # 暫不實作:
        # "market", "expired", "version", "updateDate",
        # "nextTradingDate", "currentRebalanceDate", "nextRebalanceDate",
        # "livePerformanceStart"
    },
    "profitability": {
        "annualReturn": float,      # CAGR
        "avgNStock": float,         # 平均持股數
        "maxNStock": int,           # 最大持股數
        "alpha": float,             # Alpha (需要 benchmark)
        "beta": float,              # Beta (需要 benchmark)
    },
    "risk": {
        "maxDrawdown": float,       # 最大回撤
        "avgDrawdown": float,       # 平均回撤
        "avgDrawdownDays": float,   # 平均回撤天數
        "valueAtRisk": float,       # VaR (5% percentile)
        "cvalueAtRisk": float,      # CVaR (expected shortfall)
    },
    "ratio": {
        "sharpeRatio": float,       # 夏普比率
        "sortinoRatio": float,      # 索提諾比率
        "calmarRatio": float,       # 卡瑪比率
        "volatility": float,        # 年化波動率
        "profitFactor": float,      # 獲利因子 (總獲利/總虧損)
        "tailRatio": float,         # 尾部風險比率
    },
    "winrate": {
        "winRate": float,           # 勝率
        "m12WinRate": float,        # 12個月滾動勝率 (需要 benchmark)
        "expectancy": float,        # 期望值 (平均每筆交易報酬)
        "mae": float,               # 平均 MAE
        "mfe": float,               # 平均 MFE
    },
    "liquidity": {                  # 需要額外資料，可能需改 Rust
        "capacity": float,          # 策略容量
        "disposalStockRatio": float,    # 處置股比例 帶入 data 要有 這個flag 才需要算
        "warningStockRatio": float,     # 警示股比例 帶入 data 要有 這個flag 才需要算
        "fullDeliveryStockRatio": float, # 全額交割股比例 帶入 data 要有 這個flag 才需要算
        "buyHigh": float,           # 買在漲停比例 (漲停根本買不到 流動性問題) 沒辦法成交策略會有誤差
        "sellLow": float,           # 賣在跌停比例 (跌停根本賣不掉 流動性問題)
    }
}
```

**Implementation**:

Phase 2a - 核心指標 (可立即實作):
1. Implement `get_metrics(riskfree_rate=0.02, benchmark=None)` returning nested dict
2. Reuse Stage 1 calculations where possible
3. Add new calculations:
   - `avgNStock`, `maxNStock` - from position DataFrame
   - `avgDrawdownDays` - from drawdown_details
   - `valueAtRisk`, `cvalueAtRisk` - from daily return percentiles
   - `profitFactor` - sum(positive returns) / abs(sum(negative returns))
   - `tailRatio` - 95th percentile / abs(5th percentile)
   - `expectancy` - mean trade return
   - `mae`, `mfe` - mean from trades DataFrame

Phase 2b - Benchmark 相關 (略過，等 long format):
- Benchmark 應在資料層傳入（如 DataFrame 包含 0050 欄位）
- Wide format 不方便處理，等 long format 再實作
- 待實作: alpha, beta, m12WinRate

Phase 2c - Liquidity 相關 (略過，等 long format):
- 需要額外資料：成交量、處置/警示/全額交割股清單
- Wide format 不方便處理，等 long format 再實作
- 待實作: capacity, disposalStockRatio, warningStockRatio, fullDeliveryStockRatio, buyHigh, sellLow

**Dependencies**: Stage 1

**Tests**:
- [x] `get_metrics()` returns all expected sections
- [x] Nested values are correct types (float, str, etc.)
- [x] Metrics match Finlab output for same data
- [ ] Alpha/Beta calculation correct with benchmark (deferred to long format)
- [ ] Liquidity metrics correct when data available (deferred to long format)

**Status**: ✅ Complete (Phase 2a)

**Implemented**:
- `get_metrics(sections, riskfree_rate)` - Returns single-row DataFrame
- Uses Literal type for sections parameter (backtest, profitability, risk, ratio, winrate)
- Includes paper returns for open positions (like Finlab)
- Uses Polars native patterns: `daily_with_return.select(exprs)`
- Semi join for position filtering to creturn date range
- Join for paper returns calculation (no Python loop)

**Metrics implemented**:
- backtest: startDate, endDate, feeRatio, taxRatio, freq, tradeAt, stopLoss, takeProfit, trailStop
- profitability: annualReturn, avgNStock, maxNStock
- risk: maxDrawdown, avgDrawdown, avgDrawdownDays, valueAtRisk, cvalueAtRisk
- ratio: sharpeRatio, sortinoRatio, calmarRatio, volatility, profitFactor, tailRatio
- winrate: winRate, expectancy, mae, mfe

---

## Stage 3: Visualization

**Goal**: Implement Jupyter notebook display and performance charts

**Success Criteria**:
- `_repr_html_()` shows performance summary in Jupyter
- `display()` shows interactive Plotly charts
- Yearly/monthly return heatmaps work

**Implementation**:
1. Add `benchmark` and `daily_benchmark` properties (optional, default to None)
2. Implement `_repr_html_()`:
   - Generate HTML summary table
   - Embed performance chart
3. Implement `display(save_report_path=None)`:
   - Show performance figures using Plotly
4. Implement static methods:
   - `create_performance_figure(performance, nstocks)` - Main performance chart
   - `create_yearly_return_figure(return_table)` - Yearly heatmap
   - `create_monthly_return_figure(return_table)` - Monthly heatmap
5. Add `display_mae_mfe_analysis()` for MAE/MFE scatter plot

**Dependencies**: Stage 1 (for return_table)

**Tests**:
- [ ] `_repr_html_()` returns valid HTML string
- [ ] `display()` doesn't raise errors
- [ ] Charts render correctly with sample data

**Status**: Not Started

---

## Stage 4: Position Information

**Goal**: Implement current position and trade tracking

**Success Criteria**:
- `current_trades` shows active positions
- `position_info()` returns position dict for API

**Implementation**:
1. Add properties:
   - `weights` - Current weights Series
   - `next_weights` - Next period weights
   - `actions` - Trade actions (enter/exit/hold)
2. Implement `calculate_current_trades()`:
   - Find trades without exit or exiting today
   - Add weight columns
3. Implement `position_info()`:
   - Return dict with status, weight, entry_date, exit_date, return
4. Implement `position_info2()`:
   - Return detailed position info for dashboard
5. Add helper methods:
   - `is_rebalance_due()` - Check if rebalance needed
   - `is_stop_triggered()` - Check if stop was hit

**Dependencies**: None

**Tests**:
- [ ] `current_trades` filters correctly
- [ ] `position_info()` returns expected format
- [ ] `is_stop_triggered()` detects SL/TP trades

**Status**: Not Started

---

## Stage 5: Trade Data Enhancement

**Goal**: Allow adding extra columns to trades DataFrame

**Success Criteria**:
- `add_trade_info()` can add market_value, industry, etc.
- Works with Polars DataFrames

**Implementation**:
1. Implement `add_trade_info(name, df, date_col='entry_sig_date')`:
   - Join external data to trades by date and stock_id
   - Support multiple date columns
2. Implement `remove_trade_info(name)`:
   - Remove added columns
3. Add simple getters:
   - `get_trades()` - Return trades DataFrame
   - `get_mae_mfe()` - Return MAE/MFE subset

**Dependencies**: None

**Tests**:
- [ ] `add_trade_info()` adds columns correctly
- [ ] Handles missing data with null fill
- [ ] `remove_trade_info()` cleans up

**Status**: Not Started

---

## Stage 6: Serialization

**Goal**: Implement save/load functionality

**Success Criteria**:
- Can save Report to pickle/JSON/HTML
- Can load Report from pickle

**Implementation**:
1. Implement `to_pickle(file_path)`:
   - Serialize entire Report object
2. Implement `from_pickle(file_path)` classmethod:
   - Deserialize Report object
3. Implement `to_json()`:
   - Return dict with timestamps, strategy, metrics, trades (gzip compressed)
4. Implement `to_html(path)`:
   - Generate standalone HTML file with charts
5. Implement `to_text(name=None)`:
   - Generate text summary for notifications

**Dependencies**: Stage 1, Stage 3

**Tests**:
- [ ] Pickle round-trip preserves data
- [ ] JSON output is valid and parseable
- [ ] HTML file is viewable in browser

**Status**: Not Started

---

## Stage 7: Analysis Plugins (Optional)

**Goal**: Implement extensible analysis system

**Success Criteria**:
- `run_analysis()` can run different analysis types
- Supports Drawdown, MaeMfe analysis

**Implementation**:
1. Create analysis base class interface
2. Implement `run_analysis(analysis, display=True, **kwargs)`:
   - Load analysis module by name
   - Run analysis and return results
3. Implement built-in analyses:
   - DrawdownAnalysis - Largest drawdowns detail
   - MaeMfeAnalysis - MAE/MFE distribution analysis
4. Add `liquidity` cached_property (if needed)

**Dependencies**: Stage 1, Stage 3

**Tests**:
- [ ] `run_analysis("Drawdown")` returns drawdown details
- [ ] `run_analysis("MaeMfe")` returns MAE/MFE stats
- [ ] Custom analysis can be plugged in

**Status**: Not Started

---

## Implementation Order

```
Stage 1 (Core Statistics)
    |
    v
Stage 2 (Structured Metrics) -----> Stage 3 (Visualization)
    |                                    |
    v                                    v
Stage 4 (Position Info)           Stage 6 (Serialization)
    |
    v
Stage 5 (Trade Enhancement)
    |
    v
Stage 7 (Analysis Plugins) [Optional]
```

## Notes

- Use Polars for all DataFrame operations (not pandas)
- Maintain compatibility with existing `trades` property
- Consider lazy evaluation for expensive calculations
- Add optional dependencies (plotly) as extras in pyproject.toml
