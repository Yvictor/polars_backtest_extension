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
- `get_stats(riskfree_rate=0.02)` - Returns DataFrame with all stats
- `get_monthly_stats(riskfree_rate=0.02)` - Returns DataFrame with monthly stats
- `get_return_table()` - Returns pivoted year/month return DataFrame
- `get_ndays_return(creturn_df, n)` - Static method for N-day return

**Tested Metrics**:
- ✓ cagr, total_return, max_drawdown, avg_drawdown, calmar
- ✓ daily_sharpe, daily_sortino, daily_mean, daily_vol, best_day, worst_day
- ✓ monthly_sharpe, monthly_sortino, monthly_mean, monthly_vol, best_month, worst_month
- ✓ win_ratio, return_table

---

## Stage 2: Structured Metrics

**Goal**: Implement `get_metrics()` for structured metrics dictionary

**Success Criteria**:
- Returns nested dict with profitability, risk, ratio, winrate sections
- Compatible with dashboard/API consumption

**Implementation**:
1. Implement `get_metrics(riskfree_rate=0.02)`:
   - `backtest`: startDate, endDate, feeRatio, taxRatio, freq
   - `profitability`: annualReturn, avgNStock, maxNStock
   - `risk`: maxDrawdown, avgDrawdown, avgDrawdownDays, valueAtRisk
   - `ratio`: sharpeRatio, sortinoRatio, calmarRatio, volatility, profitFactor, tailRatio
   - `winrate`: winRate, expectancy, mae, mfe
2. Add `metrics` cached_property returning self (for chained access)

**Dependencies**: Stage 1

**Tests**:
- [ ] `get_metrics()` returns all expected sections
- [ ] Nested values are correct types (float, str, etc.)
- [ ] Metrics match Finlab output for same data

**Status**: Not Started

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
