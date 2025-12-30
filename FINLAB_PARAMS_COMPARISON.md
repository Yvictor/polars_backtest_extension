# Finlab backtest.sim() Parameters Comparison

## Parameter Status Overview

| Status | Count | Parameters |
|--------|-------|------------|
| ✅ Supported | 14 | position, resample, resample_offset, trade_at_price, position_limit, fee_ratio, tax_ratio, stop_loss, take_profit, trail_stop, touched_exit, retain_cost_when_rebalance, stop_trading_next_period |
| ❌ Missing | 9 | mae_mfe_*, fast_mode, name, upload, notification_enable, line_access_token, live_performance_start, market |

---

## ✅ Supported Parameters (14/23)

| Parameter | Finlab Default | Notes |
|-----------|----------------|-------|
| `position` | (required) | Position signals/weights |
| `resample` | None | D/W/M/Q/Y, W-FRI, W-MON, MS, QS |
| `resample_offset` | None | e.g., '1D', '-1D' |
| `trade_at_price` | 'close' | Trading execution price |
| `position_limit` | 1.0 | Max weight per stock |
| `fee_ratio` | 0.001425 | Transaction fee |
| `tax_ratio` | 0.003 | Transaction tax |
| `stop_loss` | None | Verified: max diff 2.22e-16 |
| `take_profit` | None | Verified: max diff 2.22e-16 |
| `trail_stop` | None | Formula: `(max_price - current_price) / entry_price` |
| `retain_cost_when_rebalance` | False | Keep entry price on rebalance |
| `stop_trading_next_period` | True | Block re-entry after stop |
| `touched_exit` | False | Use high/low for intraday stop detection |

**Additional Parameters (Not in Finlab):**

| Parameter | Purpose |
|-----------|---------|
| `open`, `high`, `low` | OHLC price DataFrames |
| `factor` | Price adjustment factor |
| `rebalance_indices` | Manual rebalance indices |

---

## ❌ Missing Parameters (9/23)

### MEDIUM Priority

| Parameter | Description |
|-----------|-------------|
| `mae_mfe_window` | MAE/MFE calculation window |
| `mae_mfe_window_step` | MAE/MFE window step |
| `fast_mode` | Skip detailed calculations |

### LOW Priority (Metadata/Service)

| Parameter | Description |
|-----------|-------------|
| `name` | Backtest name |
| `upload` | Upload to Finlab server |
| `notification_enable` | Enable notifications |
| `line_access_token` | LINE token |
| `live_performance_start` | Live trading start |
| `market` | Market data source |

---

## Implementation Priority

1. **`mae_mfe_*`** - Enhance trade analysis
2. **`fast_mode`** - Performance optimization
3. Metadata parameters - Can add as no-op
