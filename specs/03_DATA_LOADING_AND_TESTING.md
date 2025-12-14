# 資料載入與回測測試指南

## 概述

本文檔說明如何載入自有資料並與 Finlab 回測結果進行交叉驗證。

## 資料來源

### S3 Parquet 資料結構

```
s3://trading-data/stock/TwStkPriceDaily/*.parquet
```

| 欄位 | 類型 | 說明 |
|------|------|------|
| `ymdOn` | Date | 交易日期 |
| `listCode` | String | 股票代號 (需 strip) |
| `closePrice` | Float64 | 收盤價 |
| `openPrice` | Float64 | 開盤價 |
| `highPrice` | Float64 | 最高價 |
| `lowPrice` | Float64 | 最低價 |
| `volume` | Int64 | 成交量 |

### 環境變數設定

```bash
# .env
S3_ACCESS_KEY_ID=your_access_key
S3_SECRET_ACCESS_KEY=your_secret_key
S3_ENDPOINT_URL=https://your-endpoint.com
```

## 資料載入 (Polars)

### 基本載入

```python
import polars as pl
import os
from dotenv import load_dotenv

load_dotenv()

def get_storage_options():
    return {
        "access_key_id": os.getenv("S3_ACCESS_KEY_ID"),
        "secret_access_key": os.getenv("S3_SECRET_ACCESS_KEY"),
        "endpoint_url": os.getenv("S3_ENDPOINT_URL"),
    }


def load_prices(
    start_date: str,
    end_date: str,
    symbols: list[str] | None = None,
) -> pl.DataFrame:
    """
    載入價格資料 (long format)

    Returns:
        DataFrame with columns: [date, symbol, close, open, high, low, volume]
    """
    storage_options = get_storage_options()

    query = (
        pl.scan_parquet(
            "s3://trading-data/stock/TwStkPriceDaily/*.parquet",
            storage_options=storage_options
        )
        .with_columns([
            pl.col("ymdOn").cast(pl.Date).alias("date"),
            pl.col("listCode").str.strip_chars(" ").alias("symbol"),
        ])
        .filter(
            (pl.col("date") >= pl.lit(start_date).str.to_date()) &
            (pl.col("date") <= pl.lit(end_date).str.to_date())
        )
        .filter(pl.col("symbol").str.len_chars() == 4)  # 只要 4 碼股票
    )

    if symbols:
        query = query.filter(pl.col("symbol").is_in(symbols))

    df = (
        query
        .unique(subset=["symbol", "date"])
        .sort("symbol", "date")
        .select([
            "date",
            "symbol",
            pl.col("closePrice").alias("close"),
            pl.col("openPrice").alias("open"),
            pl.col("highPrice").alias("high"),
            pl.col("lowPrice").alias("low"),
            pl.col("volume"),
        ])
        .collect()
    )

    return df
```

### 轉換為寬格式 (Finlab 相容)

```python
def to_wide_format(df: pl.DataFrame, value_col: str = "close") -> pl.DataFrame:
    """
    Long format → Wide format (Finlab 相容)

    Input:  [date, symbol, close, ...]
    Output: [date, 2330, 2317, ...]
    """
    return (
        df
        .pivot(
            index="date",
            on="symbol",
            values=value_col,
        )
        .sort("date")
    )


def to_pandas_finlab(df: pl.DataFrame, value_col: str = "close"):
    """
    轉換為 Finlab 格式的 pandas DataFrame

    Output: pd.DataFrame with DatetimeIndex, columns=symbols
    """
    import pandas as pd

    wide = to_wide_format(df, value_col)
    pdf = wide.to_pandas()
    pdf.set_index("date", inplace=True)
    pdf.columns.name = "symbol"

    return pdf
```

## 策略信號計算

### 使用 Polars (推薦)

```python
def compute_signals_polars(prices: pl.DataFrame) -> pl.DataFrame:
    """
    計算策略信號 (long format)

    策略: close > SMA(20)
    """
    return (
        prices
        .sort("symbol", "date")
        .with_columns([
            pl.col("close")
              .rolling_mean(window_size=20)
              .over("symbol")
              .alias("sma20"),
        ])
        .with_columns([
            (pl.col("close") > pl.col("sma20")).alias("signal"),
        ])
    )


def compute_position_wide(prices: pl.DataFrame) -> pl.DataFrame:
    """
    計算持倉信號 (wide format, bool)

    Output: DataFrame [date, 2330, 2317, ...] with bool values
    """
    signals = compute_signals_polars(prices)

    return (
        signals
        .pivot(
            index="date",
            on="symbol",
            values="signal",
        )
        .sort("date")
    )
```

### 使用 Pandas (Finlab 相容)

```python
def compute_signals_pandas(close_pdf):
    """
    使用 pandas 計算信號 (與 Finlab 相同方式)

    Input: pd.DataFrame (Finlab 格式)
    Output: pd.DataFrame of bool
    """
    sma20 = close_pdf.rolling(window=20).mean()
    position = close_pdf > sma20
    return position
```

## 回測執行

### 使用 Finlab (驗證基準)

```python
from finlab import backtest

def run_finlab_backtest(position_pdf, resample='M'):
    """
    執行 Finlab 回測

    Args:
        position_pdf: pandas DataFrame (bool), Finlab 格式
        resample: 再平衡頻率 ('M', 'W', 'Q', etc.)

    Returns:
        finlab Report object
    """
    report = backtest.sim(position_pdf, resample=resample)
    return report
```

### 使用 polars_backtest (我們的實作)

```python
import polars_backtest as pbt

def run_polars_backtest(prices: pl.DataFrame, signals: pl.DataFrame):
    """
    執行 Polars 回測 (待實作)

    Args:
        prices: 價格資料 (long format)
        signals: 策略信號 (long format)

    Returns:
        BacktestResult
    """
    # 計算每日報酬
    returns = (
        prices
        .sort("symbol", "date")
        .with_columns([
            pbt.daily_returns(pl.col("close")).over("symbol").alias("return"),
        ])
    )

    # 計算權重
    weighted = (
        signals
        .join(returns, on=["date", "symbol"])
        .with_columns([
            pbt.equal_weight(pl.col("signal")).over("date").alias("weight"),
        ])
    )

    # 計算投資組合報酬
    portfolio = (
        weighted
        .group_by("date")
        .agg([
            (pl.col("weight") * pl.col("return")).sum().alias("portfolio_return"),
        ])
        .sort("date")
    )

    # 計算累積報酬
    result = portfolio.with_columns([
        pbt.cumulative_returns(pl.col("portfolio_return")).alias("creturn"),
    ])

    return result
```

## 完整測試腳本

```python
"""
test_backtest_comparison.py

比較 Finlab 與 polars_backtest 結果
"""

import polars as pl
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# === 1. 載入資料 ===
print("1. 載入資料...")

START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

prices = load_prices(START_DATE, END_DATE)
print(f"   載入 {prices.height} 筆資料, {prices['symbol'].n_unique()} 檔股票")

# === 2. 轉換格式 ===
print("2. 轉換格式...")

close_pdf = to_pandas_finlab(prices, "close")
print(f"   Pandas 格式: {close_pdf.shape[0]} 天 × {close_pdf.shape[1]} 檔")

# === 3. 計算策略 ===
print("3. 計算策略信號...")

# Pandas 方式 (Finlab)
position_pdf = compute_signals_pandas(close_pdf)

# Polars 方式
signals = compute_signals_polars(prices)
position_pl = compute_position_wide(prices)

# 驗證信號一致性
position_pl_pdf = position_pl.to_pandas().set_index("date")
common_cols = position_pdf.columns.intersection(position_pl_pdf.columns)
common_dates = position_pdf.index.intersection(position_pl_pdf.index)

signal_diff = (
    position_pdf.loc[common_dates, common_cols].fillna(False) !=
    position_pl_pdf.loc[common_dates, common_cols].fillna(False)
).sum().sum()

print(f"   信號差異: {signal_diff} 個")
assert signal_diff == 0, "信號計算不一致!"

# === 4. 執行 Finlab 回測 ===
print("4. 執行 Finlab 回測...")

from finlab import backtest
finlab_report = backtest.sim(position_pdf, resample='M')

print(f"   creturn: {finlab_report.creturn.iloc[-1]:.6f}")
print(f"   trades: {len(finlab_report.trades)} 筆")

# === 5. 執行 Polars 回測 ===
print("5. 執行 Polars 回測...")

import polars_backtest as pbt

# 計算每日報酬
prices_with_returns = (
    prices
    .sort("symbol", "date")
    .with_columns([
        pbt.daily_returns(pl.col("close")).over("symbol").alias("daily_return"),
    ])
)

# 合併信號
backtest_data = (
    signals
    .select(["date", "symbol", "signal"])
    .join(
        prices_with_returns.select(["date", "symbol", "daily_return"]),
        on=["date", "symbol"],
    )
)

# 計算等權重
backtest_data = (
    backtest_data
    .with_columns([
        # 每日信號數量
        pl.col("signal").sum().over("date").alias("signal_count"),
    ])
    .with_columns([
        # 等權重
        pl.when(pl.col("signal"))
          .then(1.0 / pl.col("signal_count"))
          .otherwise(0.0)
          .alias("weight"),
    ])
)

# 計算投資組合每日報酬
portfolio_returns = (
    backtest_data
    .group_by("date")
    .agg([
        (pl.col("weight") * pl.col("daily_return").fill_null(0)).sum().alias("portfolio_return"),
    ])
    .sort("date")
)

# 計算累積報酬
portfolio_returns = portfolio_returns.with_columns([
    pbt.cumulative_returns(pl.col("portfolio_return")).alias("creturn"),
])

print(f"   creturn: {portfolio_returns['creturn'][-1]:.6f}")

# === 6. 比較結果 ===
print("6. 比較結果...")

# 對齊日期
finlab_creturn = finlab_report.creturn
our_creturn = portfolio_returns.to_pandas().set_index("date")["creturn"]

common_dates = finlab_creturn.index.intersection(our_creturn.index)
finlab_aligned = finlab_creturn.loc[common_dates]
our_aligned = our_creturn.loc[common_dates]

diff = abs(finlab_aligned - our_aligned)
max_diff = diff.max()

print(f"   共同日期: {len(common_dates)}")
print(f"   最大差異: {max_diff:.10f}")

if max_diff < 1e-10:
    print("   ✅ 驗證通過!")
else:
    print(f"   ❌ 驗證失敗 (差異 > 1e-10)")

    # 找出差異最大的位置
    worst_date = diff.idxmax()
    print(f"   最大差異日期: {worst_date}")
    print(f"   Finlab: {finlab_aligned.loc[worst_date]:.10f}")
    print(f"   Ours:   {our_aligned.loc[worst_date]:.10f}")

# === 7. 統計指標比較 ===
print("7. 統計指標比較...")

finlab_stats = finlab_report.get_stats()

# 計算我們的統計
our_returns = portfolio_returns["portfolio_return"].to_numpy()
our_creturn_arr = portfolio_returns["creturn"].to_numpy()

our_sharpe = pbt.sharpe_ratio(pl.Series(our_returns), rf=0.0, annualize=252.0)
our_max_dd = pbt.max_drawdown(pl.Series(our_creturn_arr))

print(f"   Sharpe (Finlab): {finlab_stats['daily_sharpe']:.6f}")
print(f"   Sharpe (Ours):   {our_sharpe:.6f}")
print(f"   Max DD (Finlab): {finlab_stats['max_drawdown']:.6f}")
print(f"   Max DD (Ours):   {our_max_dd:.6f}")
```

## 測試資料產生

### 產生 Finlab 基準資料

```python
"""
generate_baseline.py

產生 Finlab 回測基準資料供驗證使用
"""

import os
import json
import pickle

def generate_baseline(output_dir: str = "output/baseline"):
    os.makedirs(output_dir, exist_ok=True)

    # 載入資料並執行回測
    prices = load_prices("2015-01-01", "2025-01-01")
    close_pdf = to_pandas_finlab(prices, "close")
    position_pdf = compute_signals_pandas(close_pdf)

    from finlab import backtest
    report = backtest.sim(position_pdf, resample='M')

    # 儲存完整 report
    report.to_pickle(f"{output_dir}/finlab_report.pkl")

    # 儲存個別輸出
    report.creturn.to_pickle(f"{output_dir}/creturn.pkl")
    report.daily_creturn.to_pickle(f"{output_dir}/daily_creturn.pkl")
    report.position.to_pickle(f"{output_dir}/position.pkl")
    report.trades.to_pickle(f"{output_dir}/trades.pkl")

    # 儲存統計 (JSON)
    stats = report.get_stats()
    stats_json = {k: float(v) if isinstance(v, (int, float)) else str(v)
                  for k, v in stats.items()}

    with open(f"{output_dir}/stats.json", 'w') as f:
        json.dump(stats_json, f, indent=2)

    print(f"基準資料已儲存至: {output_dir}/")

    return report


if __name__ == "__main__":
    generate_baseline()
```

### 載入基準資料

```python
def load_baseline(baseline_dir: str = "output/baseline"):
    """載入 Finlab 基準資料"""
    import pickle
    import json

    with open(f"{baseline_dir}/creturn.pkl", 'rb') as f:
        creturn = pickle.load(f)

    with open(f"{baseline_dir}/position.pkl", 'rb') as f:
        position = pickle.load(f)

    with open(f"{baseline_dir}/trades.pkl", 'rb') as f:
        trades = pickle.load(f)

    with open(f"{baseline_dir}/stats.json", 'r') as f:
        stats = json.load(f)

    return {
        "creturn": creturn,
        "position": position,
        "trades": trades,
        "stats": stats,
    }
```

## pytest 測試範例

```python
# tests/python/test_against_finlab.py

import pytest
import polars as pl
import pandas as pd
import numpy as np

@pytest.fixture(scope="module")
def baseline():
    """載入 Finlab 基準資料"""
    return load_baseline("output/baseline")


@pytest.fixture(scope="module")
def prices():
    """載入價格資料"""
    return load_prices("2015-01-01", "2025-01-01")


class TestCreturn:
    """測試累積報酬計算"""

    def test_creturn_length(self, baseline, prices):
        """驗證 creturn 長度"""
        our_creturn = compute_our_creturn(prices)
        assert len(our_creturn) == len(baseline["creturn"])

    def test_creturn_values(self, baseline, prices):
        """驗證 creturn 數值"""
        our_creturn = compute_our_creturn(prices)
        finlab_creturn = baseline["creturn"]

        diff = abs(our_creturn.values - finlab_creturn.values).max()
        assert diff < 1e-10, f"creturn 差異: {diff}"


class TestStats:
    """測試統計指標"""

    @pytest.mark.parametrize("stat_name", [
        "total_return",
        "cagr",
        "max_drawdown",
        "daily_sharpe",
    ])
    def test_stat_value(self, baseline, prices, stat_name):
        """驗證統計指標"""
        our_stats = compute_our_stats(prices)
        finlab_stats = baseline["stats"]

        diff = abs(our_stats[stat_name] - float(finlab_stats[stat_name]))
        assert diff < 1e-6, f"{stat_name} 差異: {diff}"
```

## 執行測試

```bash
# 1. 產生基準資料 (只需執行一次)
uv run python generate_baseline.py

# 2. 執行測試
uv run pytest tests/python/ -v

# 3. 執行完整比較腳本
uv run python test_backtest_comparison.py
```
