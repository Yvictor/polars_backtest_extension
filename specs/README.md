# Specs - Polars Backtest Extension

本專案目標：使用 Rust + Polars 實作與 Finlab `backtest.sim()` 完全一致的回測引擎。

## 文件列表

| 文件 | 說明 |
|------|------|
| [01_FINLAB_BACKTEST_SPEC.md](01_FINLAB_BACKTEST_SPEC.md) | Finlab 回測引擎逆向分析 |
| [02_POLARS_RUST_EXTENSION_SPEC.md](02_POLARS_RUST_EXTENSION_SPEC.md) | Polars Rust Extension 實作規格 |
| [03_DATA_LOADING_AND_TESTING.md](03_DATA_LOADING_AND_TESTING.md) | 資料載入與回測測試指南 |
| [../BACKTEST_TEST_SPEC.md](../BACKTEST_TEST_SPEC.md) | 測試配置與驗證標準 |

## 架構概覽

```
┌─────────────────────────────────────────────────────────────┐
│           github.com/Yvictor/polars_backtest_extension      │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         │                                         │
         ▼                                         ▼
┌─────────────────────┐              ┌─────────────────────────┐
│      btcore         │              │    polars_backtest      │
│   (crates.io)       │◄─────────────│    (PyPI)               │
│                     │  path dep    │                         │
│ Pure Rust           │              │ Polars Expressions      │
│ 回測引擎            │              │ PyO3 + pyo3-polars     │
└─────────────────────┘              └─────────────────────────┘
```

## 驗證目標

對齊 `finlab.backtest.sim()` 輸出：

| 輸出 | 差異容忍度 |
|------|------------|
| creturn | < 1e-10 |
| position | < 1e-10 |
| trades.return | < 1e-6 |
| get_stats() | < 1e-6 |

## 開發環境

- **Rust**: 2024 edition
- **Python**: >= 3.9
- **Build**: uv + maturin
- **Test**: pytest + Finlab baseline comparison
