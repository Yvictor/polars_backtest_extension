# Golden fixtures for finlab parity tests

Generated on 2026-07-04 with finlab 1.5.5, pandas 2.3.3,
polars 1.36.1, by `scripts/generate_golden_fixtures.py`.

These fixtures let `tests/test_golden_fixtures.py` verify that
`polars_backtest.backtest_with_report` reproduces finlab `backtest.sim`
results without a finlab account, API token, or network access.

## Data

- Symbols (20 liquid TW large caps): 2330, 2317, 2454, 2308, 2303, 2881, 2882, 2891, 2886, 2884, 1301, 1303, 1326, 2002, 2412, 3008, 2357, 2382, 3711, 1216
- Window: 2022-01-01 .. 2023-12-31 (signal warmup from 2021-06-01)
- Strategy: `close >= close.rolling(60).max()` on raw close,
  equal-weighted; all weights forced to 0 from 2023-11-01 so every trade
  closes before the window ends (finlab simulates to 'today', so a flat
  cash tail makes truncating its creturn at the window end lossless).
- Weights are also forced to 0 up to 2022-02-07: finlab's engine treats
  the FIRST resample date of the window as the simulation start and never
  trades on it (verified empirically: shifting the window start one month
  earlier makes finlab take the trade), while polars_backtest rebalances
  on it. A flat head keeps the first resample date at zero position in
  both engines, so the fixtures don't encode this boundary difference.
- Fees: fee_ratio=0.001425, tax_ratio=0.003 (passed explicitly to both engines)

## Files

- `golden_input.parquet`: long-format input
  (date, symbol, close=adjusted close, factor=adj/raw, raw_close, weight).
- `golden_<scenario>_creturn.parquet`: finlab creturn (date, creturn),
  truncated to the input window.
- `golden_<scenario>_trades.parquet`: finlab trades (stock_id, entry_date,
  exit_date, entry_sig_date, exit_sig_date, position, return);
  stock_id normalized to the bare ticker.

## Scenarios

| scenario | sim params | creturn rows | final creturn | trades |
|---|---|---|---|---|
| monthly | resample=M | 430 | 1.583057 | 22 |
| monthly_stop_loss | resample=M, stop_loss=0.1 | 430 | 1.592246 | 22 |
| weekly | resample=W | 463 | 1.047754 | 128 |
| monthly_offset_5d | resample=M, resample_offset=5D | 429 | 0.969854 | 25 |

## Regenerating

```bash
# from an environment with finlab + FINLAB_API_TOKEN (e.g. via .env)
python scripts/generate_golden_fixtures.py
```

## Known engine differences (accounted for by the parity test)

1. First resample date: finlab treats the first resample date of the
   window as the simulation start and never trades on it, while
   polars_backtest rebalances on it. The fixtures avoid this boundary
   by keeping weights at 0 up to 2022-02-07 (see above).
2. Trade-level `return` fee convention for rolled trades: when a stock
   stays in the portfolio across a rebalance, both engines split the
   trade record at the rebalance date, but polars_backtest charges the
   full round trip (entry fee, exit fee + tax) on every record, while
   finlab omits the entry fee on records that start as a roll and the
   exit fee + tax on records that end as a roll:
   `finlab_r + 1 = (ours_r + 1) / (1-fee)^entry_rolled / (1-fee-tax)^exit_rolled`.
   The parity test asserts this exact relationship within 1e-6.
   Portfolio-level creturn is identical (max abs diff < 1e-6); only the
   per-trade fee attribution differs.

Otherwise all scenarios match finlab within the tolerances asserted in
`tests/test_golden_fixtures.py` (creturn max abs diff < 1e-6, trade
count/exit dates exact, trade returns within 1e-6 after the roll fee
adjustment). If a future engine change breaks a scenario, document the
discrepancy here instead of loosening tolerances silently.
