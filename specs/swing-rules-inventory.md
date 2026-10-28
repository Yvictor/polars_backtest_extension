# Swing feature inventory

The swing API provides share and cash accounting alongside the existing
weight-based engine. Existing APIs and defaults remain unchanged.

| Capability | Public implementation | Synthetic coverage |
| --- | --- | --- |
| Signed 1,000-share lot sizing from signal-close equity | `btcore/src/swing.rs` | Signed truncation and next-session prices |
| Persistent targets and daily retries | `btcore/src/swing.rs` | Limit auctions, missing bars, book replacement |
| Take-profit, stop, gap fills, and exit slippage | `btcore/src/swing.rs` | Strict crossings, stop precedence, locked stops |
| Open/close sequencing and reversal cancellation | `btcore/src/swing.rs` | Same-day open exits and delayed close exits |
| Fees, sales tax, and short surcharge | `btcore/src/swing.rs` | Per-fill charges, reversals, short covers |
| Borrow permissions and calendar-day financing | `btcore/src/swing.rs` | Fill-day permission and overnight costs |
| Cash dividends and finite marks | `btcore/src/swing.rs` | Signed distributions and missing closes |
| Futures return, resize, and two-sided rolls | `btcore/src/swing.rs` | Actual exposure and caller-supplied costs |
| Typed transport and GIL release | `polars_backtest/src/swing.rs` | Production adapter with compiled binding |
| Frame validation and typed result schemas | `polars_backtest/python/polars_backtest/swing.py` | Duplicate keys, invalid inputs, empty output |
| Calendar-week phases and benchmark targets | `polars_backtest/python/polars_backtest/swing.py` | Empty weeks and each weekday |
| Top-N selection and rank retention | `polars_backtest/python/polars_backtest/swing.py` | Ties, eligibility, fixed weights, overlap rejection |
| Public Python 3.9 imports | `polars_backtest/tests/swing_import_smoke.py` | Required cost construction without newer dataclass options |

Rust rule fixtures live in `btcore/src/swing_tests.rs`; Python fixtures live in
`polars_backtest/tests/test_swing.py`. The optional isolated binding builder in
`scripts/build_swing_binding.py` uses the production sources without any datasets.
