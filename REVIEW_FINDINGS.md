# Deep Code Review Findings — 2026-07-04

> **Fix status (updated 2026-07-04, same session):**
> - **H1 FIXED** — `finlab_mode` removed from the long-format Python API
>   (`backtest`/`backtest_with_report`); the long path always uses finlab-style
>   accounting and now says so. Wide API keeps the flag (it works there).
> - **H2 FIXED (as designed trade-off)** — negative/unparseable `resample_offset`
>   now raises `ValueError` in both Python and Rust layers instead of being
>   silently ignored. Negative offsets remain unsupported in the long engine by
>   deliberate choice; the wide API supports them.
> - **H3 FIXED** — namespace validates date/symbol for nulls (raise) and converts
>   nulls in price/OHLC/factor/position to NaN (the engine's missing marker)
>   before FFI; the Rust layer independently rejects date/symbol nulls.
> - **M4 FIXED** — `MS`/`QS` now raise `ValueError` in the long format instead of
>   silently rebalancing at period END.
> - **M7 FIXED** — `skip_sort=true` with an unsorted date column now errors in
>   the Rust layer instead of producing garbage.
> - **P-1 FIXED** — Python pinned to 3.13 (`.python-version`), unused
>   `yfinance` dev dep and unused `pyarrow` runtime dep removed; `uv sync`,
>   fast pytest suite (140 passed) and `cargo test` all green.
> - Also fixed (from capacity research, Stage A): `trading_value` column-name
>   parameter is now forwarded by the namespace (the `.pyi` already advertised
>   it); limit/trading-value frames are deduped on `(date, symbol)` before the
>   liquidity joins. Tests: `tests/test_input_validation.py`.
> - **M5 FIXED** — `btcore::sortino_ratio` aligned with report.rs (ffn-style
>   clamped std ddof=1; affects the `pl_bt.sortino_ratio` expression);
>   `TradeBook::stats().profit_factor` is now sum-based, old avg-based value
>   kept as `payoff_ratio`. Hand-computed unit tests added.
> - **M6 DOCUMENTED** — sparse-weights + `resample="D"` liquidation footgun
>   written into the skill (SKILL.md pitfalls + execution-semantics.md).
> - **Capacity Stages B & C DONE** — direction-aware epsilon-tolerant
>   buyHigh/sellLow; `report.capacity(method="finlab"|"min_leg"|"adv")` and
>   `report.capacity_by_date()`; default get_metrics capacity unchanged
>   (finlab byte-compatible). Tests: `tests/test_capacity_methods.py`.
> - **Golden fixtures DONE** — committed 20-symbol × 2y fixtures +
>   `tests/test_golden_fixtures.py` (fast, no finlab needed): creturn parity at
>   machine precision (≤6.7e-16) for M / M+stop_loss / W / M+offset5D. Two real
>   engine differences documented in `tests/fixtures/README.md`: finlab skips
>   trading on the first resample date of the window; rolled-trade fee
>   attribution differs in trade records (creturn unaffected, exact
>   relationship asserted).
> - Wide-format findings **M1, M2, M3 WON'T FIX** — the wide path is frozen;
>   the framework core is the long format (owner decision 2026-07-04).
> - **L5 FIXED** — tail_ratio/profit_factor return NaN (not +inf) when inputs
>   are NaN. **L6 FIXED** — `verify_ffi_compatibility()` now runs at module
>   init. **L7 FIXED** — the stops.rs FP-precision trick is guarded with
>   `std::hint::black_box`; golden-fixture parity still at machine precision,
>   confirming bit-compatibility is preserved.
> - **L4 WON'T FIX (finlab-compatible by design)** — the first-value rebase of
>   creturn matches finlab's behavior: the golden fixtures pass at machine
>   precision *with* the rebase, so "fixing" it would break parity.
> - Open: L1 (window>holding fee indexing — unreachable via defaults),
>   L2 (dead code + stale comments), L3 (W-MON + offset anchor),
>   L8 (period units documented in the skill; wide frozen).

Adversarial review of `btcore` + `polars_backtest` (Rust engine, pyo3 bindings,
Python layer, tests, packaging). `cargo test -p btcore` passes (101+19 tests).
Suspected engine bugs were verified by driving `btcore` directly from a scratch
crate. The Python test suite could not run on this machine (see P-1).

## HIGH

### H1. `finlab_mode` is silently ignored by the long-format engine; defaults differ per entry point
- `btcore/src/simulation/long.rs`: `backtest_impl` never reads `config.finlab_mode`
  (only `#[cfg(test)]` configs mention it, lines 2105–2219). The long path always
  runs finlab-style accounting.
- `namespace.py:106` `backtest(..., finlab_mode=False)` vs `namespace.py:258`
  `backtest_with_report(..., finlab_mode=True)`. The wide path
  (`lib.rs:1376`, default `false`) genuinely changes behavior — same flag,
  different meaning per API.
- Fix: remove the flag from the long API (or hard-error on `False`), unify
  defaults across `backtest` / `backtest_with_report` / `backtest_wide` and the
  Rust `config=None` fallbacks (`lib.rs:547-552` vs `lib.rs:1376,1433`).

### H2. Negative `resample_offset` is silently dropped in the long path
- `long.rs:147-174`: `ResampleOffset::from_str("-1D")` returns `None` → no offset,
  no error. `namespace.py` does not validate; the wide path (`utils.py:parse_offset`)
  *does* support `-1D` and is finlab-verified.
- `pb.backtest(df, resample="M", resample_offset="-1D")` rebalances at month end —
  silently different from finlab and from `backtest_wide` with identical args.
- Fix: support negative offsets in `ResampleOffset` or raise `ValueError` on
  unparseable/negative strings. `ResampleOffset::new()` also clamps negatives
  with `days.max(0)` silently.

### H3. Arrow validity bitmaps (nulls) ignored across the FFI boundary
- `long.rs:803-815`, `lib.rs:695-702, 1076-1083`: every array read uses
  `.value(i)`, never `is_null(i)`. Python fills nulls only for the position
  column (`namespace.py:173-182`); nulls in price/date/symbol/OHLC/factor pass
  through as undefined buffer bytes.
- A null `close` (suspension day — normal polars representation) can surface as
  stale nonzero garbage; a null `date` corrupts day grouping with no error.
- Fix: reject nulls in date/symbol, `fill_null(NaN)` prices/OHLC before FFI, or
  check validity in accessor closures.

## MEDIUM

### M1. Standard mode silently ignores short positions; stop logic inverted for shorts
- Verified: weight `-0.5` with `finlab_mode:false` → position never opened, no
  warning (`wide.rs:733-877` only buys positive diffs). `detect_stops`
  (`stops.rs:32-76`) has no direction awareness: for shorts, take_profit fires on
  loss and stop_loss on profit.
- Fix: error on negative weights in standard mode or implement short handling as
  in `detect_stops_finlab`.

### M2. `calculate_target_weights` / `apply_position_limit` violate `position_limit`
- `weights.rs:120-190` + duplicate `portfolio.rs:222-244`. Verified:
  `calculate_target_weights([true,true], [], 0.3)` → `[0.5, 0.5]`. Cap→renormalize→cap
  oscillates; when `n*limit < 1` it exits after 100 iterations in the violating
  state. End-to-end results are rescued only because `normalize_weights_finlab`
  re-clamps later — but both functions are public btcore API.
- Fix: after `min(1/n, limit)` do not renormalize back to 1.0; delete the
  oscillating loop and the dead duplicate in `portfolio.rs`.

### M3. Wide-format `touched_exit` trade records use close price, not touched price
- `wide.rs:117-119` passes `trade_prices[t]` to `tracker.close_trade`; the
  `exit_ratio` applied to cash (line 106) never reaches the record. Long path is
  correct (`long.rs:1701-1702`). Touched stop at open 88 / close 95 → cash exits
  ~88 but the trade shows exit 95. Wide touched-exit tests compare creturn only.
- Fix: multiply by `touched.exit_ratio` as in the long path.

### M4. `MS`/`QS` resample are aliases of `M`/`Q`, not month/quarter start
- `long.rs:449-466, 489-492`: `Monthly | MonthStart` share one arm keyed on the
  month-end boundary. `namespace.py:185-197` and `FINLAB_PARAMS_COMPARISON.md:17`
  advertise `MS`/`QS`. The wide Python path (`utils.py:183-231`) implements
  `1mo_start` correctly — another wide/long divergence.
- Fix: implement real start-of-period boundaries or reject `MS`/`QS` in the long path.

### M5. Metric definitions inconsistent between btcore and the report layer
- `trades.rs:169-173`: `profit_factor = avg_win/|avg_loss|` (that's payoff ratio);
  `report.rs:1160-1199` uses the correct `sum(pos)/|sum(neg)|`.
- `stats.rs:55-83` sortino ≠ `report.rs:234-240` / `wide.py:563-571` (ffn-style).
  The polars expression `sortino_ratio` exposes the btcore one.
- Fix: one definition per metric (report.rs matches finlab/ffn); delete or rename
  the others.

### M6. Long-format: missing/NaN weight row means "exit", not "no signal"
- `long.rs:586-588` drops NaN/zero weights; under `resample="D"` (the default)
  `active_weights` is replaced wholesale daily (`long.rs:424-431`), so a symbol
  absent for one day is fully sold the next. Doc comment (`long.rs:215`,
  "NaN = no signal") suggests the opposite; wide path forward-fills
  (`wide.py:1328-1335`), long does not.
- Fix: document loudly and/or make Daily forward-fill semantics explicit.

### M7. `skip_sort=true` / direct Rust calls with unsorted data → silent garbage
- `long.rs:281` treats any date *change* (including decreases) as a new day;
  `lib.rs:576-585` trusts `skip_sort` blindly.
- Fix: cheap monotonicity check inside the loop (`date < current_date` → error).

## LOW

- **L1** `mae_mfe.rs:183-224`: with `window > holding period`, exit fee lands on
  the wrong element (`exit_max` vs `exit_w` indexing). Default `window=0` unaffected.
- **L2** `long.rs:376-382`: `let delayed_triggered = false;` makes STEP 3's skip
  branch dead code; comments describe logic that no longer exists.
- **L3** `long.rs:1317-1332`: `get_all_period_boundaries` uses Sunday boundaries
  for `WeeklyOn(weekday)` too → `W-MON` + offset anchors wrong.
- **L4** `lib.rs:849-858, 1305-1314`: creturn is rebased by its first value, which
  hides a first-day fee from total_return/CAGR.
- **L5** `report.rs:1210-1214`: `calc_tail_ratio` / `calc_profit_factor` return
  `+inf` on NaN quantiles/sums; should be NaN.
- **L6** `ffi_convert.rs:32-33`: `transmute` between polars-arrow and arrow-rs FFI
  structs; `verify_ffi_compatibility()` exists but is never called.
- **L7** `stops.rs:121`: `cr * current_price / current_price` FP-precision trick
  for finlab bit-compat — guard with `std::hint::black_box` or it may be optimized
  away.
- **L8** `TradeRecord.period` is calendar days (long) vs trading-day rows (wide) —
  same column name, different units.

## Test quality

1. **All finlab reference-value tests are unrunnable by default**:
   `test_wide_vs_finlab.py` / `test_long_vs_wide.py` are `slow`-marked (excluded
   by `addopts`), require `FINLAB_API_TOKEN` + live downloads. No committed golden
   fixtures. Recommendation: snapshot a small real dataset (20 symbols × 2 years,
   parquet) + finlab creturn/trades outputs as fixtures; assert in the fast suite.
2. `test_trades_match` asserts only trade *count* — entry/exit prices and returns
   are never compared.
3. Untested reachable behaviors: negative offsets (H2), `MS`/`QS` (M4), wide
   touched-exit prices (M3), shorts in standard mode (M1), `position_limit` via
   btcore bool API (M2), null columns (H3), duplicate `(date,symbol)` rows (last
   silently wins), unknown resample strings silently → Daily
   (`ResampleFreq::from_str`), `skip_sort` misuse (M7).

## API / performance suggestions

- Unify `benchmark` semantics: Rust `backtest_with_report` treats a str as a
  *column name* (`lib.rs:1183-1195`); the namespace treats it as a *symbol value*.
- `ResampleFreq::from_str` / `ResampleOffset::from_str` should return `Result`.
- Wide path perf: `rebalance_indices.contains(&t)` is O(R)/day; daily
  `prev_prices` clone; `run_backtest_with_trades` re-gathers full per-stock price
  series per trade — transpose once.
- `tracker.rs:948-953`: `record_price(sym, price, price)` stores two identical
  Vecs per open trade.
- `report.rs:1142-1156` `calc_position_stats` is O(dates × trades) — sort ranges
  and sweep.
- Dead/duplicated code: `portfolio.rs` weight fns; `balance_finlab`'s `_prices`;
  unused `rayon`/`speedate`/`thiserror`/`serde` deps in btcore.
- `wide.py` `Report.get_stats` reimplements report.rs stats in Python — two
  sources of truth already drifting (`first_signal_index` handling).

## Packaging / build

- **P-1 (blocks dev):** `.python-version` pins `python3.14t` (free-threaded); dev
  dep `yfinance` → `curl-cffi` has no 3.14t wheel → `uv sync` / `just test` /
  `just build` fail. Pin stable CPython (3.12/3.13) or move yfinance to an extra.
- **P-2:** `polars_backtest/Cargo.toml` lacks `license`/`license-file` (btcore
  inherits from workspace; the plugin crate doesn't) — must agree with pyproject's
  PolyForm Noncommercial.
- **P-3:** pyproject hard-depends on `pyarrow>=21.0.0` but nothing imports it —
  drop if truly unused (~45 MB install).
- **P-4:** `just ci` = `check test-rust build test` — the uv half fails via P-1;
  clippy/fmt exist but aren't in `ci`.
