# Using the swing engine

The swing API models a whole-lot cash book. It leaves the existing `backtest`,
`backtest_with_report`, and wide APIs unchanged and never fetches data.

## Minimal example

```python
from datetime import date

import polars as pl
import polars_backtest as pbt

bars = pl.DataFrame({
    "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
    "code": ["EXAMPLE"] * 3,
    "open": [100.0, 100.0, 102.0],
    "high": [100.0, 102.0, 103.0],
    "low": [100.0, 99.0, 101.0],
    "close": [100.0, 101.0, 102.0],
    "limit_up": [110.0] * 3,
    "limit_down": [90.0] * 3,
})
targets = pl.DataFrame({
    "date": [date(2024, 1, 2)], "code": ["EXAMPLE"], "weight": [0.5],
})
result = pbt.backtest_swing(
    targets, bars, capital=1_000_000,
    costs=pbt.SwingCosts(fee_discount=0.2, hedge_cost=0.0),
    rules=pbt.SwingRules(execution="open", take_profit=0.1),
)
print(result.daily)
print(result.fills)
print(result.positions)
```

The decision follows the first close; the next session buys 5,000 shares.
`daily` includes equity, P&L, return, costs, long/short values, cash, dividends,
borrow cost, hedge P&L, notional, and cost. `fills` records dates, codes, signed
shares, prices, costs, and reasons. `positions` records nonzero end-of-day shares,
weighted entry prices, and marks. Empty results retain the same typed schemas.

## Inputs and execution

`code` must be a string, preserving leading zeros. Dates must be castable to
Polars Date. All prices, limits, and dividends must use consistent unadjusted
share units. Supply cash distributions in the optional `dividend` column;
missing/null dividends become zero. If a reference-price gap approximates a
cash distribution, convert it before calling the engine and avoid also applying
a total-return adjustment.

Targets are signed absolute weights without normalization. Sizing truncates
target-date equity * weight / close to 1,000-share lots. A present target date
replaces the whole pending book, including zero targets for omitted held names.
An explicit zero-weight row denotes an empty book. An absent target date keeps
pending orders. Target dates outside the bars' session calendar raise `ValueError`.
Missing auction prices/bars defer orders; missing closes retain prior marks.
The book has no cash-availability or margin constraint.

`borrowable`, when supplied, has `date`, `code`, and Boolean `can_borrow`.
Missing or false keys deny newly opened shorts; omitting the table allows them.
Borrow costs apply to overnight shorts over elapsed calendar days. Sales tax
applies to every negative share delta; the short surcharge applies only to newly
opened shorts. Fees apply to both sides with the configured discount.

Take-profit and stop rules use strict intraday crossings, gap-open fills,
stop-first resolution on ambiguous bars, adverse intraday exit slippage, and
limit-lock deferral. Open entries can exit that day; close entries cannot.

## Futures overlay and costs

`costs` is required and must be a `SwingCosts` instance. Its required
`hedge_cost` is the per-side futures cost (fraction of notional), supplied from
the caller's own cost model. Set it to zero for an unhedged book. For example,
a cost model returning 0.0002 per side can be passed as:

```python
costs = pbt.SwingCosts(fee_discount=0.2, hedge_cost=0.0002)
rules = pbt.SwingRules(hedge_ratio=1.0)
```

With a nonzero hedge ratio, also supply `hedge`: `date`, finite daily `ret`, and
Boolean `roll`. Returns must belong to the contract actually held. The overlay
tracks actual net stock value, earns P&L on the prior notional, pays twice on
rolls, and pays to resize. A zero hedge cost with a nonzero hedge ratio raises
`ValueError` unless `allow_zero_hedge_cost=True` explicitly enables an experiment.
The flag does not bypass other input checks. Keyword construction of
`SwingCosts` is recommended and supports Python 3.9.

## Selection and scheduling

`swing_books(scores, panel, days, weekday=0..4, top=N, kind="long"|"ls",
hold=rank, every=k, phase=p)` builds targets. Scores need `decision_date`,
`instrument_id` (`STK:code`), `score`, and timezone-aware `available_at`. The
panel needs `decision_date`, `instrument_id`, `top500`, `disposal`, `can_short`.
Each rebalance uses next-session scores available by 08:30 Asia/Taipei.

`hold` is a turnover rank buffer. Long ties use ascending instrument IDs;
short ties use descending IDs. Insufficient names causing overlap raise
`ValueError`. Weights stay fixed even with fewer eligible names. Calendar phases
count empty weeks so holidays do not shift the schedule. `bench_book` creates
one `0050` target on the first scheduled date; disable its exit rules for a
buy-and-hold benchmark.

## Development checks

From the repository root, run `cargo test -p btcore`. Build the complete extension
with `cd polars_backtest && uv run maturin develop`, then run
`uv run pytest tests/test_swing.py -q` from that directory.

When checking only the swing path, set `CARGO_TARGET_DIR` to an absolute build
output directory, then run
`uv run --no-sync python ../scripts/build_swing_binding.py --out ../target/swing-check`
from `polars_backtest`. Set `SWING_EXTENSION` to the resulting native library
and run the same Python tests. This compiles the unchanged production core and
PyO3 bridge; it does not verify the complete legacy extension or Arrow ABI.

`just test-swing-core` runs the same production Rust module's self-contained
rule tests with `rustc`. `tests/swing_import_smoke.py` checks public imports and
required cost construction on Python 3.9. Its optional `--stub-native` mode
isolates Python compatibility from legacy native exports and does not verify
the native ABI. All swing fixtures are synthetic.
