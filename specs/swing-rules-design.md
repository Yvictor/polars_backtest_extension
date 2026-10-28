# Swing cash-book execution

## Public interface

`btcore::swing` and `polars_backtest.backtest_swing` provide opt-in whole-share
cash accounting. `SwingCosts` and `SwingRules` configure execution; `SwingResult`
contains daily accounting, fills, and end-of-day positions. Existing weight-based
backtest, report, long, and wide APIs retain their defaults.

The Python adapter validates Polars frames, maps string symbols to stable integer
IDs, and passes typed rows to PyO3. The Rust core owns all event ordering and
accounting and uses only the standard library. No data is loaded implicitly.
The isolated binding builder compiles the same production sources when only the
new API needs checking; it does not verify the complete extension.

## State and daily order

State consists of cash, signed shares, weighted entry prices, last finite marks,
pending absolute share targets, overnight hedge notional, and prior equity/date.
Stable symbol ordering makes execution repeatable.

1. Credit signed shares times the supplied cash dividend. Charge overnight
   shorts the previous mark times annual borrow rate times calendar days / 365.
2. For open execution, retry pending targets. Check raw auction prices against
   limits before applying slippage. Buys/covers at the upper limit and
   sales/shorts at the lower limit remain pending. Only newly opened short
   shares need fill-day borrow permission. A denied reversal can close a long
   while leaving its short portion pending.
3. Evaluate exits on positions now held. Opening gaps fill at the open.
   Intraday exits require strict crossing; stops take precedence when both
   levels cross. Apply adverse exit slippage only to intraday levels. An
   adverse one-price limit lock blocks a stop. Exits cancel same-side pending
   targets and preserve reversals.
4. For close execution, retry pending targets after exits. New close positions
   cannot exit using that day's earlier high/low.
5. Mark shares at finite close or the last finite mark. Hedge P&L uses the
   previous held notional and caller-supplied same-contract return. Accrete
   notional by P&L, charge two sides on a roll, then pay to resize to minus
   hedge ratio times actual net stock value. Unfilled orders create no hedge.
6. Equity equals cash plus signed stock value. After a target-date close,
   replace the entire pending book, targeting zero for omitted held symbols.
   Truncate weight * equity / close / 1000 toward zero to size whole lots.
   An explicit zero-weight row represents an empty target book.

Every fill charges abs(delta shares) * price * fee_rate * fee_discount.
Every sale also pays sales tax; newly opened short shares pay the short-sale
surcharge. Cash costs are reported daily and per fill. The caller supplies the
per-side futures cost as a fraction of notional from its own cost model.
There is no cash-availability or contract-margin constraint.

## Selection and calendars

`phase_rebalance_dates` schedules the last session on or before a weekday in
every k calendar weeks. Empty weeks do not shift phases. `swing_books` uses the
next session's scores, requiring timezone-aware availability by that session's
08:30 Asia/Taipei cutoff. It filters universe membership, disposal status, and
short eligibility. Ties use instrument IDs ascending for longs and descending
for shorts. A hold rank retains prior selections within that rank before filling
to top N. Weights remain 1/N for longs or 1/(2N) for each long/short side even
when fewer names qualify. Overlapping sides raise `ValueError`.

`bench_book` submits one `0050` target on the first scheduled rebalance. Disable
exit rules to maintain buy-and-hold behavior. A hedge ratio optionally adds the
same futures overlay used for other books.

## Verification

Self-contained Rust and Python fixtures cover signed lot sizing, fee/tax/short
costs, limit fills and retries, missing marks, borrow permissions, calendar-day
financing, dividends, take-profit/stop ordering, gaps, pending replacement,
close sequencing, hedge return/resize/roll charges, calendar phases, rank buffers,
input validation, and Python 3.9 imports.

Behavior was checked for exact fill/position parity against an independent
reference daily swing simulator on historical data. Floating aggregate sums can
differ with addition order; exact shares, prices, per-fill costs, and stock
exposures provide separate checks. No historical datasets or reference simulator
sources are required by the public tests.
