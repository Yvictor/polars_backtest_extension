# Long-format execution

Signals execute in the following session at `trade_at_price`. Set
`trade_at_price="open", mark_price="close"` to trade at that session's open and
value the remaining positions at its close. Newly entered positions earn the
open-to-close return. `mark_price` accepts a column name or Polars expression.
The report's `trade_at` identifies the execution column (an expression uses its
resolved internal column name).

`mark_price=None` uses the execution price for valuation. Selecting the same
column or expression as execution keeps default timing. Selecting a separate
mark enables closing valuation even when every value happens to equal execution;
timing never depends on future prices. Missing marks leave positions at the
latest valid observed price.

`slippage_ratio` defaults to zero and must be finite in `[0, 1)`. Buys fill at
`price * (1 + ratio)`; sells fill at `price * (1 - ratio)`, including short sales
and stop exits. A buy keeps the legacy fee base (its allocated budget), then
slippage shrinks its post-fee mark value. Sells receive their filled notional
less costs. This is continuous at zero slippage and preserves default arithmetic.
Trade rows keep the legacy per-rebalance grouping: entries from zero and
reversals use a slipped entry basis; continuing segments reset at the unslipped
market price. Slippage charges only the net traded fraction, never the unsold
position. Trade rows can therefore differ from portfolio daily returns.
Closed trade rows retain a hypothetical full-position slipped exit even for a
partial rebalance; the fills ledger charges only the actual net traded amount.
With positive slippage, reversals book a full exit before allocating the opposite
entry budget, in either direction. This prevents an underfunded buy budget from
leaving a short behind. Both legs keep the legacy reversal entry tax convention.

With separate marks, close-based stops are detected after closing valuation and
execute at the next session's execution price. With `touched_exit=True`, carried
opening gaps are checked before a rebalance resets their stop basis. Remaining
positions and new opening fills then use open, high, low order: an opening stop
uses only the peak known at the open, a high can trigger take-profit or tighten
a long trailing stop, and the low is checked last. OHLC bars cannot reveal the
actual path, so this ordering is deterministic. Only surviving positions receive
the closing mark; a closing stop signal executes next session. Changing that
close or future rows cannot change an earlier touched fill.
Trailing protection uses the favorable price extreme with separate marks:
a maximum for longs and a minimum for shorts, across opens, intraday extremes
and closing marks. A short's low tightens its subsequent closing and next-session
checks; it cannot change the earlier high check. Adverse prices never loosen the
trail. A slipped entry starts its trailing reference at the entry fill in both
directions, so its immediate execution loss remains subject to protection.
With no separate mark, the existing stop order is preserved, including detection
before rebalancing. Separate marks or positive slippage enable final execution
observations in trade metrics, including stop exits. With separate marks, touched
exits observe their touched fill and exclude the later closing mark.

`ledger=True` adds the actual slippage cost to `report.daily.slippage`.
`report.fills.price` is the slipped price. Signed `notional` keeps the legacy
budget base for buys and uses filled value for sells. Equity already includes
these costs; do not subtract them again.
Trade rows retain their existing per-rebalance grouping; the fills table records
only the net legs that actually executed.

Short costs are opt-in: `short_sale_fee_ratio=0.0`, `borrow_fee_ratio=0.0`, and
`tax_side="exit"` preserve the existing calculations. Ratios must be finite and
nonnegative. A short-sale surcharge applies only to newly added short notional.
`tax_side="sell"` taxes sells, including new short sales, and exempts covers.
When surcharge or sell-side taxation is enabled, a direction reversal books the
closing leg before the new opening leg. Same-direction changes charge only the
net traded value.

Borrowing costs `abs(previous_mark_value) * borrow_fee_ratio * calendar_days / 365`
at the start of a session, before updating prices or closing positions. Friday
to Monday counts three days, including on a Monday cover; entry day has no
borrow charge. Borrowing costs debit cash and affect subsequent sizing.

With `ledger=True`, `daily.surcharge` and `daily.borrow_cost` report those actual
costs. `cashflows` contains signed borrow debits with date, symbol, kind and amount;
it is empty when no borrow cost accrued and `None` when the ledger is disabled.
The existing per-rebalance trade return convention is retained; surcharge and
borrow charges are reported through daily equity and the ledger.
