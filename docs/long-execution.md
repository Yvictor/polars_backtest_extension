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

With separate marks, close-based stops are detected after closing valuation and
execute at the next session's execution price. With `touched_exit=True`, carried
opening gaps are checked before a rebalance resets their stop basis. Remaining
positions and new opening fills then use open, high, low order: an opening stop
uses only the peak known at the open, a high can trigger take-profit or tighten
a long trailing stop, and the low is checked last. OHLC bars cannot reveal the
actual path, so this ordering is deterministic. Only surviving positions receive
the closing mark; a closing stop signal executes next session. Changing that
close or future rows cannot change an earlier touched fill.
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
