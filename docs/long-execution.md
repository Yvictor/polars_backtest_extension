# Long-format execution

Signals execute in the following session at `trade_at_price`. Set
`trade_at_price="open", mark_price="close"` to trade at that session's open and
value the remaining positions at its close. Newly entered positions earn the
open-to-close return. `mark_price` accepts a column name or Polars expression.
The report's `trade_at` identifies the execution column (an expression uses its
resolved internal column name).

`mark_price=None` uses the execution price for valuation. An identical mark
column keeps the default calculation and stop timing exactly. Missing marks
leave positions at the latest valid observed price.

`slippage_ratio` defaults to zero and must be finite in `[0, 1)`. Buys fill at
`price * (1 + ratio)`; sells fill at `price * (1 - ratio)`, including short sales
and stop exits. With slippage enabled, a buy's budget includes commission and
applicable tax; both costs use its actual filled notional. Sells receive their
filled notional less costs. Stop return tracking starts at the slipped fill.
Zero slippage retains the existing fee arithmetic exactly.

With separate marks, close-based stops are detected after closing valuation and
execute at the next session's execution price. `touched_exit=True` still uses
the supplied OHLC range for same-session stops. With no separate mark, the
existing stop order is preserved, including detection before rebalancing.

`ledger=True` adds the actual slippage cost to `report.daily.slippage`.
`report.fills.price` is the slipped price and `notional` is the signed filled
value. Equity already includes these costs; do not subtract them again.
Trade rows retain their existing per-rebalance grouping; the fills table records
only the net legs that actually executed.
