"""Synthetic portfolios holding many names at once, with unequal market values.

Unlike `legacy_fixtures`, these results depend on the order positions are summed
in, so they exercise the iteration order of the engine. Regenerate a snapshot with
`uv run python tests/multi_name_fixtures.py <output.json>`.
"""

import datetime as dt
import json
import random
import sys

import polars as pl
from legacy_fixtures import frame

import polars_backtest as bt

SYMBOLS = [f"S{i:02d}" for i in range(16)]
DAY_COUNT = 90


def trading_days():
    days = (dt.date(2021, 1, 4) + dt.timedelta(days=i) for i in range(DAY_COUNT * 7 // 5))
    return [d for d in days if d.weekday() < 5][:DAY_COUNT]


def inputs(kind="long", seed=11, symbols=SYMBOLS):
    rng = random.Random(seed)
    last = {s: 40.0 + 7.0 * i + rng.random() for i, s in enumerate(symbols)}
    rows = []
    for i, day in enumerate(trading_days()):
        held = rng.sample(symbols, len(symbols) // 2) if i % 4 == 0 else held
        for symbol in symbols:
            last[symbol] *= 1.0 + rng.gauss(0.0, 0.025)
            close = last[symbol]
            high = close * (1.0 + rng.random() * 0.03)
            low = close * (1.0 - rng.random() * 0.03)
            open_ = low + (high - low) * rng.random()
            weight = (0.5 + rng.random()) if symbol in held else 0.0
            if kind == "signed" and symbols.index(symbol) % 3 == 0:
                weight = -weight
            if kind == "missing" and i % 11 == 3 and symbols.index(symbol) % 5 == 0:
                close = float("nan")
            rows.append((day, symbol, close, open_, high, low, weight))
    return pl.DataFrame(
        rows, schema=["date", "symbol", "close", "open", "high", "low", "weight"], orient="row"
    )


STOPS = {"stop_loss": 0.04, "take_profit": 0.05}

CASES = {
    "daily": ("long", {}),
    "weekly": ("long", {"resample": "W"}),
    "monthly": ("long", {"resample": "M"}),
    "position_change": ("long", {"resample": None}),
    "offset": ("long", {"resample": "M", "resample_offset": "2D"}),
    "weekly_offset": ("long", {"resample": "W", "resample_offset": "2D"}),
    "position_limit": ("long", {"resample": "W", "position_limit": 0.1}),
    "retain": ("long", {"resample": "W", "retain_cost_when_rebalance": True}),
    "stops": ("long", {"resample": "W", **STOPS}),
    "stops_keep_trading": ("long", {"resample": "W", "stop_trading_next_period": False, **STOPS}),
    "trail": ("long", {"resample": "W", "trail_stop": 0.03}),
    "touched": ("long", {"resample": "W", "touched_exit": True, **STOPS}),
    "signed": ("signed", {"resample": "W"}),
    "signed_stops": ("signed", {"resample": "W", **STOPS}),
    "missing": ("missing", {"resample": "W"}),
}


def capture(name):
    """Return every number a report exposes for one case, as exact bit strings."""
    kind, options = CASES[name]
    df = inputs(kind)
    report = bt.backtest_with_report(df, **options)
    return {
        "backtest": frame(bt.backtest(df, **options)),
        "creturn": frame(report.creturn),
        "trades": frame(report.trades, keyed=True),
        "weights": frame(report.weights(), keyed=True),
        "next_weights": frame(report.next_weights(), keyed=True),
    }


def capture_all():
    return json.loads(json.dumps({name: capture(name) for name in CASES}))


if __name__ == "__main__":
    with open(sys.argv[1], "w") as output:
        json.dump(capture_all(), output, sort_keys=True)
