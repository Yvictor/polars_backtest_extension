"""Bitwise snapshots of the long and wide APIs on synthetic inputs.

Float values, nulls, schemas and dates are compared exactly. Trade rows are
compared by key because the reference build wrote them in unordered map order.
"""

import datetime as dt
import json
import struct

import polars as pl

import polars_backtest as bt


def exact(value):
    if isinstance(value, float):
        return {"f64": struct.pack(">d", value).hex()}
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: exact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [exact(v) for v in value]
    return value


def frame(df, keyed=False):
    rows = exact(df.rows())
    if keyed:
        rows.sort(key=lambda row: json.dumps(row, sort_keys=True))
    return {"schema": [(k, str(v)) for k, v in df.schema.items()], "rows": rows}


def inputs(kind):
    days = [dt.date(2020, 1, 2) + dt.timedelta(days=i) for i in range(65)]
    days = [d for d in days if d.weekday() < 5]
    rows = []
    for i, day in enumerate(days):
        for j, symbol in enumerate(["A", "B"]):
            price = float(64 + j * 64 + (i % 9) * (1 if i % 17 < 8 else -1))
            # One active name, so these snapshots were bit-stable even on the
            # reference build, whose map reduction order varied by process.
            weight = (0.5 if i % 12 < 6 else 0.25) if j == 0 else 0.0
            if kind == "signed":
                weight *= -1 if i % 12 < 6 else 1
            if kind == "boolean":
                weight = bool(weight)
            rows.append((day, symbol, price, price * 0.98, price * 1.05, price * 0.94, weight))
    df = pl.DataFrame(
        rows, schema=["date", "symbol", "close", "open", "high", "low", "weight"], orient="row"
    )
    if kind == "missing":
        df = df.with_columns(
            pl.when((pl.col("date") == days[9]) & (pl.col("symbol") == "B"))
            .then(float("nan"))
            .otherwise(pl.col("close"))
            .alias("close")
        )
    if kind == "empty":
        df = df.head(0)
    wide = {
        name: df.pivot(on="symbol", index="date", values=name).sort("date")
        for name in ["close", "open", "high", "low", "weight"]
    }
    if kind == "empty":
        wide = {
            name: pl.DataFrame(schema={"date": pl.Date, "A": pl.Float64, "B": pl.Float64})
            for name in wide
        }
    return df, wide


CASES = {
    "default": ("partial", {}),
    "boolean": ("boolean", {}),
    "signed": ("signed", {}),
    "zero_cost": ("partial", {"fee_ratio": 0.0, "tax_ratio": 0.0}),
    "limit": ("partial", {"position_limit": 0.3}),
    "retain": ("partial", {"retain_cost_when_rebalance": True}),
    "weekly": ("partial", {"resample": "W"}),
    "offset": ("partial", {"resample": "M", "resample_offset": "2D"}),
    "trail": ("partial", {"trail_stop": 0.03}),
    "stops": ("partial", {"stop_loss": 0.03, "take_profit": 0.04}),
    "touch": ("partial", {"touched_exit": True, "stop_loss": 0.03, "take_profit": 0.04}),
    "missing": ("missing", {}),
    "empty": ("empty", {}),
}


def report_output(report):
    return {
        "creturn": frame(report.creturn),
        "trades": frame(report.trades, keyed=True),
        "metadata": exact(
            {
                k: getattr(report, k)
                for k in [
                    "fee_ratio",
                    "tax_ratio",
                    "trade_at",
                    "stop_loss",
                    "take_profit",
                    "trail_stop",
                ]
            }
        ),
    }


def capture(name, api, cases=CASES, build=inputs):
    kind, options = cases[name]
    options = dict(options)
    df, wide = build(kind)
    try:
        if api == "namespace":
            return {"creturn": frame(df.bt.backtest(**options))}
        if api == "standalone":
            return {"creturn": frame(bt.backtest(df, **options))}
        if api == "namespace_report":
            return report_output(df.bt.backtest_with_report(**options))
        if api == "standalone_report":
            return report_output(bt.backtest_with_report(df, **options))
        if api == "wide_report":
            return report_output(
                bt.backtest_with_report_wide(
                    wide["close"],
                    wide["weight"],
                    open=wide["open"],
                    high=wide["high"],
                    low=wide["low"],
                    **options,
                )
            )
        options.pop("touched_exit", None)  # not an existing return-only wide option
        result = bt.backtest_wide(
            wide["close"], wide["weight"], finlab_mode=api == "wide_finlab", **options
        )
        return {"creturn": frame(result)}
    except (ValueError, IndexError, KeyError, pl.exceptions.PolarsError) as error:
        return {"error": type(error).__name__, "message": str(error)}
    except BaseException as error:
        # Empty Arrow buffers panic in the reference extension. Keep recording
        # that contract rather than changing it here.
        if type(error).__name__ != "PanicException":
            raise
        return {"error": type(error).__name__, "message": str(error)}


APIS = [
    "namespace",
    "standalone",
    "namespace_report",
    "standalone_report",
    "wide_standard",
    "wide_finlab",
    "wide_report",
]
