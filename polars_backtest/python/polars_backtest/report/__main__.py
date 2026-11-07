"""Command line reader for bundles, for agents and shells.

    python -m polars_backtest.report summary my_run [--md]
    python -m polars_backtest.report schema  my_run
    python -m polars_backtest.report day     my_run 2024-03-15
    python -m polars_backtest.report symbol  my_run FAKE001
    python -m polars_backtest.report sql     my_run "select ..." --limit 50
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from typing import Any, Sequence

import polars as pl

from polars_backtest.report.bundle import DEFAULT_LIMIT, MissingTableError, open_bundle
from polars_backtest.report.schema import SchemaError


def _plain(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _shaped(frames: dict[str, pl.DataFrame], limit: int) -> dict[str, Any]:
    out = {}
    for name, frame in frames.items():
        rows = frame.head(limit)
        out[name] = {
            "rows": rows.to_dicts(),
            "returned": rows.height,
            "total": frame.height,
            "truncated": frame.height > rows.height,
        }
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m polars_backtest.report")
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summary", help="the fixed-key summary")
    summary.add_argument("bundle")
    summary.add_argument("--md", action="store_true", help="the Markdown brief")
    schema = commands.add_parser("schema", help="tables, states, rows and columns")
    schema.add_argument("bundle")
    for name, key in (("day", "date"), ("symbol", "symbol"), ("sql", "query")):
        command = commands.add_parser(name, help=f"look up by {key}")
        command.add_argument("bundle")
        command.add_argument(key)
        command.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    return parser


def _run(args: argparse.Namespace) -> Any:
    db = open_bundle(args.bundle)
    if args.command == "summary":
        return db.summary("md") if args.md else db.summary()
    if args.command == "schema":
        return {"run": db.manifest["run"], "tables": db.manifest["tables"]}
    if args.command == "day":
        return _shaped(db.day(args.date), args.limit)
    if args.command == "symbol":
        return _shaped(db.symbol(args.symbol), args.limit)
    return db.query(args.query, limit=args.limit)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _run(args)
    except (FileNotFoundError, MissingTableError, SchemaError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(_plain(result), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
