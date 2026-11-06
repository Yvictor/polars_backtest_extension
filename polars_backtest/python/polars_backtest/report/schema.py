"""The bundle contract: tables, columns, sort keys and versions.

Every column name a bundle exposes is declared here. Engine names are mapped to bundle
names in one place (``ENGINE_ALIASES``), so an engine rename touches this table only.

Version rules: a minor bump adds tables or columns and old readers ignore them. A major
bump renames or redefines something and old readers refuse the bundle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import polars as pl

SCHEMA_VERSION = "1.0"
STATES = ("present", "partial", "absent", "empty", "not_applicable")

DTYPES: dict[str, Any] = {
    "bool": pl.Boolean,
    "date": pl.Date,
    "float": pl.Float64,
    "int": pl.Int64,
    "str": pl.String,
}

# Engine column -> bundle column.
ENGINE_ALIASES = {
    "stock_id": "symbol",
    "return": "ret",
    "position": "weight",
    "entry_sig_date": "entry_signal_date",
    "exit_sig_date": "exit_signal_date",
    "entry_raw_price": "entry_price",
    "exit_raw_price": "exit_price",
    "lim_entry": "limit_entry",
    "lim_exit": "limit_exit",
    "sell_tax": "tax",
}


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    optional: bool = False


@dataclass(frozen=True)
class Table:
    name: str
    sort: tuple[str, ...]
    columns: tuple[Column, ...]
    required: bool = False
    # Optional columns a complete table carries. Missing any of them makes it "partial".
    expected: tuple[str, ...] = ()
    absent_reason: str = "not provided"

    @property
    def file(self) -> str:
        return f"{self.name}.parquet"

    def column(self, name: str) -> Column | None:
        return next((c for c in self.columns if c.name == name), None)


def _cols(spec: str) -> tuple[Column, ...]:
    """Parse ``"name:type name?:type"``. A trailing ``?`` marks an optional column."""
    out = []
    for item in spec.split():
        name, dtype = item.split(":")
        out.append(Column(name.rstrip("?"), dtype, name.endswith("?")))
    return tuple(out)


_TRADE_COLUMNS = (
    "symbol:str side:str status:str entry_signal_date?:date entry_date:date "
    "exit_signal_date?:date exit_date:date ret:float weight:float pdays?:int "
    "entry_price?:float exit_price?:float mae?:float gmfe?:float bmfe?:float mdd?:float"
)
_LEDGER = "ledger=False"

TABLES: dict[str, Table] = {
    table.name: table
    for table in (
        Table(
            "curve",
            ("date",),
            _cols(
                "date:date creturn:float ret:float drawdown:float bench_creturn?:float "
                "bench_drawdown?:float gross_creturn?:float"
            ),
            required=True,
        ),
        Table(
            "metrics",
            ("section", "key"),
            _cols("section:str key:str value:float unit:str source:str"),
            required=True,
        ),
        Table(
            "monthly",
            ("year", "month"),
            _cols("year:int month:int ret:float bench_ret?:float"),
            required=True,
        ),
        Table(
            "yearly",
            ("year",),
            _cols("year:int ret:float bench_ret?:float mdd:float"),
            required=True,
        ),
        Table(
            "drawdowns",
            ("depth",),
            _cols(
                "rank:int start:date trough:date end:date depth:float days:int "
                "recovery_days?:int series:str"
            ),
            required=True,
        ),
        Table(
            "day_index",
            ("date",),
            _cols(
                "date:date ret:float n_entries:int n_exits:int n_fills?:int n_blocks?:int "
                "traded_notional?:float cost_total?:float"
            ),
            required=True,
        ),
        Table(
            "trades",
            ("entry_date", "symbol"),
            _cols(
                "trade_id:int "
                + _TRADE_COLUMNS
                + " limit_entry?:bool limit_exit?:bool entry_kind?:str exit_kind?:str "
                "n_segments:int"
            ),
            absent_reason="no trades",
        ),
        Table(
            "trade_segments",
            ("entry_date", "symbol"),
            _cols(_TRADE_COLUMNS),
            absent_reason="no segments merged",
        ),
        Table(
            "symbol_index",
            ("contrib_proxy",),
            _cols(
                "symbol:str name?:str n_trades:int win_rate:float contrib_proxy:float "
                "n_fills?:int n_blocks?:int traded_notional?:float cost_total?:float"
            ),
            absent_reason="no trades",
        ),
        Table("symbols", ("symbol",), _cols("symbol:str name:str")),
        Table(
            "fill_scenarios",
            ("name",),
            _cols(
                "name:str cagr?:float total_return?:float max_drawdown?:float "
                "daily_sharpe?:float calmar?:float blocked_n?:int"
            ),
        ),
        Table(
            "ledger_daily",
            ("date",),
            _cols(
                "date:date equity?:float cash?:float commission?:float tax?:float "
                "surcharge?:float borrow_cost?:float hedge_cost?:float gross_long?:float "
                "gross_short?:float net_exposure?:float gross_exposure?:float "
                "traded_notional?:float turnover?:float pnl?:float"
            ),
            expected=(
                "equity",
                "cash",
                "commission",
                "tax",
                "gross_long",
                "gross_short",
                "net_exposure",
                "gross_exposure",
                "traded_notional",
                "turnover",
            ),
            absent_reason=_LEDGER,
        ),
        Table(
            "fills",
            ("date", "symbol"),
            _cols(
                "date:date symbol:str notional?:float price?:float reason?:str qty?:float "
                "cost?:float side?:str"
            ),
            expected=("notional", "price", "reason"),
            absent_reason=_LEDGER,
        ),
        Table(
            "blocks",
            ("date", "symbol"),
            _cols("date:date symbol:str reason:str side?:str notional?:float"),
            absent_reason=_LEDGER,
        ),
        Table(
            "cashflows",
            ("date", "symbol"),
            _cols("date:date symbol:str kind?:str amount?:float"),
            absent_reason=_LEDGER,
        ),
        Table(
            "orders",
            ("date", "symbol"),
            _cols("date:date symbol:str status?:str notional?:float"),
            absent_reason=_LEDGER,
        ),
    )
}

# The tables that come from the engine's ledger, in the order the page reports them.
LEDGER_TABLES = ("ledger_daily", "fills", "blocks", "cashflows", "orders")


class SchemaError(ValueError):
    """A bundle or table does not match this library's contract."""


def conform(name: str, frame: pl.DataFrame) -> pl.DataFrame:
    """Rename, cast and sort a frame to its table contract.

    Registered columns come first in registry order. Columns the registry does not know
    are kept after them, so a newer engine loses nothing on the way to the bundle.
    """
    table = TABLES[name]
    renames = {
        old: new
        for old, new in ENGINE_ALIASES.items()
        if old in frame.columns and new not in frame.columns
    }
    frame = frame.rename(renames)
    missing = [c.name for c in table.columns if not c.optional and c.name not in frame.columns]
    if missing:
        raise SchemaError(f"table {name!r} is missing required columns: {missing}")
    known = [c for c in table.columns if c.name in frame.columns]
    extra = [c for c in frame.columns if table.column(c) is None]
    frame = frame.select(
        *[pl.col(c.name).cast(DTYPES[c.dtype]) for c in known],
        *[pl.col(c) for c in extra],
    )
    return frame.sort(list(table.sort), nulls_last=True, maintain_order=True)


def validate(name: str, frame: pl.DataFrame) -> list[str]:
    """Return what is wrong with a frame against its contract, empty when it conforms."""
    table = TABLES[name]
    problems = []
    for column in table.columns:
        if column.name not in frame.columns:
            if not column.optional:
                problems.append(f"{name}.{column.name} is missing")
        elif frame.schema[column.name] != DTYPES[column.dtype]:
            found = frame.schema[column.name]
            problems.append(f"{name}.{column.name} is {found}, expected {column.dtype}")
    keys = list(table.sort)
    if not frame.equals(frame.sort(keys, nulls_last=True, maintain_order=True)):
        problems.append(f"{name} is not sorted by {keys}")
    return problems


def check_version(version: str) -> None:
    """Refuse a bundle written by a newer major version of this contract."""
    try:
        major = int(str(version).split(".")[0])
    except ValueError as exc:
        raise SchemaError(f"unreadable bundle schema_version: {version!r}") from exc
    if major > int(SCHEMA_VERSION.split(".")[0]):
        raise SchemaError(
            f"bundle schema_version {version} is newer than this library reads "
            f"({SCHEMA_VERSION}); upgrade polars_backtest to open it"
        )


def json_schema() -> dict[str, Any]:
    """The contract as data. ``specs/report_bundle.schema.json`` is this, serialized."""
    return {
        "schema_version": SCHEMA_VERSION,
        "states": list(STATES),
        "tables": {
            table.name: {
                "file": table.file,
                "required": table.required,
                "sort": list(table.sort),
                "expected": list(table.expected),
                "columns": [
                    {"name": c.name, "type": c.dtype, "optional": c.optional}
                    for c in table.columns
                ],
            }
            for table in TABLES.values()
        },
    }


if __name__ == "__main__":
    print(json.dumps(json_schema(), indent=2))
