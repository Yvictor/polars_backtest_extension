"""``ReportBundle``: one set of tables behind every way of reading a backtest."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from polars_backtest.report.schema import TABLES, SchemaError, check_version
from polars_backtest.report.summary import render_markdown, to_json

MANIFEST = "manifest.json"
SUMMARY_JSON = "summary.json"
SUMMARY_MD = "SUMMARY.md"
ROW_GROUP = 100_000
DEFAULT_LIMIT = 100


class MissingTableError(KeyError):
    """A table was asked for that this bundle does not hold."""

    def __str__(self) -> str:
        return str(self.args[0])


class BundleCheckError(ValueError):
    """Ledger tables disagree and ``strict=True`` asked for that to be an error."""


def _as_date(value: str | dt.date) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReportBundle:
    """Tables, a manifest and a summary. Build with ``report.to_bundle()``.

    Tables are lazy, so a bundle opened from disk reads only what a query needs.
    """

    def __init__(
        self,
        manifest: dict[str, Any],
        summary: dict[str, Any],
        tables: dict[str, pl.LazyFrame],
        path: Path | None = None,
    ) -> None:
        self.manifest = manifest
        self._summary = summary
        self._tables = tables
        self.path = path

    def __repr__(self) -> str:
        run = self.manifest["run"]
        return (
            f"ReportBundle({run['title']!r}, {run['start']} to {run['end']}, "
            f"{len(self._tables)} tables)"
        )

    @property
    def coverage(self) -> dict[str, str]:
        """Table name to state: present, partial, absent, empty or not_applicable."""
        return {name: entry["state"] for name, entry in self.manifest["tables"].items()}

    @property
    def checks(self) -> list[dict[str, Any]]:
        return self.manifest["checks"]

    def has(self, name: str) -> bool:
        return name in self._tables

    def summary(self, format: str = "dict") -> dict[str, Any] | str:
        """The fixed-key summary as a dict, or the same content as Markdown with ``"md"``."""
        if format == "dict":
            return self._summary
        if format == "md":
            return render_markdown(self._summary)
        raise ValueError(f"format must be 'dict' or 'md', got {format!r}")

    def table(self, name: str) -> pl.LazyFrame:
        """A table as a LazyFrame. Raises ``MissingTableError`` with the reason it is absent."""
        if name in self._tables:
            return self._tables[name]
        entry = self.manifest["tables"].get(name)
        if entry is None:
            raise MissingTableError(f"unknown table {name!r}; see manifest['tables']")
        raise MissingTableError(
            f"table {name!r} is {entry['state']}: {entry.get('reason', 'no detail')}"
        )

    def _query(self, query: str) -> pl.LazyFrame:
        context = pl.SQLContext(frames=self._tables, eager=False)
        return context.execute(query, eager=False)

    def sql(self, query: str, limit: int | None = DEFAULT_LIMIT) -> pl.DataFrame:
        """Run SQL over every present table. ``limit=None`` returns all rows."""
        frame = self._query(query)
        return frame.collect() if limit is None else frame.limit(limit).collect()

    def query(self, query: str, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
        """SQL with a shape that says whether the caller saw everything."""
        frame = self._query(query)
        total = frame.select(pl.len()).collect().item()
        rows = frame.limit(limit).collect()
        return {
            "rows": rows.to_dicts(),
            "returned": rows.height,
            "total": total,
            "truncated": total > rows.height,
        }

    def _rows(self, name: str, predicate: pl.Expr) -> pl.DataFrame:
        return self.table(name).filter(predicate).collect()

    def _holdings(self, date: dt.date) -> pl.DataFrame:
        held = (
            (pl.col("status") != "pending")
            & (pl.col("entry_date") <= date)
            & (pl.col("exit_date").is_null() | (pl.col("exit_date") > date))
        )
        columns = ["symbol", "side", "weight", "entry_date", "exit_date", "ret"]
        return self._rows("trades", held).select(columns)

    def day(self, date: str | dt.date) -> dict[str, pl.DataFrame]:
        """One day's detail. A table this bundle lacks leaves its key out, never empty.

        ``holdings`` comes from the trades table: which names were open, and their entry
        weights. That weight is the weight at entry, not the weight on this day.
        """
        day = _as_date(date)
        same_day = pl.col("date") == day
        out = {"summary": self._rows("day_index", same_day)}
        for name in ("fills", "blocks", "cashflows"):
            if self.has(name):
                out[name] = self._rows(name, same_day)
        if self.has("trades"):
            out["holdings"] = self._holdings(day)
        return out

    def symbol(self, symbol: str) -> dict[str, pl.DataFrame]:
        """One symbol's detail. A table this bundle lacks leaves its key out, never empty."""
        same = pl.col("symbol") == symbol
        out = {}
        for key, name in (
            ("summary", "symbol_index"),
            ("trades", "trades"),
            ("fills", "fills"),
            ("blocks", "blocks"),
        ):
            if self.has(name):
                out[key] = self._rows(name, same)
        return out

    def write(self, path: str | Path) -> Path:
        """Write the bundle as a folder of Parquet tables, JSON and Markdown."""
        out = Path(path)
        if self.path is not None and out.resolve() == self.path.resolve():
            raise ValueError(f"bundle is already stored at {out}")
        out.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(json.dumps(self.manifest))
        for name, lazy in self._tables.items():
            target = out / TABLES[name].file
            lazy.collect().write_parquet(
                target, compression="zstd", statistics=True, row_group_size=ROW_GROUP
            )
            manifest["tables"][name]["sha256"] = _sha256(target)
        (out / SUMMARY_JSON).write_text(to_json(self._summary), encoding="utf-8")
        (out / SUMMARY_MD).write_text(render_markdown(self._summary), encoding="utf-8")
        # Last, so a folder with a manifest is a finished bundle.
        (out / MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return out


def open_bundle(path: str | Path, *, verify: bool = False) -> ReportBundle:
    """Open a bundle folder. No engine is needed, only polars.

    Args:
        verify: also compare each file with its recorded sha256.
    """
    root = Path(path)
    manifest_path = root / MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"no {MANIFEST} in {root}, is this a bundle folder?")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check_version(manifest.get("schema_version", ""))
    summary = json.loads((root / SUMMARY_JSON).read_text(encoding="utf-8"))
    tables = {}
    for name, entry in manifest["tables"].items():
        if "file" not in entry:
            continue
        target = root / entry["file"]
        if verify and _sha256(target) != entry.get("sha256"):
            raise SchemaError(f"{target} does not match its recorded sha256")
        tables[name] = pl.scan_parquet(target)
    return ReportBundle(manifest, summary, tables, path=root)
