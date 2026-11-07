"""Report bundle: the tables behind the dashboard, readable without it.

>>> db = report.to_bundle("my_run", title="demo")
>>> db.summary()["performance"]["max_drawdown"]
>>> pl_bt.open_bundle("my_run").sql("select * from trades order by ret limit 5")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polars_backtest.report.bundle import (
    BundleCheckError,
    MissingTableError,
    ReportBundle,
    open_bundle,
)
from polars_backtest.report.schema import SCHEMA_VERSION, SchemaError

__all__ = [
    "SCHEMA_VERSION",
    "BundleCheckError",
    "MissingTableError",
    "ReportBundle",
    "SchemaError",
    "attach",
    "open_bundle",
    "show",
    "summary",
    "to_bundle",
    "to_html",
]


def to_bundle(report: Any, path: str | Path | None = None, **kwargs: Any) -> ReportBundle:
    """Build a report's bundle, in memory, or on disk too when ``path`` is given.

    Args:
        path: folder to write. Without it nothing touches the disk.
        title: display name of the run.
        input_df: the backtest input; ``limit_up``/``limit_down`` columns add limit flags.
        symbol_names: ``{symbol: display name}``.
        fill_scenarios: output of the fill-stress scenarios.
        strict: raise ``BundleCheckError`` when ledger tables do not reconcile.
    """
    from polars_backtest.report.build import build_bundle

    bundle = build_bundle(report, **kwargs)
    if path is not None:
        bundle.write(path)
        return open_bundle(path)
    return bundle


def summary(report: Any, format: str = "dict", **kwargs: Any) -> dict[str, Any] | str:
    """A report's summary as a dict, or as Markdown with ``format="md"``."""
    return to_bundle(report, **kwargs).summary(format)


def to_html(
    report: Any,
    path: str | Path | None = None,
    *,
    detail: str = "full",
    max_mb: float = 20,
    capital: float | None = None,
    **kwargs: Any,
) -> str | Path:
    """The report's dashboard as one offline file. See ``ReportBundle.to_html``."""
    return to_bundle(report, **kwargs).to_html(
        path, detail=detail, max_mb=max_mb, capital=capital
    )


def show(
    report: Any,
    *,
    height: int = 900,
    inline: bool | None = None,
    day: str | None = None,
    detail: str = "full",
    capital: float | None = None,
    **kwargs: Any,
) -> None:
    """Open the report's dashboard. See ``ReportBundle.show``."""
    to_bundle(report, **kwargs).show(
        height=height, inline=inline, day=day, detail=detail, capital=capital
    )


def attach(*classes: type) -> None:
    """Add the bundle methods to report classes.

    ``BacktestReport`` is a pyo3 class and accepts attributes from Python, so long and
    wide reports share these functions without new Rust.
    """
    for cls in classes:
        cls.to_bundle = to_bundle
        cls.summary = summary
        cls.to_html = to_html
        cls.show = show
