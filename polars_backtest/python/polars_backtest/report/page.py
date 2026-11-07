"""Render a bundle as one offline HTML file.

``embed`` chooses the rows, this module joins them with the page assets. Size limits are
measured on the final bytes, because notebook escaping inflates the output.
"""

from __future__ import annotations

import base64
import html as html_lib
import json
import tempfile
import webbrowser
from pathlib import Path
from typing import Any

from polars_backtest.report.bundle import ReportBundle
from polars_backtest.report.embed import NOTEBOOK_LIMITS, PAGE_LIMITS, build_payload

DASHBOARD = Path(__file__).parent / "dashboard"
PAGE_MB = 20
NOTEBOOK_MB = 8
MAX_HALVINGS = 8
HALVING_ORDER = ("orders", "fills", "trades")


class ReportTooLargeError(ValueError):
    """The page stayed over its size limit after every reduction."""


def _asset(name: str) -> str:
    return (DASHBOARD / name).read_text(encoding="utf-8")


def _font() -> str:
    raw = (DASHBOARD / "vendor" / "manrope-latin.woff2").read_bytes()
    return base64.b64encode(raw).decode("ascii")


def render(payload: dict[str, Any]) -> str:
    """Fill the page template. Parts are joined, so user text is never substituted again."""
    data = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    # A raw "<" could close the script element from inside a string.
    data = data.replace("<", "\\u003c")
    parts = {
        "__VERSION__": html_lib.escape(payload["library_version"]),
        "__TITLE__": html_lib.escape(payload["run"]["title"]),
        "__CSS__": _asset("app.css").replace("__FONT__", _font()),
        "__PAYLOAD__": data,
        "__CORE__": _asset("core.js"),
        "__STRINGS__": _asset("strings.js"),
        "__APP__": _asset("app.js"),
    }
    page = _asset("index.html")
    pieces = []
    for marker, value in parts.items():
        before, page = page.split(marker, 1)
        pieces += [before, value]
    return "".join(pieces) + page


def srcdoc(page: str, height: int) -> str:
    """The page as a notebook iframe, which is what a notebook stores."""
    escaped = page.replace("&", "&amp;").replace('"', "&quot;")
    style = f"width:100%;height:{height}px;border:none;border-radius:8px;"
    return f'<iframe srcdoc="{escaped}" style="{style}"></iframe>'


def build_html(
    bundle: ReportBundle,
    *,
    detail: str = "full",
    max_mb: float = PAGE_MB,
    capital: float | None = None,
    limits: dict[str, int] | None = None,
    day: str | None = None,
    measure: Any = None,
) -> str:
    """Render the page, halving row limits until it fits ``max_mb``.

    Args:
        measure: maps the page to the text whose size counts, for notebook escaping.
    """
    limits = dict(limits or PAGE_LIMITS)
    measure = measure or (lambda page: page)
    budget = max_mb * 1024 * 1024
    for attempt in range(MAX_HALVINGS + 1):
        page = render(
            build_payload(bundle, limits=limits, detail=detail, capital=capital, day=day)
        )
        size = len(measure(page).encode("utf-8"))
        if size <= budget:
            return page
        limits[HALVING_ORDER[attempt % len(HALVING_ORDER)]] //= 2
    raise ReportTooLargeError(
        f"report is {size / 1024 / 1024:.1f} MB after reductions, over max_mb={max_mb}; "
        'use detail="summary" or a larger max_mb'
    )


def _in_notebook() -> bool:
    try:
        import IPython
    except ImportError:
        return False
    return IPython.get_ipython() is not None


def show(
    bundle: ReportBundle,
    *,
    height: int = 900,
    inline: bool | None = None,
    day: str | None = None,
    detail: str = "full",
    capital: float | None = None,
) -> None:
    """Show the page in a notebook iframe, or in a browser tab outside a notebook."""
    inline = _in_notebook() if inline is None else inline
    if not inline:
        page = build_html(bundle, detail=detail, capital=capital, day=day)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".html", prefix="polars_backtest_", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(page)
        webbrowser.open(f"file://{handle.name}")
        return
    from IPython.display import HTML, display

    page = build_html(
        bundle,
        detail=detail,
        capital=capital,
        day=day,
        max_mb=NOTEBOOK_MB,
        limits=NOTEBOOK_LIMITS,
        measure=lambda text: srcdoc(text, height),
    )
    display(HTML(srcdoc(page, height)))
