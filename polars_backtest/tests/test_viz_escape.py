"""Script-context escaping of the embedded report payload."""

import datetime

import polars as pl
import polars_backtest as pl_bt
from polars_backtest import viz


def _tiny_report(symbol: str):
    rows = []
    start = datetime.date(2024, 1, 1)
    for i in range(70):
        date = start + datetime.timedelta(days=i)
        if date.weekday() >= 5:
            continue
        rows.append({
            "date": date, "symbol": symbol,
            "close": 100.0 * (1 + 0.002 * i),
            "weight": 1.0 if date < datetime.date(2024, 2, 15) else 0.0,
        })
    return pl_bt.backtest_with_report(pl.DataFrame(rows), resample="M")


def test_payload_cannot_break_out_of_script_tag():
    hostile = "</script><script>alert(1)</script>"
    report = _tiny_report(hostile)
    html = viz.report_html(report, title=hostile)

    # the template's own closing tags only — no user string may add one
    baseline = viz.report_html(_tiny_report("SAFE"), title="safe")
    assert html.count("</script>") == baseline.count("</script>")
    assert "\\u003c/script" in html  # hostile payload strings are JSON-escaped


def test_title_containing_placeholder_stays_literal():
    report = _tiny_report("AAA")
    html = viz.report_html(report, title="__PAYLOAD__")

    assert html.count('"schema":3') == 1  # payload substituted exactly once
