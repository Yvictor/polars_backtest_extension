"""Tests for the self-contained HTML visualization (pl_bt.viz)."""

import datetime
import json

import pytest
import polars as pl

import polars_backtest as pl_bt
from polars_backtest import viz


@pytest.fixture(scope="module")
def report():
    """A small multi-month report with trades."""
    rows = []
    start = datetime.date(2024, 1, 1)
    for i in range(180):
        date = start + datetime.timedelta(days=i)
        if date.weekday() >= 5:
            continue
        for j, symbol in enumerate(["AAPL", "GOOG", "MSFT"]):
            base = 100.0 * (j + 1)
            price = base * (1 + 0.002 * i + 0.05 * ((i + j) % 7 == 0))
            rows.append({
                "date": str(date),
                "symbol": symbol,
                "open": price * 0.99,
                "high": price * 1.02,
                "low": price * 0.98,
                "close": price,
                "weight": (i + j) % 5 != 0,
            })
    df = pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())
    return pl_bt.backtest_with_report(df, resample="M", stop_loss=0.1)


def test_report_data_structure(report):
    data = viz.report_data(report, title="t")

    assert data["title"] == "t"
    assert set(data) >= {"stats", "daily", "return_table", "trades", "stat_groups"}
    n = len(data["daily"]["dates"])
    assert n > 0 and len(data["daily"]["creturn"]) == n
    assert data["daily"]["benchmark"] is None
    assert data["return_table"] and len(data["return_table"][0]["months"]) == 12
    assert data["trades"]["total"] == len(data["trades"]["ret"])
    assert data["trade_summary"]["n"] == data["trades"]["total"]


def test_report_data_is_strict_json(report):
    # json.dumps with allow_nan=False rejects NaN/inf — payload must be clean.
    json.dumps(viz.report_data(report), allow_nan=False)


def test_report_html_placeholders_filled(report):
    html = viz.report_html(report, title="My <Strategy>")

    for placeholder in ("__PAYLOAD__", "__TITLE__", "__VERSION__"):
        assert placeholder not in html
    assert "My &lt;Strategy&gt;" in html
    assert html.startswith("<!DOCTYPE html>")


def test_save_html(report, tmp_path):
    out = viz.save_html(report, tmp_path / "r.html")

    assert out.exists() and out.stat().st_size > 10_000


def test_trade_sampling_cap(report, monkeypatch):
    monkeypatch.setattr(viz, "MAX_EMBEDDED_TRADES", 5)
    data = viz.report_data(report)

    assert data["trades"]["sampled"] is True
    assert len(data["trades"]["ret"]) == 5
    assert data["trades"]["total"] > 5
    # summary is computed from all trades, not the sample
    assert data["trade_summary"]["n"] == data["trades"]["total"]
