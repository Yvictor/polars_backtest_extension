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


def test_schema_v3_fields(report):
    data = viz.report_data(report)

    assert data["schema"] == 3
    assert data["version"]
    assert isinstance(data["metrics"], dict) and "annualReturn" in data["metrics"]
    assert data["daily"]["series"][0]["creturn"] == data["daily"]["creturn"]
    assert isinstance(data["dd_episodes"], list) and data["dd_episodes"]
    ep = data["dd_episodes"][0]
    assert set(ep) >= {"start", "trough", "end", "depth", "days", "recovery_days"}
    assert ep["depth"] < 0
    # deepest episode first, and its depth matches max_drawdown
    assert ep["depth"] == pytest.approx(data["stats"]["max_drawdown"], abs=1e-9)


def test_quality_scores(report):
    quality = viz.report_data(report)["quality"]

    assert set(quality) == {"profitability", "risk", "ratio", "winrate", "liquidity"}
    prof = quality["profitability"]
    assert prof["label"] == "獲利"
    checks = {c["key"]: c for c in prof["checks"]}
    # no benchmark in fixture: alpha/beta have no value and are excluded
    assert checks["alpha"]["pass"] is None and checks["alpha"]["value"] is None
    evaluated = [c for c in prof["checks"] if c["pass"] is not None]
    assert evaluated, "some profitability checks must evaluate"
    # score equals FinLab formula over evaluated checks
    expected = round(100 * sum(c["pass"] for c in evaluated) / len(evaluated))
    assert prof["score"] == expected
    # capacity carries 11x weight: with no liquidity data at all, score is None
    liq = quality["liquidity"]
    if all(c["value"] is None for c in liq["checks"]):
        assert liq["score"] is None


def test_trade_detail_fields(report):
    trades = viz.report_data(report)["trades"]

    n = len(trades["ret"])
    for key in ("pos", "entry_px", "exit_px", "gmfe", "mdd", "lim_entry", "lim_exit"):
        assert len(trades[key]) == n
    # without input_df the limit flags are unknown, not false
    assert all(v is None for v in trades["lim_entry"])


def test_limit_flags_from_input_df(report):
    import polars_backtest as _pb  # noqa: F401

    # rebuild the same input and mark every entry price as exactly limit-up
    entry_px = [p for p in viz.report_data(report)["trades"]["entry_px"] if p is not None]
    assert entry_px, "fixture must have executed trades"

    # craft input_df: limit_up equal to raw price on every (date,symbol) → all entries flagged
    trades_df = report.trades
    input_df = trades_df.select(
        pl.col("entry_date").alias("date"),
        pl.col("stock_id").alias("symbol"),
        pl.col("entry_raw_price").alias("limit_up"),
        pl.col("entry_raw_price").alias("limit_down") * 0.0,
    ).drop_nulls("date")
    data = viz.report_data(report, input_df=input_df)
    entered = [f for f, e in zip(data["trades"]["lim_entry"], data["trades"]["entry"]) if e]
    assert entered and all(f is True for f in entered)
    assert data["trade_summary"]["buy_high_ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# V2 template (FinLab-parity layout, specs/VIZ_V2_FINLAB_PARITY_SPEC.md)
# ---------------------------------------------------------------------------


def test_template_v2_sections(report):
    html = viz.report_html(report)

    # hero / score chips / metric row / tab bar containers
    for marker in ('id="hero"', 'id="qchips"', 'id="mrow"', 'id="tabbar"'):
        assert marker in html
    # one panel per tab, switchable by dimension
    for panel in ("perf", "monthly", "yearly", "trades", "dd", "dist", "liq"):
        assert f'id="panel-{panel}"' in html


def test_template_v2_zh_labels(report):
    html = viz.report_html(report)

    for label in (
        "歷史績效", "月報酬", "年度比較", "交易明細", "虧損歷史", "跌幅排名",
        "報酬分布", "模擬停損", "流動性", "漲跌停成交明細", "年度回報",
        "最大回檔", "夏普值", "逐筆交易勝率",
    ):
        assert label in html, f"missing zh-TW label: {label}"


def test_template_v2_script_syntax(report, tmp_path):
    """The embedded JS must be syntactically valid (checked with node when present)."""
    import re
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    html = viz.report_html(report)
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert scripts, "template must embed a script"
    js = tmp_path / "viz.js"
    js.write_text("\n".join(scripts), encoding="utf-8")
    subprocess.run([node, "--check", str(js)], check=True)
