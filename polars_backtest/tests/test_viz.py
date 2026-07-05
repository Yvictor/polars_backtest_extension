"""Tests for the self-contained HTML visualization (pl_bt.viz)."""

import datetime
import json

import polars as pl
import polars_backtest as pl_bt
import pytest
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
# V3 template (quant-narrative one-page scroll, no tabs)
# ---------------------------------------------------------------------------


def test_template_v3_one_page_sections(report):
    html = viz.report_html(report)

    # six narrative sections in reading order, no tab bar
    for marker in (
        'id="sec-verdict"', 'id="sec-perf"', 'id="sec-structure"',
        'id="sec-dd"', 'id="sec-live"', 'id="sec-micro"',
    ):
        assert marker in html
    assert 'id="tabbar"' not in html and 'class="panel"' not in html
    # verdict strip: hero numbers, tradability flags, collapsible quality check
    for marker in ('id="hero"', 'id="flags"', 'id="qcheck"'):
        assert marker in html
    # new panels: rolling 1Y, long/short split, concentration, underwater tiles
    for marker in (
        'id="rolling-card"', 'id="roll-ret"', 'id="roll-sharpe"',
        'id="ls-card"', 'id="contrib-hist"', 'id="uw-tiles"',
        'id="lim-tiles"', 'id="cap-tiles"', 'id="cost-tiles"',
        'id="fill-scn"', 'id="lim-note"',
    ):
        assert marker in html
    # trade microscope is a collapsed <details>
    assert '<details class="card" id="micro">' in html
    # params chips live in the footer now
    assert html.index('id="chips"') > html.index('id="rpt-footer"')


def test_template_v3_zh_labels(report):
    html = viz.report_html(report)

    for label in (
        # section titles
        "判決", "績效軌跡", "報酬結構", "回檔與痛苦", "實盤可行性", "交易顯微鏡",
        # kept card titles
        "歷史績效", "月報酬", "年度比較", "交易明細", "虧損歷史", "跌幅排名",
        "模擬停損", "漲跌停成交明細", "最大回檔", "夏普值",
        # new panels
        "滾動 1 年表現", "多空拆解", "報酬集中度", "水下時間比例",
        "最長水下天數", "平均修復天數", "漲停依賴", "胃納量", "成本結構",
        "年換手率", "賣在跌停",
        # three-method capacity tiles (rendered when payload carries the keys)
        "保守估計（min-leg 法）", "ADV 法",
        # limit-lock classification + fill-stress scenario table
        "一字鎖死進場", "盤中觸及進場", "買不到情境模擬",
        "一字漲停進", "盤中漲停進", "一字跌停出", "盤中跌停出",
        "排除一字鎖死（有監控）", "排除所有漲停進場（保守）",
    ):
        assert label in html, f"missing zh-TW label: {label}"


def test_template_v3_flags_and_takeaways(report):
    html = viz.report_html(report)

    # tradability flag labels (多空平衡 is conditional, injected by JS)
    for label in ("資金容量", "漲停依賴", "成本敏感", "Alpha衰減", "多空平衡"):
        assert label in html, f"missing flag label: {label}"
    # one takeaway anchor per narrative section
    for marker in (
        'id="tk-verdict"', 'id="tk-perf"', 'id="tk-structure"',
        'id="tk-dd"', 'id="tk-live"',
    ):
        assert marker in html


def test_template_v3_log_scale_default(report):
    html = viz.report_html(report)

    assert 'scale: "log"' in html


def test_report_html_escapes_hostile_symbol():
    """Hostile symbol / display names must never reach the HTML unescaped."""
    from polars_backtest._viz_template import TEMPLATE

    evil = "</script><img src=x onerror=alert(1)>"
    rows = []
    start = datetime.date(2024, 1, 1)
    for i in range(90):
        date = start + datetime.timedelta(days=i)
        if date.weekday() >= 5:
            continue
        price = 100.0 * (1 + 0.002 * i)
        rows.append({
            "date": str(date),
            "symbol": evil,
            "open": price * 0.99,
            "high": price * 1.02,
            "low": price * 0.98,
            "close": price,
            "weight": (i % 5) != 0,
        })
    df = pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())
    hostile_report = pl_bt.backtest_with_report(df, resample="M")
    data = viz.report_data(hostile_report)
    assert data["trades"] and evil in data["trades"]["stock"], "fixture must trade the symbol"

    html = viz.report_html(hostile_report, symbol_names={evil: "<b>bad</b>"})

    # no script breakout: the </script> count stays at the template's baseline
    assert html.count("</script>") == TEMPLATE.count("</script>")
    # the raw injection payloads never appear unescaped anywhere in the document
    assert "</script><img" not in html
    assert "<img" not in html
    assert "<b>bad</b>" not in html


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
