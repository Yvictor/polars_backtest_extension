"""The dashboard page: rendering, size limits, escaping, offline use and the JS formulas."""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import polars as pl
import polars_backtest as pl_bt
import pytest
from polars_backtest import viz
from polars_backtest.report import embed, open_bundle, page
from polars_backtest.report import metrics as M
from report_fixtures import (
    NAMES,
    LedgerReport,
    benchmark,
    fake_ledger,
    huge_fills_report,
    long_report,
    prices,
    weekdays,
)

DASHBOARD = page.DASHBOARD
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not available")


@pytest.fixture(scope="module")
def plain():
    return long_report()


@pytest.fixture(scope="module")
def full(plain):
    report = LedgerReport(plain, benchmark=benchmark(), **fake_ledger(plain))
    return pl_bt.report.to_bundle(report, title="full", input_df=prices(), symbol_names=NAMES)


def run_node(tmp_path: Path, script: str) -> dict:
    path = tmp_path / "check.cjs"
    path.write_text(script, encoding="utf-8")
    done = subprocess.run([NODE, str(path)], check=True, capture_output=True, text=True)
    return json.loads(done.stdout)


def decoded(column: object) -> list:
    """A payload column as a list, the same step the page takes when it reads one."""
    if isinstance(column, dict):
        return [None if i is None else column["d"][i] for i in column["i"]]
    return column


def payload_of(html: str) -> dict:
    raw = re.search(r'id="payload">(.*?)</script>', html, flags=re.S).group(1)
    return json.loads(raw)


class TestPage:
    def test_viz_functions_keep_working(self, plain, tmp_path):
        html = viz.report_html(plain, title="t")
        assert html.startswith("<!DOCTYPE html>")
        assert viz.save_html(plain, tmp_path / "r.html").stat().st_size > 10_000

    def test_wide_report_renders(self):
        df = prices(200).with_columns(pl.col("date").cast(pl.String))
        close = df.pivot(on="symbol", index="date", values="close").sort("date")
        weight = df.pivot(on="symbol", index="date", values="weight").sort("date")
        wide = pl_bt.backtest_with_report_wide(close, weight, resample="M")
        assert payload_of(viz.report_html(wide))["coverage"]["trades"] == "present"

    def test_report_methods(self, plain, tmp_path):
        assert plain.to_html().startswith("<!DOCTYPE html>")
        assert plain.to_html(tmp_path / "x.html").exists()

    def test_opened_bundle_renders_without_the_engine(self, full, tmp_path):
        full.write(tmp_path / "run")
        assert "Fake Co 1" in open_bundle(tmp_path / "run").to_html()

    def test_summary_detail_carries_no_names(self, full):
        html = full.to_html(detail="summary")
        assert "FAKE001" not in html and "Fake Co" not in html

    def test_notebook_iframe(self, full, monkeypatch):
        shown = {}
        ipython = pytest.importorskip("IPython.display")
        monkeypatch.setattr(ipython, "display", lambda obj: shown.update(data=obj.data))
        full.show(inline=True, day="2022-03-01")
        assert shown["data"].startswith("<iframe srcdoc=")


class TestEscaping:
    HOSTILE = "</script><img src=x onerror=alert(1)>"

    @pytest.fixture(scope="class")
    def html(self, plain):
        trades = plain.trades.with_columns(pl.lit(self.HOSTILE).alias("stock_id"))
        blocks = pl.DataFrame(
            {"date": [dt.date(2022, 3, 1)], "symbol": [self.HOSTILE], "reason": [self.HOSTILE]}
        )
        report = LedgerReport(plain, trades=trades, blocks=blocks)
        names = {self.HOSTILE: self.HOSTILE}
        return pl_bt.report.to_html(report, title=self.HOSTILE, symbol_names=names)

    def test_no_script_breakout(self, html):
        assert html.count("</script>") == 2

    def test_no_raw_markup(self, html):
        assert "<img" not in html

    def test_title_is_escaped(self, html):
        assert "<title>&lt;/script&gt;&lt;img" in html

    def test_text_still_round_trips(self, html):
        assert payload_of(html)["run"]["title"] == self.HOSTILE

    def test_placeholder_in_a_title_stays_literal(self, plain):
        html = plain.to_html(title="__PAYLOAD__ __CSS__")
        assert html.count('id="payload"') == 1
        assert payload_of(html)["run"]["title"] == "__PAYLOAD__ __CSS__"

    def test_page_escapes_every_inserted_field(self):
        """Fields from a run reach innerHTML only through esc()."""
        source = (DASHBOARD / "app.js").read_text(encoding="utf-8")
        for field in ("symbol", "reason", "kind", "name", "title"):
            assert re.findall(rf"\$\{{[a-z]\.{field}\}}", source) == [], field


class TestOffline:
    @pytest.fixture(scope="class")
    def html(self, full):
        return full.to_html()

    def test_nothing_is_loaded_from_elsewhere(self, html):
        assert re.findall(r"""@import|https?://""", html) == []

    def test_no_network_calls_in_the_script(self, html):
        for call in ("fetch(", "XMLHttpRequest", "WebSocket", "EventSource", "importScripts"):
            assert call not in html

    def test_font_is_embedded(self, html):
        assert "data:font/woff2;base64," in html


class TestSizeLimit:
    @pytest.fixture(scope="class")
    def huge(self, plain):
        return pl_bt.report.to_bundle(huge_fills_report(plain))

    def test_output_stays_under_the_limit(self, huge):
        assert len(huge.to_html(max_mb=8).encode("utf-8")) <= 8 * 1024 * 1024

    def test_page_reports_what_the_limit_cut(self, huge):
        note = payload_of(huge.to_html(max_mb=8))["embed"]["fills"]
        assert note["rows_embedded"] < note["rows_total"]

    def test_impossible_limit_says_what_to_do(self, full):
        with pytest.raises(page.ReportTooLargeError, match="detail"):
            full.to_html(max_mb=0.01)

    def test_notebook_size_counts_the_escaped_text(self, full):
        html = page.build_html(full, measure=lambda text: page.srcdoc(text, 900))
        assert len(page.srcdoc(html, 900).encode()) <= page.PAGE_MB * 1024 * 1024


@needs_node
class TestJsFormulas:
    CORE = (DASHBOARD / "core.js").as_posix()

    def call(self, tmp_path, expression: str) -> dict:
        script = f"""
const BT = require({json.dumps(self.CORE)});
console.log(JSON.stringify({expression}));
"""
        return run_node(tmp_path, script)

    @staticmethod
    def series(curve: pl.DataFrame) -> str:
        dates = curve["date"].cast(pl.String).to_list()
        return json.dumps(curve["creturn"].to_list()) + "," + json.dumps(dates)

    def test_whole_range_matches_the_summary(self, full, tmp_path):
        curve = full.table("curve").collect()
        args = f"{self.series(curve)}, 0, {curve.height - 1}, 0.02"
        got = self.call(tmp_path, f"BT.rangeKpi({args})")
        for key, value in full.summary()["performance"].items():
            assert got[key] == pytest.approx(value, abs=1e-9), key

    @pytest.mark.parametrize(("a", "b"), [(0, 100), (50, 320), (200, 590)])
    def test_sub_range_matches_python(self, full, tmp_path, a, b):
        curve = full.table("curve").collect()
        part = curve[a : b + 1].with_columns(
            (pl.col("creturn") / pl.col("creturn").shift(1) - 1).alias("ret"),
            (pl.col("creturn") / pl.col("creturn").cum_max() - 1).alias("drawdown"),
        )
        got = self.call(tmp_path, f"BT.rangeKpi({self.series(curve)}, {a}, {b}, 0.02)")
        for key, value in M.performance(part, 0.02).items():
            assert got[key] == pytest.approx(value, abs=1e-9), key

    def test_track_matches_underwater(self, full, tmp_path):
        curve = full.table("curve").collect()
        n = curve.height - 1
        script = f"""
const BT = require({json.dumps(self.CORE)});
const run = (eq, dates) => {{
  const T = BT.track(eq, dates, 0, {n});
  return {{mdd: T.mdd[{n}], start: dates[T.uwA[{n}]], end: dates[T.uwB[{n}]]}};
}};
console.log(JSON.stringify(run({self.series(curve)})));
"""
        got = run_node(tmp_path, script)
        dd = full.summary()["drawdown"]
        assert got["mdd"] == pytest.approx(full.summary()["performance"]["max_drawdown"], abs=1e-12)
        assert (got["start"], got["end"]) == (dd["underwater_start"], dd["underwater_end"])

    def test_drawdown_to_date_never_improves(self, tmp_path):
        eq = [1.0, 1.2, 0.9, 1.1, 0.8, 1.3, 1.25]
        dates = [d.isoformat() for d in weekdays(len(eq))]
        args = f"{json.dumps(eq)}, {json.dumps(dates)}, 0, {len(eq) - 1}"
        got = self.call(tmp_path, f"[...BT.track({args}).mdd]")
        assert got == sorted(got, reverse=True)
        assert got[-1] == pytest.approx(0.8 / 1.2 - 1)

    def test_flat_best_matches_the_summary(self, full, tmp_path):
        curve = full.table("curve").collect()
        rets = full.table("monthly").collect()["ret"].to_list()
        conc = full.summary()["concentration"]
        years = (curve["date"][-1] - curve["date"][0]).days / 365.25
        growth = curve["creturn"][-1] / curve["creturn"][0]
        steps = self.call(
            tmp_path, f"BT.flatBest({json.dumps(rets)}, {conc['k']}, {growth}, {years})"
        )
        assert steps[0] == pytest.approx(conc["cagr"], abs=1e-9)
        assert steps[-1] == pytest.approx(conc["cagr_ex_best"], abs=1e-9)

    def test_rolling_hold_matches_the_summary(self, full, tmp_path):
        eq = json.dumps(full.table("curve").collect()["creturn"].to_list())
        entry = full.summary()["entry_date"]
        got = self.call(
            tmp_path,
            f"(() => {{ const v = BT.rollHold({eq}, {entry['window_days']})"
            ".filter(x => x != null); return {worst: Math.min(...v), median: BT.median(v)}; })()",
        )
        assert got["worst"] == pytest.approx(entry["worst"], abs=1e-9)
        assert got["median"] == pytest.approx(entry["median"], abs=1e-9)

    @pytest.mark.parametrize("rf", [0.0, 0.02, 0.05])
    def test_rolling_sharpe_uses_excess_return(self, full, tmp_path, rf):
        rets = full.table("curve").collect()["ret"].fill_null(0.0)
        window = 63
        args = f"{json.dumps(rets.to_list())}, {window}, 'sharpe', {rf}"
        got = self.call(tmp_path, f"BT.rolling({args})")
        daily = (1 + rf) ** (1 / 252) - 1
        expected = (
            (rets.rolling_mean(window) - daily) / rets.rolling_std(window, ddof=1)
        ) * math.sqrt(252)
        assert got[-1] == pytest.approx(expected[-1], abs=1e-9)
        assert got[window - 1] == pytest.approx(expected[window - 1], abs=1e-9)
        assert got[window - 2] is None

    def test_rolling_sharpe_on_one_window_matches_the_range_figure(self, full, tmp_path):
        curve = full.table("curve").collect()
        n = curve.height
        rets = curve["ret"].fill_null(0.0).to_list()
        got = self.call(tmp_path, f"BT.rolling({json.dumps(rets)}, {n}, 'sharpe', 0.02)[{n - 1}]")
        assert got == pytest.approx(full.summary()["performance"]["sharpe"], abs=1e-9)

    def test_drawing_keeps_every_extreme(self, tmp_path):
        values = [1 + ((i * 7) % 101) / 100 for i in range(50_000)]
        values[12_345] = 9.0
        values[40_000] = 0.1
        got = self.call(tmp_path, f"BT.drawIndices([{json.dumps(values)}], 0, 49999, 500)")
        assert len(got) < 5_000
        assert {0, 49_999, 12_345, 40_000} <= set(got)
        assert got == sorted(got)

    def test_drawing_short_ranges_keeps_every_point(self, tmp_path):
        assert self.call(tmp_path, "BT.drawIndices([[1,2,3,4]], 1, 3, 500)") == [1, 2, 3]

    def test_drawing_skips_gaps_in_a_series(self, tmp_path):
        got = self.call(tmp_path, "BT.drawIndices([[null,null,null,null,2,null]], 0, 5, 1)")
        assert 4 in got

    @staticmethod
    def biased(plain) -> tuple:
        """A bundle whose page rows are nothing like its trades: tails only, kept extreme."""
        db = pl_bt.report.to_bundle(plain)
        limits = dict(embed.PAGE_LIMITS, trades=12)
        return db, embed.build_payload(db, limits=limits)

    @staticmethod
    def rows(table: dict) -> str:
        names = list(table)
        columns = [decoded(table[n]) for n in names]
        return json.dumps([dict(zip(names, row)) for row in zip(*columns)])

    def window(self, tmp_path, data: dict, start: str, end: str) -> dict:
        args = f"{self.rows(data['trade_days'])}, {start!r}, {end!r}, {data['stops']}"
        return self.call(tmp_path, f"BT.windowTrades({args})")

    @pytest.mark.parametrize(
        ("start", "end"), [("2021-01-01", "2030-01-01"), ("2021-06-01", "2022-03-31")]
    )
    def test_window_figures_ignore_trimming(self, plain, tmp_path, monkeypatch, start, end):
        """Win rate and mean come from every trade, however few rows the page holds."""
        monkeypatch.setattr(embed, "TRADE_TAIL", 3)
        db, data = self.biased(plain)
        assert data["embed"]["trades"]["rows_embedded"] < data["embed"]["trades"]["rows_total"]
        got = self.window(tmp_path, data, start, end)
        closed = M.closed_trades(db.table("trades").collect()).filter(
            pl.col("exit_date").is_between(dt.date.fromisoformat(start), dt.date.fromisoformat(end))
        )
        assert got["n"] == closed.height
        assert got["wins"] == (closed["ret"] > 0).sum()
        assert got["mean"] == pytest.approx(closed["ret"].mean(), abs=1e-12)

    def test_side_figures_match_the_summary(self, full, tmp_path):
        data = embed.build_payload(full)
        got = self.window(tmp_path, data, "2000-01-01", "2100-01-01")
        for side, value in full.summary()["structure"]["sides"].items():
            for key in ("n", "win_rate", "avg_ret", "contrib"):
                assert got[side][key] == pytest.approx(value[key], abs=1e-9), key

    @pytest.mark.parametrize("i", [0, 1, 2])
    def test_stop_replay_uses_every_trade(self, full, tmp_path, i):
        data = embed.build_payload(full)
        stop = data["stops"][i]
        got = self.window(tmp_path, data, "2000-01-01", "2100-01-01")["stops"][i]
        closed = M.closed_trades(full.table("trades").collect()).drop_nulls("mae")
        hit = closed["mae"] <= -stop
        clipped = closed.with_columns(
            pl.when(pl.col("mae") <= -stop).then(pl.lit(-stop)).otherwise(pl.col("ret")).alias("r")
        )["r"]
        assert got["hit"] == hit.sum()
        assert got["win"] == pytest.approx((clipped > 0).mean(), abs=1e-12)
        assert got["avg"] == pytest.approx(clipped.mean(), abs=1e-9)

    def test_window_histogram_over_all_days_is_the_whole_histogram(self, full, tmp_path):
        data = embed.build_payload(full)
        args = f"{json.dumps(data['hist'])}, {self.rows(data['hist_days'])}"
        got = self.call(tmp_path, f"BT.windowHist({args}, '2000-01-01', '2100-01-01')")
        assert got["counts"] == data["hist"]["counts"]
        assert got["n"] == data["hist"]["n"]

    def test_periods_are_clipped_to_the_range(self, tmp_path):
        periods = json.dumps([{"a": 0, "b": 9}, {"a": 10, "b": 19}, {"a": 20, "b": 29}])
        got = self.call(tmp_path, f"BT.periodsIn({periods}, 5, 12)")
        assert got == [{"a": 5, "b": 9}, {"a": 10, "b": 12}]

    @pytest.mark.parametrize(
        ("embedded", "complete"), [(100, True), (99, False)]
    )
    def test_detail_state_comes_from_counts(self, tmp_path, embedded, complete):
        embed = json.dumps({"trades": {"rows_total": 100, "rows_embedded": embedded}})
        got = self.call(tmp_path, f"BT.embedStatus({embed})")
        assert got["complete"] is complete
        assert got["parts"] == [{"name": "trades", "embedded": embedded, "total": 100}]


@needs_node
class TestSources:
    STRINGS = json.dumps((DASHBOARD / "strings.js").as_posix())

    @pytest.mark.parametrize("name", ["core.js", "strings.js", "app.js"])
    def test_syntax(self, name):
        subprocess.run([NODE, "--check", str(DASHBOARD / name)], check=True)

    def test_languages_have_the_same_keys(self, tmp_path):
        script = f"""
const {{I18N}} = require({self.STRINGS});
const keys = (o) => Object.keys(o).sort();
console.log(JSON.stringify({{
  en: keys(I18N.en), zh: keys(I18N["zh-TW"]),
  enM: keys(I18N.en.m), zhM: keys(I18N["zh-TW"].m),
  enF: keys(I18N.en.fmt), zhF: keys(I18N["zh-TW"].fmt),
}}));
"""
        got = run_node(tmp_path, script)
        assert got["en"] == got["zh"]
        assert got["enM"] == got["zhM"]
        assert got["enF"] == got["zhF"]

    def test_every_string_the_page_asks_for_exists(self, tmp_path):
        source = (DASHBOARD / "app.js").read_text(encoding="utf-8")
        wanted = sorted(set(re.findall(r'\bt\("([A-Za-z_0-9]+)"[,)]', source)))
        wanted += ["held_trades", "held_fills", "held_blocks"]
        script = f"""
const {{I18N}} = require({self.STRINGS});
console.log(JSON.stringify({json.dumps(wanted)}.filter(k => !(k in I18N.en))));
"""
        assert run_node(tmp_path, script) == []

    def test_every_quality_check_has_a_label_and_a_target(self, tmp_path):
        keys = [check[0] for checks in viz._QUALITY_CHECKS.values() for check in checks]
        script = f"""
const {{I18N, TARGETS}} = require({self.STRINGS});
console.log(JSON.stringify({json.dumps(keys)}.filter(k => !I18N.en.m[k] || !TARGETS[k])));
"""
        assert run_node(tmp_path, script) == []

    def test_every_flag_has_a_label(self, full, tmp_path):
        codes = [f"flag_{flag['code']}" for flag in full.summary()["flags"]]
        codes.append("flag_long_short_balance")
        script = f"""
const {{I18N}} = require({self.STRINGS});
console.log(JSON.stringify({json.dumps(codes)}.filter(k => !I18N.en[k])));
"""
        assert run_node(tmp_path, script) == []


class TestPackaging:
    def test_page_assets_are_listed_with_the_package(self):
        names = {p.name for p in DASHBOARD.rglob("*") if p.is_file()}
        assert {"index.html", "app.css", "core.js", "strings.js", "app.js"} <= names
        assert {"manrope-latin.woff2", "OFL.txt"} <= names

    @pytest.mark.slow
    @pytest.mark.skipif(shutil.which("maturin") is None, reason="maturin not available")
    def test_wheel_carries_the_page(self, tmp_path):
        """A full release build, so it runs with the slow tests."""
        root = Path(page.__file__).parents[3]
        subprocess.run(
            ["maturin", "build", "--out", str(tmp_path), "--quiet"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        names = zipfile.ZipFile(next(tmp_path.glob("*.whl"))).namelist()
        assert any(n.endswith("report/dashboard/app.js") for n in names)
        assert any(n.endswith("report/dashboard/vendor/manrope-latin.woff2") for n in names)
