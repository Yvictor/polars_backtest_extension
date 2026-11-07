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


def test_report_html_escapes_hostile_symbol():
    """Hostile symbol / display names must never reach the HTML unescaped."""
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

    # no script breakout: only the page's own two script elements close
    assert html.count("</script>") == 2
    # the raw injection payloads never appear unescaped anywhere in the document
    assert "</script><img" not in html
    assert "<img" not in html
    assert "<b>bad</b>" not in html


def test_trades_payload_includes_signal_dates(report):
    payload = viz.report_data(report)
    t = payload["trades"]
    assert "entry_sig" in t and "exit_sig" in t
    assert len(t["entry_sig"]) == len(t["entry"])
    # signal precedes execution (T+1)
    for sig, ent in zip(t["entry_sig"], t["entry"]):
        if sig and ent:
            assert sig <= ent


class TestMergeRolledTrades:
    def _seg(self, stock, entry, exit, ret, sig=None, mae=None, pos=0.01):
        import datetime as dt
        d = lambda s: dt.date.fromisoformat(s) if s else None
        return {
            "stock_id": stock, "entry_date": d(entry), "exit_date": d(exit),
            "entry_sig_date": d(sig or entry), "exit_sig_date": d(exit),
            "return": ret, "mae": mae if mae is not None else min(ret, 0.0),
            "gmfe": max(ret, 0.0), "bmfe": 0.0, "mdd": min(ret, 0.0),
            "position": pos, "period": 1, "pdays": 1 if ret > 0 else 0,
            "entry_price": 100.0, "exit_price": 100.0 * (1 + ret),
            "entry_raw_price": 100.0, "exit_raw_price": 100.0 * (1 + ret),
        }

    def test_chained_segments_merge_into_round_trip(self):
        import polars as pl
        segs = pl.DataFrame([
            self._seg("2330", "2024-01-02", "2024-01-03", 0.10),
            self._seg("2330", "2024-01-03", "2024-01-04", -0.05),
            self._seg("2330", "2024-01-04", "2024-01-05", 0.02),
            # gap → separate round trip
            self._seg("2330", "2024-01-10", "2024-01-11", 0.01),
            self._seg("1101", "2024-01-03", "2024-01-04", 0.03),
        ])
        merged = viz._merge_rolled_trades(segs).sort("stock_id", "entry_date")
        assert merged.height == 3
        trip = merged.filter(
            (pl.col("stock_id") == "2330") & (pl.col("entry_date") == pl.date(2024, 1, 2))
        ).to_dicts()[0]
        assert trip["exit_date"].isoformat() == "2024-01-05"
        assert trip["return"] == pytest.approx(1.10 * 0.95 * 1.02 - 1)
        assert trip["n_segments"] == 3
        assert trip["period"] == 3
        # never below trip entry (seg maes are 0/-5%/0 but compounding keeps
        # the envelope at/above entry) -> whole-trip MAE is 0
        assert trip["mae"] == pytest.approx(0.0)
        assert trip["gmfe"] == pytest.approx(0.10)

    def test_open_position_terminates_chain(self):
        import polars as pl
        segs = pl.DataFrame([
            self._seg("2330", "2024-01-02", "2024-01-03", 0.10),
            {**self._seg("2330", "2024-01-03", None, -0.001), "exit_sig_date": None},
        ])
        merged = viz._merge_rolled_trades(segs)
        assert merged.height == 1
        row = merged.to_dicts()[0]
        assert row["exit_date"] is None and row["n_segments"] == 2
        assert row["return"] == pytest.approx(1.10 * 0.999 - 1)
