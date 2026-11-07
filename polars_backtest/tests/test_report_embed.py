"""What a page carries from a bundle: complete daily inputs, trimmed detail rows."""

from __future__ import annotations

import datetime as dt
import json

import polars as pl
import polars_backtest as pl_bt
import pytest
from polars_backtest.report import embed, open_bundle
from report_fixtures import (
    NAMES,
    LedgerReport,
    benchmark,
    fake_ledger,
    huge_fills_report,
    long_report,
    prices,
)

ROOMY = dict(embed.PAGE_LIMITS, fills=50_000)


def decoded(column: object) -> list:
    """A payload column as a list, the same step a page takes when it reads one."""
    if isinstance(column, dict):
        return [None if i is None else column["d"][i] for i in column["i"]]
    return column


@pytest.fixture(scope="module")
def plain():
    return long_report()


@pytest.fixture(scope="module")
def full(plain):
    report = LedgerReport(plain, benchmark=benchmark(), **fake_ledger(plain))
    return pl_bt.report.to_bundle(report, title="full", input_df=prices(), symbol_names=NAMES)


@pytest.fixture(scope="module")
def huge(plain):
    return pl_bt.report.to_bundle(huge_fills_report(plain))


class TestPayload:
    def test_payload_is_strict_json(self, full):
        json.dumps(embed.build_payload(full), allow_nan=False)

    def test_payload_survives_disk(self, full, tmp_path):
        full.write(tmp_path / "run")
        assert embed.build_payload(open_bundle(tmp_path / "run")) == embed.build_payload(full)

    def test_summary_detail_leaves_rows_out(self, full):
        data = embed.build_payload(full, detail="summary")
        assert data["tables"] == {}
        assert data["embed"]["trades"]["policy"] == "left out"
        assert data["embed"]["trades"]["rows_embedded"] == 0

    def test_summary_detail_keeps_the_curve_and_totals(self, full):
        data = embed.build_payload(full, detail="summary")
        assert len(data["curve"]["creturn"]) == full.table("curve").collect().height
        assert data["summary"] == full.summary()

    def test_bad_detail_is_refused(self, full):
        with pytest.raises(ValueError, match="detail"):
            embed.build_payload(full, detail="everything")

    def test_requested_day_and_capital_are_carried(self, full):
        data = embed.build_payload(full, day="2022-03-01", capital=1e8)
        assert (data["day"], data["capital"]) == ("2022-03-01", 1e8)

    def test_summary_written_before_a_field_was_added_still_reads(self, full, tmp_path):
        full.write(tmp_path / "run")
        path = tmp_path / "run" / "summary.json"
        older = json.loads(path.read_text(encoding="utf-8"))
        del older["structure"]
        path.write_text(json.dumps(older), encoding="utf-8")
        assert embed.build_payload(open_bundle(tmp_path / "run"))["summary"]["structure"] == {}


class TestDailyInputs:
    def test_curve_is_exact(self, full):
        data = embed.build_payload(full)
        assert data["curve"]["creturn"] == full.table("curve").collect()["creturn"].to_list()

    def test_long_curve_stays_complete(self):
        n = 30_000
        days = [dt.date(1900, 1, 1) + dt.timedelta(days=i) for i in range(n)]
        values = [1.0 + ((i * 7) % 101) / 100 for i in range(n)]

        class Long:
            def __init__(self, inner):
                self.inner = inner

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def daily_creturn(self):
                return pl.DataFrame({"date": days, "creturn": values})

        db = pl_bt.report.to_bundle(Long(long_report(120)))
        assert embed.build_payload(db)["curve"]["creturn"] == values

    def test_rounding_keeps_eight_digits_on_detail_rows(self):
        assert embed._round(1.23456789123) == 1.2345679


class TestTrimming:
    def test_bundle_stays_complete(self, huge):
        embed.build_payload(huge, limits=ROOMY)
        assert huge.table("fills").collect().height == huge.manifest["tables"]["fills"]["rows"]
        assert huge.manifest["tables"]["fills"]["rows"] > 4_000_000

    def test_a_day_is_all_in_or_all_out(self, huge):
        data = embed.build_payload(huge, limits=ROOMY)
        kept = pl.DataFrame({"date": decoded(data["tables"]["fills"]["date"])})
        counts = dict(kept.group_by("date").len().iter_rows())
        whole = dict(
            huge.table("fills")
            .group_by("date")
            .len()
            .collect()
            .with_columns(pl.col("date").dt.strftime("%Y-%m-%d"))
            .iter_rows()
        )
        assert counts
        for day, rows in counts.items():
            assert rows == whole[day]

    def test_day_totals_stay_complete(self, huge):
        data = embed.build_payload(huge, limits=ROOMY)
        assert sum(data["day_index"]["n_fills"]) == huge.manifest["tables"]["fills"]["rows"]

    def test_notes_say_what_was_cut(self, huge):
        note = embed.build_payload(huge, limits=ROOMY)["embed"]["fills"]
        assert note["policy"] == "whole days"
        assert 0 < note["rows_embedded"] <= 50_000 < note["rows_total"]

    def test_notes_say_when_nothing_was_cut(self, full):
        note = embed.build_payload(full)["embed"]["trades"]
        assert note["policy"] == "complete"
        assert note["rows_embedded"] == note["rows_total"]

    def test_choice_of_days_repeats(self, huge):
        assert embed._priority_days(huge) == embed._priority_days(huge)

    def test_drawdown_troughs_come_first(self, huge):
        frame = huge.table("drawdowns").filter(pl.col("series") == "strategy").sort("rank")
        trough = frame.collect()["trough"][0]
        assert trough in embed._priority_days(huge)[:11]

    def test_trade_trim_keeps_tails_and_open_trades(self):
        n = 20_000
        trades = pl.DataFrame(
            {
                "status": ["closed"] * (n - 1) + ["open"],
                "ret": [float(i - n / 2) for i in range(n)],
                "weight": [1.0] * n,
                "exit_date": [dt.date(2024, 1, 1)] * n,
            }
        )
        kept = embed._trim_trades(trades, 3_000)
        assert kept.height <= 3_000
        assert kept["ret"].min() == trades["ret"].min()
        assert (kept["status"] == "open").sum() == 1
        assert trades.filter(pl.col("status") == "closed")["ret"].max() in kept["ret"]

    def test_histogram_counts_every_trade(self, full):
        hist = embed.build_payload(full)["hist"]
        closed = full.table("trades").filter(pl.col("status") == "closed").collect()
        assert sum(hist["counts"]) == closed.height == hist["n"]
