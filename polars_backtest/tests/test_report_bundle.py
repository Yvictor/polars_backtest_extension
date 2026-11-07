"""The report bundle: contract, round trip, absent tables, queries and the CLI."""

from __future__ import annotations

import datetime as dt
import itertools
import json
from pathlib import Path

import polars as pl
import polars_backtest as pl_bt
import pytest
from polars.testing import assert_frame_equal
from polars_backtest import viz
from polars_backtest.report import (
    SCHEMA_VERSION,
    BundleCheckError,
    MissingTableError,
    SchemaError,
    open_bundle,
)
from polars_backtest.report import metrics as M
from polars_backtest.report import schema as S
from polars_backtest.report.__main__ import main
from polars_backtest.report.bundle import MANIFEST
from polars_backtest.viz import _merge_rolled_trades
from report_fixtures import (
    LEDGER_TABLES,
    NAMES,
    SYMBOLS,
    LedgerReport,
    benchmark,
    fake_ledger,
    long_report,
    prices,
    weekdays,
)

SPEC = Path(__file__).parents[2] / "specs" / "report_bundle.schema.json"
ATTRS = {
    "ledger_daily": "daily",
    "fills": "fills",
    "blocks": "blocks",
    "cashflows": "cashflows",
    "orders": "orders",
}


@pytest.fixture(scope="module")
def plain():
    return long_report()


@pytest.fixture(scope="module")
def ledger(plain):
    return fake_ledger(plain)


@pytest.fixture(scope="module")
def full_report(plain, ledger):
    return LedgerReport(plain, benchmark=benchmark(), **ledger)


@pytest.fixture(scope="module")
def full(full_report):
    return pl_bt.report.to_bundle(
        full_report, title="full", input_df=prices(), symbol_names=NAMES
    )


def curve_frame(values: list[float], start: dt.date = dt.date(2024, 1, 1)) -> pl.DataFrame:
    dates = [start + dt.timedelta(days=i) for i in range(len(values))]
    frame = pl.DataFrame({"date": dates, "creturn": [float(v) for v in values]})
    return frame.with_columns(
        (pl.col("creturn") / pl.col("creturn").shift(1) - 1).alias("ret"),
        (pl.col("creturn") / pl.col("creturn").cum_max() - 1).alias("drawdown"),
    )


class TestContract:
    def test_every_table_matches_the_registry(self, full):
        for name in full.manifest["tables"]:
            if full.has(name):
                assert S.validate(name, full.table(name).collect()) == []

    def test_required_tables_always_exist(self, plain):
        db = pl_bt.report.to_bundle(plain)
        for table in S.TABLES.values():
            if table.required:
                assert db.coverage[table.name] == "present"

    def test_optional_column_is_absent_not_null(self, plain):
        columns = pl_bt.report.to_bundle(plain).table("curve").collect_schema().names()
        assert "bench_creturn" not in columns
        assert "gross_creturn" not in columns

    def test_spec_file_matches_the_registry(self):
        assert json.loads(SPEC.read_text(encoding="utf-8")) == S.json_schema()

    def test_engine_names_are_translated(self, full):
        columns = full.table("trades").collect_schema().names()
        assert {"symbol", "ret", "weight"} <= set(columns)
        assert not {"stock_id", "return", "position"} & set(columns)

    def test_unknown_engine_column_is_kept(self):
        frame = pl.DataFrame(
            {"date": [dt.date(2024, 1, 2)], "symbol": ["FAKE001"], "new_engine_field": [1.5]}
        )
        assert "new_engine_field" in S.conform("fills", frame).columns

    def test_missing_required_column_names_it(self):
        with pytest.raises(SchemaError, match="symbol"):
            S.conform("fills", pl.DataFrame({"date": [dt.date(2024, 1, 2)]}))

    def test_newer_major_is_refused(self, tmp_path, plain):
        pl_bt.report.to_bundle(plain).write(tmp_path)
        manifest = json.loads((tmp_path / MANIFEST).read_text())
        manifest["schema_version"] = "2.0"
        (tmp_path / MANIFEST).write_text(json.dumps(manifest))
        with pytest.raises(SchemaError, match="upgrade"):
            open_bundle(tmp_path)

    def test_newer_minor_is_read(self, tmp_path, plain):
        pl_bt.report.to_bundle(plain).write(tmp_path)
        manifest = json.loads((tmp_path / MANIFEST).read_text())
        manifest["schema_version"] = "1.9"
        (tmp_path / MANIFEST).write_text(json.dumps(manifest))
        assert open_bundle(tmp_path).has("curve")


class TestRoundTrip:
    def test_summary_survives_disk(self, full, tmp_path):
        full.write(tmp_path)
        assert open_bundle(tmp_path).summary() == full.summary()

    def test_markdown_survives_disk(self, full, tmp_path):
        full.write(tmp_path)
        assert open_bundle(tmp_path).summary("md") == full.summary("md")

    def test_tables_survive_disk(self, full, tmp_path):
        full.write(tmp_path)
        back = open_bundle(tmp_path)
        for name in full.manifest["tables"]:
            if full.has(name):
                assert_frame_equal(
                    full.table(name).collect(), back.table(name).collect(), check_exact=True
                )

    def test_files_written(self, full, tmp_path):
        full.write(tmp_path)
        names = {p.name for p in tmp_path.iterdir()}
        assert {"manifest.json", "summary.json", "SUMMARY.md", "curve.parquet"} <= names

    def test_to_bundle_with_path_returns_the_opened_bundle(self, plain, tmp_path):
        db = plain.to_bundle(tmp_path / "run", title="on disk")
        assert db.path == tmp_path / "run"

    def test_sha_detects_a_changed_file(self, full, tmp_path):
        full.write(tmp_path)
        open_bundle(tmp_path, verify=True)
        (tmp_path / "curve.parquet").write_bytes(b"changed")
        with pytest.raises(SchemaError, match="sha256"):
            open_bundle(tmp_path, verify=True)

    def test_write_over_own_source_is_refused(self, full, tmp_path):
        full.write(tmp_path)
        with pytest.raises(ValueError, match="already stored"):
            open_bundle(tmp_path).write(tmp_path)

    def test_rewrite_from_disk(self, full, tmp_path):
        full.write(tmp_path / "a")
        open_bundle(tmp_path / "a").write(tmp_path / "b")
        assert open_bundle(tmp_path / "b").summary() == full.summary()

    def test_config_hash_ignores_results(self, plain):
        other = pl_bt.backtest_with_report(prices(400), resample="M", stop_loss=0.1)
        first = pl_bt.report.to_bundle(plain).manifest["run"]["config_hash"]
        assert pl_bt.report.to_bundle(other).manifest["run"]["config_hash"] == first


class TestAbsentMatrix:
    @pytest.mark.parametrize(
        "present",
        [c for n in range(6) for c in itertools.combinations(LEDGER_TABLES, n)],
        ids=lambda c: "+".join(c) or "none",
    )
    def test_any_subset_builds_with_true_coverage(self, plain, ledger, present):
        tables = {ATTRS[name]: ledger[ATTRS[name]] for name in present}
        db = pl_bt.report.to_bundle(LedgerReport(plain, **tables))

        for name in LEDGER_TABLES:
            assert db.coverage[name] == ("present" if name in present else "absent")
            assert db.summary()["coverage"][name] == db.coverage[name]
        assert (db.summary()["costs"]["source"] == "ledger") == ("ledger_daily" in present)
        json.dumps(db.summary(), allow_nan=False)

    def test_absent_table_says_why(self, plain):
        db = pl_bt.report.to_bundle(plain)
        with pytest.raises(MissingTableError, match="ledger=False"):
            db.table("fills")

    def test_missing_columns_are_partial(self, plain, ledger):
        daily = ledger["daily"].drop("cash", "gross_short")
        db = pl_bt.report.to_bundle(LedgerReport(plain, daily=daily))
        entry = db.manifest["tables"]["ledger_daily"]
        assert entry["state"] == "partial"
        assert entry["columns_missing"] == ["cash", "gross_short"]

    def test_partial_daily_still_reports_what_it_can(self, plain, ledger):
        daily = ledger["daily"].drop("cash")
        summary = pl_bt.report.to_bundle(LedgerReport(plain, daily=daily)).summary()
        assert summary["exposure"] == {"source": "absent"}
        assert summary["costs"]["source"] == "ledger"

    def test_zero_row_ledger_with_a_schema_builds(self, plain, ledger):
        report = LedgerReport(plain, daily=ledger["daily"].clear())
        db = pl_bt.report.to_bundle(report)
        summary = db.summary()
        assert db.coverage["ledger_daily"] == "empty"
        assert summary["costs"]["source"] == "estimate"
        assert summary["exposure"] == {"source": "absent"}
        assert summary["turnover"]["source"] == "estimate"
        assert summary["net_of_costs"] is None
        assert "gross_creturn" not in db.table("curve").collect_schema().names()

    def test_ledger_without_positive_equity_builds(self, plain, ledger):
        daily = ledger["daily"].with_columns(pl.lit(0.0).alias("equity"))
        summary = pl_bt.report.to_bundle(LedgerReport(plain, daily=daily)).summary()
        assert summary["costs"]["source"] == "estimate"
        assert summary["exposure"] == {"source": "absent"}

    def test_zero_row_table_is_empty(self, plain, ledger):
        db = pl_bt.report.to_bundle(LedgerReport(plain, blocks=ledger["blocks"].clear()))
        assert db.coverage["blocks"] == "empty"
        assert db.table("blocks").collect().height == 0

    def test_no_trades_builds(self, plain):
        db = pl_bt.report.to_bundle(LedgerReport(plain, trades=pl.DataFrame()))
        assert db.coverage["trades"] == "absent"
        assert db.summary()["trades"]["n_round_trips"] == 0
        json.dumps(db.summary(), allow_nan=False)

    def test_empty_curve_fails_the_same_way_as_viz(self):
        rows = prices(40).with_columns(pl.lit(0.0).alias("weight"))
        with pytest.raises(ValueError, match="no equity curve"):
            pl_bt.report.to_bundle(pl_bt.backtest_with_report(rows, resample="M"))


class TestReconcile:
    def test_synthetic_ledger_passes(self, full):
        assert {c["id"]: c["status"] for c in full.checks} == {
            "curve_vs_equity": "pass",
            "traded_vs_fills": "pass",
            "daily_cost_vs_fills": "pass",
        }

    @pytest.mark.parametrize(
        ("column", "check"),
        [("traded_notional", "traded_vs_fills"), ("commission", "daily_cost_vs_fills")],
    )
    def test_one_changed_row_fails_its_check(self, plain, ledger, column, check):
        row = ledger["fills"]["date"][0]
        daily = ledger["daily"].with_columns(
            pl.when(pl.col("date") == row)
            .then(pl.col(column) + 0.5)
            .otherwise(pl.col(column))
            .alias(column)
        )
        report = LedgerReport(plain, daily=daily, fills=ledger["fills"])
        db = pl_bt.report.to_bundle(report)
        assert {c["id"]: c["status"] for c in db.checks}[check] == "fail"
        assert db.summary()["checks"]["fail"] == 1

    def test_equity_off_curve_fails(self, plain, ledger):
        daily = ledger["daily"].with_columns(pl.col("equity") * (1 + 0.001 * pl.col("cash")))
        db = pl_bt.report.to_bundle(LedgerReport(plain, daily=daily))
        assert {c["id"]: c["status"] for c in db.checks}["curve_vs_equity"] == "fail"

    def test_strict_raises(self, plain, ledger):
        daily = ledger["daily"].with_columns(pl.col("commission") + 0.1)
        report = LedgerReport(plain, daily=daily, fills=ledger["fills"])
        with pytest.raises(BundleCheckError, match="daily_cost_vs_fills"):
            pl_bt.report.to_bundle(report, strict=True)

    def test_ledger_starting_before_the_curve_is_fine(self, plain, ledger):
        early = ledger["daily"].head(1).with_columns(pl.lit(dt.date(2020, 1, 2)).alias("date"))
        daily = pl.concat([early, ledger["daily"]])
        db = pl_bt.report.to_bundle(LedgerReport(plain, daily=daily, fills=ledger["fills"]))
        assert db.summary()["checks"]["fail"] == 0


class TestSummary:
    def test_size_limits(self, full):
        assert len(json.dumps(full.summary()).encode()) <= 8 * 1024
        assert len(full.summary("md").encode()) <= 2 * 1024

    def test_many_block_reasons_stay_bounded(self, plain, ledger):
        day = ledger["blocks"]["date"][0]
        blocks = pl.DataFrame(
            {
                "date": [day] * 500,
                "symbol": ["FAKE001"] * 500,
                "reason": [f"reason_{i}" for i in range(500)],
            }
        )
        summary = pl_bt.report.to_bundle(LedgerReport(plain, blocks=blocks)).summary()
        assert len(summary["execution"]["blocks_by_reason"]) == 8
        assert len(json.dumps(summary).encode()) <= 8 * 1024

    def test_performance_matches_get_stats(self, plain):
        stats = plain.get_stats().to_dicts()[0]
        perf = pl_bt.report.to_bundle(plain).summary()["performance"]
        pairs = {
            "total_return": "total_return",
            "cagr": "cagr",
            "volatility": "daily_vol",
            "sharpe": "daily_sharpe",
            "sortino": "daily_sortino",
            "calmar": "calmar",
            "max_drawdown": "max_drawdown",
        }
        for ours, theirs in pairs.items():
            assert perf[ours] == pytest.approx(stats[theirs], rel=1e-10)

    def test_monthly_matches_the_return_table(self, plain):
        table = plain.get_return_table()
        monthly = pl_bt.report.to_bundle(plain).table("monthly").collect()
        for row in monthly.to_dicts():
            year = table.filter(pl.col("year") == row["year"])
            assert year[str(row["month"])][0] == pytest.approx(row["ret"], abs=1e-12)

    def test_win_rates_are_named_apart(self, plain):
        trades = pl_bt.report.to_bundle(plain).summary()["trades"]
        assert trades["segment_win_rate"] == plain.get_metrics()["winRate"][0]
        closed = _merge_rolled_trades(plain.trades).filter(pl.col("exit_date").is_not_null())
        assert trades["win_rate"] == pytest.approx((closed["return"] > 0).mean())

    def test_open_positions_stay_out_of_the_win_rate(self, plain):
        trades = pl_bt.report.to_bundle(plain).table("trades").collect()
        opened = trades.filter(pl.col("status") == "open")
        assert opened.height > 0
        assert opened["exit_date"].null_count() == opened.height

    def test_structure_comes_from_every_trade(self, full):
        structure = full.summary()["structure"]
        assert structure["sides"]["long"]["n"] > 0
        assert "top10_contrib_ratio" in structure

    def test_structure_counts_the_trades_table(self, full):
        closed = full.table("trades").filter(pl.col("status") == "closed").collect()
        sides = full.summary()["structure"]["sides"]
        assert sum(side["n"] for side in sides.values()) == closed.height

    def test_open_marks_stay_out_of_flags_and_structure(self, plain):
        """A huge open mark must not move figures that describe closed round trips."""
        marked = plain.trades.with_columns(
            pl.when(pl.col("exit_date").is_null() & pl.col("entry_date").is_not_null())
            .then(pl.lit(50.0))
            .otherwise(pl.col("return"))
            .alias("return")
        )
        base = pl_bt.report.to_bundle(plain).summary()
        changed = pl_bt.report.to_bundle(LedgerReport(plain, trades=marked)).summary()
        assert changed["structure"] == base["structure"]

    def test_episode_trades_are_not_merged_behind_the_summary(self, plain):
        """With the engine's own episodes, summary figures still follow the trades table."""
        db = pl_bt.report.to_bundle(LedgerReport(plain, execution_model="episode"))
        trades = db.table("trades").collect()
        assert db.manifest["trade_granularity"] == "episode"
        assert trades["n_segments"].sum() == trades.height
        closed = trades.filter(pl.col("status") == "closed")
        sides = db.summary()["structure"]["sides"]
        assert sum(side["n"] for side in sides.values()) == closed.height
        assert closed.height > pl_bt.report.to_bundle(plain).summary()["trades"]["n_closed"]

    def test_non_finite_returns_stay_out(self):
        trades = pl.DataFrame(
            {
                "status": ["closed", "closed"],
                "side": ["long", "long"],
                "ret": [0.1, float("nan")],
                "weight": [1.0, 1.0],
                "entry_date": [dt.date(2024, 1, 1)] * 2,
                "exit_date": [dt.date(2024, 2, 1)] * 2,
            }
        )
        assert M.trade_structure(trades, None, None)["sides"]["long"]["contrib"] == 0.1

    def test_ledger_costs_replace_the_estimate(self, full, ledger):
        costs = full.summary()["costs"]
        assert costs["source"] == "ledger"
        assert costs["annual_drag"] == pytest.approx(costs["commission"] + costs["tax"])
        flag = next(f for f in full.summary()["flags"] if f["code"] == "cost_sensitivity")
        assert flag["source"] == "ledger"

    def test_estimate_is_labelled(self, plain):
        summary = pl_bt.report.to_bundle(plain).summary()
        assert summary["costs"]["source"] == "estimate"
        assert summary["turnover"] == {
            "source": "estimate",
            "annual_entry_weight": pytest.approx(summary["turnover"]["annual_entry_weight"]),
        }
        assert "annual_two_sided" not in summary["turnover"]


class TestDefinitions:
    def test_underwater_measures_high_to_recovery(self):
        result = M.underwater(curve_frame([1, 2, 1.5, 1.8, 2.1, 1.9]))
        assert result["longest_underwater_days"] == 3
        assert result["underwater_start"] == dt.date(2024, 1, 2)
        assert result["underwater_end"] == dt.date(2024, 1, 5)
        assert result["underwater_ongoing"] is False

    def test_underwater_never_recovered_runs_to_the_end(self):
        result = M.underwater(curve_frame([1, 2, 1.5, 1.4]))
        assert result["longest_underwater_days"] == 2
        assert result["underwater_ongoing"] is True

    def test_rising_curve_is_never_under_water(self):
        result = M.underwater(curve_frame([1, 1.1, 1.2]))
        assert result["longest_underwater_days"] == 0
        assert result["underwater_start"] is None

    def test_entry_date_windows(self):
        result = M.entry_date_risk(curve_frame([2, 4, 1, 3, 1.5]), window=2)
        assert result["n_windows"] == 3
        assert result["worst"] == pytest.approx(-0.5)
        assert result["best"] == pytest.approx(0.5)
        assert result["median"] == pytest.approx(-0.25)
        assert result["positive_share"] == pytest.approx(1 / 3)
        assert result["worst_start"] == dt.date(2024, 1, 1)

    def test_entry_date_needs_a_full_window(self):
        assert M.entry_date_risk(curve_frame([1, 2]), window=2) is None

    def test_concentration_sets_best_months_flat(self):
        curve = curve_frame([1.0, 2.0], start=dt.date(2024, 1, 1))
        days = dt.date(2025, 1, 1) - dt.date(2024, 1, 1)
        curve = curve.with_columns(
            pl.Series("date", [dt.date(2024, 1, 1), dt.date(2024, 1, 1) + days])
        )
        monthly = pl.DataFrame({"ret": [1.0] + [0.0] * 19})
        result = M.concentration(monthly, curve)
        assert result["k"] == 1
        assert result["cagr_ex_best"] == pytest.approx(0.0, abs=1e-12)
        assert result["level"] == "red"

    def test_concentration_green_when_spread(self):
        curve = curve_frame([1.0, 1.01, 1.02])
        monthly = pl.DataFrame({"ret": [0.01] * 24})
        assert M.concentration(monthly, curve)["level"] in ("green", "amber", "red")

    def test_gross_curve_adds_costs_back(self):
        curve = curve_frame([1.0, 0.99])
        ledger = pl.DataFrame(
            {
                "date": curve["date"],
                "equity": [1.0, 0.99],
                "commission": [0.0, 0.004],
                "tax": [0.0, 0.006],
            }
        )
        gross = M.with_gross_curve(curve, ledger)["gross_creturn"].to_list()
        assert gross == pytest.approx([1.0, 1.0])

    def test_limit_fills_share(self):
        trades = pl.DataFrame(
            {
                "status": ["closed", "closed", "open"],
                "ret": [0.1, 0.3, 9.0],
                "weight": [1.0, 1.0, 1.0],
                "limit_entry": [True, False, True],
            }
        )
        result = M.limit_fills(trades)
        assert result["n"] == 1
        assert result["share_return"] == pytest.approx(0.25)
        assert result["level"] == "amber"

    def test_limit_fills_need_flags(self):
        assert M.limit_fills(pl.DataFrame({"status": ["closed"], "ret": [0.1]})) is None


class TestFlags:
    @pytest.mark.parametrize(
        ("capacity", "level"),
        [(None, "na"), (5e7, "green"), (1e7, "amber"), (9e6, "red")],
    )
    def test_capacity(self, capacity, level):
        assert M._capacity_flag({"capacity": capacity})["level"] == level

    @pytest.mark.parametrize(
        ("summary", "metrics", "level", "green"),
        [
            ({"entry_locked_contrib_ratio": 0.05}, {}, "green", 0.05),
            ({"entry_locked_contrib_ratio": 0.15}, {}, "amber", 0.05),
            ({"entry_locked_contrib_ratio": 0.16}, {}, "red", 0.05),
            ({"buy_high_contrib_ratio": 0.2}, {}, "amber", 0.10),
            ({"buy_high_ratio": 0.2}, {}, "red", 0.05),
            ({}, {"buyHigh": 0.04}, "green", 0.05),
            ({}, {}, "na", 0.05),
        ],
    )
    def test_limit_dependence_prefers_the_strongest_evidence(self, summary, metrics, level, green):
        flag = M._limit_flag(metrics, summary)
        assert (flag["level"], flag["green_max"]) == (level, green)

    @pytest.mark.parametrize(
        ("drag", "cagr", "level"),
        [(0.02, 0.1, "green"), (0.05, 0.1, "amber"), (0.06, 0.1, "red"), (0.05, -0.1, "na")],
    )
    def test_cost_sensitivity(self, drag, cagr, level):
        assert M._cost_flag(drag, "estimate", cagr)["level"] == level

    @pytest.mark.parametrize(
        ("ratio", "level"), [(0.7, "green"), (0.4, "amber"), (0.39, "red"), (None, "na")]
    )
    def test_alpha_decay(self, ratio, level):
        assert M._decay_flag({"ratio": ratio, "recent": 1, "full": 1})["level"] == level

    def test_balance_needs_shorts(self):
        assert M._balance_flag({"sides": {"long": {"contrib": 1.0}}}) is None

    @pytest.mark.parametrize(
        ("long_c", "short_c", "level"),
        [(1.0, 0.5, "green"), (1.0, -0.05, "amber"), (1.0, -0.5, "red")],
    )
    def test_balance(self, long_c, short_c, level):
        sides = {"long": {"contrib": long_c}, "short": {"contrib": short_c}}
        assert M._balance_flag({"sides": sides})["level"] == level


class TestQuery:
    def test_day_returns_every_table_it_has(self, full):
        day = full.table("fills").collect()["date"][0]
        result = full.day(day)
        assert set(result) == {"summary", "fills", "blocks", "cashflows", "holdings"}
        assert result["summary"].height == 1
        assert result["fills"].height > 0

    def test_day_accepts_text(self, full):
        day = full.table("fills").collect()["date"][0]
        assert full.day(day.isoformat())["summary"].height == 1

    def test_day_leaves_out_what_the_bundle_lacks(self, plain):
        result = pl_bt.report.to_bundle(plain).day(plain.creturn["date"][-1])
        assert set(result) == {"summary", "holdings"}

    def test_holdings_are_the_open_names(self, plain):
        db = pl_bt.report.to_bundle(plain)
        trade = db.table("trades").filter(pl.col("status") == "closed").collect().row(0, named=True)
        inside = trade["entry_date"] + (trade["exit_date"] - trade["entry_date"]) / 2
        assert trade["symbol"] in db.day(inside)["holdings"]["symbol"].to_list()
        assert db.day(trade["exit_date"])["holdings"].filter(
            (pl.col("symbol") == trade["symbol"]) & (pl.col("entry_date") == trade["entry_date"])
        ).is_empty()

    def test_symbol_lookup(self, full):
        result = full.symbol(SYMBOLS[0])
        assert result["summary"]["name"][0] == NAMES[SYMBOLS[0]]
        assert result["trades"]["symbol"].unique().to_list() == [SYMBOLS[0]]

    def test_sql_applies_the_limit(self, full):
        assert full.sql("select * from trades", limit=3).height == 3

    def test_sql_joins_tables(self, full):
        rows = full.sql(
            "select t.symbol, s.name from trades t join symbols s on t.symbol = s.symbol", limit=1
        )
        assert rows["name"][0].startswith("Fake Co")

    def test_query_says_when_truncated(self, full):
        result = full.query("select * from trades", limit=2)
        assert result["returned"] == 2
        assert result["truncated"] is True
        assert result["total"] > 2

    def test_day_index_matches_trades(self, full):
        trades = full.table("trades").collect()
        index = full.table("day_index").collect()
        entered = trades.filter(pl.col("entry_date").is_in(index["date"].implode()))
        assert index["n_entries"].sum() == entered.height

    def test_symbol_index_sums_to_closed_trades(self, full):
        closed = full.table("trades").filter(pl.col("status") == "closed").collect()
        index = full.table("symbol_index").collect()
        assert index["n_trades"].sum() == closed.height


class TestReportKinds:
    def test_long_report_has_the_methods(self, plain):
        assert isinstance(plain.summary(), dict)
        assert plain.summary("md").startswith("# ")

    def test_wide_report_has_the_methods(self):
        df = prices(200).with_columns(pl.col("date").cast(pl.String))
        close = df.pivot(on="symbol", index="date", values="close").sort("date")
        weight = df.pivot(on="symbol", index="date", values="weight").sort("date")
        wide = pl_bt.backtest_with_report_wide(close, weight, resample="M")
        assert wide.summary()["performance"]["cagr"] is not None
        assert wide.summary()["trades"]["n_round_trips"] > 0
        assert viz.report_data(wide)["daily"]["dates"]

    def test_namespace_report_has_the_methods(self):
        report = prices(200).bt.backtest_with_report(position="weight", resample="M")
        assert report.to_bundle().has("trades")

    def test_ledgerless_report_in_a_notebook_is_not_written(self, plain, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        plain.to_bundle()
        assert list(tmp_path.iterdir()) == []


class TestCli:
    @pytest.fixture()
    def run(self, full, tmp_path):
        full.write(tmp_path / "run")
        return str(tmp_path / "run")

    def test_summary(self, run, capsys):
        assert main(["summary", run]) == 0
        assert json.loads(capsys.readouterr().out)["run"]["title"] == "full"

    def test_summary_markdown(self, run, capsys):
        assert main(["summary", run, "--md"]) == 0
        assert capsys.readouterr().out.startswith("# full")

    def test_schema(self, run, capsys):
        assert main(["schema", run]) == 0
        assert json.loads(capsys.readouterr().out)["tables"]["curve"]["state"] == "present"

    def test_day_has_the_query_shape(self, run, capsys):
        day = weekdays(1)[0].isoformat()
        assert main(["day", run, day, "--limit", "1"]) == 0
        assert set(json.loads(capsys.readouterr().out)["summary"]) == {
            "rows",
            "returned",
            "total",
            "truncated",
        }

    def test_symbol(self, run, capsys):
        assert main(["symbol", run, SYMBOLS[0]]) == 0
        assert json.loads(capsys.readouterr().out)["trades"]["returned"] > 0

    def test_sql_limit(self, run, capsys):
        assert main(["sql", run, "select * from trades", "--limit", "2"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["returned"] == 2 and result["truncated"] is True

    def test_output_is_strict_json_with_nan_prices(self, tmp_path, plain, ledger, capsys):
        fills = ledger["fills"].with_columns(pl.lit(float("nan")).alias("price"))
        pl_bt.report.to_bundle(LedgerReport(plain, fills=fills)).write(tmp_path / "run")
        assert main(["sql", str(tmp_path / "run"), "select price from fills"]) == 0
        assert json.loads(capsys.readouterr().out)["rows"][0]["price"] is None

    def test_missing_bundle_is_an_error_not_a_trace(self, tmp_path, capsys):
        assert main(["summary", str(tmp_path / "nope")]) == 1
        assert "no manifest.json" in capsys.readouterr().err


def test_schema_version_is_the_documented_one():
    assert SCHEMA_VERSION == "1.0"
