"""Tests for input validation added after the 2026-07 code review.

Covers: negative/unparseable resample_offset (H2), null handling at the FFI
boundary (H3), MS/QS rejection (M4), skip_sort misuse guard (M7), and the
removal of the no-op finlab_mode parameter (H1).
"""

import datetime

import pytest
import polars as pl

import polars_backtest as pl_bt
from polars_backtest._polars_backtest import backtest as _rust_backtest


@pytest.fixture(scope="module")
def df():
    rows = []
    start = datetime.date(2024, 1, 1)
    for i in range(90):
        date = start + datetime.timedelta(days=i)
        if date.weekday() >= 5:
            continue
        for j, symbol in enumerate(["AAPL", "GOOG"]):
            price = 100.0 * (j + 1) * (1 + 0.002 * i)
            rows.append({
                "date": date,
                "symbol": symbol,
                "close": price,
                "weight": 0.5,
            })
    return pl.DataFrame(rows)


class TestResampleOffsetValidation:
    def test_negative_offset_rejected(self, df):
        # Used to be silently ignored (no offset applied at all).
        with pytest.raises(ValueError, match="resample_offset"):
            pl_bt.backtest(df, resample="M", resample_offset="-1D")

    def test_garbage_offset_rejected(self, df):
        with pytest.raises(ValueError, match="resample_offset"):
            pl_bt.backtest(df, resample="M", resample_offset="tomorrow")

    def test_positive_offsets_accepted(self, df):
        for offset in ("1D", "5d", "1W", "2w"):
            result = pl_bt.backtest(df, resample="M", resample_offset=offset)
            assert result.height > 0


class TestResampleValidation:
    def test_month_start_rejected(self, df):
        # MS/QS used to be silently treated as month/quarter END.
        with pytest.raises(ValueError, match="MS"):
            pl_bt.backtest(df, resample="MS")

    def test_quarter_start_rejected(self, df):
        with pytest.raises(ValueError, match="QS"):
            pl_bt.backtest_with_report(df, resample="QS")

    def test_unknown_resample_rejected_in_report(self, df):
        # backtest_with_report previously skipped resample validation entirely
        with pytest.raises(ValueError, match="resample"):
            pl_bt.backtest_with_report(df, resample="fortnight")


class TestNullHandling:
    def test_null_date_rejected(self, df):
        bad = pl.concat([df, df.head(1).with_columns(pl.lit(None, dtype=pl.Date).alias("date"))])
        with pytest.raises(ValueError, match="date"):
            pl_bt.backtest(bad, resample="M")

    def test_null_symbol_rejected(self, df):
        bad = pl.concat([df, df.head(1).with_columns(pl.lit(None, dtype=pl.String).alias("symbol"))])
        with pytest.raises(ValueError, match="symbol"):
            pl_bt.backtest(bad, resample="M")

    def test_null_price_treated_as_missing(self, df):
        # A null close (e.g. suspension day) must not corrupt results — it is
        # converted to NaN, the engine's missing marker, instead of reaching the
        # FFI boundary as undefined buffer bytes.
        bad = df.with_columns(
            pl.when((pl.col("symbol") == "GOOG") & (pl.col("date") == datetime.date(2024, 2, 1)))
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close")
        )
        result = pl_bt.backtest(bad, resample="M")
        assert result.get_column("creturn").is_finite().all()

    def test_float32_price_accepted(self, df):
        result = pl_bt.backtest(df.with_columns(pl.col("close").cast(pl.Float32)), resample="M")
        expected = pl_bt.backtest(df, resample="M")
        assert result.get_column("creturn").to_list() == pytest.approx(
            expected.get_column("creturn").to_list()
        )


class TestSkipSortGuard:
    def test_unsorted_with_skip_sort_rejected(self, df):
        shuffled = df.sort("symbol", "date")  # date no longer globally sorted
        assert not shuffled.get_column("date").is_sorted()
        with pytest.raises(ValueError, match="not sorted"):
            _rust_backtest(
                shuffled, "date", "symbol", "close", "weight",
                "open", "high", "low", None, "M", None, None, True,
            )

    def test_unsorted_input_auto_sorted_via_namespace(self, df):
        # The namespace detects unsorted input and sorts — same result either way.
        shuffled = df.sort("symbol", "date")
        a = pl_bt.backtest(shuffled, resample="M")
        b = pl_bt.backtest(df, resample="M")
        assert a.get_column("creturn").to_list() == b.get_column("creturn").to_list()


class TestFinlabModeRemoved:
    def test_finlab_mode_no_longer_accepted(self, df):
        # The parameter never had any effect on the long-format engine (H1).
        with pytest.raises(TypeError):
            pl_bt.backtest(df, resample="M", finlab_mode=True)
        with pytest.raises(TypeError):
            pl_bt.backtest_with_report(df, resample="M", finlab_mode=True)


class TestTradingValueForwarding:
    def test_custom_trading_value_column(self, df):
        # capacity only counts closed trades, so make positions exit mid-way
        named = df.with_columns(
            (pl.col("close") * 1_000_000).alias("tv"),
            pl.when(pl.col("date") < datetime.date(2024, 2, 15))
            .then(pl.col("weight"))
            .otherwise(0.0)
            .alias("weight"),
        )
        report = pl_bt.backtest_with_report(named, resample="M", trading_value="tv")
        metrics = report.get_metrics(sections=["liquidity"])
        assert metrics.to_dicts()[0]["capacity"] is not None

    def test_capacity_null_without_column(self, df):
        report = pl_bt.backtest_with_report(df, resample="M")
        metrics = report.get_metrics(sections=["liquidity"])
        assert metrics.to_dicts()[0]["capacity"] is None
