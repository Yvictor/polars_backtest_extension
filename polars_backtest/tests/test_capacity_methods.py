"""Tests for the capacity()/capacity_by_date() report methods and the
direction-aware, epsilon-tolerant buyHigh/sellLow liquidity metrics.

All expected values are hand-derived constants (see
specs/CAPACITY_IMPROVEMENT_PLAN.md test plan) - they are never recomputed
with the same formula as the implementation.

Fixture timeline (weekday-only calendar, resample="M", 90 calendar days):
- data spans 2024-01-01 .. 2024-03-30
- first signal 2024-01-31 (month end)  -> entry executes 2024-02-01
- weights set to 0 from 2024-02-15     -> exit signal 2024-02-29,
                                          exit executes 2024-03-01
Each capacity test asserts these engine-produced dates as a precondition so
the hand-derived constants stay valid if engine timing ever changes.
"""

import datetime

import pytest
import polars as pl

import polars_backtest as pl_bt

START = datetime.date(2024, 1, 1)
N_DAYS = 90
CUTOFF = datetime.date(2024, 2, 15)  # weights go to 0 at/after this date

ENTRY_SIG = datetime.date(2024, 1, 31)
ENTRY = datetime.date(2024, 2, 1)
EXIT_SIG = datetime.date(2024, 2, 29)
EXIT = datetime.date(2024, 3, 1)


def weekdays():
    days = [START + datetime.timedelta(days=i) for i in range(N_DAYS)]
    return [d for d in days if d.weekday() < 5]


def entered_trades(report):
    """Trades that actually entered.

    Besides executed trades, the engine emits pending re-entry rows
    (entry_date null, entry_sig_date at the resample label after the data
    end, e.g. Sunday 2024-03-31). Those never join any execution-date or
    trading-day data, so every capacity/liquidity computation ignores them.
    """
    return report.trades.filter(pl.col("entry_date").is_not_null())


def assert_fixture_trades(report, n_trades):
    """Precondition: the fixture produced the expected closed-trade timeline."""
    trades = entered_trades(report)
    assert trades.height == n_trades
    assert trades["entry_sig_date"].unique().to_list() == [ENTRY_SIG]
    assert trades["entry_date"].unique().to_list() == [ENTRY]
    assert trades["exit_sig_date"].unique().to_list() == [EXIT_SIG]
    assert trades["exit_date"].unique().to_list() == [EXIT]
    # any remaining rows must be pending entries only (never executed)
    pending = report.trades.filter(pl.col("entry_date").is_null())
    assert pending["exit_date"].is_null().all()


# ---------------------------------------------------------------------------
# Capacity fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tv_step_report():
    """Two closed trades with known trading values and positions.

    AAA: |position| = 0.5, TV = 1M before CUTOFF (covers entry 2024-02-01),
         2M from CUTOFF (covers exit 2024-03-01).
    BBB: |position| = 0.25, TV = 1M on every day.
    Weight sum is 0.75 <= 1.0, so positions are not renormalized.
    """
    rows = []
    for date in weekdays():
        active = date < CUTOFF
        rows.append({
            "date": date, "symbol": "AAA", "close": 100.0,
            "weight": 0.5 if active else 0.0,
            "trading_value": 1_000_000.0 if date < CUTOFF else 2_000_000.0,
        })
        rows.append({
            "date": date, "symbol": "BBB", "close": 200.0,
            "weight": 0.25 if active else 0.0,
            "trading_value": 1_000_000.0,
        })
    report = pl_bt.backtest_with_report(pl.DataFrame(rows), resample="M")
    assert_fixture_trades(report, n_trades=2)
    assert sorted(entered_trades(report)["position"].to_list()) == [0.25, 0.5]
    return report


@pytest.fixture(scope="module")
def const_tv_report():
    """Same two trades but constant TV = 1M for both symbols."""
    rows = []
    for date in weekdays():
        active = date < CUTOFF
        rows.append({
            "date": date, "symbol": "AAA", "close": 100.0,
            "weight": 0.5 if active else 0.0,
            "trading_value": 1_000_000.0,
        })
        rows.append({
            "date": date, "symbol": "BBB", "close": 200.0,
            "weight": 0.25 if active else 0.0,
            "trading_value": 1_000_000.0,
        })
    report = pl_bt.backtest_with_report(pl.DataFrame(rows), resample="M")
    assert_fixture_trades(report, n_trades=2)
    return report


@pytest.fixture(scope="module")
def tv_spike_report():
    """Single trade (|position| = 1.0) whose entry- and exit-day trading
    values are 50x spikes (e.g. the strategy's own volume on a rebalance
    day); every other day trades 1M."""
    rows = []
    for date in weekdays():
        rows.append({
            "date": date, "symbol": "AAA", "close": 100.0,
            "weight": 1.0 if date < CUTOFF else 0.0,
            "trading_value": 50_000_000.0 if date in (ENTRY, EXIT) else 1_000_000.0,
        })
    report = pl_bt.backtest_with_report(pl.DataFrame(rows), resample="M")
    assert_fixture_trades(report, n_trades=1)
    assert entered_trades(report)["position"].to_list() == [1.0]
    return report


@pytest.fixture(scope="module")
def no_tv_report():
    """Report without any trading_value column."""
    rows = []
    for date in weekdays():
        rows.append({
            "date": date, "symbol": "AAA", "close": 100.0,
            "weight": 1.0 if date < CUTOFF else 0.0,
        })
    return pl_bt.backtest_with_report(pl.DataFrame(rows), resample="M")


# ---------------------------------------------------------------------------
# capacity()
# ---------------------------------------------------------------------------


class TestCapacityFinlab:
    def test_finlab_value(self, tv_step_report):
        # AAA: amf = (1M*0.05/0.5 + 2M*0.05/0.5) / 2 = (100k + 200k) / 2 = 150k
        # BBB: amf = (1M*0.05/0.25 + 1M*0.05/0.25) / 2 = 200k
        # quantile(0.1, linear) of [150k, 200k] = 150k + 0.1 * 50k = 155k
        assert tv_step_report.capacity() == 155_000.0

    def test_min_leg_value(self, tv_step_report):
        # AAA: amf = 0.05 * min(1M, 2M) / 0.5 = 100k
        # BBB: amf = 0.05 * min(1M, 1M) / 0.25 = 200k
        # quantile(0.1, linear) of [100k, 200k] = 110k
        assert tv_step_report.capacity(method="min_leg") == 110_000.0

    def test_pov_and_quantile_forwarded(self, tv_step_report):
        # Doubling pov doubles every amf: [300k, 400k] -> q0.1 = 310k
        assert tv_step_report.capacity(percentage_of_volume=0.10) == 310_000.0
        # quantile(1.0) = max(amf) = 200k
        assert tv_step_report.capacity(quantile=1.0) == 200_000.0

    def test_get_metrics_capacity_unchanged(self, tv_step_report):
        # Default get_metrics capacity must equal the finlab method with the
        # default pov/quantile (and the hand-derived value).
        metrics = tv_step_report.get_metrics(sections=["liquidity"]).to_dicts()[0]
        assert metrics["capacity"] == 155_000.0
        assert metrics["capacity"] == tv_step_report.capacity(
            percentage_of_volume=0.05, quantile=0.1, method="finlab"
        )

    def test_unknown_method_rejected(self, tv_step_report):
        with pytest.raises(ValueError, match="method"):
            tv_step_report.capacity(method="vwap")

    def test_none_without_trading_value(self, no_tv_report):
        assert no_tv_report.capacity() is None
        assert no_tv_report.capacity(method="min_leg") is None
        assert no_tv_report.capacity(method="adv") is None


class TestCapacityAdv:
    def test_constant_tv_equals_min_leg(self, const_tv_report):
        # With constant TV the rolling median equals the TV itself, so the
        # adv method reduces to min_leg:
        # amfs = [0.05*1M/0.5, 0.05*1M/0.25] = [100k, 200k] -> q0.1 = 110k
        assert const_tv_report.capacity(method="adv") == 110_000.0
        assert const_tv_report.capacity(method="adv") == const_tv_report.capacity(
            method="min_leg"
        )

    def test_adv_median_ignores_own_volume_spike(self, tv_spike_report):
        # TV spikes to 50M on the entry and exit execution days. The raw-TV
        # methods swallow the spike:
        #   min_leg = finlab = 0.05 * 50M / 1.0 = 2.5M
        # while the 20-day rolling median at the SIGNAL dates stays 1M:
        #   adv = 0.05 * 1M / 1.0 = 50k
        assert tv_spike_report.capacity(method="min_leg") == 2_500_000.0
        assert tv_spike_report.capacity(method="finlab") == 2_500_000.0
        assert tv_spike_report.capacity(method="adv") == 50_000.0


# ---------------------------------------------------------------------------
# capacity_by_date()
# ---------------------------------------------------------------------------


class TestCapacityByDate:
    def test_joint_entries_take_min(self, const_tv_report):
        # Both trades enter on the same signal date; per-trade entry caps are
        # 0.05*1M/0.5 = 100k (AAA) and 0.05*1M/0.25 = 200k (BBB); the joint
        # constraint is the min.
        df = const_tv_report.capacity_by_date()  # default method="adv"
        assert df is not None
        assert df.height == 1
        assert df["date"].to_list() == [ENTRY_SIG]
        assert df["capacity"].to_list() == [100_000.0]
        assert df["n_entries"].to_list() == [2]

    def test_schema(self, const_tv_report):
        df = const_tv_report.capacity_by_date()
        assert df.columns == ["date", "capacity", "n_entries"]
        assert df.schema["date"] == pl.Date
        assert df.schema["capacity"] == pl.Float64
        assert df.schema["n_entries"] == pl.UInt32

    def test_tv_method(self, const_tv_report):
        # finlab/min_leg use raw TV at the entry execution date; with constant
        # TV the values match the adv result.
        df = const_tv_report.capacity_by_date(method="finlab")
        assert df["date"].to_list() == [ENTRY_SIG]
        assert df["capacity"].to_list() == [100_000.0]
        assert df["n_entries"].to_list() == [2]

    def test_none_without_trading_value(self, no_tv_report):
        assert no_tv_report.capacity_by_date() is None

    def test_unknown_method_rejected(self, const_tv_report):
        with pytest.raises(ValueError, match="method"):
            const_tv_report.capacity_by_date(method="vwap")


# ---------------------------------------------------------------------------
# Stage B: direction-aware, epsilon-tolerant buyHigh / sellLow
# ---------------------------------------------------------------------------


def _limit_report(limit_up, limit_down):
    """Single long trade entering at raw price 100.0.

    close = 103.0 (adjusted), factor = 1.03 -> entry_raw_price
    round-trips through adj/factor back to exactly 100.0.
    """
    rows = []
    for date in weekdays():
        rows.append({
            "date": date, "symbol": "AAA", "close": 103.0, "factor": 1.03,
            "weight": 1.0 if date < CUTOFF else 0.0,
            "limit_up": limit_up, "limit_down": limit_down,
        })
    report = pl_bt.backtest_with_report(pl.DataFrame(rows), resample="M")
    assert_fixture_trades(report, n_trades=1)
    return report


class TestBuyHighSellLow:
    def test_long_entry_exactly_at_limit_up(self):
        # Entry raw price 100.0 == limit_up -> counted as buying into the
        # limit; the exit at 100.0 is far above limit_down 50 -> sellLow 0.
        # Long-only: the short legs of both metrics are absent and must not
        # crash or dilute the ratios.
        report = _limit_report(limit_up=100.0, limit_down=50.0)
        metrics = report.get_metrics(sections=["liquidity"]).to_dicts()[0]
        assert metrics["buyHigh"] == 1.0
        assert metrics["sellLow"] == 0.0

    def test_epsilon_catches_float_round_trip(self):
        # Fill at 100.0 with the limit a hair (1e-7 relative) above: an exact
        # >= comparison misses it, the 1e-6-relative tolerance counts it.
        report = _limit_report(limit_up=100.00001, limit_down=50.0)
        metrics = report.get_metrics(sections=["liquidity"]).to_dicts()[0]
        assert metrics["buyHigh"] == 1.0

    def test_epsilon_not_too_loose(self):
        # 0.1% below the limit is a normal fill, not a limit lock.
        report = _limit_report(limit_up=100.1, limit_down=50.0)
        metrics = report.get_metrics(sections=["liquidity"]).to_dicts()[0]
        assert metrics["buyHigh"] == 0.0

    def test_long_exit_at_limit_down_via_stop_loss(self):
        # Price gaps from 100 to 89 (-11%) at CUTOFF, breaching stop_loss=0.10;
        # from CUTOFF on, close stays 89.0 and limit_down is 89.0, so whichever
        # day the stop executes the exit fills at the limit-down price.
        rows = []
        for date in weekdays():
            crashed = date >= CUTOFF
            rows.append({
                "date": date, "symbol": "AAA",
                "close": 89.0 if crashed else 100.0,
                "weight": 1.0 if date < datetime.date(2024, 3, 15) else 0.0,
                "limit_up": 500.0,
                "limit_down": 89.0 if crashed else 60.0,
            })
        report = pl_bt.backtest_with_report(
            pl.DataFrame(rows), resample="M", stop_loss=0.10
        )
        trades = entered_trades(report)
        assert trades.height == 1
        exit_date = trades["exit_date"][0]
        assert exit_date is not None and exit_date >= CUTOFF
        assert trades["exit_raw_price"][0] == 89.0

        metrics = report.get_metrics(sections=["liquidity"]).to_dicts()[0]
        assert metrics["sellLow"] == 1.0
        assert metrics["buyHigh"] == 0.0

    def test_null_without_limit_columns(self, const_tv_report):
        # No limit_up/limit_down in the input -> both metrics stay null.
        metrics = const_tv_report.get_metrics(sections=["liquidity"]).to_dicts()[0]
        assert metrics["buyHigh"] is None
        assert metrics["sellLow"] is None
