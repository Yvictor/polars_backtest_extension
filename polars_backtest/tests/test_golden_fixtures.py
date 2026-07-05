"""Golden-fixture parity tests against finlab backtest.sim.

Fast tests: no finlab, no FINLAB_API_TOKEN, no network, no slow marker.

The fixtures under tests/fixtures/ were produced by
scripts/generate_golden_fixtures.py, which ran finlab 1.5.5 `backtest.sim`
on 20 liquid TW symbols over 2022-01-01..2023-12-31 and stored:

- golden_input.parquet: long-format input
  (date, symbol, close=adjusted, factor=adj/raw, raw_close, weight)
- golden_<scenario>_creturn.parquet: finlab creturn (date, creturn)
- golden_<scenario>_trades.parquet: normalized finlab trades

Each test runs pl_bt.backtest_with_report on the golden input with the same
parameters finlab used and asserts parity. See fixtures/README.md for the
scenario definitions and generation details.
"""

from __future__ import annotations

import math
import os

import polars as pl
import pytest

import polars_backtest as pl_bt

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# Same fee settings the generator passed explicitly to finlab backtest.sim.
FEE_RATIO = 0.001425
TAX_RATIO = 0.003

# Scenario name -> backtest params (must mirror SCENARIOS in
# scripts/generate_golden_fixtures.py).
SCENARIOS: dict[str, dict] = {
    "monthly": {"resample": "M"},
    "monthly_stop_loss": {"resample": "M", "stop_loss": 0.1},
    "weekly": {"resample": "W"},
    "monthly_offset_5d": {"resample": "M", "resample_offset": "5D"},
}

CRETURN_ATOL = 1e-6
RETURN_ATOL = 1e-6


def _load_fixture(filename: str) -> pl.DataFrame:
    path = os.path.join(FIXTURE_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"golden fixture missing: {path}")
    return pl.read_parquet(path)


@pytest.fixture(scope="module")
def golden_input() -> pl.DataFrame:
    return _load_fixture("golden_input.parquet")


def _run_scenario(golden_input: pl.DataFrame, scenario: str):
    params = SCENARIOS[scenario]
    return pl_bt.backtest_with_report(
        golden_input,
        trade_at_price="close",  # adjusted close, finlab convention
        position="weight",
        factor="factor",
        fee_ratio=FEE_RATIO,
        tax_ratio=TAX_RATIO,
        **params,
    )


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_creturn_matches_finlab(golden_input: pl.DataFrame, scenario: str) -> None:
    golden = _load_fixture(f"golden_{scenario}_creturn.parquet")
    report = _run_scenario(golden_input, scenario)

    ours = report.creturn.with_columns(pl.col("date").cast(pl.Date))
    golden = golden.with_columns(pl.col("date").cast(pl.Date))

    # Both series must end on the same date (the fixture is truncated to the
    # input window and is flat/cash at the tail by construction).
    assert ours.get_column("date").max() == golden.get_column("date").max()

    # The two engines may trim the leading all-cash stretch differently; any
    # dates only one side reports must sit at creturn == 1.0.
    common_start = max(ours.get_column("date").min(), golden.get_column("date").min())
    for label, df in (("finlab", golden), ("ours", ours)):
        leading = df.filter(pl.col("date") < common_start)
        assert leading.filter(
            (pl.col("creturn") - 1.0).abs() > CRETURN_ATOL
        ).is_empty(), f"{label} creturn moved before the common start date"

    joined = golden.join(ours.rename({"creturn": "creturn_ours"}), on="date", how="inner")
    # After the leading flat stretch both engines must report the same dates.
    n_golden = golden.filter(pl.col("date") >= common_start).height
    n_ours = ours.filter(pl.col("date") >= common_start).height
    assert joined.height == n_ours, "our creturn has dates finlab lacks"
    assert joined.height == n_golden, "finlab creturn has dates we lack"
    assert joined.height > 400  # ~2 years of trading days, sanity check

    max_diff = (
        joined.select((pl.col("creturn") - pl.col("creturn_ours")).abs().max()).item()
    )
    assert max_diff < CRETURN_ATOL, f"creturn max abs diff {max_diff:.3e} >= {CRETURN_ATOL}"


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_trades_match_finlab(golden_input: pl.DataFrame, scenario: str) -> None:
    golden = _load_fixture(f"golden_{scenario}_trades.parquet")
    report = _run_scenario(golden_input, scenario)

    # The report may include pending signal-only records (null entry_date) for
    # the period after the last rebalance; the fixture stores executed finlab
    # trades only, so compare completed trades.
    ours = report.trades.filter(
        pl.col("entry_date").is_not_null() & pl.col("exit_date").is_not_null()
    )
    assert ours.height == golden.height, (
        f"trade count mismatch: finlab={golden.height}, ours={ours.height}"
    )
    assert golden.height > 0

    joined = golden.rename(
        {"exit_date": "exit_date_finlab", "return": "return_finlab"}
    ).join(
        ours.select("stock_id", "entry_date", "exit_date", "return"),
        on=["stock_id", "entry_date"],
        how="left",
    )

    unmatched = joined.filter(pl.col("exit_date").is_null() & pl.col("return").is_null())
    assert unmatched.is_empty(), f"finlab trades without a match:\n{unmatched}"

    exit_mismatch = joined.filter(pl.col("exit_date_finlab") != pl.col("exit_date"))
    assert exit_mismatch.is_empty(), f"exit_date mismatches:\n{exit_mismatch}"

    # Trade return fee convention (documented in fixtures/README.md):
    # polars_backtest charges the full round trip (entry fee, exit fee + tax)
    # on every trade record. finlab omits the entry fee when the record starts
    # as a roll (the stock was already held and the trade was split at a
    # rebalance) and omits the exit fee + tax when it ends as a roll:
    #   finlab_return + 1 =
    #       (ours_return + 1) / (1-fee)^entry_rolled / (1-fee-tax)^exit_rolled
    # A roll is identified by another trade of the same stock exiting
    # (entering) on this trade's entry (exit) date. creturn is unaffected.
    roll_points = golden.select(
        pl.col("stock_id"), pl.col("exit_date").alias("roll_date")
    ).unique()
    joined = (
        joined.join(
            roll_points.with_columns(pl.lit(True).alias("entry_rolled")),
            left_on=["stock_id", "entry_date"],
            right_on=["stock_id", "roll_date"],
            how="left",
        )
        .join(
            golden.select(
                pl.col("stock_id"), pl.col("entry_date").alias("roll_date")
            )
            .unique()
            .with_columns(pl.lit(True).alias("exit_rolled")),
            left_on=["stock_id", "exit_date_finlab"],
            right_on=["stock_id", "roll_date"],
            how="left",
        )
        .with_columns(
            pl.col("entry_rolled").fill_null(False),
            pl.col("exit_rolled").fill_null(False),
        )
        .with_columns(
            (
                (1.0 + pl.col("return"))
                / pl.when(pl.col("entry_rolled"))
                .then(1.0 - FEE_RATIO)
                .otherwise(1.0)
                / pl.when(pl.col("exit_rolled"))
                .then(1.0 - FEE_RATIO - TAX_RATIO)
                .otherwise(1.0)
                - 1.0
            ).alias("return_expected")
        )
    )

    return_mismatch = joined.filter(
        (pl.col("return_finlab") - pl.col("return_expected")).abs() > RETURN_ATOL
    )
    max_return_diff = joined.select(
        (pl.col("return_finlab") - pl.col("return_expected")).abs().max()
    ).item()
    assert return_mismatch.is_empty(), (
        f"trade return max abs diff {max_return_diff:.3e} >= {RETURN_ATOL}:\n"
        f"{return_mismatch}"
    )


def test_golden_input_schema(golden_input: pl.DataFrame) -> None:
    """The committed input frame must stay well-formed and finlab-consistent."""
    assert set(golden_input.columns) >= {
        "date", "symbol", "close", "factor", "raw_close", "weight",
    }
    assert golden_input.get_column("date").dtype == pl.Date
    # factor = adjusted / raw, so close == raw_close * factor must hold exactly
    bad = golden_input.filter(
        (pl.col("close") - pl.col("raw_close") * pl.col("factor")).abs() > 1e-9
    )
    assert bad.is_empty()
    # no NaN/nulls anywhere (the FFI layer treats NaN as "missing")
    for col in ("close", "factor", "raw_close", "weight"):
        s = golden_input.get_column(col)
        assert s.null_count() == 0
        assert not any(math.isnan(v) for v in (s.min(), s.max()))
