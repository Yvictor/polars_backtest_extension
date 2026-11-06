"""Two simultaneously held names with equal market values, so no reduction order."""

import json
from pathlib import Path

import polars as pl
import pytest
from legacy_fixtures import APIS, capture
from legacy_fixtures import inputs as single_name_inputs

CASES = {
    "monthly": ("partial", {"resample": "M"}),
    "weekly_boolean": ("boolean", {"resample": "W"}),
    "offset_cap_retain": (
        "partial",
        {
            "resample": "M",
            "resample_offset": "2D",
            "position_limit": 0.25,
            "retain_cost_when_rebalance": True,
        },
    ),
    "touched": ("partial", {"touched_exit": True, "stop_loss": 0.03, "take_profit": 0.04}),
    "offset_touched": (
        "partial",
        {
            "resample": "W",
            "resample_offset": "2D",
            "touched_exit": True,
            "stop_loss": 0.03,
            "take_profit": 0.04,
        },
    ),
    "short": ("signed", {}),
}


def inputs(kind):
    data, _ = single_name_inputs(kind)
    first = data.filter(pl.col("symbol") == "A")
    second = first.with_columns(
        pl.lit("B").alias("symbol"),
        *[(pl.col(name) * 2).alias(name) for name in ("close", "open", "high", "low")],
    )
    data = pl.concat([first, second]).sort(["date", "symbol"])
    # The wide resampler accepts its documented ISO date input.
    wide = {
        name: data.pivot(on="symbol", index="date", values=name)
        .sort("date")
        .with_columns(pl.col("date").dt.to_string("%Y-%m-%d"))
        for name in ("close", "open", "high", "low", "weight")
    }
    return data, wide


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("api", APIS)
def test_simultaneous_legacy_positions_exact(name, api):
    fixtures = Path(__file__).parent / "fixtures"
    reference = json.loads((fixtures / "legacy_paired_exact.json").read_text())
    reference.update(json.loads((fixtures / "legacy_offset_touched_exact.json").read_text()))
    actual = json.loads(json.dumps(capture(name, api, CASES, inputs)))
    assert actual == reference[f"{api}/{name}"]
