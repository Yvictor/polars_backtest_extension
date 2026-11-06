"""viz.report_data() is an external contract: the bundle work must not change it.

The fixture was written before the bundle existed. To regenerate it on purpose, run
this file directly; a change here means the public payload changed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from report_fixtures import NAMES, long_report, prices

from polars_backtest import viz

SNAPSHOT = Path(__file__).parent / "fixtures" / "report_data_v3.json"
UNSTABLE = ("version",)


def payload() -> dict[str, Any]:
    data = viz.report_data(
        long_report(),
        title="snapshot",
        input_df=prices(),
        symbol_names=NAMES,
        fill_scenarios=[{"name": "baseline", "cagr": 0.1, "blocked_n": 0}],
    )
    return {k: v for k, v in json.loads(json.dumps(data)).items() if k not in UNSTABLE}


def assert_same(actual: Any, expected: Any, path: str = "$") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), path
        for key, value in expected.items():
            assert_same(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, value in enumerate(expected):
            assert_same(actual[i], value, f"{path}[{i}]")
    elif isinstance(expected, float) and not math.isclose(expected, 0.0):
        assert actual == pytest.approx(expected, rel=1e-12), path
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-12), path
    else:
        assert actual == expected, path


def test_report_data_matches_snapshot():
    assert_same(payload(), json.loads(SNAPSHOT.read_text(encoding="utf-8")))


if __name__ == "__main__":
    SNAPSHOT.write_text(json.dumps(payload(), indent=None, sort_keys=True), encoding="utf-8")
