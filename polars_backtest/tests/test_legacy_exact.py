"""Bit-exact snapshots captured on master 62fd52f, one active name at a time."""

import json
from pathlib import Path

import pytest
from legacy_fixtures import APIS, CASES, capture


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("api", APIS)
def test_legacy_exact(name, api):
    reference = json.loads((Path(__file__).parent / "fixtures" / "legacy_exact.json").read_text())
    actual = json.loads(json.dumps(capture(name, api)))
    assert actual == reference[f"{api}/{name}"]
