"""Portfolios of many names: stable bits, and equal to master up to rounding noise.

`multi_name_master.json` was captured on master 62fd52f, whose unordered position
maps made the last bits vary between processes (observed spread 2.2e-15). It is
compared within `ROUNDING`. `multi_name_exact.json` was captured after positions
became ordered; every later change must keep it bit for bit.
"""

import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from multi_name_fixtures import CASES, capture

FIXTURES = Path(__file__).parent / "fixtures"
ROUNDING = 1e-14


@pytest.fixture(scope="module")
def master():
    return json.loads((FIXTURES / "multi_name_master.json").read_text())


@pytest.fixture(scope="module")
def exact():
    return json.loads((FIXTURES / "multi_name_exact.json").read_text())


def assert_close(actual, reference):
    if isinstance(reference, dict) and set(reference) == {"f64"}:
        a, b = (struct.unpack(">d", bytes.fromhex(v["f64"]))[0] for v in (actual, reference))
        assert a == b or abs(a - b) <= ROUNDING or (a != a and b != b)
    elif isinstance(reference, dict):
        assert actual.keys() == reference.keys()
        for key in reference:
            assert_close(actual[key], reference[key])
    elif isinstance(reference, list):
        assert len(actual) == len(reference)
        for a, b in zip(actual, reference):
            assert_close(a, b)
    else:
        assert actual == reference


@pytest.mark.parametrize("name", CASES)
def test_matches_master_up_to_rounding(name, master):
    assert_close(json.loads(json.dumps(capture(name))), master[name])


@pytest.mark.parametrize("name", CASES)
def test_bit_exact(name, exact):
    assert json.loads(json.dumps(capture(name))) == exact[name]


def test_results_do_not_vary_between_processes():
    script = (
        "import hashlib, json\n"
        "from multi_name_fixtures import capture_all\n"
        "print(hashlib.sha256(json.dumps(capture_all()).encode()).hexdigest())\n"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent)}
    runs = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for _ in range(4)
    ]
    digests = set()
    for run in runs:
        output, errors = run.communicate()
        assert run.returncode == 0, errors
        digest = output.strip()
        assert re.fullmatch("[0-9a-f]{64}", digest), output
        digests.add(digest)
    assert len(digests) == 1
