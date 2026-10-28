"""Reproducible offline check of the new binding when legacy crates are missing.

This compiles the unmodified production Rust sources with only PyO3. It does
not replace a full workspace build and must not be reported as one.
Use uv run --no-sync python ../scripts/build_swing_binding.py --out target/swing-check.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get("CARGO_TARGET_DIR"):
        ap.error("CARGO_TARGET_DIR is required")
    root = Path(__file__).resolve().parents[1]
    cargo = args.out.resolve() / "isolated-binding"
    (cargo / "src").mkdir(parents=True, exist_ok=True)
    (cargo / "Cargo.toml").write_text(
        '[package]\nname="swing-binding-check"\nversion="0.0.0"\nedition="2021"\n'
        "[workspace]\n"
        '[lib]\nname="_swing_core"\ncrate-type=["cdylib"]\n'
        '[dependencies]\npyo3={version="=0.26.0",features=["extension-module"]}\n'
    )
    core = json.dumps(str(root / "btcore/src/swing.rs"))
    binding = json.dumps(str(root / "polars_backtest/src/swing.rs"))
    (cargo / "src/lib.rs").write_text(
        f"extern crate self as btcore;\n#[path={core}] pub mod swing;\n"
        f"#[path={binding}] mod binding;\n"
        "use pyo3::prelude::*;\n#[pymodule]\n"
        "fn _swing_core(m: &Bound<'_, PyModule>) -> PyResult<()> {\n"
        "m.add_function(wrap_pyfunction!(binding::swing_impl, m)?)?; Ok(()) }\n"
    )
    subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "maturin",
            "build",
            "--manifest-path",
            str(cargo / "Cargo.toml"),
            "--out",
            str(args.out.resolve() / "wheels"),
        ],
        cwd=root / "polars_backtest",
        check=True,
    )


if __name__ == "__main__":
    main()
