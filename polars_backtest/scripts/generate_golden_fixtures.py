#!/usr/bin/env python3
"""Generate golden fixtures for finlab parity tests.

This script downloads real TW market data via finlab, runs finlab's
``backtest.sim`` for a fixed set of scenarios, and stores both the
long-format input frame and finlab's outputs (creturn + trades) as small
parquet fixtures under ``tests/fixtures/``.

The committed fixtures let ``tests/test_golden_fixtures.py`` verify
finlab-parity WITHOUT a finlab account, API token, or network access.

Requirements to (re)generate:
    - finlab installed and FINLAB_API_TOKEN available (e.g. via a .env file)
    - polars + pyarrow + pandas

Usage (from an environment that has finlab, e.g. finlab_research):
    cd /path/to/finlab_research && uv run python \
        /path/to/polars_backtest/scripts/generate_golden_fixtures.py

Design notes:
    - Weights are forced to 0 from FLAT_FROM onward so every trade closes
      well before the end of the fixture data. finlab simulates with price
      data extending to "today", while the parity test only has data up to
      END; a flat (all-cash) tail makes truncating finlab's creturn at END
      lossless.
    - The input frame stores the finlab-convention columns used by the
      slow reference tests (tests/test_wide_vs_finlab.py):
        close     = adjusted close (etl:adj_close)  -> trade_at_price
        raw_close = actual close   (price:收盤價)
        factor    = close / raw_close (adjustment factor)
        weight    = strategy signal (identical for all scenarios)
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl
from dotenv import find_dotenv, load_dotenv

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# 20 liquid TW large caps, chosen for COMPLETE price data over the whole
# window (e.g. 2603 is excluded: trading was halted 2022-09 for a capital
# reduction, leaving NaN prices).
SYMBOLS = [
    "2330", "2317", "2454", "2308", "2303",
    "2881", "2882", "2891", "2886", "2884",
    "1301", "1303", "1326", "2002", "2412",
    "3008", "2357", "2382", "3711", "1216",
]

WARMUP_START = "2021-06-01"  # extra history so the rolling signal is valid at START
START = "2022-01-01"
END = "2023-12-31"
# All weights are zeroed up to and including this date. finlab's engine treats the FIRST
# resample date of the window as the simulation start and never trades on it
# (verified empirically: shifting the window start one month earlier makes
# finlab take the trade), while polars_backtest rebalances on it. Keeping the
# first weeks flat means every scenario's first resample date carries a zero
# position in both engines, sidestepping the boundary difference.
FLAT_UNTIL = "2022-02-07"
# All weights are zeroed from this date on -> last entries exit in early
# December at the latest and creturn is flat (cash) afterwards. See module
# docstring for why this is required.
FLAT_FROM = "2023-11-01"

ROLLING_WINDOW = 60  # momentum: close >= rolling 60-day max

FEE_RATIO = 0.001425
TAX_RATIO = 0.003

# Scenario name -> kwargs passed identically to finlab backtest.sim and
# (by the parity test) to polars_backtest.backtest_with_report.
SCENARIOS: dict[str, dict] = {
    "monthly": {"resample": "M"},
    "monthly_stop_loss": {"resample": "M", "stop_loss": 0.1},
    "weekly": {"resample": "W"},
    "monthly_offset_5d": {"resample": "M", "resample_offset": "5D"},
}


def _login() -> None:
    import finlab

    load_dotenv(find_dotenv(usecwd=True))  # pick up .env from cwd if present
    token = os.getenv("FINLAB_API_TOKEN")
    if not token:
        sys.exit("FINLAB_API_TOKEN not set (put it in .env or the environment)")
    finlab.login(token)


def _wide_to_long(wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    out = wide.reset_index().melt(
        id_vars="date", var_name="symbol", value_name=value_name
    )
    return out


def build_position(raw_close: pd.DataFrame) -> pd.DataFrame:
    """Momentum signal on raw close, zeroed at both window edges (see FLAT_UNTIL / FLAT_FROM)."""
    signal = raw_close >= raw_close.rolling(ROLLING_WINDOW).max()
    position = signal.loc[START:END].astype(float)
    position.loc[:FLAT_UNTIL] = 0.0
    position.loc[FLAT_FROM:] = 0.0
    return position


def build_long_input(
    adj_close: pd.DataFrame, raw_close: pd.DataFrame, position: pd.DataFrame
) -> pl.DataFrame:
    adj_slice = adj_close.loc[START:END, SYMBOLS]
    raw_slice = raw_close.loc[START:END, SYMBOLS]

    df = (
        _wide_to_long(adj_slice, "close")
        .merge(_wide_to_long(raw_slice, "raw_close"), on=["date", "symbol"])
        .merge(_wide_to_long(position, "weight"), on=["date", "symbol"])
    )

    n_nan = int(df[["close", "raw_close", "weight"]].isna().sum().sum())
    if n_nan:
        raise AssertionError(
            f"Found {n_nan} NaN values in the fixture window; pick symbols with"
            " complete data so long/wide inputs are trivially equivalent"
        )

    df["factor"] = df["close"] / df["raw_close"]

    return (
        pl.from_pandas(df)
        .with_columns(pl.col("date").cast(pl.Date))
        .select("date", "symbol", "close", "factor", "raw_close", "weight")
        .sort("date", "symbol")
    )


def normalize_trades(trades: pd.DataFrame) -> pl.DataFrame:
    trades = trades.reset_index(drop=True)
    trades = trades[trades["entry_date"].notna()]
    out = pl.from_pandas(trades).select(
        pl.col("stock_id").cast(pl.Utf8).str.split(" ").list.first().alias("stock_id"),
        pl.col("entry_date").cast(pl.Date),
        pl.col("exit_date").cast(pl.Date),
        pl.col("entry_sig_date").cast(pl.Date),
        pl.col("exit_sig_date").cast(pl.Date),
        pl.col("position").cast(pl.Float64),
        pl.col("return").cast(pl.Float64),
    )
    n_open = out.filter(pl.col("exit_date").is_null()).height
    if n_open:
        raise AssertionError(
            f"{n_open} finlab trades are still open at the end of the window;"
            " FLAT_FROM should force everything to close before END"
        )
    return out


def main() -> None:
    _login()

    import finlab
    from finlab import backtest as finlab_backtest
    from finlab import data as finlab_data

    raw_close = finlab_data.get("price:收盤價").loc[WARMUP_START:END, SYMBOLS]
    adj_close = finlab_data.get("etl:adj_close").loc[WARMUP_START:END, SYMBOLS]

    position = build_position(raw_close)
    df_long = build_long_input(adj_close, raw_close, position)

    first_date = df_long.get_column("date").min()
    last_date = df_long.get_column("date").max()
    print(f"input frame: {df_long.height} rows, {first_date} .. {last_date}")

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    df_long.write_parquet(FIXTURE_DIR / "golden_input.parquet", compression="zstd")

    summary = []
    for name, params in SCENARIOS.items():
        print(f"\n--- scenario: {name} params={params}")
        report = finlab_backtest.sim(
            position,
            upload=False,
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            **params,
        )

        creturn = report.creturn
        creturn = creturn.loc[
            pd.Timestamp(first_date) : pd.Timestamp(last_date)
        ]
        df_creturn = pl.DataFrame(
            {
                "date": [d.date() for d in creturn.index],
                "creturn": creturn.to_numpy().astype(float),
            }
        )
        df_trades = normalize_trades(report.trades)

        df_creturn.write_parquet(
            FIXTURE_DIR / f"golden_{name}_creturn.parquet", compression="zstd"
        )
        df_trades.write_parquet(
            FIXTURE_DIR / f"golden_{name}_trades.parquet", compression="zstd"
        )

        final = df_creturn.get_column("creturn")[-1]
        print(
            f"    creturn rows={df_creturn.height} final={final:.6f}"
            f" trades={df_trades.height}"
        )
        summary.append((name, params, df_creturn.height, final, df_trades.height))

    write_readme(finlab.__version__, df_long, summary)
    print("\nfixtures written to", FIXTURE_DIR)


def write_readme(finlab_version: str, df_long: pl.DataFrame, summary: list) -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    lines = [
        "# Golden fixtures for finlab parity tests",
        "",
        f"Generated on {today} with finlab {finlab_version}, pandas {pd.__version__},",
        f"polars {pl.__version__}, by `scripts/generate_golden_fixtures.py`.",
        "",
        "These fixtures let `tests/test_golden_fixtures.py` verify that",
        "`polars_backtest.backtest_with_report` reproduces finlab `backtest.sim`",
        "results without a finlab account, API token, or network access.",
        "",
        "## Data",
        "",
        f"- Symbols (20 liquid TW large caps): {', '.join(SYMBOLS)}",
        f"- Window: {START} .. {END} (signal warmup from {WARMUP_START})",
        f"- Strategy: `close >= close.rolling({ROLLING_WINDOW}).max()` on raw close,",
        f"  equal-weighted; all weights forced to 0 from {FLAT_FROM} so every trade",
        "  closes before the window ends (finlab simulates to 'today', so a flat",
        "  cash tail makes truncating its creturn at the window end lossless).",
        f"- Weights are also forced to 0 up to {FLAT_UNTIL}: finlab's engine treats",
        "  the FIRST resample date of the window as the simulation start and never",
        "  trades on it (verified empirically: shifting the window start one month",
        "  earlier makes finlab take the trade), while polars_backtest rebalances",
        "  on it. A flat head keeps the first resample date at zero position in",
        "  both engines, so the fixtures don't encode this boundary difference.",
        f"- Fees: fee_ratio={FEE_RATIO}, tax_ratio={TAX_RATIO} (passed explicitly to both engines)",
        "",
        "## Files",
        "",
        "- `golden_input.parquet`: long-format input",
        "  (date, symbol, close=adjusted close, factor=adj/raw, raw_close, weight).",
        "- `golden_<scenario>_creturn.parquet`: finlab creturn (date, creturn),",
        "  truncated to the input window.",
        "- `golden_<scenario>_trades.parquet`: finlab trades (stock_id, entry_date,",
        "  exit_date, entry_sig_date, exit_sig_date, position, return);",
        "  stock_id normalized to the bare ticker.",
        "",
        "## Scenarios",
        "",
        "| scenario | sim params | creturn rows | final creturn | trades |",
        "|---|---|---|---|---|",
    ]
    for name, params, n_rows, final, n_trades in summary:
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"| {name} | {params_str} | {n_rows} | {final:.6f} | {n_trades} |")
    lines += [
        "",
        "## Regenerating",
        "",
        "```bash",
        "# from an environment with finlab + FINLAB_API_TOKEN (e.g. via .env)",
        "python scripts/generate_golden_fixtures.py",
        "```",
        "",
        "## Known engine differences (accounted for by the parity test)",
        "",
        "1. First resample date: finlab treats the first resample date of the",
        "   window as the simulation start and never trades on it, while",
        "   polars_backtest rebalances on it. The fixtures avoid this boundary",
        f"   by keeping weights at 0 up to {FLAT_UNTIL} (see above).",
        "2. Trade-level `return` fee convention for rolled trades: when a stock",
        "   stays in the portfolio across a rebalance, both engines split the",
        "   trade record at the rebalance date, but polars_backtest charges the",
        "   full round trip (entry fee, exit fee + tax) on every record, while",
        "   finlab omits the entry fee on records that start as a roll and the",
        "   exit fee + tax on records that end as a roll:",
        "   `finlab_r + 1 = (ours_r + 1) / (1-fee)^entry_rolled / (1-fee-tax)^exit_rolled`.",
        "   The parity test asserts this exact relationship within 1e-6.",
        "   Portfolio-level creturn is identical (max abs diff < 1e-6); only the",
        "   per-trade fee attribution differs.",
        "",
        "Otherwise all scenarios match finlab within the tolerances asserted in",
        "`tests/test_golden_fixtures.py` (creturn max abs diff < 1e-6, trade",
        "count/exit dates exact, trade returns within 1e-6 after the roll fee",
        "adjustment). If a future engine change breaks a scenario, document the",
        "discrepancy here instead of loosening tolerances silently.",
        "",
    ]
    (FIXTURE_DIR / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
