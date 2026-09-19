#!/usr/bin/env python3
"""Merge phase-I sweep CSVs and add within-run deltas from identity."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="Repeat for each sweep result table; LABEL becomes sweep_group.",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def _epoch(value: str) -> int:
    if value == "base":
        return 0
    match = re.fullmatch(r"epoch_(\d+)", str(value))
    if match is None:
        raise ValueError(f"Unknown state {value!r}")
    return int(match.group(1))


def main() -> None:
    args = parse_args()
    destination = Path(args.out_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    parts = []
    for spec in args.input:
        if "=" not in spec:
            raise ValueError(f"--input expects LABEL=CSV, got {spec!r}")
        label, raw_path = spec.split("=", 1)
        if not label:
            raise ValueError("Sweep label cannot be empty.")
        table = pd.read_csv(raw_path)
        if "run_name" not in table or "state" not in table:
            raise ValueError(f"{raw_path} is not a sweep result CSV.")
        table.insert(0, "sweep_group", label)
        table.insert(1, "run_uid", label + "/" + table["run_name"].astype(str))
        table["epoch"] = table["state"].map(_epoch)
        parts.append(table)
    combined = pd.concat(parts, ignore_index=True)
    if combined.duplicated(["run_uid", "state"]).any():
        raise ValueError("Merged inputs contain duplicate run/state rows.")
    metric_columns = [
        column for column in (
            "anchor_held_out", "drift_max", "drift_p95", "strat_A", "strat_B",
            "strat_A_block_offset", "strat_B_block_offset", "mmd_A_eval",
            "scib_stratum_asw", "signature_spearman_all", "signature_spearman_disease",
        )
        if column in combined
    ]
    base = combined[combined["epoch"] == 0].loc[:, ["run_uid", *metric_columns]].copy()
    if base["run_uid"].duplicated().any():
        # Every state has a base duplicate only if a caller included the same
        # run twice; the run/state uniqueness check above catches that.
        raise ValueError("Each run must have exactly one identity row.")
    base = base.rename(columns={column: f"base_{column}" for column in metric_columns})
    combined = combined.merge(base, on="run_uid", how="left", validate="many_to_one")
    if combined[[f"base_{column}" for column in metric_columns]].isna().all(axis=None):
        raise ValueError("No identity/base states available for delta calculation.")
    for column in metric_columns:
        combined[f"delta_{column}"] = combined[column] - combined[f"base_{column}"]
    combined = combined.sort_values(["sweep_group", "lambda_mmd", "split_seed", "epoch"], kind="stable")
    combined.to_csv(destination / "all_sweep_states_with_deltas.csv", index=False)
    selected = combined[combined["epoch"].isin((0, 2, 5, 10))].copy()
    selected.to_csv(destination / "checkpoint_table_with_deltas.csv", index=False)
    summary = (
        selected[selected["epoch"] > 0]
        .groupby(["sweep_group", "lambda_mmd", "epoch"], observed=True)
        .agg(
            n_runs=("run_uid", "nunique"),
            mean_delta_strat_B=("delta_strat_B", "mean"),
            min_delta_strat_B=("delta_strat_B", "min"),
            max_delta_strat_B=("delta_strat_B", "max"),
            mean_delta_mmd_A_eval=("delta_mmd_A_eval", "mean"),
            mean_anchor_held_out=("anchor_held_out", "mean"),
            mean_signature_spearman_disease=("signature_spearman_disease", "mean"),
        )
        .reset_index()
        .sort_values(["sweep_group", "lambda_mmd", "epoch"], kind="stable")
    )
    summary.to_csv(destination / "within_run_delta_summary.csv", index=False)
    (destination / "README.md").write_text(
        """# Merged phase-I sweep results

`delta_*` is always state minus the identity/base state from the same
`run_uid`.  This is essential when different healthy donor A/B splits have
different base distances; raw held-out B numbers should not be compared across
splits without this normalization.  The tables are descriptive and do not
choose a model or checkpoint automatically.
""",
        encoding="utf-8",
    )
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
