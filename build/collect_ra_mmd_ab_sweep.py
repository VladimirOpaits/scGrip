"""Collate the RA LoRA MMD x A/B-split sweep into one flat CSV."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DELTA_COLUMNS = [
    "anchor_mse_sigma",
    "anchor_drift_max",
    "ifn_spearman_disease",
    "ifn_pearson_disease",
    "scib_bio_cell_type_asw",
    "scib_bio_compartment_asw",
    "scib_batch_healthy_asw",
    "scib_block_healthy_asw",
    "strat_healthy_A_to_census",
    "strat_healthy_B_to_census",
    "strat_healthy_B_minus_A_to_census",
    "scib_batch_healthy_A_asw",
    "scib_batch_healthy_B_asw",
    "scib_block_healthy_A_asw",
    "scib_block_healthy_B_asw",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/ra_mmd_ab_sweep")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def mmd_weight(state):
    if state == "base":
        return np.nan
    match = re.fullmatch(r"mmd([0-9]+(?:\.[0-9]+)?)", str(state))
    if match is None:
        raise ValueError(f"Cannot infer MMD weight from state label {state!r}")
    return float(match.group(1))


def main():
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / "results.csv"
    sweep_paths = sorted(root.glob("eval_b*/metrics.csv"))
    validation_paths = sorted(root.glob("validation_*/metrics.csv"))
    path_specs = [(path, "sweep") for path in sweep_paths] + [
        (path, "validation") for path in validation_paths
    ]
    if not path_specs:
        raise FileNotFoundError(f"No evaluation metrics.csv files under {root}")

    tables = []
    for path, evaluation_kind in path_specs:
        table = pd.read_csv(path)
        if "state" not in table:
            raise ValueError(f"{path} does not contain a state column")
        manifest_path = path.parent / "manifest.json"
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        table["eval_dir"] = path.parent.name
        table["evaluation_kind"] = evaluation_kind
        table["scib_sample_cap"] = manifest["args"]["scib_max_cells"]
        table["mmd_weight"] = table["state"].map(mmd_weight)
        table["evaluation_epoch"] = np.where(table["state"] == "base", 0, 5)
        tables.append(table)
    results = pd.concat(tables, ignore_index=True, sort=False)

    for column in DELTA_COLUMNS:
        if column not in results:
            continue
        results[f"delta_vs_base_{column}"] = np.nan
        results[f"delta_vs_mmd0_{column}"] = np.nan

    for _, positions in results.groupby("eval_dir", sort=False).groups.items():
        index = list(positions)
        frame = results.loc[index]
        base = frame.loc[frame["state"] == "base"]
        control = frame.loc[frame["state"] == "mmd0"]
        if len(base) != 1 or len(control) != 1:
            raise ValueError(
                f"{frame['eval_dir'].iloc[0]} must contain exactly one base and one mmd0 row"
            )
        base_row = base.iloc[0]
        control_row = control.iloc[0]
        for column in DELTA_COLUMNS:
            if column not in results:
                continue
            results.loc[index, f"delta_vs_base_{column}"] = (
                results.loc[index, column] - base_row[column]
            )
            results.loc[index, f"delta_vs_mmd0_{column}"] = (
                results.loc[index, column] - control_row[column]
            )

    results["_sort_weight"] = results["mmd_weight"].fillna(-1.0)
    results = results.sort_values(
        ["ab_holdout_frac", "scib_sample_cap", "_sort_weight", "state"],
        kind="stable",
    ).drop(columns="_sort_weight")
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    summary_columns = [
        "ab_holdout_frac",
        "ab_n_healthy_donors_A",
        "ab_n_healthy_donors_B",
        "evaluation_kind",
        "scib_sample_cap",
        "state",
        "mmd_weight",
        "checkpoint",
        "anchor_mse_sigma",
        "ifn_spearman_disease",
        "ifn_pearson_disease",
        "scib_bio_cell_type_asw",
        "scib_bio_compartment_asw",
        "scib_batch_healthy_asw",
        "scib_batch_healthy_A_asw",
        "scib_batch_healthy_B_asw",
        "scib_block_healthy_asw",
        "scib_block_healthy_A_asw",
        "scib_block_healthy_B_asw",
        "strat_healthy_A_to_census",
        "strat_healthy_B_to_census",
        "strat_healthy_B_minus_A_to_census",
        "delta_vs_mmd0_anchor_mse_sigma",
        "delta_vs_mmd0_scib_batch_healthy_asw",
        "delta_vs_mmd0_scib_batch_healthy_A_asw",
        "delta_vs_mmd0_scib_batch_healthy_B_asw",
        "delta_vs_mmd0_scib_block_healthy_asw",
        "delta_vs_mmd0_scib_block_healthy_A_asw",
        "delta_vs_mmd0_scib_block_healthy_B_asw",
        "delta_vs_mmd0_strat_healthy_A_to_census",
        "delta_vs_mmd0_strat_healthy_B_to_census",
        "delta_vs_base_scib_block_healthy_asw",
        "delta_vs_base_scib_block_healthy_A_asw",
        "delta_vs_base_scib_block_healthy_B_asw",
    ]
    summary_columns = [column for column in summary_columns if column in results]
    summary_path = output.with_name("summary.csv")
    results.loc[:, summary_columns].to_csv(summary_path, index=False)
    print(f"wrote compact summary to {summary_path}")
    print(f"wrote {len(results)} rows x {len(results.columns)} columns to {output}")


if __name__ == "__main__":
    main()
