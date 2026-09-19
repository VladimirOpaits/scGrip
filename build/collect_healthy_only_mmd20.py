"""Collate the four-cohort healthy-only LoRA MMD=20 screen into CSV files."""

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
    "strat_healthy_A_block_offset_to_census",
    "strat_healthy_B_block_offset_to_census",
    "strat_healthy_A_donor_spread_ref_ratio",
    "strat_healthy_B_donor_spread_ref_ratio",
    "scib_batch_healthy_A_asw",
    "scib_batch_healthy_B_asw",
    "scib_block_healthy_A_asw",
    "scib_block_healthy_B_asw",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/healthy_only_mmd20_e5")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def cohort_from_directory(directory):
    match = re.fullmatch(r"eval_(lupus_c[234]|ra)_b[0-9]+", directory)
    if match is None:
        raise ValueError(f"Cannot infer cohort from evaluation directory {directory!r}")
    return match.group(1)


def mmd_weight(state):
    if state == "base":
        return np.nan
    match = re.search(r"mmd([0-9]+(?:\.[0-9]+)?)", str(state))
    if match is None:
        raise ValueError(f"Cannot infer MMD weight from state label {state!r}")
    return float(match.group(1))


def training_counts(root, cohort):
    log_path = root / "logs" / f"{cohort}_train.log"
    if not log_path.exists():
        return np.nan, np.nan
    match = re.search(
        r"healthy-only ELBO: (\d+) cells \| excluded disease cells: (\d+)",
        log_path.read_text(),
    )
    if match is None:
        return np.nan, np.nan
    return int(match.group(1)), int(match.group(2))


def main():
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / "results.csv"
    metric_paths = sorted(root.glob("eval_*/metrics.csv"))
    if not metric_paths:
        raise FileNotFoundError(f"No evaluation metrics.csv files under {root}")

    tables = []
    for path in metric_paths:
        with open(path.parent / "manifest.json") as handle:
            manifest = json.load(handle)
        cohort = cohort_from_directory(path.parent.name)
        n_train_healthy, n_train_disease_excluded = training_counts(root, cohort)
        table = pd.read_csv(path)
        table["cohort"] = cohort
        table["disease"] = manifest["disease"]
        table["training_scope"] = "healthy_only"
        table["evaluation_epoch"] = np.where(table["state"] == "base", 0, 5)
        table["mmd_weight"] = table["state"].map(mmd_weight)
        table["eval_dir"] = path.parent.name
        table["new_adata"] = manifest["new_adata"]
        table["scib_sample_cap"] = manifest["args"]["scib_max_cells"]
        table["n_train_healthy_cells"] = n_train_healthy
        table["n_train_disease_cells_excluded"] = n_train_disease_excluded
        table["signature_name"] = manifest["signature_name"]
        table["signature_n_present"] = manifest["signature"].get("n_present")
        table["signature_n_total"] = manifest["signature"].get("n_total")
        tables.append(table)

    results = pd.concat(tables, ignore_index=True, sort=False)
    for column in DELTA_COLUMNS:
        if column in results:
            results[f"delta_vs_base_{column}"] = np.nan
    for cohort, index in results.groupby("cohort", sort=False).groups.items():
        index = list(index)
        base = results.loc[index].query("state == 'base'")
        if len(base) != 1:
            raise ValueError(f"{cohort} must contain exactly one base row")
        base_row = base.iloc[0]
        for column in DELTA_COLUMNS:
            if column in results:
                results.loc[index, f"delta_vs_base_{column}"] = (
                    results.loc[index, column] - base_row[column]
                )

    results["_state_order"] = np.where(results["state"] == "base", 0, 1)
    results = results.sort_values(["cohort", "_state_order"], kind="stable").drop(
        columns="_state_order"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    compact_columns = [
        "cohort",
        "disease",
        "training_scope",
        "state",
        "evaluation_epoch",
        "mmd_weight",
        "n_train_healthy_cells",
        "n_train_disease_cells_excluded",
        "signature_n_present",
        "signature_n_total",
        "anchor_mse_sigma",
        "ifn_spearman_disease",
        "ifn_pearson_disease",
        "scib_bio_cell_type_asw",
        "scib_batch_healthy_asw",
        "scib_block_healthy_asw",
        "scib_block_healthy_A_asw",
        "scib_block_healthy_B_asw",
        "strat_healthy_A_to_census",
        "strat_healthy_B_to_census",
        "strat_healthy_A_block_offset_to_census",
        "strat_healthy_B_block_offset_to_census",
        "strat_healthy_A_donor_spread_ref_ratio",
        "strat_healthy_B_donor_spread_ref_ratio",
    ] + [f"delta_vs_base_{column}" for column in DELTA_COLUMNS]
    compact_columns = [column for column in compact_columns if column in results]
    results.loc[:, compact_columns].to_csv(output.with_name("summary.csv"), index=False)
    print(f"wrote {len(results)} rows to {output}")
    print(f"wrote compact summary to {output.with_name('summary.csv')}")


if __name__ == "__main__":
    main()
