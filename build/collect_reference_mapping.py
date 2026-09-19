"""Collate LoRA and scArches reference-mapping evaluations into one CSV."""

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
    "scib_batch_healthy_A_asw",
    "scib_batch_healthy_B_asw",
    "scib_block_healthy_A_asw",
    "scib_block_healthy_B_asw",
    "strat_healthy_A_to_census",
    "strat_healthy_B_to_census",
    "strat_healthy_B_minus_A_to_census",
    "strat_healthy_A_block_offset_to_census",
    "strat_healthy_B_block_offset_to_census",
    "strat_healthy_A_donor_spread_ref_ratio",
    "strat_healthy_B_donor_spread_ref_ratio",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Experiment root containing evaluator dirs.")
    parser.add_argument("--output", default=None, help="Full CSV path; defaults to ROOT/results.csv.")
    return parser.parse_args()


def read_json(path):
    if path.exists():
        with open(path) as handle:
            return json.load(handle)
    return {}


def checkpoint_context(raw_checkpoint):
    if raw_checkpoint == "base":
        return {}, {}, ""
    path = Path(raw_checkpoint)
    run_dir = path if path.is_dir() else path.parent
    config = read_json(run_dir / "run_config.json")
    roles = read_json(run_dir / "reference_roles.json")
    return config, roles, str(run_dir)


def infer_cohort(new_adata):
    path = str(new_adata).lower()
    match = re.search(r"cohort([234])", path)
    if match:
        return f"lupus_c{match.group(1)}"
    if "/ra/" in path or "ra_full" in path:
        return "ra"
    return Path(new_adata).stem


def epoch_from_state(state, config):
    if state == "base":
        return 0
    match = re.fullmatch(r"epoch_(\d+)", str(state))
    if match:
        return int(match.group(1))
    return int(config.get("epochs", np.nan))


def main():
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / "results.csv"
    metric_paths = sorted(root.rglob("metrics.csv"))
    if not metric_paths:
        raise FileNotFoundError(f"No evaluator metrics.csv files under {root}")

    tables = []
    for metric_path in metric_paths:
        manifest_path = metric_path.parent / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing evaluator manifest next to {metric_path}")
        manifest = read_json(manifest_path)
        table = pd.read_csv(metric_path)
        if "state" not in table:
            raise ValueError(f"{metric_path} has no state column")

        evaluation_id = str(metric_path.parent.relative_to(root))
        contexts = {
            state: checkpoint_context(checkpoint)
            for state, checkpoint in zip(table["state"], table["checkpoint"], strict=True)
        }
        nonbase_contexts = [
            context for state, context in contexts.items() if state != "base"
        ]
        default_config, default_roles, default_run = (
            nonbase_contexts[0] if nonbase_contexts else ({}, {}, "")
        )
        for row_index, row in table.iterrows():
            state = row["state"]
            config, roles, run_dir = contexts[state]
            if state == "base":
                config, roles, run_dir = default_config, default_roles, default_run
            table.loc[row_index, "cohort"] = infer_cohort(manifest["new_adata"])
            table.loc[row_index, "evaluation_id"] = evaluation_id
            table.loc[row_index, "disease"] = manifest["disease"]
            table.loc[row_index, "training_run"] = run_dir
            table.loc[row_index, "training_epoch"] = epoch_from_state(state, config)
            table.loc[row_index, "training_scope"] = roles.get(
                "training_scope",
                "healthy_only" if config else "not_trained",
            )
            table.loc[row_index, "mmd_reference"] = roles.get(
                "mmd_reference",
                "not_trained" if state == "base" else "unknown",
            )
            table.loc[row_index, "mmd_query_group"] = roles.get(
                "mmd_query_group",
                "A" if config else "",
            )
            table.loc[row_index, "elbo_query_groups"] = ";".join(
                roles.get("elbo_query_groups", ["A", "B"] if config else [])
            )
            table.loc[row_index, "disease_has_gradients"] = roles.get(
                "disease_has_gradients",
                False if config else np.nan,
            )
            table.loc[row_index, "lam_anchor"] = config.get("lam", np.nan)
            table.loc[row_index, "lam_mmd"] = config.get("lam_mmd", np.nan)
            table.loc[row_index, "learning_rate"] = config.get("lr", np.nan)
            table.loc[row_index, "n_replay"] = config.get("n_replay", np.nan)
            table.loc[row_index, "n_eval"] = config.get("n_eval", np.nan)
            table.loc[row_index, "metric_ref_n"] = config.get("metric_ref_n", np.nan)
            table.loc[row_index, "mmd_k_per_compartment"] = config.get(
                "mmd_k_per_compartment",
                np.nan,
            )
            table.loc[row_index, "training_split_seed"] = config.get("split_seed", np.nan)
            table.loc[row_index, "training_seed"] = config.get("seed", np.nan)
            table.loc[row_index, "training_arches_mode"] = config.get(
                "arches_mode",
                row.get("arches_mode"),
            )
        tables.append(table)

    results = pd.concat(tables, ignore_index=True, sort=False)
    for column in DELTA_COLUMNS:
        if column in results:
            results[f"delta_vs_base_{column}"] = np.nan

    for evaluation_id, positions in results.groupby("evaluation_id", sort=False).groups.items():
        index = list(positions)
        frame = results.loc[index]
        base = frame.loc[frame["state"] == "base"]
        if len(base) != 1:
            raise ValueError(
                f"{evaluation_id} must contain exactly one base row, found {len(base)}."
            )
        base_row = base.iloc[0]
        for column in DELTA_COLUMNS:
            if column in results:
                results.loc[index, f"delta_vs_base_{column}"] = (
                    results.loc[index, column] - base_row[column]
                )

    results["_method_order"] = results["method"].map({"lora": 0, "scarches": 1}).fillna(9)
    results["_epoch_order"] = results["training_epoch"].fillna(np.inf)
    results = results.sort_values(
        ["cohort", "_method_order", "arches_mode", "evaluation_id", "_epoch_order", "state"],
        kind="stable",
    ).drop(columns=["_method_order", "_epoch_order"])

    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    summary_columns = [
        "cohort",
        "disease",
        "method",
        "arches_mode",
        "training_epoch",
        "state",
        "mmd_reference",
        "mmd_query_group",
        "elbo_query_groups",
        "disease_has_gradients",
        "lam_anchor",
        "lam_mmd",
        "learning_rate",
        "anchor_mse_sigma",
        "anchor_drift_max",
        "strat_healthy_A_to_census",
        "strat_healthy_B_to_census",
        "strat_healthy_B_minus_A_to_census",
        "scib_bio_cell_type_asw",
        "scib_bio_compartment_asw",
        "scib_batch_healthy_asw",
        "scib_batch_healthy_A_asw",
        "scib_batch_healthy_B_asw",
        "scib_block_healthy_asw",
        "scib_block_healthy_A_asw",
        "scib_block_healthy_B_asw",
        "ifn_spearman_disease",
        "ifn_pearson_disease",
    ] + [f"delta_vs_base_{column}" for column in DELTA_COLUMNS]
    summary_columns = [column for column in summary_columns if column in results]
    summary_path = output.with_name("summary.csv")
    results.loc[:, summary_columns].to_csv(summary_path, index=False)
    print(f"wrote {len(results)} rows x {len(results.columns)} columns to {output}")
    print(f"wrote compact summary to {summary_path}")


if __name__ == "__main__":
    main()

