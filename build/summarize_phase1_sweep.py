#!/usr/bin/env python3
"""Render a non-selective table and trade-off plot from phase-I sweep CSVs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def _epoch(state: str) -> int:
    if state == "base":
        return 0
    match = re.fullmatch(r"epoch_(\d+)", str(state))
    if match is None:
        raise ValueError(f"Unrecognised state {state!r}")
    return int(match.group(1))


def _markdown(table: pd.DataFrame) -> str:
    columns = [
        "lambda_mmd", "epoch", "anchor_held_out", "strat_A", "strat_B",
        "mmd_A_eval", "scib_stratum_asw", "signature_spearman_all",
        "signature_spearman_disease",
    ]
    view = table.loc[:, [column for column in columns if column in table]].copy()
    for column in view.select_dtypes(include="number"):
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    header = "| " + " | ".join(view.columns) + " |"
    separator = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows]) + "\n"


def main() -> None:
    args = parse_args()
    source = Path(args.results_csv)
    destination = Path(args.out_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(source)
    if table.empty:
        raise ValueError(f"{source} has no rows")
    table["epoch"] = table["state"].map(_epoch)
    table = table.sort_values(["lambda_mmd", "split_seed", "epoch"], kind="stable")
    table.to_csv(destination / "sweep_all_states.csv", index=False)
    selected = table[table["epoch"].isin((0, 2, 5, 10))].copy()
    selected.to_csv(destination / "sweep_epoch_table.csv", index=False)
    (destination / "sweep_epoch_table.md").write_text(_markdown(selected), encoding="utf-8")

    values = sorted(selected["lambda_mmd"].drop_duplicates())
    positions = np.arange(len(values))
    styles = {0: "--", 2: "-", 5: "-.", 10: ":"}
    panels = [
        ("strat_B", "held-out healthy B distance ↓"),
        ("mmd_A_eval", "fixed MMD² on healthy A ↓"),
        ("anchor_held_out", "reference anchor drift ↓"),
        ("signature_spearman_disease", "SLE IFN-18 distance Spearman ↑"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, (metric, title) in zip(axes.flat, panels, strict=True):
        for epoch in sorted(selected["epoch"].unique()):
            subset = selected[selected["epoch"] == epoch]
            # A base row is duplicated once per run but should be numerically
            # identical; mean makes that fact explicit without hiding split
            # variation at adapted epochs.
            summary = subset.groupby("lambda_mmd", observed=True)[metric].mean().reindex(values)
            axis.plot(positions, summary.to_numpy(), marker="o", linestyle=styles.get(epoch, "-"), label=f"epoch {epoch}")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xticks(positions, [f"{value:g}" for value in values])
        axis.set_xlabel("λ MMD")
    axes[0, 0].legend(title="state", fontsize=8)
    figure.suptitle("Phase-I trade-offs; reporting only, no automatic winner selection")
    figure.tight_layout()
    figure.savefig(destination / "mmd_tradeoffs.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    (destination / "README.md").write_text(
        """# Phase-I sweep summary

This folder is a visualization of immutable run outputs.  It does not select a
checkpoint or an MMD strength.  Interpret held-out healthy B and held-out
reference anchor jointly; disease IFN-18 is a post-hoc positive control only.
The current GSE298578 run has five healthy donors, so each A/B split remains a
small-cohort diagnostic and requires repeat-split confirmation.
""",
        encoding="utf-8",
    )
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
