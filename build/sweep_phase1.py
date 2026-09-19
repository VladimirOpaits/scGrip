#!/usr/bin/env python3
"""Run a reproducible phase-I MMD-strength sweep and collect evaluation CSVs.

Every run preserves the phase-I invariant implemented by ``train_phase1``:
ELBO uses healthy A+B only, MMD uses healthy A only, held-out healthy B has no
gradient, and disease cells are only encoded during evaluation.  This wrapper
does not select a winner; it makes the trade-off table available for review.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--reference-adata", required=True)
    parser.add_argument("--query-adata", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--mmd-values", type=float, nargs="+", required=True)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=(0,))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-epochs", type=int, nargs="+", default=(2, 5, 10))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-anchor", type=float, default=20.0)
    parser.add_argument("--healthy-label", action="append", default=None)
    parser.add_argument("--healthy-holdout-fraction", type=float, default=0.4)
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--input-panel-policy", choices=("strict", "audited_imputation"), default="strict")
    parser.add_argument("--mmd-k-per-stratum", type=int, default=64)
    parser.add_argument("--mmd-min-per-side", type=int, default=32)
    parser.add_argument("--signature", default=None)
    parser.add_argument("--signature-cache", default=None)
    parser.add_argument("--signature-disease-label", default=None)
    parser.add_argument(
        "--skip-existing-training",
        action="store_true",
        help="Do not retrain an already nonempty run directory; still evaluate it into a fresh full-IFN folder.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _number(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def _run(command: list[str], *, log_path: Path, dry_run: bool) -> None:
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log_path}")


def _collect(rows: list[dict[str, object]], output: Path) -> None:
    if not rows:
        return
    # Keep ``rows`` intact because this function is deliberately called after
    # every completed run to leave a crash-resilient partial CSV overnight.
    table = pd.concat([row["metrics"] for row in rows], ignore_index=True)
    configs = pd.DataFrame([{key: value for key, value in row.items() if key != "metrics"} for row in rows])
    final = configs.merge(table, on="run_name", how="right", validate="one_to_many")
    final = final.sort_values(["lambda_mmd", "split_seed", "state"], kind="stable")
    final.to_csv(output, index=False)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    logs = out_root / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    plan: list[dict[str, object]] = []
    for lambda_mmd in args.mmd_values:
        for split_seed in args.split_seeds:
            run_name = f"{args.run_prefix}_mmd{_number(lambda_mmd)}_s{split_seed}"
            run_dir = out_root / run_name
            eval_dir = out_root / f"eval_fullifn_{run_name}"
            plan.append(
                {
                    "run_name": run_name,
                    "lambda_anchor": args.lambda_anchor,
                    "lambda_mmd": lambda_mmd,
                    "split_seed": split_seed,
                    "epochs": args.epochs,
                    "healthy_holdout_fraction": args.healthy_holdout_fraction,
                    "run_dir": str(run_dir),
                    "eval_dir": str(eval_dir),
                }
            )
            if run_dir.exists() and any(run_dir.iterdir()):
                if not args.skip_existing_training:
                    raise FileExistsError(
                        f"{run_dir} already exists; use --skip-existing-training to retain it."
                    )
                print(f"retaining existing training run: {run_name}", flush=True)
            else:
                command = [
                    sys.executable, "build/train_phase1.py",
                    "--base-model", args.base_model,
                    "--reference-adata", args.reference_adata,
                    "--query-adata", args.query_adata,
                    "--out-dir", str(out_root),
                    "--run-name", run_name,
                    "--device", args.device,
                    "--counts-layer", args.counts_layer,
                    "--input-panel-policy", args.input_panel_policy,
                    "--healthy-holdout-fraction", str(args.healthy_holdout_fraction),
                    "--split-seed", str(split_seed),
                    "--seed", str(args.seed),
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--lr", str(args.lr),
                    "--lambda-anchor", str(args.lambda_anchor),
                    "--lambda-mmd", str(lambda_mmd),
                    "--mmd-k-per-stratum", str(args.mmd_k_per_stratum),
                    "--mmd-min-per-side", str(args.mmd_min_per_side),
                    "--save-epochs", *[str(epoch) for epoch in args.eval_epochs if epoch <= args.epochs],
                ]
                for label in args.healthy_label or ("normal",):
                    command.extend(("--healthy-label", label))
                _run(command, log_path=logs / f"{run_name}.train.log", dry_run=args.dry_run)
            if eval_dir.exists() and any(eval_dir.iterdir()):
                raise FileExistsError(f"Evaluation destination already exists: {eval_dir}")
            command = [
                sys.executable, "build/evaluate_phase1.py",
                "--run-dir", str(run_dir),
                "--out-dir", str(eval_dir),
                "--device", args.device,
                "--no-umap",
            ]
            for epoch in args.eval_epochs:
                checkpoint = run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
                if checkpoint.exists() or args.dry_run:
                    command.extend(("--checkpoint", f"epoch_{epoch:03d}={checkpoint}"))
            if args.signature is not None:
                command.extend(("--signature", args.signature))
            if args.signature_cache is not None:
                command.extend(("--signature-cache", args.signature_cache))
            if args.signature_disease_label is not None:
                command.extend(("--signature-disease-label", args.signature_disease_label))
            _run(command, log_path=logs / f"{run_name}.eval.log", dry_run=args.dry_run)
            if not args.dry_run:
                metrics = pd.read_csv(eval_dir / "reference_mapping_metrics.csv")
                metrics.insert(0, "run_name", run_name)
                rows.append({**plan[-1], "metrics": metrics})
                _collect(rows, out_root / "nightly_sweep_results_partial.csv")
    pd.DataFrame(plan).to_csv(out_root / "nightly_sweep_plan.csv", index=False)
    _collect(rows, out_root / "nightly_sweep_results.csv")
    print(f"wrote {out_root / 'nightly_sweep_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
