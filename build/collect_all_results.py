#!/usr/bin/env python
"""Collect every nightly sweep result into one master CSV.

Run from the scGRIP repo root:

    PYTHONPATH=$PWD python build/collect_all_results.py \
        --root runs/night_20260919 \
        --out runs/night_20260919/master_all_results.csv

The script is tolerant: missing groups, partial files and differing column
sets are reported, not fatal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------
# Group definitions.
#
# `sweep_group` is the label that ends up in the master CSV.  `patterns` are
# globs relative to --root.  `prefer_corrected` means: if both the raw and a
# *_corrected* sibling exist, drop the raw one (ratio60 disease-label bug).
# --------------------------------------------------------------------------
GROUPS: list[dict] = [
    {
        "sweep_group": "primary",
        "patterns": ["nightly_sweep_results.csv"],
        "prefer_corrected": False,
    },
    {
        "sweep_group": "primary_mmd0",
        "patterns": ["gse298578_published_cd4_w0_lam20_mmd0_s0/**/reference_mapping_metrics.csv"],
        "prefer_corrected": False,
        "eval_only": True,
    },
    {
        "sweep_group": "repeat_splits",
        "patterns": ["repeat_splits/nightly_sweep_results.csv"],
        "prefer_corrected": False,
    },
    {
        "sweep_group": "ratio60",
        "patterns": [
            "ratio_ablation_ifn18/nightly_sweep_results.csv",
            "ratio_ablation_ifn18/nightly_sweep_results_corrected.csv",
        ],
        "prefer_corrected": True,
    },
    {
        "sweep_group": "anchor5",
        "patterns": ["anchor_ablation/a5/nightly_sweep_results.csv"],
        "prefer_corrected": False,
    },
    {
        "sweep_group": "anchor50",
        "patterns": ["anchor_ablation/a50/nightly_sweep_results.csv"],
        "prefer_corrected": False,
    },
    {
        "sweep_group": "anchor100",
        "patterns": ["anchor_ablation/a100/nightly_sweep_results.csv"],
        "prefer_corrected": False,
    },
    {
        "sweep_group": "temporal_a100_mmd500",
        "patterns": ["temporal_a100_mmd500/nightly_sweep_results.csv"],
        "prefer_corrected": False,
    },
    {
        "sweep_group": "anchor100_repeat_splits",
        "patterns": [
            "anchor100_repeat_splits/nightly_sweep_results.csv",
            "anchor100_repeat_splits/nightly_sweep_results_partial.csv",
        ],
        "prefer_corrected": False,
    },
    {
        "sweep_group": "anchor100_missing_pairs",
        "patterns": ["anchor100_missing_pairs/nightly_sweep_results.csv"],
        "prefer_corrected": False,
    },
    {
        "sweep_group": "anchor100_mmd0_all_pairs",
        "patterns": ["anchor100_mmd0_all_pairs/nightly_sweep_results.csv"],
        "prefer_corrected": False,
    },
]

# Standalone aggregates that are already long-format and worth keeping.
EXTRA_FILES: list[dict] = [
    {"sweep_group": "agg_all_pairs",      "path": "all_pairs_anchor100_mmd500.csv"},
    {"sweep_group": "agg_all_pairs_core", "path": "all_pairs_anchor100_mmd500_core_B.csv"},
    {"sweep_group": "agg_b_pair_plan",    "path": "all_b_pair_plan.csv"},
]

# Columns that we always want present, even if a group lacks them.
CANONICAL_COLUMNS = [
    "sweep_group", "source_file", "run_name",
    "lambda_anchor", "lambda_mmd", "split_seed", "epochs",
    "healthy_holdout_fraction", "run_dir", "eval_dir",
    "state", "checkpoint",
    "anchor_held_out", "drift_max", "drift_p95",
    "strat_A", "strat_B", "strat_A_block_offset", "strat_B_block_offset",
    "strat_common_n", "mmd_A_eval",
    "scib_n_query_cells", "scib_stratum_asw",
    "scib_batch_healthy_A_asw", "scib_batch_healthy_B_asw",
    "scib_block_healthy_A_asw", "scib_block_healthy_B_asw",
    "signature_n_all_donors", "signature_spearman_all", "signature_spearman_all_p",
    "signature_pearson_all", "signature_pearson_all_p",
    "signature_n_disease_donors", "signature_spearman_disease",
    "signature_spearman_disease_p", "signature_pearson_disease",
    "signature_pearson_disease_p",
]


def _pick_sources(root: Path, patterns: list[str], prefer_corrected: bool) -> list[Path]:
    """Resolve one group's input files, honouring prefer_corrected."""
    found: list[Path] = []
    for pat in patterns:
        found.extend(sorted(root.glob(pat)))
    if not found:
        return []
    if not prefer_corrected:
        return found
    corrected = [p for p in found if "corrected" in p.name]
    if corrected:
        # Keep only the corrected variant(s).
        return corrected
    return found


def _read_one(path: Path, sweep_group: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] cannot read {path}: {exc}", file=sys.stderr)
        return None
    if df.empty:
        print(f"  [warn] empty file {path}", file=sys.stderr)
        return None
    df = df.copy()
    df.insert(0, "sweep_group", sweep_group)
    df.insert(1, "source_file", str(path))
    return df


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing canonical columns as NA so concat is stable."""
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    # Put canonical columns first, keep the rest at the end.
    extra = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    return df[CANONICAL_COLUMNS + extra]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/night_20260919", type=Path)
    ap.add_argument("--out", default="runs/night_20260919/master_all_results.csv", type=Path)
    ap.add_argument("--out-extras", default="runs/night_20260919/master_all_aggregates.csv", type=Path)
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero if any declared group produced zero rows.")
    args = ap.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(f"root does not exist: {root}", file=sys.stderr)
        return 2

    frames: list[pd.DataFrame] = []
    report: list[tuple[str, int, list[str]]] = []

    print(f"scanning {root}")
    for spec in GROUPS:
        group = spec["sweep_group"]
        sources = _pick_sources(root, spec["patterns"], spec.get("prefer_corrected", False))
        if not sources:
            report.append((group, 0, [f"no file matched {spec['patterns']}"]))
            print(f"  {group:28s} -> no file")
            continue
        group_frames: list[pd.DataFrame] = []
        for path in sources:
            df = _read_one(path, group)
            if df is not None:
                group_frames.append(df)
        if not group_frames:
            report.append((group, 0, [str(p) for p in sources]))
            print(f"  {group:28s} -> unreadable")
            continue
        merged = pd.concat(group_frames, ignore_index=True, sort=False)
        frames.append(merged)
        report.append((group, len(merged), [str(p) for p in sources]))
        print(f"  {group:28s} -> {len(merged):6d} rows from {len(sources)} file(s)")

    if not frames:
        print("no sweep rows collected", file=sys.stderr)
        return 3

    master = _normalise(pd.concat(frames, ignore_index=True, sort=False))

    # ---- derived deltas -------------------------------------------------
    # within (sweep_group, run_name): delta vs that run's own `base` state.
    def _add_deltas(df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for (grp, run), sub in df.groupby(["sweep_group", "run_name"], dropna=False):
            sub = sub.copy()
            base_rows = sub.loc[sub["state"].astype(str).eq("base")]
            for metric in ("strat_A", "strat_B", "anchor_held_out", "mmd_A_eval"):
                if metric not in sub.columns:
                    continue
                bval = pd.to_numeric(base_rows[metric], errors="coerce").iloc[0] \
                    if not base_rows.empty else pd.NA
                sub[f"delta_{metric}"] = pd.to_numeric(sub[metric], errors="coerce") - bval
            out.append(sub)
        return pd.concat(out, ignore_index=True) if out else df

    master = _add_deltas(master)

    # stable ordering
    order = ["sweep_group", "run_name", "split_seed", "state"]
    master = master.sort_values(order, kind="stable", na_position="last")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(master)} rows, {master['sweep_group'].nunique()} groups)")

    # ---- extras ---------------------------------------------------------
    extra_frames = []
    for spec in EXTRA_FILES:
        p = root / spec["path"]
        if not p.exists():
            print(f"  [extras] missing {p}")
            continue
        try:
            df = pd.read_csv(p)
        except Exception as exc:  # noqa: BLE001
            print(f"  [extras] unreadable {p}: {exc}")
            continue
        df.insert(0, "sweep_group", spec["sweep_group"])
        df.insert(1, "source_file", str(p))
        extra_frames.append(df)
        print(f"  [extras] {spec['sweep_group']:22s} -> {len(df)} rows")
    if extra_frames:
        extras = pd.concat(extra_frames, ignore_index=True, sort=False)
        extras.to_csv(args.out_extras, index=False)
        print(f"wrote {args.out_extras}  ({len(extras)} rows)")
    else:
        print("no aggregate extras found")

    # ---- report ---------------------------------------------------------
    print("\n=== coverage report ===")
    empty: list[str] = []
    for group, n, files in report:
        flag = "OK " if n > 0 else "MISS"
        print(f"  [{flag}] {group:28s} rows={n:6d}  files={len(files)}")
        if n == 0:
            empty.append(group)
    if empty:
        print(f"\nempty groups: {', '.join(empty)}")
        if args.strict:
            return 1

    print("\n=== per-group epoch coverage (state counts) ===")
    cov = (master
           .assign(state=master["state"].astype(str))
           .groupby(["sweep_group", "state"])
           .size()
           .unstack(fill_value=0))
    print(cov.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())