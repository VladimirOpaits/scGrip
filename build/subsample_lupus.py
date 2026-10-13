#!/usr/bin/env python3
"""Subsample the large lupus atlas to a manageable size.

Keeps at most --max-cells-per-donor-type per (donor, cell_type) pair,
and optionally caps the total number of cells per cell type.

Raw counts are read from a.raw.X (the .X slot in this file is log1p +
z-scored and must not be used for training).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--donor-col", default="donor_id")
    ap.add_argument("--cell-type-col", default="cell_type")
    ap.add_argument("--disease-col", default="disease")
    ap.add_argument("--max-cells-per-donor-type", type=int, default=50)
    ap.add_argument("--max-cells-per-type", type=int, default=None)
    ap.add_argument("--healthy-labels", nargs="+", default=["normal"])
    ap.add_argument("--use-raw", action="store_true",
                    help="Use a.raw.X instead of a.X (recommended for this file).")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def materialize_raw_chunk(a: ad.AnnData, chunk: np.ndarray) -> sparse.csr_matrix:
    """Read raw counts for the given row indices as a CSR matrix."""
    raw = a.raw.X[chunk]  # _CSRDataset (lazy)
    # Different anndata versions expose different materialization methods
    if hasattr(raw, "to_memory"):
        raw = raw.to_memory()
    elif hasattr(raw, "to_scipy"):
        raw = raw.to_scipy()
    elif hasattr(raw, "toarray"):
        raw = raw.toarray()
    if not sparse.issparse(raw):
        raw = sparse.csr_matrix(raw)
    return raw.astype(np.float32).tocsr()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"opening {args.input} (backed)")
    a = ad.read_h5ad(args.input, backed="r")
    print(f"shape: {a.shape}")

    obs = a.obs.copy()
    obs["_row"] = np.arange(len(obs), dtype=np.int64)
    obs["_donor"] = obs[args.donor_col].astype(str)
    obs["_type"] = obs[args.cell_type_col].astype(str)
    obs["_disease"] = obs[args.disease_col].astype(str)

    healthy = obs["_disease"].isin(args.healthy_labels).to_numpy()
    print(f"healthy cells: {healthy.sum():,}")
    print(f"disease cells: {(~healthy).sum():,}")

    rng = np.random.default_rng(args.seed)

    # 1. per (donor, type) cap
    keep_rows: list[np.ndarray] = []
    for _, group in obs.groupby(["_donor", "_type"], observed=True):
        idx = group["_row"].to_numpy()
        if idx.size > args.max_cells_per_donor_type:
            idx = rng.choice(idx, size=args.max_cells_per_donor_type, replace=False)
        keep_rows.append(idx)
    kept = np.sort(np.concatenate(keep_rows))
    print(f"after per (donor, type) cap: {kept.size:,}")

    # 2. per-type cap
    if args.max_cells_per_type is not None:
        sub_obs = obs[obs["_row"].isin(kept)]
        keep_rows = []
        for _, group in sub_obs.groupby("_type", observed=True):
            idx = group["_row"].to_numpy()
            if idx.size > args.max_cells_per_type:
                idx = rng.choice(idx, size=args.max_cells_per_type, replace=False)
            keep_rows.append(idx)
        kept = np.sort(np.concatenate(keep_rows))
        print(f"after per-type cap: {kept.size:,}")

    # 3. read subset in chunks
    print("reading subset in chunks")
    chunk_size = 10_000
    parts = []
    for start in range(0, len(kept), chunk_size):
        chunk = kept[start:start + chunk_size]
        sub = a[chunk].to_memory()

        if args.use_raw:
            sub.X = materialize_raw_chunk(a, chunk)
        else:
            if sparse.issparse(sub.X):
                sub.X = sub.X.astype(np.float32)

        parts.append(sub)
        print(f"  read {min(start + chunk_size, len(kept)):,}/{len(kept):,}")

    combined = ad.concat(parts, axis=0, join="inner", merge="same")
    print(f"combined: {combined.shape}")

    # sanity check on X values
    sample = combined.X[:100]
    if hasattr(sample, "toarray"):
        sample = sample.toarray()
    print(f"X max: {np.asarray(sample).max()}")

    combined.uns["lupus_subsample"] = {
        "source": str(args.input),
        "max_cells_per_donor_type": args.max_cells_per_donor_type,
        "max_cells_per_type": args.max_cells_per_type,
        "used_raw": args.use_raw,
        "n_cells_final": int(combined.n_obs),
        "seed": args.seed,
    }
    combined.write_h5ad(args.out, compression="gzip")
    print(f"wrote {args.out}")

    print("\ncell_type distribution:")
    print(combined.obs[args.cell_type_col].value_counts().to_string())
    print("\ndisease distribution:")
    print(combined.obs[args.disease_col].value_counts().to_string())
    print("\ndonor count:", combined.obs[args.donor_col].nunique())


if __name__ == "__main__":
    main()