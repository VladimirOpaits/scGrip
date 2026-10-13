#!/usr/bin/env python3
"""Merge Census and lupus healthy banks into one 8000-HVG bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--census", required=True, type=Path)
    ap.add_argument("--lupus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-hvg", type=int, default=8000)
    ap.add_argument("--batch-key", default="dataset_id")
    ap.add_argument("--donor-col", default="donor_id")
    ap.add_argument("--celltype-col", default="celltypist")
    ap.add_argument("--min-cells-per-gene", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def harmonize_var_names(a: ad.AnnData, source: str) -> ad.AnnData:
    """Keep var_names as they are (should be gene symbols after CellTypist)."""
    first = str(a.var_names[0])
    print(f"  [{source}] var_names[0]={first!r}, keeping as is")
    a.var_names_make_unique()
    return a


def to_csr(a: ad.AnnData) -> ad.AnnData:
    if not sparse.issparse(a.X):
        a.X = sparse.csr_matrix(a.X)
    a.X = a.X.astype(np.float32).tocsr()
    return a


def harmonize_obs(a: ad.AnnData, source: str, args: argparse.Namespace) -> ad.AnnData:
    obs = a.obs.copy()
    obs[args.donor_col] = source + "|" + obs[args.donor_col].astype(str)
    if args.batch_key in obs.columns:
        obs[args.batch_key] = source + "|" + obs[args.batch_key].astype(str)
    else:
        obs[args.batch_key] = source
    obs["_source"] = source
    a.obs = obs
    return a


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading Census: {args.census}")
    census = ad.read_h5ad(args.census, backed="r")
    print(f"  shape: {census.shape}")

    print(f"loading lupus: {args.lupus}")
    lupus = ad.read_h5ad(args.lupus, backed="r")
    print(f"  shape: {lupus.shape}")

    # --- harmonize var_names BEFORE intersecting --------------------------
    print("\nharmonizing var_names")
    census = harmonize_var_names(census, "census")
    lupus = harmonize_var_names(lupus, "lupus")

    # --- intersect --------------------------------------------------------
    census_genes = pd.Index([str(g) for g in census.var_names])
    lupus_genes = pd.Index([str(g) for g in lupus.var_names])
    shared = census_genes.intersection(lupus_genes)
    print(f"\nshared genes: {len(shared)}")
    if len(shared) < 5000:
        raise ValueError(f"Too few shared genes ({len(shared)}).")

    print("\nreading to memory")
    census = census[:, shared].to_memory()
    lupus = lupus[:, shared].to_memory()
    print(f"  census: {census.shape}")
    print(f"  lupus: {lupus.shape}")

    # --- harmonize obs ----------------------------------------------------
    census = harmonize_obs(census, "census", args)
    lupus = harmonize_obs(lupus, "lupus_c55dc602", args)

    # --- concat -----------------------------------------------------------
    print("\nconcatenating")
    combined = ad.concat(
        [census, lupus],
        axis=0,
        join="inner",
        merge="same",
        index_unique="__grip__",
    )
    print(f"  combined: {combined.shape}")
    print(f"  sources: {combined.obs['_source'].value_counts().to_dict()}")

    combined = to_csr(combined)

    # --- filter genes -----------------------------------------------------
    counts = combined.X
    n_cells_per_gene = np.asarray((counts > 0).sum(axis=0)).ravel()
    keep = n_cells_per_gene >= args.min_cells_per_gene
    print(f"\ngenes with >= {args.min_cells_per_gene} expressing cells: "
          f"{int(keep.sum())}/{len(keep)}")
    combined = combined[:, keep].copy()
    print(f"  after gene filter: {combined.shape}")

    # --- HVG --------------------------------------------------------------
    print(f"\nselecting {args.n_hvg} HVGs")
    sc.pp.highly_variable_genes(
        combined,
        n_top_genes=args.n_hvg,
        flavor="seurat",
        batch_key=args.batch_key,
    )
    hvg_mask = combined.var["highly_variable"].to_numpy()
    print(f"  selected {int(hvg_mask.sum())} HVGs")

    final = combined[:, hvg_mask].copy()
    print(f"  final bank: {final.shape}")

    # --- save -------------------------------------------------------------
    final.uns["grip_bank_merge"] = json.dumps({
        "census": str(args.census),
        "lupus": str(args.lupus),
        "n_hvg": args.n_hvg,
        "min_cells_per_gene": args.min_cells_per_gene,
        "batch_key": args.batch_key,
        "celltype_col": args.celltype_col,
        "n_cells_final": int(final.n_obs),
        "n_genes_final": int(final.n_vars),
        "n_donors_final": int(final.obs[args.donor_col].nunique()),
        "sources": final.obs["_source"].value_counts().to_dict(),
    }, sort_keys=True)

    final.write_h5ad(args.out, compression="gzip")
    print(f"\nwrote {args.out}")

    print("\ncelltype distribution:")
    print(final.obs[args.celltype_col].value_counts().to_string())
    print("\nsource distribution:")
    print(final.obs["_source"].value_counts().to_string())


if __name__ == "__main__":
    main()