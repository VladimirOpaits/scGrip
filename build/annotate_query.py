#!/usr/bin/env python3
"""Annotate an AnnData with CellTypist (immune cell classifier).

CellTypist requires gene symbols in var_names. This script:
  1. Loads the input AnnData.
  2. If var_names are Ensembl IDs or a positional index, replaces them
     with gene symbols from var['feature_name'] (or var['GeneSymbol']).
  3. Normalizes counts (CP10K + log1p) unless the input is already
     normalized.
  4. Runs CellTypist in chunks to keep memory bounded.
  5. Writes the annotated AnnData with obs['celltypist'] labels.

majority_voting is disabled in chunked mode because it requires a
neighborhood graph over the full dataset. Per-cell labels are used.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import anndata as ad
import celltypist
from celltypist import models
import numpy as np
import scanpy as sc


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="Immune_All_High.pkl")
    ap.add_argument("--counts-layer", default=None,
                    help="Layer with raw counts; None uses .X")
    ap.add_argument("--chunk-size", type=int, default=20_000,
                    help="Cells per CellTypist call; keeps peak memory bounded.")
    ap.add_argument("--majority-voting", action="store_true",
                    help="Only used when input is smaller than chunk-size.")
    ap.add_argument("--already-normalized", action="store_true",
                    help="Skip normalize_total + log1p if input is already log1p.")
    return ap.parse_args()


def ensure_gene_symbols(a: ad.AnnData) -> ad.AnnData:
    """CellTypist expects gene symbols; Ensembl IDs and positional indices
    are not accepted."""
    first = str(a.var_names[0])

    is_ensembl = first.startswith("ENSG")
    is_positional = first.isdigit()
    is_symbol_like = first[:1].isalpha() and not is_ensembl

    if is_symbol_like:
        print(f"var_names look like gene symbols (first: {first!r})")
        a.var_names_make_unique()
        return a

    if "feature_name" in a.var.columns:
        print("using var['feature_name'] as var_names (gene symbols)")
        a.var_names = a.var["feature_name"].astype(str)
    elif "GeneSymbol" in a.var.columns:
        print("using var['GeneSymbol'] as var_names")
        a.var_names = a.var["GeneSymbol"].astype(str)
    else:
        raise ValueError(
            f"var_names are not symbols (first: {first!r}) and no symbol column found. "
            f"var columns: {a.var.columns.tolist()}"
        )

    a.var_names_make_unique()
    return a


def annotate_chunked(a: ad.AnnData, model: str, chunk_size: int, majority_voting: bool) -> np.ndarray:
    """Run CellTypist on chunks of cells and return per-cell labels."""
    n = a.n_obs
    if n <= chunk_size:
        print(f"running CellTypist on {n} cells (single chunk)")
        preds = celltypist.annotate(
            a,
            model=model,
            majority_voting=majority_voting,
        )
        if majority_voting and "majority_voting" in preds.predicted_labels.columns:
            return preds.predicted_labels["majority_voting"].to_numpy()
        return preds.predicted_labels["predicted_labels"].to_numpy()

    print(f"running CellTypist in chunks of {chunk_size}")
    n_chunks = (n + chunk_size - 1) // chunk_size
    labels = []
    for i, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        print(f"  chunk {i + 1}/{n_chunks}: cells {start}:{end}")
        chunk = a[start:end].copy()
        preds = celltypist.annotate(
            chunk,
            model=model,
            majority_voting=False,
        )
        labels.append(preds.predicted_labels["predicted_labels"].to_numpy())
        del chunk, preds
        gc.collect()
    return np.concatenate(labels)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.input}")
    a = ad.read_h5ad(args.input)
    print(f"shape: {a.shape}")

    # CellTypist needs raw counts in X.
    if args.counts_layer is not None:
        if args.counts_layer not in a.layers:
            raise ValueError(f"Layer {args.counts_layer!r} not in {list(a.layers)}")
        a.X = a.layers[args.counts_layer].copy()

    a = ensure_gene_symbols(a)

    if args.already_normalized:
        print("skipping normalization (--already-normalized)")
    else:
        print("normalizing (CP10K + log1p)")
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)

    models.download_models(model=args.model)
    print(f"running CellTypist with {args.model}")

    labels = annotate_chunked(
        a,
        model=args.model,
        chunk_size=args.chunk_size,
        majority_voting=args.majority_voting,
    )
    a.obs["celltypist"] = labels

    print(f"\ncell types: {a.obs['celltypist'].nunique()}")
    print(a.obs["celltypist"].value_counts().to_string())

    # Avoid name clash between var index and a var column.
    a.var.index.name = None

    a.write_h5ad(args.out, compression="gzip")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()