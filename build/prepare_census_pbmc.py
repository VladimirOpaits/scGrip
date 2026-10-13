#!/usr/bin/env python3
"""Prepare a small healthy PBMC reference bank from CELLxGENE Census.

Pipeline:
  1. Fetch metadata only (fast) and select (donor, cell_type) pairs that pass
     the per-donor threshold. Keep a cell type only if enough distinct donors
     pass. Optionally cap donors per type.
  2. Fetch raw counts for the selected cells and save them immediately to
     ``<out>_raw.h5ad``.
  3. If ``--model-panel`` is given, project the raw AnnData onto the shared
     gene panel and save the final artifact to ``<out>``.

Saving raw before projection means a projection failure does not lose the
downloaded expression. The raw file can be re-projected without re-downloading.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-adata", required=True, type=Path)
    ap.add_argument("--census-version", default="2024-07-01")
    ap.add_argument(
        "--tissues", nargs="+", default=["blood"],
        help="Census tissue_general values. Default: blood.",
    )
    ap.add_argument(
        "--min-cells-per-donor-type", type=int, default=30,
        help="Minimum cells of a given type in a given donor for that pair to be kept.",
    )
    ap.add_argument(
        "--max-cells-per-donor-type", type=int, default=100,
        help="Upper cap on cells sampled per (donor, type) pair after filtering.",
    )
    ap.add_argument(
        "--min-donors-per-type", type=int, default=20,
        help="A type is kept only if at least this many donors pass the per-donor threshold.",
    )
    ap.add_argument(
        "--max-donors-per-type", type=int, default=100,
        help="Cap on the number of donors sampled per cell type.",
    )
    ap.add_argument(
        "--cell-types", nargs="+", default=None,
        help="Optional whitelist of cell_type values (Census labels).",
    )
    ap.add_argument(
        "--model-panel", default=None, type=Path,
        help="Optional frozen scVI model.pt; projects output to its gene panel.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--report", default=None, type=Path,
        help="Optional CSV path for per-(donor,type) support after filtering.",
    )
    ap.add_argument(
        "--metadata-only", action="store_true",
        help="Only run the metadata step and print the plan; do not download expression.",
    )
    ap.add_argument(
        "--throughput-cells-per-minute", type=float, default=100_000.0,
        help="Calibrated throughput for the time estimate.",
    )
    ap.add_argument(
        "--skip-if-raw-exists", action="store_true",
        help="If <out>_raw.h5ad already exists, reuse it instead of re-downloading.",
    )
    return ap.parse_args()


def open_census(version: str):
    import cellxgene_census
    return cellxgene_census.open_soma(census_version=version)


def load_obs(args: argparse.Namespace) -> pd.DataFrame:
    """Fetch only the metadata columns we need for planning."""
    census = open_census(args.census_version)
    obs_filter = (
        "is_primary_data == True"
        " and disease == 'normal'"
        f" and tissue_general in {args.tissues!r}"
    )
    columns = [
        "soma_joinid",
        "dataset_id",
        "donor_id",
        "cell_type",
        "cell_type_ontology_term_id",
        "tissue",
        "tissue_general",
        "assay",
        "suspension_type",
        "disease",
        "development_stage",
        "raw_sum",
    ]
    available = set(census["census_data"]["homo_sapiens"].obs.schema.names)
    columns = [c for c in columns if c in available]
    obs = (
        census["census_data"]["homo_sapiens"]
        .obs.read(value_filter=obs_filter, column_names=columns)
        .concat()
        .to_pandas()
    )
    return obs


def plan_selection(obs: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (selected_obs, support) after per-donor-per-type filtering."""
    if args.cell_types is not None:
        obs = obs[obs["cell_type"].isin(args.cell_types)].copy()
        if obs.empty:
            raise ValueError("No cells left after --cell-types filter.")

    # Census donor_id is dataset-scoped: combine with dataset_id.
    obs["_donor_key"] = obs["dataset_id"].astype(str) + "|" + obs["donor_id"].astype(str)

    pair_counts = (
        obs.groupby(["_donor_key", "cell_type"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )

    valid_pairs = pair_counts[pair_counts["n_cells"] >= args.min_cells_per_donor_type]

    donors_per_type = (
        valid_pairs.groupby("cell_type", observed=True)["_donor_key"]
        .nunique()
        .rename("n_donors")
    )
    keep_types = donors_per_type[donors_per_type >= args.min_donors_per_type].index
    if len(keep_types) == 0:
        raise ValueError(
            "No cell type meets --min-donors-per-type after the per-donor threshold."
        )

    obs = obs[obs["cell_type"].isin(keep_types)].copy()
    valid_pair_index = pd.MultiIndex.from_frame(
        valid_pairs.loc[valid_pairs["cell_type"].isin(keep_types), ["_donor_key", "cell_type"]]
    )
    obs_index = pd.MultiIndex.from_frame(obs[["_donor_key", "cell_type"]])
    obs = obs[obs_index.isin(valid_pair_index)].copy()

    rng = np.random.default_rng(args.seed)

    # Optional cap: at most N donors per cell type.
    if args.max_donors_per_type is not None:
        keep_rows: list[np.ndarray] = []
        for _, group in obs.groupby("cell_type", observed=True):
            donors = group["_donor_key"].unique()
            if donors.size > args.max_donors_per_type:
                donors = rng.choice(donors, size=args.max_donors_per_type, replace=False)
            keep_rows.append(group[group["_donor_key"].isin(donors)].index.to_numpy())
        obs = obs.loc[np.concatenate(keep_rows)].copy()

    # Cap per (donor, type).
    kept_rows: list[np.ndarray] = []
    for _, group in obs.groupby(["_donor_key", "cell_type"], observed=True):
        idx = group.index.to_numpy()
        if idx.size > args.max_cells_per_donor_type:
            idx = rng.choice(idx, size=args.max_cells_per_donor_type, replace=False)
        kept_rows.append(idx)
    kept = np.concatenate(kept_rows) if kept_rows else np.empty(0, dtype=int)
    obs = obs.loc[kept].copy()

    support = (
        obs.groupby(["dataset_id", "donor_id", "cell_type"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values(["cell_type", "dataset_id", "donor_id"], kind="stable")
    )
    return obs, support


def estimate_time(n_cells: int, throughput_per_min: float) -> dict[str, float]:
    minutes = n_cells / max(throughput_per_min, 1.0)
    return {
        "n_cells": int(n_cells),
        "throughput_cells_per_min": float(throughput_per_min),
        "estimated_minutes": minutes,
        "estimated_hours": minutes / 60.0,
    }


def fetch_expression(obs, args):
    import cellxgene_census
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path
    import gc

    census = open_census(args.census_version)
    joinids = obs["soma_joinid"].astype(int).tolist()
    chunk = 1024
    chunks = [joinids[i:i+chunk] for i in range(0, len(joinids), chunk)]
    total = len(joinids)

    # папка для временных чанков
    tmp_dir = args.out_adata.parent / f"{args.out_adata.stem}_chunks"
    tmp_dir.mkdir(exist_ok=True)

    def fetch_one(idx_batch):
        idx, batch = idx_batch
        adata = cellxgene_census.get_anndata(
            census=census,
            organism="homo_sapiens",
            measurement_name="RNA",
            X_name="raw",
            obs_coords=batch,
            obs_column_names=[
                "dataset_id", "donor_id", "cell_type", "cell_type_ontology_term_id",
                "tissue", "tissue_general", "assay", "suspension_type",
                "disease", "is_primary_data", "development_stage", "raw_sum",
            ],
            var_column_names=["feature_id", "feature_name"],
        )
        chunk_path = tmp_dir / f"chunk_{idx:05d}.h5ad"
        adata.write_h5ad(chunk_path, compression="gzip")
        return chunk_path

    n_workers = 4
    print(f"  fetching {len(chunks)} chunks with {n_workers} workers to {tmp_dir}", flush=True)
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(fetch_one, (i, c)): i for i, c in enumerate(chunks)}
        for future in as_completed(futures):
            chunk_path = future.result()
            # читаем только n_obs для прогресса
            a = ad.read_h5ad(chunk_path, backed="r")
            done += a.n_obs
            a.file.close()
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-9)
            eta = (total - done) / max(rate, 1e-9)
            print(f"  fetched {done:,}/{total:,} | {rate:,.0f} cells/s | ETA {eta/60:.1f} min", flush=True)

    # финальная склейка по одному чанку за раз
    print(f"  concatenating {len(chunks)} chunks", flush=True)
    chunk_files = sorted(tmp_dir.glob("chunk_*.h5ad"))
    combined = None
    for i, cf in enumerate(chunk_files):
        part = ad.read_h5ad(cf)
        if combined is None:
            combined = part
        else:
            combined = ad.concat([combined, part], axis=0, join="outer", merge="same")
        del part
        gc.collect()
        if (i + 1) % 10 == 0:
            print(f"    concatenated {i+1}/{len(chunk_files)}", flush=True)
    return combined


def normalise_var_names(adata: ad.AnnData) -> ad.AnnData:
    """Ensure var_names are Ensembl IDs (feature_id), not gene symbols."""
    first = str(adata.var_names[0])
    if first.startswith("ENSG"):
        return adata
    if "feature_id" not in adata.var.columns:
        raise ValueError(
            f"adata.var_names[0]={first!r} is not an Ensembl ID and "
            "adata.var has no 'feature_id' column."
        )
    print(f"renaming var_names from {first!r} to feature_id")
    adata = adata.copy()
    adata.var_names = adata.var["feature_id"].astype(str)
    adata.var_names_make_unique()
    return adata


def project_to_model_panel(adata: ad.AnnData, model_path: Path) -> ad.AnnData:
    """Project to model panel via intersection; drop missing genes with audit."""
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    panel = pd.Index(np.asarray(payload["var_names"], dtype=str))
    observed = pd.Index(np.asarray(adata.var_names, dtype=str))

    shared = panel.intersection(observed)
    missing = panel.difference(observed)

    coverage = len(shared) / len(panel)
    print(
        f"panel projection: {len(shared)}/{len(panel)} genes present "
        f"({coverage:.1%}), {len(missing)} missing"
    )
    if len(missing) > 0:
        print(f"  first 10 missing: {missing.tolist()[:10]}")

    if coverage < 0.95:
        raise ValueError(
            f"Only {len(shared)}/{len(panel)} model genes present in Census. "
            "Too few to proceed; check that the model and Census use the same annotation."
        )

    result = adata[:, shared].copy()
    result.uns["grip2_panel_projection"] = json.dumps(
        {
            "n_model_genes": int(len(panel)),
            "n_shared_genes": int(len(shared)),
            "n_missing_genes": int(len(missing)),
            "missing_genes": missing.tolist(),
        },
        sort_keys=True,
    )
    return result


def save_raw(adata: ad.AnnData, raw_path: Path) -> None:
    """Save immediately after download, before any projection."""
    print(f"saving raw AnnData to {raw_path}", flush=True)
    adata.write_h5ad(raw_path, compression="gzip")
    print(f"wrote {raw_path}", flush=True)


def main() -> None:
    args = parse_args()
    args.out_adata.parent.mkdir(parents=True, exist_ok=True)

    raw_path = args.out_adata.with_name(args.out_adata.stem + "_raw.h5ad")

    # Fast path: if raw exists and we are told to reuse it, skip metadata+download.
    if args.skip_if_raw_exists and raw_path.exists():
        print(f"reusing existing raw file {raw_path}", flush=True)
        adata = ad.read_h5ad(raw_path)
    else:
        print(f"opening census {args.census_version}", flush=True)
        obs = load_obs(args)
        print(f"healthy primary cells in {args.tissues}: {len(obs):,}", flush=True)

        obs, support = plan_selection(obs, args)
        print(
            f"after per-donor-per-type >= {args.min_cells_per_donor_type} and "
            f">= {args.min_donors_per_type} donors/type: {len(obs):,} cells, "
            f"{obs['cell_type'].nunique()} types, "
            f"{obs['_donor_key'].nunique()} donors",
            flush=True,
        )

        time_plan = estimate_time(len(obs), args.throughput_cells_per_minute)
        print(
            f"estimated expression download: "
            f"{time_plan['estimated_minutes']:.1f} min "
            f"({time_plan['estimated_hours']:.2f} h) "
            f"for {time_plan['n_cells']:,} cells",
            flush=True,
        )

        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            support.to_csv(args.report, index=False)
            print(f"wrote support report to {args.report}", flush=True)

        if args.metadata_only:
            print("--metadata-only set: stopping before expression download.", flush=True)
            return

        print("fetching expression", flush=True)
        adata = fetch_expression(obs, args)
        print(f"raw AnnData: {adata.shape}", flush=True)

        # Canonical columns. No library_uuid in Census 2024-07-01: use donor_id as batch.
        adata.obs["donor_id"] = (
            adata.obs["dataset_id"].astype(str) + "|" + adata.obs["donor_id"].astype(str)
        )
        adata.obs["batch"] = adata.obs["donor_id"]
        adata.obs["disease"] = "normal"

        # Save raw BEFORE projection; projection failure cannot lose the data.
        save_raw(adata, raw_path)

    # --- Projection to model panel (optional) ---
    adata = normalise_var_names(adata)

    if args.model_panel is not None:
        adata = project_to_model_panel(adata, args.model_panel)

    adata.uns["grip2_census_download"] = json.dumps(
        {
            "census_version": args.census_version,
            "tissues": args.tissues,
            "min_cells_per_donor_type": args.min_cells_per_donor_type,
            "max_cells_per_donor_type": args.max_cells_per_donor_type,
            "min_donors_per_type": args.min_donors_per_type,
            "max_donors_per_type": args.max_donors_per_type,
            "cell_types_whitelist": args.cell_types,
            "model_panel": str(args.model_panel) if args.model_panel else None,
            "seed": args.seed,
            "n_cells_final": int(adata.n_obs),
            "n_genes_final": int(adata.n_vars),
            "n_types_final": int(adata.obs["cell_type"].nunique()),
            "n_donors_final": int(adata.obs["donor_id"].nunique()),
            "batch_source": "donor_id (dataset_id|donor_id; no library_uuid in Census 2024-07-01)",
        },
        sort_keys=True,
    )

    adata.write_h5ad(args.out_adata, compression="gzip")
    print(f"wrote {args.out_adata}", flush=True)


if __name__ == "__main__":
    main()