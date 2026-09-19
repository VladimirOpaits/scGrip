#!/usr/bin/env python3
"""Join published Seurat labels to a prepared raw-count H5AD by sample/barcode.

Only exact barcode matches are retained.  This is intentionally a subsetter,
not a label propagator: unlabeled raw cells never receive a guessed cell type.
The supplied crosswalk turns a publication label into an exact frozen-reference
stratum and is persisted beside the output for auditability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scgrip.preparation import harmonise_strata
from scgrip.reference import fingerprint_adata, validate_raw_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-adata", required=True)
    parser.add_argument("--labels-tsv", required=True)
    parser.add_argument("--label-map", required=True, help="CSV source,target: publication label → frozen reference stratum.")
    parser.add_argument("--out-adata", required=True)
    parser.add_argument("--query-sample-col", default="sample_id")
    parser.add_argument("--query-barcode-col", default="cell_barcode")
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument(
        "--input-panel-policy",
        choices=("strict", "audited_imputation"),
        default="strict",
        help="Must match the audit status of the prepared raw query.",
    )
    parser.add_argument("--published-label-col", default="published_label")
    parser.add_argument("--out-stratum-col", default="cell_type")
    return parser.parse_args()


def _json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _key(frame: pd.DataFrame, sample_col: str, barcode_col: str) -> pd.Index:
    for column in (sample_col, barcode_col):
        if column not in frame:
            raise ValueError(f"Missing required metadata column {column!r}.")
        values = frame[column].astype(str).str.strip()
        if (values == "").any() or values.str.lower().isin({"nan", "none"}).any():
            raise ValueError(f"Metadata column {column!r} contains blank values.")
    return pd.Index(
        frame[sample_col].astype(str).str.strip() + "\t" + frame[barcode_col].astype(str).str.strip(),
        dtype=str,
    )


def main() -> None:
    args = parse_args()
    output_path = Path(args.out_adata)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    query = ad.read_h5ad(args.query_adata)
    validate_raw_counts(
        query,
        args.counts_layer,
        dataset="prepared raw query",
        input_panel_policy=args.input_panel_policy,
    )
    labels = pd.read_csv(args.labels_tsv, sep="\t", dtype=str, keep_default_na=False)
    required = {"sample_id", "cell_barcode", args.published_label_col}
    if not required.issubset(labels.columns):
        raise ValueError(f"{args.labels_tsv} misses columns {sorted(required - set(labels.columns))}")
    label_keys = _key(labels, "sample_id", "cell_barcode")
    if label_keys.has_duplicates:
        raise ValueError("Published label table contains duplicate (sample_id, cell_barcode) keys.")
    query_keys = _key(query.obs, args.query_sample_col, args.query_barcode_col)
    if query_keys.has_duplicates:
        raise ValueError("Prepared query contains duplicate (sample_id, cell_barcode) keys.")
    positions = label_keys.get_indexer(query_keys)
    keep = positions >= 0
    if not keep.any():
        raise ValueError("No published labels match the prepared query by (sample_id, cell_barcode).")
    matched = query[keep].copy()
    source_labels = labels.iloc[positions[keep]][args.published_label_col].reset_index(drop=True)
    if source_labels.isna().any() or (source_labels.astype(str).str.strip() == "").any():
        raise ValueError("Matched published labels contain missing/blank values.")
    matched.obs["cell_type_published"] = source_labels.astype(str).to_numpy()
    matched.obs[args.out_stratum_col] = harmonise_strata(
        matched.obs["cell_type_published"],
        mapping_path=args.label_map,
        column_name="cell_type_published",
    ).to_numpy()
    for column in ("published_label_score", "published_mapping_score", "published_subcluster", "source_cell_name"):
        if column in labels:
            matched.obs[column] = labels.iloc[positions[keep]][column].to_numpy()
    matched.uns["scgrip_published_label_join"] = json.dumps(
        {
            "format": "scgrip_published_label_join_v1",
            "query_adata": str(Path(args.query_adata).resolve()),
            "labels_tsv": str(Path(args.labels_tsv).resolve()),
            "label_map": str(Path(args.label_map).resolve()),
            "match_key": [args.query_sample_col, args.query_barcode_col],
            "publication_label_column": args.published_label_col,
            "output_stratum_column": args.out_stratum_col,
            "n_query_cells": int(query.n_obs),
            "n_published_labels": int(labels.shape[0]),
            "n_exact_matches": int(matched.n_obs),
            "unmatched_query_cells_are_excluded": True,
        },
        sort_keys=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matched.write_h5ad(output_path)
    manifest = {
        "format": "scgrip_published_label_join_v1",
        "query_adata": str(Path(args.query_adata).resolve()),
        "query_fingerprint": fingerprint_adata(query),
        "output_fingerprint": fingerprint_adata(matched),
        "labels_tsv": str(Path(args.labels_tsv).resolve()),
        "label_map": str(Path(args.label_map).resolve()),
        "n_exact_matches": int(matched.n_obs),
        "published_label_counts": matched.obs["cell_type_published"].value_counts().sort_index().to_dict(),
        "reference_stratum_counts": matched.obs[args.out_stratum_col].value_counts().sort_index().to_dict(),
        "input_panel_policy": args.input_panel_policy,
    }
    with output_path.with_suffix(".manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    print(f"wrote {output_path}: {matched.n_obs}/{query.n_obs} exact published-label matches", flush=True)


if __name__ == "__main__":
    main()
