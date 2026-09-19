#!/usr/bin/env python3
"""Build compact reproducible reference assets from a frozen scVI/source pair.

The source H5AD is read once.  It is then split into three disjoint roles:

* ``metric_latents.npz``: coordinate-only cells for healthy MMD/geometry;
* ``anchor.h5ad`` + ``anchor_base_qz.npy``: raw-count functional replay pack;
* ``anchor_heldout.h5ad`` + ``anchor_heldout_base_qz.npy``: independent
  functional-retention check.

The command never trains a model and never changes the source H5AD.  It is
intentionally useful for a model release: a large Census source can be reduced
to a small anchor pack while the much larger metric bank needs only latents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scvi

from scgrip.reference import (
    align_to_model_genes,
    count_matrix,
    fingerprint_adata,
    model_setup_args,
    model_var_names,
    require_obs_columns,
    sample_reference_roles,
    validate_raw_counts,
)
from scgrip.config import ReferenceSplitConfig
from scgrip.strata import normalise_strata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--source-adata", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--counts-layer", default="counts", help="Raw count layer in source, or literal X.")
    parser.add_argument("--donor-col", default="donor_id")
    parser.add_argument("--stratum-col", default="cell_type")
    parser.add_argument(
        "--metadata-col",
        action="append",
        default=[],
        help="Additional .obs column retained in metric/anchor metadata; repeatable.",
    )
    parser.add_argument("--metric-per-donor", type=int, default=50)
    parser.add_argument("--anchor-per-donor", type=int, default=25)
    parser.add_argument("--heldout-per-donor", type=int, default=25)
    parser.add_argument("--metric-seed", type=int, default=12345)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    def default(value: object):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=default)
        handle.write("\n")


def _encode(model, adata, indices: np.ndarray, batch_size: int) -> np.ndarray:
    return np.asarray(
        model.get_latent_representation(
            adata,
            indices=np.asarray(indices, dtype=int),
            batch_size=batch_size,
        ),
        dtype=np.float32,
    )


def _anchor_adata(source, positions: np.ndarray, *, counts_layer: str, role: str):
    """Make a raw-count-only artifact with stable unique IDs for a role."""

    selected = source[np.asarray(positions, dtype=int)].copy()
    matrix = count_matrix(selected, counts_layer).copy()
    result = ad.AnnData(X=None, obs=selected.obs.copy(), var=selected.var.copy())
    if counts_layer == "X":
        result.X = matrix
    else:
        result.layers[counts_layer] = matrix
    result.obs.insert(0, "source_row", np.asarray(positions, dtype=int))
    result.obs.insert(1, "source_obs_name", selected.obs_names.astype(str).to_numpy())
    # Source public/Census identifiers are sometimes non-unique.  Anchor rows
    # are position-addressed, so make a stable unique artifact ID without
    # rewriting the original identifier stored above.
    result.obs_names = [f"{role}_{int(position):09d}" for position in positions]
    return result


def main() -> None:
    args = parse_args()
    destination = Path(args.out_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty asset directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    setup = model_setup_args(args.base_model)
    expected_layer = setup.get("layer")
    expected_layer = None if expected_layer is None else str(expected_layer)
    requested_layer = None if args.counts_layer == "X" else args.counts_layer
    if expected_layer != requested_layer:
        raise ValueError(
            f"Base model expects layer={expected_layer!r}, but --counts-layer={args.counts_layer!r}."
        )

    source = ad.read_h5ad(args.source_adata)
    required = [args.donor_col, args.stratum_col, *args.metadata_col]
    require_obs_columns(source, required, dataset="source")
    input_audit = validate_raw_counts(source, args.counts_layer, dataset="source", input_panel_policy="strict")
    source = align_to_model_genes(source, model_var_names(args.base_model), dataset="source")
    strata = normalise_strata(source.obs[args.stratum_col], name="reference stratum")
    roles = sample_reference_roles(
        source.obs,
        args.donor_col,
        ReferenceSplitConfig(
            metric_per_donor=args.metric_per_donor,
            replay_per_donor=args.anchor_per_donor,
            heldout_per_donor=args.heldout_per_donor,
            metric_seed=args.metric_seed,
            split_seed=args.split_seed,
        ),
    )

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    model = scvi.model.SCVI.load(str(args.base_model))
    model.module.to(device)
    model.module.eval()
    metric_latents = _encode(model, source, roles.metric, args.batch_size)
    anchor_latents = _encode(model, source, roles.replay, args.batch_size)
    heldout_latents = _encode(model, source, roles.heldout, args.batch_size)

    metric_meta = source.obs.iloc[roles.metric].copy()
    metric_meta.insert(0, "source_row", roles.metric)
    metric_meta.insert(1, "source_obs_name", source.obs_names[roles.metric].astype(str).to_numpy())
    metric_meta.insert(2, "stratum", strata[roles.metric])
    metric_meta.to_csv(destination / "metric_metadata.csv", index=False)
    np.savez_compressed(
        destination / "metric_latents.npz",
        qz_mean=metric_latents,
        source_row=roles.metric,
        stratum=strata[roles.metric],
        donor=source.obs.iloc[roles.metric][args.donor_col].astype(str).to_numpy(),
    )

    anchor = _anchor_adata(source, roles.replay, counts_layer=args.counts_layer, role="anchor")
    heldout = _anchor_adata(source, roles.heldout, counts_layer=args.counts_layer, role="heldout")
    anchor.write_h5ad(destination / "anchor.h5ad")
    heldout.write_h5ad(destination / "anchor_heldout.h5ad")
    np.save(destination / "anchor_base_qz.npy", anchor_latents)
    np.save(destination / "anchor_heldout_base_qz.npy", heldout_latents)
    roles.per_donor.to_csv(destination / "role_support_by_donor.csv")

    _write_json(
        destination / "manifest.json",
        {
            "format": "scgrip_reference_assets_v1",
            "base_model": str(Path(args.base_model).resolve()),
            "source_adata": str(Path(args.source_adata).resolve()),
            "source_fingerprint": fingerprint_adata(source),
            "model_setup": setup,
            "model_panel_sha256": fingerprint_adata(source)["var_names_sha256"],
            "counts_layer": args.counts_layer,
            "donor_col": args.donor_col,
            "stratum_col": args.stratum_col,
            "metadata_columns": args.metadata_col,
            "input_audit": input_audit,
            "sampling": {
                "metric_per_donor": args.metric_per_donor,
                "anchor_per_donor": args.anchor_per_donor,
                "heldout_per_donor": args.heldout_per_donor,
                "metric_seed": args.metric_seed,
                "split_seed": args.split_seed,
            },
            "roles": roles.summary(),
            "files": {
                "metric": "metric_latents.npz + metric_metadata.csv (coordinate-only)",
                "anchor": "anchor.h5ad + anchor_base_qz.npy (raw counts + frozen targets)",
                "heldout": "anchor_heldout.h5ad + anchor_heldout_base_qz.npy (raw counts + frozen targets)",
            },
            "note": "Metric cells are never anchor cells; held-out anchor cells are never replay cells.",
        },
    )
    print(
        f"wrote reference assets to {destination}: metric={roles.metric.size}, "
        f"anchor={roles.replay.size}, heldout={roles.heldout.size}",
        flush=True,
    )


if __name__ == "__main__":
    main()
