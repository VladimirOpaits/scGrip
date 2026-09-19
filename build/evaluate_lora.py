"""Evaluate LoRA or scArches reference-mapping states on one disease cohort.

The script compares the original backbone with named checkpoints, reports scIB
scores, uses one fixed Census geometry for donor-level signature analyses, and
makes one shared UMAP coordinate system for every state. LoRA adapter weights
and the z_encoder portion of scArches checkpoints are reconstructed explicitly.
"""

import argparse
import gc
import h5py
import json
import os
import re
import warnings
from collections import OrderedDict
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import scib
import scvi
import torch
import torch.nn as nn
import umap
from matplotlib import pyplot as plt
from scipy import sparse, stats
from sceris.arches import ARCHES_MODES, arches_options
from sceris.functions import (
    compartment,
    reference_geometry,
    sample_per_donor,
    stratified_dist,
    stratified_geometry,
    summarize_stratified,
)
from sceris.lora import attach
from sceris.signatures import DISEASE_SPECS, get_disease_spec



SIGNATURE_TARGET_SUM = 10_000.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("lora", "scarches"),
        default="lora",
        help="Checkpoint family to evaluate. Default preserves the existing LoRA CLI.",
    )
    parser.add_argument(
        "--arches-mode",
        choices=sorted(ARCHES_MODES),
        default=None,
        help=(
            "scArches update mode for the base state. Trained checkpoint metadata "
            "wins and must agree when this option is supplied."
        ),
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--old-adata",
        required=True,
        help="Healthy Census data used to train the backbone and to define fixed geometry.",
    )
    parser.add_argument("--new-adata", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--disease", choices=sorted(DISEASE_SPECS), required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Repeat for each trained state. PATH is a final run directory containing "
            "model.pt/adapters.pt, or an epoch_XXX.pt evaluator checkpoint. "
            "The base state is always included automatically."
        ),
    )
    parser.add_argument("--donor", default="donor_id")
    parser.add_argument("--disease-col", default="disease")
    parser.add_argument("--disease-label", default=None)
    parser.add_argument("--healthy-label", default="normal")
    parser.add_argument("--cell-type-col", default="cell_type")
    parser.add_argument(
        "--batch-key",
        default=None,
        help="Technical batch for within-cohort scIB. Default depends on --disease.",
    )
    parser.add_argument(
        "--signature-adata",
        default=None,
        help=(
            "Optional full-gene raw-count h5ad used only for the expression signature. "
            "Its cells must match --new-adata by obs_names or --signature-id-col."
        ),
    )
    parser.add_argument(
        "--signature-id-col",
        default=None,
        help="Column shared by --new-adata and --signature-adata for cell alignment.",
    )
    parser.add_argument(
        "--signature-layer",
        default="counts",
        help="Raw-count layer in --signature-adata; falls back to X when unavailable.",
    )
    parser.add_argument("--signature-batch-size", type=int, default=1024)
    parser.add_argument("--signature-min-cells", type=int, default=5)
    parser.add_argument("--min-compartments", type=int, default=2)
    parser.add_argument("--metric-ref-n", type=int, default=50)
    parser.add_argument("--anchor-n", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--scib-max-cells", type=int, default=30_000)
    parser.add_argument(
        "--ab-holdout-frac",
        type=float,
        default=None,
        help=(
            "Optional held-out-B fraction of healthy donors, reconstructed exactly as "
            "build/finetune_lora.py does. Enables separate A (MMD-seen) and B "
            "(MMD-held-out) healthy metrics."
        ),
    )
    parser.add_argument(
        "--ab-split-seed",
        type=int,
        default=0,
        help="Split seed used by build/finetune_lora.py for the healthy donor A/B split.",
    )
    parser.add_argument("--umap-cells", type=int, default=20_000)
    parser.add_argument("--umap-neighbors", type=int, default=20)
    parser.add_argument("--umap-min-dist", type=float, default=0.3)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument(
        "--activation",
        choices=("auto", "identity", "gelu"),
        default="auto",
        help="Adapter activation. auto reads saved metadata, otherwise defaults to identity.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-umap", action="store_true")
    parser.add_argument("--no-save-latents", dest="save_latents", action="store_false")
    parser.set_defaults(save_latents=True)
    return parser.parse_args()


def require_columns(adata, columns, label):
    missing = [column for column in columns if column not in adata.obs]
    if missing:
        raise ValueError(f"{label} lacks required obs columns: {missing}")


def healthy_ab_groups(query_obs, holdout_frac, split_seed):
    """Reproduce finetune_lora.py's donor-level healthy A/B split."""
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError("--ab-holdout-frac must be strictly between 0 and 1.")
    healthy = query_obs["_healthy"].to_numpy()
    healthy_donors = np.unique(query_obs.loc[healthy, "_donor"].astype(str).to_numpy())
    if not len(healthy_donors):
        raise ValueError("Cannot make A/B groups: no healthy query donors.")
    split_rng = np.random.default_rng(split_seed + 1)
    if len(healthy_donors) == 1:
        donors_b = np.array([], dtype=healthy_donors.dtype)
    else:
        n_b = min(
            len(healthy_donors) - 1,
            max(1, round(len(healthy_donors) * holdout_frac)),
        )
        donors_b = split_rng.choice(healthy_donors, size=n_b, replace=False)
    group = np.full(len(query_obs), "disease", dtype=object)
    donor_values = query_obs["_donor"].astype(str).to_numpy()
    group[healthy & np.isin(donor_values, donors_b)] = "B"
    group[healthy & ~np.isin(donor_values, donors_b)] = "A"
    return group, healthy_donors, donors_b

def model_var_names(model_dir):
    payload = torch.load(
        os.path.join(model_dir, "model.pt"),
        map_location="cpu",
        weights_only=False,
    )
    return pd.Index(payload["var_names"].astype(str))


def ensure_model_gene_order(adata, expected, label):
    observed = pd.Index(adata.var_names.astype(str))
    if observed.equals(expected):
        return adata
    missing = expected.difference(observed)
    if len(missing):
        # The trainer uses outer concat with Census replay, which makes absent
        # query genes explicit zero-count columns before restoring model order.
        # Do the same here so evaluation encodes the identical input for a
        # query whose panel is a strict subset of the model panel (e.g. lupus).
        pad = ad.AnnData(
            X=sparse.csr_matrix((adata.n_obs, len(missing)), dtype=adata.X.dtype),
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=missing),
        )
        for layer_name in adata.layers:
            layer = adata.layers[layer_name]
            pad.layers[layer_name] = sparse.csr_matrix(
                (adata.n_obs, len(missing)), dtype=layer.dtype
            )
        adata = ad.concat(
            [adata, pad], axis=1, join="inner", merge="same", uns_merge="first"
        )
        print(
            f"{label}: zero-padded {len(missing)} absent model genes before encoding.",
            flush=True,
        )
    reordered = adata[:, expected].copy()
    if not pd.Index(reordered.var_names.astype(str)).equals(expected):
        raise AssertionError(f"Could not restore model gene order for {label}.")
    return reordered

def parse_states(entries):
    states = OrderedDict([("base", None)])
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"--checkpoint must have LABEL=PATH form, got {entry!r}")
        label, raw_path = entry.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
            raise ValueError(f"Unsafe checkpoint label {label!r}")
        if label == "base" or label in states:
            raise ValueError(f"Duplicate/reserved checkpoint label {label!r}")
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        states[label] = path
    return states


def read_run_config(path):
    for candidate in (path / "run_config.json", path.parent / "run_config.json"):
        if candidate.exists():
            with open(candidate) as handle:
                return json.load(handle)
    return {}


def lora_checkpoint_info(path):
    if path.is_dir():
        model_path = path / "model.pt"
        adapter_path = path / "adapters.pt"
        if not model_path.exists() or not adapter_path.exists():
            raise FileNotFoundError(
                f"Expected model.pt and adapters.pt in checkpoint directory {path}"
            )
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state = payload["model_state_dict"]
        adapter_state = torch.load(adapter_path, map_location="cpu", weights_only=False)
        metadata = read_run_config(path)
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "adapter_state_dict" not in payload:
            raise ValueError(
                f"{path} is not an evaluator epoch checkpoint; pass its run directory instead."
            )
        model_state = payload.get("model_state_dict", {})
        adapter_state = payload["adapter_state_dict"]
        metadata = {**read_run_config(path), **payload}

    state = dict(model_state)
    state.pop("pyro_param_store", None)
    down_weights = [
        value
        for key, value in adapter_state.items()
        if key.endswith(".down.weight")
    ]
    if not down_weights:
        raise ValueError(f"No LoRA down weights found in {path}")
    ranks = {weight.shape[0] for weight in down_weights}
    if len(ranks) != 1:
        raise ValueError(f"Inconsistent adapter rank in {path}")
    return {
        "path": str(path),
        "model_state": state,
        "adapter_state": adapter_state,
        "rank": int(next(iter(ranks))),
        "gelu": metadata.get("gelu"),
        "epoch": metadata.get("epoch"),
    }


def resolve_adapter_config(states, args):
    infos = {
        label: lora_checkpoint_info(path)
        for label, path in states.items()
        if path is not None
    }
    saved_ranks = {info["rank"] for info in infos.values()}
    if len(saved_ranks) > 1:
        raise ValueError("All evaluated adapter checkpoints must have the same rank.")
    saved_rank = next(iter(saved_ranks), None)
    if args.rank is not None and saved_rank is not None and args.rank != saved_rank:
        raise ValueError(f"--rank={args.rank} conflicts with checkpoint rank={saved_rank}")
    rank = args.rank or saved_rank or 8

    saved_activations = {info["gelu"] for info in infos.values() if info["gelu"] is not None}
    if len(saved_activations) > 1:
        raise ValueError("Checkpoints disagree about their adapter activation.")
    if args.activation == "auto":
        gelu = bool(next(iter(saved_activations))) if saved_activations else False
    else:
        gelu = args.activation == "gelu"
        if saved_activations and gelu != bool(next(iter(saved_activations))):
            raise ValueError("--activation conflicts with saved checkpoint metadata.")
    return infos, rank, gelu


def scarches_checkpoint_info(path):
    """Read a final scArches run directory or an epoch evaluator checkpoint."""
    state_metadata = {}
    if path.is_dir():
        model_path = path / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Expected model.pt in scArches directory {path}")
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        metadata = read_run_config(path)
        train_state_path = path / "train_state.pt"
        if train_state_path.exists():
            state_metadata = torch.load(
                train_state_path,
                map_location="cpu",
                weights_only=False,
            )
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in payload:
            raise ValueError(f"{path} has no scArches model_state_dict.")
        metadata = {**read_run_config(path), **payload}

    if "model_state_dict" not in payload:
        raise ValueError(f"{path} has no scArches model_state_dict.")
    mode = (
        payload.get("arches_mode")
        or state_metadata.get("arches_mode")
        or metadata.get("arches_mode")
    )
    if mode not in ARCHES_MODES:
        raise ValueError(
            f"{path} does not declare a recognized scArches mode; got {mode!r}."
        )
    options = (
        payload.get("arches_options")
        or state_metadata.get("arches_options")
        or arches_options(mode)
    )
    expected_options = arches_options(mode)
    if options != expected_options:
        raise ValueError(
            f"{path} scArches options do not match the declared mode {mode!r}."
        )
    state = dict(payload["model_state_dict"])
    state.pop("pyro_param_store", None)
    return {
        "path": str(path),
        "model_state": state,
        "arches_mode": mode,
        "arches_options": options,
        "epoch": payload.get("epoch", state_metadata.get("epoch")),
    }


def resolve_state_config(states, args):
    """Resolve architecture-specific checkpoint metadata once for all states."""
    if args.method == "lora":
        infos, rank, gelu = resolve_adapter_config(states, args)
        return infos, {
            "method": "lora",
            "rank": rank,
            "gelu": gelu,
            "arches_mode": None,
        }

    infos = {
        label: scarches_checkpoint_info(path)
        for label, path in states.items()
        if path is not None
    }
    saved_modes = {info["arches_mode"] for info in infos.values()}
    if len(saved_modes) > 1:
        raise ValueError("All evaluated scArches checkpoints must use one mode.")
    saved_mode = next(iter(saved_modes), None)
    if args.arches_mode is not None and saved_mode is not None and args.arches_mode != saved_mode:
        raise ValueError(
            f"--arches-mode={args.arches_mode} conflicts with checkpoint mode={saved_mode}."
        )
    mode = args.arches_mode or saved_mode or "standard"
    return infos, {
        "method": "scarches",
        "rank": None,
        "gelu": None,
        "arches_mode": mode,
    }


def decode_h5_strings(values):
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values],
        dtype=str,
    )


def h5ad_obs_values(handle, column=None):
    obs = handle["obs"]
    if column is None:
        column = obs.attrs["_index"]
        if isinstance(column, bytes):
            column = column.decode()
    if column not in obs:
        raise ValueError(f"Signature h5ad lacks obs column {column!r}")
    node = obs[column]
    if isinstance(node, h5py.Dataset):
        return decode_h5_strings(node[:])
    if "codes" in node and "categories" in node:
        categories = decode_h5_strings(node["categories"][:])
        codes = np.asarray(node["codes"][:], dtype=int)
        values = np.full(len(codes), "", dtype=object)
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        return values.astype(str)
    raise ValueError(f"Unsupported h5ad encoding for obs column {column!r}")


def h5ad_var_names(handle):
    var = handle["var"]
    index_name = var.attrs["_index"]
    if isinstance(index_name, bytes):
        index_name = index_name.decode()
    return decode_h5_strings(var[index_name][:])


def signature_source_positions(source_ids, target_ids, same_order):
    if same_order:
        if len(source_ids) != len(target_ids):
            raise ValueError("The signature source and query have different cell counts.")
        return np.arange(len(target_ids), dtype=int)
    source_index = pd.Index(source_ids)
    target_index = pd.Index(target_ids)
    if not source_index.is_unique or not target_index.is_unique:
        raise ValueError(
            "Cannot align a separate --signature-adata with duplicate IDs; pass a unique "
            "--signature-id-col or use the same h5ad for both inputs."
        )
    positions = source_index.get_indexer(target_index)
    if (positions < 0).any():
        raise ValueError(
            f"{int((positions < 0).sum())} query cells are absent from --signature-adata."
        )
    return positions


def h5ad_matrix(handle, layer):
    layer_path = f"layers/{layer}" if layer else None
    if layer_path and layer_path in handle:
        return handle[layer_path], layer_path
    return handle["X"], "X"


def h5ad_library_sizes(handle, source_positions):
    for column in ("total_counts", "n_counts", "raw_sum"):
        path = f"obs/{column}"
        if path not in handle or not isinstance(handle[path], h5py.Dataset):
            continue
        all_values = np.asarray(handle[path][:], dtype=float)
        values = all_values[source_positions]
        if np.isfinite(values).all() and (values > 0).all():
            return values, column
    return None, None


def csr_signature_log_expression(matrix, source_rows, lookup, n_genes, library):
    indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
    first, last = int(source_rows[0]), int(source_rows[-1])
    offset = indptr[first]
    data = np.asarray(matrix["data"][offset:indptr[last + 1]], dtype=np.float32)
    columns = np.asarray(matrix["indices"][offset:indptr[last + 1]], dtype=np.int64)
    values = np.zeros((len(source_rows), n_genes), dtype=np.float32)
    computed_library = np.empty(len(source_rows), dtype=np.float64) if library is None else library
    for local_row, source_row in enumerate(source_rows):
        start = indptr[source_row] - offset
        end = indptr[source_row + 1] - offset
        output_columns = lookup[columns[start:end]]
        hits = output_columns >= 0
        if hits.any():
            values[local_row, output_columns[hits]] = data[start:end][hits]
        if library is None:
            computed_library[local_row] = data[start:end].sum(dtype=np.float64)
    return np.log1p(
        values * SIGNATURE_TARGET_SUM / np.maximum(computed_library[:, None], 1.0)
    )


def csc_signature_log_expression(matrix, source_rows, gene_positions, library):
    if library is None:
        raise ValueError("A total-count obs column is required for a CSC signature source.")
    indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
    values = np.zeros((len(source_rows), len(gene_positions)), dtype=np.float32)
    for output_column, gene_position in enumerate(gene_positions):
        rows = np.asarray(matrix["indices"][indptr[gene_position]:indptr[gene_position + 1]], dtype=np.int64)
        data = np.asarray(matrix["data"][indptr[gene_position]:indptr[gene_position + 1]], dtype=np.float32)
        local_rows = np.searchsorted(source_rows, rows)
        valid = local_rows < len(source_rows)
        matched = np.zeros(len(rows), dtype=bool)
        matched[valid] = source_rows[local_rows[valid]] == rows[valid]
        valid &= matched
        values[local_rows[valid], output_column] = data[valid]
    return np.log1p(values * SIGNATURE_TARGET_SUM / np.maximum(library[:, None], 1.0))


def dense_signature_log_expression(matrix, source_rows, gene_positions, library):
    values = np.empty((len(source_rows), len(gene_positions)), dtype=np.float32)
    for output_column, gene_position in enumerate(gene_positions):
        values[:, output_column] = matrix[source_rows, gene_position]
    if library is None:
        library = np.asarray(matrix[source_rows, :].sum(axis=1)).ravel()
    return np.log1p(values * SIGNATURE_TARGET_SUM / np.maximum(library[:, None], 1.0))


def source_row_chunks(source_positions, batch_size):
    order = np.argsort(source_positions)
    start = 0
    max_span = max(batch_size * 8, batch_size)
    while start < len(order):
        stop = min(start + batch_size, len(order))
        while (
            stop > start + 1
            and source_positions[order[stop - 1]] - source_positions[order[start]] > max_span
        ):
            stop -= 1
        yield order[start:stop]
        start = stop


def score_signature_from_file(
    source_path,
    target_ids,
    genes,
    layer,
    batch_size,
    id_column=None,
    same_order=False,
):
    """Stream a per-cell log-CP10K, gene-standardized module score from raw h5ad."""
    with h5py.File(source_path, "r") as source:
        source_ids = h5ad_obs_values(source, id_column)
        positions = signature_source_positions(source_ids, target_ids, same_order)
        var_index = pd.Index(h5ad_var_names(source))
        found = [(gene_id, symbol) for gene_id, symbol in genes if gene_id in var_index]
        missing = [(gene_id, symbol) for gene_id, symbol in genes if gene_id not in var_index]
        if len(found) < 2:
            raise ValueError(
                f"Only {len(found)} signature genes are available in {source_path}; need at least two."
            )
        gene_positions = var_index.get_indexer([gene_id for gene_id, _ in found])
        lookup = np.full(len(var_index), -1, dtype=np.int32)
        lookup[gene_positions] = np.arange(len(gene_positions), dtype=np.int32)
        matrix, matrix_path = h5ad_matrix(source, layer)
        encoding = matrix.attrs.get("encoding-type", "array")
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        library, library_column = h5ad_library_sizes(source, positions)
        expression = np.empty((len(positions), len(found)), dtype=np.float32)

        for target_rows in source_row_chunks(positions, batch_size):
            source_rows = positions[target_rows]
            library_rows = library[target_rows] if library is not None else None
            if encoding == "csr_matrix":
                values = csr_signature_log_expression(
                    matrix,
                    source_rows,
                    lookup,
                    len(found),
                    library_rows,
                )
            elif encoding == "csc_matrix":
                values = csc_signature_log_expression(
                    matrix,
                    source_rows,
                    gene_positions,
                    library_rows,
                )
            else:
                values = dense_signature_log_expression(
                    matrix,
                    source_rows,
                    gene_positions,
                    library_rows,
                )
            expression[target_rows] = values

    mean = expression.mean(axis=0, dtype=np.float64)
    std = expression.std(axis=0, dtype=np.float64)
    usable = std > 1e-8
    if usable.sum() < 2:
        raise ValueError("Fewer than two signature genes have nonzero expression variance.")
    scores = ((expression[:, usable] - mean[usable]) / std[usable]).mean(axis=1).astype(np.float32)
    report = {
        "source_path": str(source_path),
        "source_layer": matrix_path,
        "library_size_column": library_column or "computed_from_matrix",
        "n_requested_genes": len(genes),
        "n_present_genes": len(found),
        "n_used_nonconstant_genes": int(usable.sum()),
        "present_genes": [
            {"ensembl": gene_id, "symbol": symbol} for gene_id, symbol in found
        ],
        "missing_genes": [
            {"ensembl": gene_id, "symbol": symbol} for gene_id, symbol in missing
        ],
        "score_definition": (
            "mean across present signature genes of per-gene z(log1p(CP10K)); "
            "z-scoring is over the evaluated query cells"
        ),
    }
    return scores, report

def balanced_sample(frame, columns, max_cells, rng):
    if len(frame) <= max_cells:
        return np.arange(len(frame), dtype=int)
    groups = frame.groupby(columns, observed=True, sort=False).indices
    per_group = max(1, max_cells // len(groups))
    picked = []
    for positions in groups.values():
        n_pick = min(per_group, len(positions))
        picked.append(rng.choice(positions, size=n_pick, replace=False))
    picked = np.concatenate(picked)
    if len(picked) < max_cells:
        keep = np.zeros(len(frame), dtype=bool)
        keep[picked] = True
        remaining = np.flatnonzero(~keep)
        extra = rng.choice(remaining, size=min(max_cells - len(picked), len(remaining)), replace=False)
        picked = np.concatenate([picked, extra])
    return rng.permutation(picked)


def metric_adata(embedding, obs):
    result = ad.AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs.copy())
    result.obsm["X_latent"] = np.asarray(embedding, dtype=np.float32)
    return result


def labels_with_support(obs, label_key, batch_key=None, min_cells=10):
    counts = obs[label_key].value_counts()
    labels = counts.index[counts >= min_cells]
    if batch_key is not None:
        batch_support = obs.groupby(label_key, observed=True)[batch_key].nunique()
        labels = labels.intersection(batch_support.index[batch_support >= 2])
    return obs[label_key].isin(labels).to_numpy()


def safe_scib(name, callback):
    try:
        value = callback()
        return float(value)
    except Exception as error:
        warnings.warn(f"scIB {name} unavailable: {error}", stacklevel=2)
        return float("nan")


def scib_metrics(
    query_latent,
    query_obs,
    query_sample,
    healthy_sample,
    reference_latent,
    reference_compartment,
    ref_compartments,
    batch_key,
    rng,
    max_cells,
    block_query_positions=None,
    include_biology=True,
):
    """scIB biological and healthy-alignment metrics on fixed cell selections."""
    out = {}
    if include_biology:
        bio_obs = query_obs.iloc[query_sample].copy().reset_index(drop=True)
        bio_obs["_cell_type"] = bio_obs["_cell_type"].astype(str)
        bio_obs["_compartment"] = bio_obs["_compartment"].astype(str)
        bio_latent = query_latent[query_sample]

        fine_keep = labels_with_support(bio_obs, "_cell_type")
        coarse_keep = labels_with_support(bio_obs, "_compartment")
        out["scib_bio_cell_type_asw"] = safe_scib(
            "cell_type silhouette",
            lambda: scib.metrics.silhouette(
                metric_adata(bio_latent[fine_keep], bio_obs.iloc[np.flatnonzero(fine_keep)]),
                "_cell_type",
                "X_latent",
            ),
        )
        out["scib_bio_compartment_asw"] = safe_scib(
            "compartment silhouette",
            lambda: scib.metrics.silhouette(
                metric_adata(bio_latent[coarse_keep], bio_obs.iloc[np.flatnonzero(coarse_keep)]),
                "_compartment",
                "X_latent",
            ),
        )
        out["scib_n_query_cells"] = int(len(bio_obs))

    healthy_obs = query_obs.iloc[healthy_sample].copy().reset_index(drop=True)
    healthy_latent = query_latent[healthy_sample]
    healthy_keep = labels_with_support(healthy_obs, "_cell_type", batch_key)
    out["scib_batch_healthy_asw"] = safe_scib(
        "healthy within-cohort batch silhouette",
        lambda: scib.metrics.silhouette_batch(
            metric_adata(
                healthy_latent[healthy_keep],
                healthy_obs.iloc[np.flatnonzero(healthy_keep)],
            ),
            batch_key=batch_key,
            label_key="_cell_type",
            embed="X_latent",
            verbose=False,
        ),
    )
    out["scib_n_healthy_cells"] = int(len(healthy_obs))

    reference_keep = np.isin(reference_compartment, ref_compartments)
    if block_query_positions is None:
        query_block_positions = np.flatnonzero(query_obs["_healthy"].to_numpy())
    else:
        query_block_positions = np.asarray(block_query_positions, dtype=int)
    query_block_positions = query_block_positions[
        np.isin(query_obs["_compartment"].to_numpy()[query_block_positions], ref_compartments)
    ]
    block_obs = pd.concat(
        [
            pd.DataFrame(
                {
                    "_compartment": reference_compartment[reference_keep],
                    "_block": "census",
                }
            ),
            pd.DataFrame(
                {
                    "_compartment": query_obs["_compartment"].to_numpy()[query_block_positions],
                    "_block": "healthy_query",
                }
            ),
        ],
        ignore_index=True,
    )
    block_latent = np.vstack(
        [reference_latent[reference_keep], query_latent[query_block_positions]]
    )
    if len(block_obs):
        block_sample = balanced_sample(
            block_obs,
            ["_block", "_compartment"],
            max_cells,
            rng,
        )
        block_obs = block_obs.iloc[block_sample].reset_index(drop=True)
        block_latent = block_latent[block_sample]
    block_keep = labels_with_support(block_obs, "_compartment", "_block")
    out["scib_block_healthy_asw"] = safe_scib(
        "fixed-Census versus healthy-query block silhouette",
        lambda: scib.metrics.silhouette_batch(
            metric_adata(block_latent[block_keep], block_obs.iloc[np.flatnonzero(block_keep)]),
            batch_key="_block",
            label_key="_compartment",
            embed="X_latent",
            verbose=False,
        ),
    )
    out["scib_n_block_cells"] = int(len(block_obs))
    return out


def ab_stratified_metrics(
    query_latent,
    query_obs,
    group_positions,
    ref_centroid,
    metric_sigma,
    reference_donor_spread,
):
    """A/B donor-macro distance, offset, and spread against fixed Census geometry."""
    summaries = {}
    for label, positions in group_positions.items():
        if len(positions):
            ct = query_obs["_compartment"].to_numpy()[positions]
            donors = query_obs["_donor"].to_numpy()[positions]
            distances = stratified_dist(
                query_latent[positions], ct, donors, ref_centroid, metric_sigma
            )
            geometry = stratified_geometry(
                query_latent[positions], ct, donors, ref_centroid, metric_sigma
            )
        else:
            distances = pd.DataFrame(columns=ref_centroid.index, dtype=float)
            geometry = pd.DataFrame(index=ref_centroid.index)
        means, support = summarize_stratified(distances)
        summaries[label] = (distances, means, support, geometry)

    _, means_a, support_a, geometry_a = summaries["A"]
    _, means_b, support_b, geometry_b = summaries["B"]
    common = means_a.index[(support_a > 0) & (support_b > 0)]
    out = {"strat_healthy_ab_n_common_compartments": int(len(common))}
    for label, (distances, means, support, geometry) in summaries.items():
        out[f"strat_healthy_{label}_to_census"] = (
            float(means.loc[common].mean()) if len(common) else float("nan")
        )
        out[f"strat_healthy_{label}_n_donors"] = int(len(distances.index))
        out[f"strat_healthy_{label}_n_supported_compartments"] = int((support > 0).sum())
        out[f"strat_healthy_{label}_block_offset_to_census"] = (
            float(geometry.loc[common, "block_offset"].mean())
            if len(common)
            else float("nan")
        )
        spread_common = common[
            geometry.loc[common, "donor_spread"].notna()
            & reference_donor_spread.loc[common].notna()
            & (reference_donor_spread.loc[common] > 1e-8)
        ]
        out[f"strat_healthy_{label}_donor_spread_ref_ratio"] = (
            float(
                (
                    geometry.loc[spread_common, "donor_spread"]
                    / reference_donor_spread.loc[spread_common]
                ).mean()
            )
            if len(spread_common)
            else float("nan")
        )
    out["strat_healthy_B_minus_A_to_census"] = (
        out["strat_healthy_B_to_census"] - out["strat_healthy_A_to_census"]
    )
    out["strat_healthy_B_minus_A_block_offset_to_census"] = (
        out["strat_healthy_B_block_offset_to_census"]
        - out["strat_healthy_A_block_offset_to_census"]
    )
    out["strat_healthy_B_minus_A_donor_spread_ref_ratio"] = (
        out["strat_healthy_B_donor_spread_ref_ratio"]
        - out["strat_healthy_A_donor_spread_ref_ratio"]
    )
    return out

def donor_compartment_scores(scores, compartments, donors, ref_compartments, min_cells):
    frame = pd.DataFrame(
        {
            "donor": np.asarray(donors).astype(str),
            "compartment": np.asarray(compartments),
            "signature_score": np.asarray(scores),
        }
    )
    frame = frame[frame["compartment"].isin(ref_compartments)]
    grouped = frame.groupby(["donor", "compartment"], observed=True)["signature_score"]
    means = grouped.mean().unstack("compartment").reindex(columns=ref_compartments)
    counts = grouped.size().unstack("compartment").reindex(columns=ref_compartments, fill_value=0)
    return means.mask(counts < min_cells), counts


def donor_distance_and_signature(
    latent,
    signature_scores,
    query_obs,
    ref_centroid,
    metric_sigma,
    disease_col,
    min_cells,
    min_compartments,
):
    ref_compartments = ref_centroid.index.to_numpy()
    distances = stratified_dist(
        latent,
        query_obs["_compartment"].to_numpy(),
        query_obs["_donor"].to_numpy(),
        ref_centroid,
        metric_sigma,
    )
    signature, signature_counts = donor_compartment_scores(
        signature_scores,
        query_obs["_compartment"].to_numpy(),
        query_obs["_donor"].to_numpy(),
        ref_compartments,
        min_cells,
    )
    donors = distances.index.union(signature.index)
    disease_by_donor = query_obs.groupby("_donor", observed=True)[disease_col].agg(
        lambda values: values.astype(str).iloc[0]
    )

    donor_rows = []
    component_rows = []
    for donor in donors:
        distance_row = distances.loc[donor] if donor in distances.index else pd.Series(index=ref_compartments, dtype=float)
        score_row = signature.loc[donor] if donor in signature.index else pd.Series(index=ref_compartments, dtype=float)
        common = distance_row.index[distance_row.notna() & score_row.notna()]
        for comp in common:
            component_rows.append(
                {
                    "donor": donor,
                    "disease": disease_by_donor.get(donor, np.nan),
                    "compartment": comp,
                    "latent_distance": float(distance_row[comp]),
                    "signature_score": float(score_row[comp]),
                    "signature_n_cells": int(signature_counts.loc[donor, comp]),
                }
            )
        if len(common) >= min_compartments:
            donor_rows.append(
                {
                    "donor": donor,
                    "disease": disease_by_donor.get(donor, np.nan),
                    "latent_distance": float(distance_row[common].mean()),
                    "signature_score": float(score_row[common].mean()),
                    "n_compartments": int(len(common)),
                    "compartments": ";".join(common),
                }
            )
    return (
        pd.DataFrame(
            donor_rows,
            columns=[
                "donor", "disease", "latent_distance", "signature_score",
                "n_compartments", "compartments",
            ],
        ),
        pd.DataFrame(
            component_rows,
            columns=[
                "donor", "disease", "compartment", "latent_distance",
                "signature_score", "signature_n_cells",
            ],
        ),
    )


def correlation_metrics(donor_table, disease_label):
    out = {}
    for name, mask in {
        "all": np.ones(len(donor_table), dtype=bool),
        "disease": donor_table["disease"].astype(str).to_numpy() == disease_label,
    }.items():
        values = donor_table.loc[mask, ["latent_distance", "signature_score"]].dropna()
        out[f"ifn_n_{name}_donors"] = int(len(values))
        if len(values) < 3 or values.nunique().min() < 2:
            out[f"ifn_spearman_{name}"] = float("nan")
            out[f"ifn_spearman_{name}_p"] = float("nan")
            out[f"ifn_pearson_{name}"] = float("nan")
            out[f"ifn_pearson_{name}_p"] = float("nan")
            continue
        spearman = stats.spearmanr(values["latent_distance"], values["signature_score"])
        pearson = stats.pearsonr(values["latent_distance"], values["signature_score"])
        out[f"ifn_spearman_{name}"] = float(spearman.statistic)
        out[f"ifn_spearman_{name}_p"] = float(spearman.pvalue)
        out[f"ifn_pearson_{name}"] = float(pearson.statistic)
        out[f"ifn_pearson_{name}_p"] = float(pearson.pvalue)
    return out


def plot_ifn_scatter(donor_tables, disease_label, output_path):
    labels = list(donor_tables)
    figure, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4), squeeze=False)
    for axis, label in zip(axes.flat, labels):
        table = donor_tables[label]
        values = table.loc[
            table["disease"].astype(str) == disease_label,
            ["latent_distance", "signature_score"],
        ].dropna()
        axis.set_title(label)
        axis.set_xlabel("donor macro latent distance to fixed Census")
        axis.set_ylabel("IFN signature score")
        if len(values):
            axis.scatter(values["latent_distance"], values["signature_score"], s=35)
        if len(values) >= 3 and values.nunique().min() >= 2:
            slope, intercept = np.polyfit(values["latent_distance"], values["signature_score"], 1)
            x_values = np.linspace(values["latent_distance"].min(), values["latent_distance"].max(), 100)
            axis.plot(x_values, slope * x_values + intercept, color="black", linewidth=1)
            result = stats.spearmanr(values["latent_distance"], values["signature_score"])
            axis.text(
                0.03,
                0.97,
                f"Spearman r={result.statistic:.2f}\nn={len(values)}, p={result.pvalue:.3g}",
                transform=axis.transAxes,
                va="top",
            )
        elif not len(values):
            axis.text(0.5, 0.5, "no eligible disease donors", transform=axis.transAxes, ha="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_umap_categorical(points, category, state_labels, output_path):
    labels = sorted(points[category].astype(str).unique())
    colors = plt.get_cmap("tab20")(np.arange(len(labels)) % 20)
    color_map = dict(zip(labels, colors))
    xlim = (points["umap1"].min(), points["umap1"].max())
    ylim = (points["umap2"].min(), points["umap2"].max())
    figure, axes = plt.subplots(1, len(state_labels), figsize=(5 * len(state_labels), 4), squeeze=False)
    handles = []
    for axis, state in zip(axes.flat, state_labels):
        subset = points[points["state"] == state]
        for value in labels:
            data = subset[subset[category].astype(str) == value]
            handle = axis.scatter(
                data["umap1"],
                data["umap2"],
                s=2,
                alpha=0.55,
                color=color_map[value],
                rasterized=True,
                label=value,
            )
            if state == state_labels[0]:
                handles.append(handle)
        axis.set_title(state)
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), markerscale=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_umap_continuous(points, state_labels, output_path):
    lower, upper = points["signature_score"].quantile([0.01, 0.99])
    figure, axes = plt.subplots(1, len(state_labels), figsize=(5 * len(state_labels), 4), squeeze=False)
    scatter = None
    for axis, state in zip(axes.flat, state_labels):
        subset = points[points["state"] == state]
        scatter = axis.scatter(
            subset["umap1"],
            subset["umap2"],
            c=subset["signature_score"],
            cmap="viridis",
            vmin=lower,
            vmax=upper,
            s=2,
            alpha=0.6,
            rasterized=True,
        )
        axis.set_title(state)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(scatter, ax=axes.ravel().tolist(), label="IFN signature score")
    figure.subplots_adjust(wspace=0.08)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_shared_umap(latents, query_obs, sample, args, out_dir):
    state_labels = list(latents)
    stacked = np.vstack([latents[label][sample] for label in state_labels])
    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.seed,
        transform_seed=args.seed,
    )
    embedding = reducer.fit_transform(stacked)
    frames = []
    n_cells = len(sample)
    metadata = query_obs.iloc[sample].reset_index(drop=True)
    for position, state in enumerate(state_labels):
        frame = metadata.copy()
        frame["state"] = state
        frame["umap1"] = embedding[position * n_cells:(position + 1) * n_cells, 0]
        frame["umap2"] = embedding[position * n_cells:(position + 1) * n_cells, 1]
        frames.append(frame)
    points = pd.concat(frames, ignore_index=True)
    points.to_parquet(os.path.join(out_dir, "umap_points.parquet"), index=False)
    plot_umap_categorical(
        points,
        "_cell_type",
        state_labels,
        os.path.join(out_dir, "umap_cell_type.png"),
    )
    plot_umap_categorical(
        points,
        "_disease",
        state_labels,
        os.path.join(out_dir, "umap_disease.png"),
    )
    plot_umap_continuous(
        points,
        state_labels,
        os.path.join(out_dir, "umap_ifn_score.png"),
    )


def build_query_model(base_model, adata, info, state_config, device):
    """Recreate the model surgery used by the state, then load its weights."""
    if state_config["method"] == "lora":
        q_model = scvi.model.SCVI.load_query_data(adata, base_model)
        q_model.is_trained_ = True
        q_module = q_model.module
        q_module.to(device)
        q_module.eval()
        for parameter in q_module.parameters():
            parameter.requires_grad_(False)
        adapters = attach(
            q_module,
            rank=state_config["rank"],
            device=device,
            act=nn.GELU if state_config["gelu"] else nn.Identity,
        )
        adapters.eval()
        if info is not None:
            # The frozen LoRA backbone state is intentionally ignored: only
            # external adapters can affect qz for this method.
            adapters.load_state_dict(info["adapter_state"], strict=True)
        return q_model, adapters

    options = (
        info["arches_options"]
        if info is not None
        else arches_options(state_config["arches_mode"])
    )
    q_model = scvi.model.SCVI.load_query_data(adata, base_model, **options)
    q_model.is_trained_ = True
    q_module = q_model.module
    q_module.to(device)
    if info is not None:
        # A saved scArches decoder is tied to the training combined registry,
        # whose donor-category width differs for separate query/anchor evaluators.
        # Latent qz uses z_encoder only, so load every compatible z_encoder tensor
        # exactly and intentionally leave decoder-only state out of this evaluator.
        current_state = q_module.state_dict()
        expected_encoder_keys = {
            key for key in current_state if key.startswith("z_encoder.")
        }
        encoder_state = {
            key: value
            for key, value in info["model_state"].items()
            if key in expected_encoder_keys and current_state[key].shape == value.shape
        }
        missing_encoder = expected_encoder_keys.difference(encoder_state)
        if missing_encoder:
            raise RuntimeError(
                "scArches checkpoint lacks compatible z_encoder tensors: "
                + ", ".join(sorted(missing_encoder))
            )
        q_module.load_state_dict(encoder_state, strict=False)
    q_module.eval()
    return q_model, None


def main():
    args = parse_args()
    spec = get_disease_spec(args.disease)
    args.disease_label = args.disease_label or spec.disease_label
    args.batch_key = args.batch_key or spec.batch_key
    if args.metric_ref_n < 5 or args.anchor_n < 1 or args.signature_batch_size < 1:
        raise ValueError(
            "--metric-ref-n must be >=5; --anchor-n and --signature-batch-size must be positive."
        )

    states = parse_states(args.checkpoint)
    state_infos, state_config = resolve_state_config(states, args)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"using device: {device}", flush=True)
    if args.method == "lora":
        print(
            f"states: {list(states)} | rank={state_config['rank']} | "
            f"activation={'gelu' if state_config['gelu'] else 'identity'}",
            flush=True,
        )
    else:
        print(
            f"states: {list(states)} | scArches mode={state_config['arches_mode']}",
            flush=True,
        )

    base_model = scvi.model.SCVI.load(args.base_model)
    base_model.module.to(device)
    base_model.module.eval()
    expected_vars = model_var_names(args.base_model)

    census = ad.read_h5ad(args.old_adata)
    census = ensure_model_gene_order(census, expected_vars, "old adata")
    require_columns(census, [args.donor, args.cell_type_col], "old adata")
    census.X = None

    metric_rng = np.random.default_rng(12345)
    metric_idx = sample_per_donor(census.obs, args.donor, args.metric_ref_n, metric_rng)
    metric_obs = census.obs.iloc[metric_idx]
    metric_latent = base_model.get_latent_representation(
        census,
        indices=metric_idx,
        batch_size=args.batch_size,
    )
    metric_compartment = compartment(metric_obs[args.cell_type_col].astype(str)).to_numpy()
    metric_donor = metric_obs[args.donor].astype(str).to_numpy()
    ref_centroid, metric_sigma, ref_support = reference_geometry(
        metric_latent,
        metric_compartment,
        metric_donor,
    )
    if ref_centroid.empty:
        raise ValueError(
            "The fixed Census reference has no supported compartments; increase --metric-ref-n."
        )
    metric_sigma = np.maximum(metric_sigma, 1e-6)
    reference_donor_spread = stratified_geometry(
        metric_latent, metric_compartment, metric_donor, ref_centroid, metric_sigma
    )["donor_spread"]
    print(
        "fixed Census reference support: "
        + ", ".join(f"{key}={value}" for key, value in ref_support.items()),
        flush=True,
    )

    anchor_idx = sample_per_donor(
        census.obs,
        args.donor,
        args.anchor_n,
        np.random.default_rng(args.seed + 991),
    )
    anchor = census[anchor_idx].copy()
    del metric_obs, metric_idx, census
    gc.collect()

    query = ad.read_h5ad(args.new_adata)
    query = ensure_model_gene_order(query, expected_vars, "new adata")
    require_columns(
        query,
        [args.donor, args.disease_col, args.cell_type_col, args.batch_key],
        "new adata",
    )
    if args.signature_id_col is not None:
        require_columns(query, [args.signature_id_col], "new adata")
        target_ids = query.obs[args.signature_id_col].astype(str).to_numpy()
    else:
        target_ids = query.obs_names.astype(str).to_numpy()

    signature_path = args.signature_adata or args.new_adata
    same_order = Path(signature_path).resolve() == Path(args.new_adata).resolve()
    signature_scores, signature_report = score_signature_from_file(
        signature_path,
        target_ids,
        spec.genes,
        args.signature_layer,
        args.signature_batch_size,
        id_column=args.signature_id_col,
        same_order=same_order,
    )
    print(
        f"{signature_report['n_present_genes']} / {signature_report['n_requested_genes']} "
        f"{spec.signature_name} genes present; source={signature_path}",
        flush=True,
    )

    query_obs = query.obs[
        [args.donor, args.disease_col, args.cell_type_col, args.batch_key]
    ].copy()
    query_obs["_donor"] = query_obs[args.donor].astype(str)
    query_obs["_disease"] = query_obs[args.disease_col].astype(str)
    query_obs["_cell_type"] = query_obs[args.cell_type_col].astype(str)
    query_obs["_batch"] = query_obs[args.batch_key].astype(str)
    query_obs["_compartment"] = compartment(query_obs["_cell_type"]).to_numpy()
    query_obs["_healthy"] = query_obs["_disease"].to_numpy() == args.healthy_label
    query_obs["signature_score"] = signature_scores
    query_obs["_cell_id"] = query.obs_names.astype(str).to_numpy()
    if not query_obs["_healthy"].any():
        raise ValueError(f"No cells have healthy label {args.healthy_label!r}.")
    if not (query_obs["_disease"] == args.disease_label).any():
        raise ValueError(
            f"No cells have disease label {args.disease_label!r}; pass --disease-label to override."
        )

    ab_group_positions = None
    ab_healthy_samples = {}
    ab_donors_a = ab_donors_b = None
    ab_manifest = None
    if args.ab_holdout_frac is not None:
        ab_group, healthy_donors, ab_donors_b = healthy_ab_groups(
            query_obs,
            args.ab_holdout_frac,
            args.ab_split_seed,
        )
        query_obs["_ab_group"] = ab_group
        ab_donors_a = healthy_donors[~np.isin(healthy_donors, ab_donors_b)]
        ab_group_positions = {
            label: np.flatnonzero(query_obs["_ab_group"].to_numpy() == label)
            for label in ("A", "B")
        }
        ab_manifest = {
            "holdout_frac": args.ab_holdout_frac,
            "split_seed": args.ab_split_seed,
            "donors_A": [str(value) for value in ab_donors_a],
            "donors_B": [str(value) for value in ab_donors_b],
        }
        print(
            f"evaluation healthy donors: A={len(ab_donors_a)} B={len(ab_donors_b)} "
            f"(B fraction={args.ab_holdout_frac:.3f})",
            flush=True,
        )

    # Separate registries avoid making a second full query counts matrix during
    # AnnData concatenation. The adapter itself is external to the registry.
    anchor.X = None
    query.X = None
    scvi.model.SCVI.setup_anndata(anchor, batch_key=args.donor, layer="counts")
    scvi.model.SCVI.setup_anndata(query, batch_key=args.donor, layer="counts")
    gc.collect()

    query_sample = balanced_sample(
        query_obs,
        ["_disease", "_cell_type"],
        args.scib_max_cells,
        rng,
    )
    healthy_positions = np.flatnonzero(query_obs["_healthy"].to_numpy())
    healthy_sample_local = balanced_sample(
        query_obs.iloc[healthy_positions],
        ["_cell_type", "_batch"],
        args.scib_max_cells,
        rng,
    )
    healthy_sample = healthy_positions[healthy_sample_local]
    if ab_group_positions is not None:
        for offset, label in enumerate(("A", "B")):
            positions = ab_group_positions[label]
            sample_local = balanced_sample(
                query_obs.iloc[positions],
                ["_cell_type", "_batch"],
                args.scib_max_cells,
                np.random.default_rng(args.seed + 710 + offset),
            )
            ab_healthy_samples[label] = positions[sample_local]

    umap_sample = balanced_sample(
        query_obs,
        ["_disease", "_cell_type"],
        args.umap_cells,
        rng,
    )

    latents = OrderedDict()
    summary_rows = []
    donor_tables = OrderedDict()
    base_anchor = None
    anchor_sigma = None

    for state, path in states.items():
        print(f"encoding state: {state}", flush=True)
        q_model, adapters = build_query_model(
            base_model,
            query,
            state_infos.get(state),
            state_config,
            device,
        )
        latent_query = q_model.get_latent_representation(
            query,
            batch_size=args.batch_size,
        )
        if adapters is not None:
            adapters.clear()
            adapters.remove()
        del q_model, adapters
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        anchor_model, anchor_adapters = build_query_model(
            base_model,
            anchor,
            state_infos.get(state),
            state_config,
            device,
        )
        latent_anchor = anchor_model.get_latent_representation(
            anchor,
            batch_size=args.batch_size,
        )
        if anchor_adapters is not None:
            anchor_adapters.clear()
            anchor_adapters.remove()
        del anchor_model, anchor_adapters
        if base_anchor is None:
            base_anchor = latent_anchor.copy()
            anchor_sigma = np.maximum(base_anchor.std(axis=0), 1e-6)
        drift = latent_anchor - base_anchor
        row = {
            "state": state,
            "method": args.method,
            "arches_mode": state_config["arches_mode"],
            "checkpoint": str(path) if path is not None else "base",
            "anchor_mse_sigma": float(((drift / anchor_sigma) ** 2).mean()),
            "anchor_drift_max": float(np.linalg.norm(drift, axis=1).max()),
        }
        donor_table, component_table = donor_distance_and_signature(
            latent_query,
            signature_scores,
            query_obs,
            ref_centroid,
            metric_sigma,
            args.disease_col,
            args.signature_min_cells,
            args.min_compartments,
        )
        row.update(correlation_metrics(donor_table, args.disease_label))
        row.update(
            scib_metrics(
                latent_query,
                query_obs,
                query_sample,
                healthy_sample,
                metric_latent,
                metric_compartment,
                ref_centroid.index.to_numpy(),
                "_batch",
                np.random.default_rng(args.seed + 300),
                args.scib_max_cells,
            )
        )
        if ab_group_positions is not None:
            row.update(
                {
                    "ab_holdout_frac": args.ab_holdout_frac,
                    "ab_split_seed": args.ab_split_seed,
                    "ab_n_healthy_donors_A": int(len(ab_donors_a)),
                    "ab_n_healthy_donors_B": int(len(ab_donors_b)),
                }
            )
            row.update(
                ab_stratified_metrics(
                    latent_query,
                    query_obs,
                    ab_group_positions,
                    ref_centroid,
                    metric_sigma,
                    reference_donor_spread,
                )
            )
            for offset, label in enumerate(("A", "B")):
                subset_metrics = scib_metrics(
                    latent_query,
                    query_obs,
                    query_sample,
                    ab_healthy_samples[label],
                    metric_latent,
                    metric_compartment,
                    ref_centroid.index.to_numpy(),
                    "_batch",
                    np.random.default_rng(args.seed + 410 + offset),
                    args.scib_max_cells,
                    block_query_positions=ab_group_positions[label],
                    include_biology=False,
                )
                row[f"scib_batch_healthy_{label}_asw"] = subset_metrics[
                    "scib_batch_healthy_asw"
                ]
                row[f"scib_block_healthy_{label}_asw"] = subset_metrics[
                    "scib_block_healthy_asw"
                ]
                row[f"scib_n_healthy_{label}_cells"] = subset_metrics[
                    "scib_n_healthy_cells"
                ]
                row[f"scib_n_block_healthy_{label}_cells"] = subset_metrics[
                    "scib_n_block_cells"
                ]

        summary_rows.append(row)
        donor_tables[state] = donor_table
        latents[state] = latent_query
        donor_table.to_parquet(os.path.join(args.out_dir, f"donor_metrics_{state}.parquet"), index=False)
        component_table.to_parquet(
            os.path.join(args.out_dir, f"donor_compartment_metrics_{state}.parquet"),
            index=False,
        )
        if args.save_latents:
            np.save(os.path.join(args.out_dir, f"latent_{state}.npy"), latent_query)
        del latent_anchor
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = pd.DataFrame(summary_rows).set_index("state")
    metrics.to_csv(os.path.join(args.out_dir, "metrics.csv"))
    print("\n" + metrics.round(4).to_string(), flush=True)
    plot_ifn_scatter(
        donor_tables,
        args.disease_label,
        os.path.join(args.out_dir, "ifn_distance_scatter.png"),
    )
    if not args.no_umap:
        make_shared_umap(latents, query_obs, umap_sample, args, args.out_dir)

    manifest = {
        "base_model": args.base_model,
        "old_adata": args.old_adata,
        "new_adata": args.new_adata,
        "disease": spec.key,
        "disease_label": args.disease_label,
        "healthy_label": args.healthy_label,
        "batch_key": args.batch_key,
        "signature_name": spec.signature_name,
        "signature_reference": spec.reference,
        "signature": signature_report,
        "metric_definitions": {
            "strat_healthy_A_to_census": "Donor-balanced, compartment-macro latent distance of A healthy donors to the fixed Census centroid; lower is closer.",
            "strat_healthy_B_to_census": "Same distance for held-out B healthy donors; it is compared with A on the same supported compartments.",
            "scib_batch_healthy_A_asw": "scIB technical batch ASW within MMD-seen A healthy cells, conditioned on fine cell type; higher mixes batches.",
            "scib_batch_healthy_B_asw": "Same technical batch ASW for MMD-held-out B healthy cells.",
            "scib_block_healthy_A_asw": "Fixed-Census versus A healthy-query block ASW, conditioned on compartment; higher aligns A to Census.",
            "scib_block_healthy_B_asw": "Fixed-Census versus held-out B healthy-query block ASW, conditioned on compartment; higher aligns B to Census.",
            "anchor_mse_sigma": "Mean squared movement of a fixed held-out Census sample, normalized per latent axis; lower is better.",
            "scib_bio_cell_type_asw": "scIB scaled ASW for query fine cell types; higher preserves biological separation.",
            "scib_batch_healthy_asw": "scIB batch ASW within healthy query cells, conditioned on fine cell type; higher mixes technical batches.",
            "scib_block_healthy_asw": "scIB block ASW between fixed base-Census latents and healthy query latents, conditioned on coarse compartment; higher aligns the healthy cohort to Census.",
            "ifn_spearman_disease": "Spearman correlation across disease donors between compartment-macro latent distance to fixed Census and compartment-macro IFN score.",
        },
        "states": {
            label: (str(path) if path is not None else "base")
            for label, path in states.items()
        },
        "method": args.method,
        "adapter_rank": state_config["rank"],
        "adapter_activation": (
            "gelu" if state_config["gelu"] else "identity"
        ) if args.method == "lora" else None,
        "arches_mode": state_config["arches_mode"],
        "metric_reference_support": {key: int(value) for key, value in ref_support.items()},
        "ab_groups": ab_manifest,
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"wrote evaluator outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
