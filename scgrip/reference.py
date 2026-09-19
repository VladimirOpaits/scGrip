"""Input validation and explicit cell/donor roles for reference mapping."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from .config import ReferenceSplitConfig

if TYPE_CHECKING:  # imports stay light for metadata-only commands/tests
    from anndata import AnnData


def model_var_names(model_dir: str | Path) -> pd.Index:
    """Read the exact ordered model panel without loading a full scVI object."""

    payload = torch.load(Path(model_dir) / "model.pt", map_location="cpu", weights_only=False)
    if "var_names" not in payload:
        raise ValueError(f"{model_dir}/model.pt does not contain var_names.")
    names = pd.Index(np.asarray(payload["var_names"], dtype=str))
    if names.has_duplicates:
        raise ValueError("The saved reference model has duplicate var_names.")
    return names


def model_setup_args(model_dir: str | Path) -> dict[str, object]:
    """Return the original scVI setup arguments stored in ``model.pt``."""

    payload = torch.load(Path(model_dir) / "model.pt", map_location="cpu", weights_only=False)
    try:
        return dict(payload["attr_dict"]["registry_"]["setup_args"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"Could not read saved scVI setup args from {model_dir}/model.pt.") from error


def _digest(values: Iterable[object]) -> str:
    digest = sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def fingerprint_adata(adata: "AnnData") -> dict[str, object]:
    """Small provenance fingerprint; this deliberately does not hash raw counts."""

    return {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "var_names_sha256": _digest(adata.var_names),
        "obs_names_sha256": _digest(adata.obs_names),
    }


def require_obs_columns(adata: "AnnData", columns: Iterable[str], *, dataset: str) -> None:
    missing = [column for column in columns if column not in adata.obs]
    if missing:
        raise ValueError(f"{dataset} is missing required .obs columns: {missing}")
    for column in columns:
        if adata.obs[column].isna().any():
            raise ValueError(f"{dataset}.obs[{column!r}] contains missing values.")


def count_matrix(adata: "AnnData", counts_layer: str):
    """Get the declared raw-count matrix; ``X`` is permitted only explicitly."""

    if counts_layer == "X":
        matrix = adata.X
    else:
        if counts_layer not in adata.layers:
            raise ValueError(
                f"Raw-count layer {counts_layer!r} is missing. Available layers: "
                f"{list(adata.layers.keys())}"
            )
        matrix = adata.layers[counts_layer]
    if matrix is None:
        raise ValueError(f"Declared raw-count matrix {counts_layer!r} is None.")
    return matrix


def validate_raw_counts(
    adata: "AnnData",
    counts_layer: str,
    *,
    dataset: str,
    input_panel_policy: str = "strict",
) -> dict[str, object]:
    """Reject normalized input and explicitly audit any known panel imputation.

    ``strict`` accepts only observed integer-like counts. ``audited_imputation``
    is a consciously weaker mode for legacy inputs which state their synthetic
    panel entries in ``.uns`` (currently ``zero_padded_genes`` and/or
    ``malat1_fill``). It never makes the imputation invisible: its exact audit
    is written into the run manifest and should be sensitivity-tested.
    """

    if input_panel_policy not in {"strict", "audited_imputation"}:
        raise ValueError("input_panel_policy must be 'strict' or 'audited_imputation'.")

    matrix = count_matrix(adata, counts_layer)
    # Work in COO for sparse inputs so every checked value has an unambiguous
    # gene column.  The raw-count check itself still touches only stored values.
    if sparse.issparse(matrix):
        coordinate = matrix.tocoo(copy=False)
        values = np.asarray(coordinate.data)
        value_columns = np.asarray(coordinate.col, dtype=int)
    else:
        values = np.asarray(matrix).ravel()
        value_columns = np.arange(values.size, dtype=int) % matrix.shape[1]
    if not np.isfinite(values).all():
        raise ValueError(f"{dataset} raw counts contain NaN or infinity.")
    if (values < 0).any():
        raise ValueError(f"{dataset} raw counts contain negative values.")
    noninteger = ~np.isclose(values, np.rint(values), rtol=0.0, atol=1e-3)
    var_names = pd.Index(np.asarray(adata.var_names, dtype=str))
    padded = np.asarray(adata.uns.get("zero_padded_genes", ()), dtype=str)
    if pd.Index(padded).has_duplicates:
        raise ValueError(f"{dataset} declares duplicate zero-padded genes in .uns.")
    missing_padded = pd.Index(padded).difference(var_names)
    if len(missing_padded):
        raise ValueError(
            f"{dataset} declares zero-padded genes absent from its panel, e.g. "
            f"{missing_padded[:5].tolist()}."
        )

    padded_audit: list[dict[str, object]] = []
    for gene in padded:
        column = var_names.get_loc(gene)
        values_in_column = (
            np.asarray(matrix[:, column].data)
            if sparse.issparse(matrix)
            else np.asarray(matrix[:, column]).ravel()
        )
        n_nonzero = int((~np.isclose(values_in_column, 0.0, rtol=0.0, atol=1e-8)).sum())
        padded_audit.append(
            {
                "gene": str(gene),
                "verified_all_zero": n_nonzero == 0,
                "n_observed_nonzero_entries": n_nonzero,
            }
        )
    contradicted_padding = [item["gene"] for item in padded_audit if not item["verified_all_zero"]]
    if contradicted_padding:
        raise ValueError(
            f"{dataset} .uns['zero_padded_genes'] contradicts the matrix; declared genes "
            f"are not all zero, e.g. {contradicted_padding[:5]}."
        )

    malat_fill = adata.uns.get("malat1_fill")
    known_fill = False
    filled_gene_audit: list[dict[str, object]] = []
    if noninteger.any() and malat_fill is not None:
        known_fill = bool(
            np.allclose(values[noninteger], float(malat_fill), rtol=0.0, atol=1e-3)
        )
        if known_fill:
            columns, counts = np.unique(value_columns[noninteger], return_counts=True)
            filled_gene_audit = [
                {
                    "gene": str(var_names[column]),
                    "n_filled_entries": int(count),
                }
                for column, count in zip(columns, counts, strict=True)
            ]
    if noninteger.any() and not known_fill:
        raise ValueError(
            f"{dataset} layer {counts_layer!r} is not integer-like raw counts. "
            "Values are neither raw counts nor a declared audited fill; do not pass "
            "normalised expression to the encoder."
        )
    needs_audit = bool(padded.size or noninteger.any())
    if needs_audit and input_panel_policy == "strict":
        notes = []
        if padded.size:
            notes.append(f"{padded.size} declared zero-padded genes")
        if noninteger.any():
            notes.append(
                f"{int(noninteger.sum())} declared filled entries at value {float(malat_fill):.6g}"
            )
        raise ValueError(
            f"{dataset} is not a strictly observed raw-count panel: " + ", ".join(notes) + ". "
            "Use a source with full observed gene coverage, or explicitly choose "
            "input_panel_policy='audited_imputation' and record a sensitivity analysis."
        )
    return {
        "input_panel_policy": input_panel_policy,
        "n_declared_zero_padded_genes": int(padded.size),
        "declared_zero_padded_genes": padded.tolist(),
        "zero_padded_gene_audit": padded_audit,
        "n_declared_noninteger_fill_entries": int(noninteger.sum()),
        "declared_noninteger_fill_value": None if malat_fill is None else float(malat_fill),
        "noninteger_fill_verified_against_declared_value": known_fill,
        "noninteger_fill_gene_audit": filled_gene_audit,
        "uses_audited_imputation": needs_audit,
    }


def align_to_model_genes(adata: "AnnData", model_genes: pd.Index, *, dataset: str) -> "AnnData":
    """Subset/reorder to the model panel, never silently zero-pad missing genes."""

    observed = pd.Index(np.asarray(adata.var_names, dtype=str))
    if observed.has_duplicates:
        duplicates = observed[observed.duplicated()].unique().tolist()[:10]
        raise ValueError(f"{dataset} has duplicate var_names, e.g. {duplicates}")
    missing = model_genes.difference(observed)
    if len(missing):
        preview = missing.tolist()[:12]
        raise ValueError(
            f"{dataset} lacks {len(missing)} model-panel genes (first: {preview}). "
            "Gene-padding is intentionally forbidden; resolve gene IDs/panel coverage first."
        )
    aligned = adata[:, model_genes].copy()
    if not np.array_equal(np.asarray(aligned.var_names, dtype=str), model_genes.to_numpy(dtype=str)):
        raise AssertionError("Gene reordering failed: aligned order differs from model order.")
    return aligned


def subset_obs_values(
    adata: "AnnData",
    *,
    column: str | None,
    values: Iterable[str] | None,
    dataset: str,
) -> "AnnData":
    """Optional explicit tissue/study filter, applied before every role split."""

    if column is None and values is None:
        return adata
    if not column or values is None:
        raise ValueError("A filter requires both a column and at least one value.")
    require_obs_columns(adata, [column], dataset=dataset)
    wanted = {str(value) for value in values}
    keep = adata.obs[column].astype(str).isin(wanted).to_numpy()
    if not keep.any():
        raise ValueError(f"{dataset} filter {column!r}={sorted(wanted)} kept no cells.")
    return adata[keep].copy()


@dataclass(frozen=True)
class ReferenceRoles:
    """Disjoint absolute positions into the supplied reference AnnData."""

    metric: np.ndarray
    replay: np.ndarray
    heldout: np.ndarray
    per_donor: pd.DataFrame

    def validate(self) -> None:
        arrays = {"metric": self.metric, "replay": self.replay, "heldout": self.heldout}
        for name, values in arrays.items():
            if values.ndim != 1:
                raise ValueError(f"{name} positions must be one-dimensional.")
            if np.unique(values).size != values.size:
                raise AssertionError(f"{name} positions contain duplicates.")
        names = list(arrays)
        for i, left_name in enumerate(names):
            for right_name in names[i + 1 :]:
                if np.intersect1d(arrays[left_name], arrays[right_name]).size:
                    raise AssertionError(f"{left_name} and {right_name} reference roles overlap.")

    def summary(self) -> dict[str, object]:
        return {
            "n_metric_cells": int(self.metric.size),
            "n_replay_cells": int(self.replay.size),
            "n_heldout_cells": int(self.heldout.size),
            "n_metric_donors": int((self.per_donor["metric"] > 0).sum()),
            "n_replay_donors": int((self.per_donor["replay"] > 0).sum()),
            "n_heldout_donors": int((self.per_donor["heldout"] > 0).sum()),
            "roles_pairwise_disjoint": True,
        }


def sample_reference_roles(
    obs: pd.DataFrame,
    donor_col: str,
    config: ReferenceSplitConfig,
) -> ReferenceRoles:
    """Sample fixed metric then independent replay/held-out roles per donor.

    Metric selection uses its own fixed generator and happens before any
    experiment-specific split. Therefore changing replay amount or training
    seed cannot change the MMD target, provided the reference input itself is
    unchanged.
    """

    config.validate()
    if donor_col not in obs:
        raise ValueError(f"Reference donor column {donor_col!r} does not exist.")
    if obs[donor_col].isna().any():
        raise ValueError(f"Reference donor column {donor_col!r} has missing values.")

    metric_rng = np.random.default_rng(config.metric_seed)
    split_rng = np.random.default_rng(config.split_seed)
    metric_parts: list[np.ndarray] = []
    replay_parts: list[np.ndarray] = []
    heldout_parts: list[np.ndarray] = []
    rows: list[dict[str, object]] = []

    groups = obs.groupby(donor_col, observed=True, sort=True).indices
    for donor, positions in groups.items():
        positions = np.asarray(positions, dtype=int)
        metric_n = min(config.metric_per_donor, positions.size)
        metric = metric_rng.choice(positions, size=metric_n, replace=False)
        remaining = np.setdiff1d(positions, metric, assume_unique=False)
        replay_n = min(config.replay_per_donor, remaining.size)
        replay = split_rng.choice(remaining, size=replay_n, replace=False)
        remaining = np.setdiff1d(remaining, replay, assume_unique=False)
        heldout_n = min(config.heldout_per_donor, remaining.size)
        heldout = split_rng.choice(remaining, size=heldout_n, replace=False)

        metric_parts.append(metric)
        replay_parts.append(replay)
        heldout_parts.append(heldout)
        rows.append(
            {
                "donor": str(donor),
                "available": int(positions.size),
                "metric": int(metric_n),
                "replay": int(replay_n),
                "heldout": int(heldout_n),
            }
        )

    def combine(parts: list[np.ndarray]) -> np.ndarray:
        return np.sort(np.concatenate(parts)).astype(int, copy=False) if parts else np.empty(0, dtype=int)

    roles = ReferenceRoles(
        metric=combine(metric_parts),
        replay=combine(replay_parts),
        heldout=combine(heldout_parts),
        per_donor=pd.DataFrame(rows).set_index("donor") if rows else pd.DataFrame(),
    )
    roles.validate()
    if not roles.metric.size or not roles.replay.size or not roles.heldout.size:
        raise ValueError(
            "Reference role split left an empty metric/replay/held-out set. "
            "Increase reference-bank coverage or reduce per-donor requests."
        )
    return roles


@dataclass(frozen=True)
class QueryGroups:
    """Absolute positions into the original query AnnData."""

    healthy_a: np.ndarray
    healthy_b: np.ndarray
    disease: np.ndarray
    donors_a: np.ndarray
    donors_b: np.ndarray

    def summary(self) -> dict[str, object]:
        return {
            "n_healthy_a_cells": int(self.healthy_a.size),
            "n_healthy_b_cells": int(self.healthy_b.size),
            "n_disease_cells_excluded_from_training": int(self.disease.size),
            "n_healthy_a_donors": int(self.donors_a.size),
            "n_healthy_b_donors": int(self.donors_b.size),
        }


def split_query_groups(
    obs: pd.DataFrame,
    *,
    donor_col: str,
    disease_col: str,
    healthy_labels: Iterable[str],
    holdout_fraction: float,
    seed: int,
) -> QueryGroups:
    """Split *healthy donors*, never cells, into MMD A and held-out B."""

    if donor_col not in obs or disease_col not in obs:
        raise ValueError(f"Query requires donor={donor_col!r} and disease={disease_col!r} columns.")
    if obs[donor_col].isna().any() or obs[disease_col].isna().any():
        raise ValueError("Query donor/disease annotations may not contain missing values.")
    labels = {str(label) for label in healthy_labels}
    if not labels:
        raise ValueError("At least one healthy label must be supplied.")
    donors = obs[donor_col].astype(str).to_numpy()
    healthy = obs[disease_col].astype(str).isin(labels).to_numpy()
    healthy_donors = np.unique(donors[healthy])
    if healthy_donors.size < 2:
        raise ValueError(
            "Phase I requires at least two healthy query donors for independent A/B validation."
        )
    n_b = min(
        healthy_donors.size - 1,
        max(1, int(round(healthy_donors.size * holdout_fraction))),
    )
    rng = np.random.default_rng(seed)
    donors_b = np.sort(rng.choice(healthy_donors, size=n_b, replace=False))
    is_b = healthy & np.isin(donors, donors_b)
    is_a = healthy & ~is_b
    disease = ~healthy
    if not is_a.any() or not is_b.any():
        raise AssertionError("Healthy A/B donor split unexpectedly produced an empty side.")
    return QueryGroups(
        healthy_a=np.flatnonzero(is_a),
        healthy_b=np.flatnonzero(is_b),
        disease=np.flatnonzero(disease),
        donors_a=np.unique(donors[is_a]),
        donors_b=donors_b,
    )


def prepare_reference_batch_key(
    reference: "AnnData",
    query: "AnnData",
    *,
    reference_donor_col: str,
    query_donor_col: str,
    model_batch_key: str | None,
) -> tuple["AnnData", "AnnData", str | None]:
    """Copy actual donor labels into the batch key expected by the base model.

    The query and reference may call their donor annotation differently. The
    base model does not care about that spelling, but its saved setup registry
    does; this bridge preserves both original columns and uses the registered
    name only for ``SCVI.setup_anndata``.
    """

    if model_batch_key is None:
        return reference, query, None
    reference = reference.copy()
    query = query.copy()
    reference.obs[model_batch_key] = reference.obs[reference_donor_col].astype(str).to_numpy()
    query.obs[model_batch_key] = query.obs[query_donor_col].astype(str).to_numpy()
    return reference, query, model_batch_key


def combine_replay_query(
    replay: "AnnData",
    query: "AnnData",
) -> tuple["AnnData", np.ndarray]:
    """Concatenate model-aligned replay then query; return query row positions."""

    import anndata as ad

    if not np.array_equal(np.asarray(replay.var_names), np.asarray(query.var_names)):
        raise ValueError("Replay and query genes differ before concatenation.")
    source_key = "_scgrip_source"
    if source_key in replay.obs or source_key in query.obs:
        raise ValueError(f"Reserved .obs column {source_key!r} already exists.")
    combined = ad.concat(
        [replay, query],
        join="inner",
        merge="same",
        label=source_key,
        keys=["replay", "query"],
        index_unique="__scgrip__",
    )
    query_positions = np.arange(replay.n_obs, combined.n_obs, dtype=int)
    if not np.all(combined.obs[source_key].to_numpy()[query_positions] == "query"):
        raise AssertionError("AnnData concat did not preserve replay/query ordering.")
    return combined, query_positions
