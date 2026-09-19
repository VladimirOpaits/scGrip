"""Build small, auditable model-panel H5AD artifacts before fitting.

Training must not be the place where gene identifiers, raw-count provenance,
or cell-type vocabularies are guessed.  These helpers deliberately make those
decisions explicit and leave a compact artifact that can be reused by phase I
and by its evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .reference import (
    align_to_model_genes,
    count_matrix,
    model_setup_args,
    model_var_names,
    require_obs_columns,
    subset_obs_values,
    validate_raw_counts,
)


def _json_default(value: object):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to preparation JSON.")


def read_label_map(path: str | Path) -> dict[str, str]:
    """Read an explicit two-column ``source,target`` stratum crosswalk.

    An unlisted source label is an error.  This is intentional: silently
    assigning unmatched labels to ``other`` would change MMD support and can
    turn a vocabulary mismatch into an apparently successful integration.
    """

    path = Path(path)
    table = pd.read_csv(path)
    required = {"source", "target"}
    if not required.issubset(table.columns):
        raise ValueError(
            f"{path} must contain CSV columns 'source' and 'target'; found {table.columns.tolist()}."
        )
    table = table.loc[:, ["source", "target"]].dropna()
    table["source"] = table["source"].astype(str)
    table["target"] = table["target"].astype(str)
    duplicates = table.groupby("source", observed=True)["target"].nunique()
    ambiguous = duplicates[duplicates > 1].index.tolist()
    if ambiguous:
        raise ValueError(f"{path} maps source labels to multiple targets, e.g. {ambiguous[:8]}.")
    mapping = dict(zip(table["source"], table["target"], strict=True))
    if not mapping:
        raise ValueError(f"{path} contains no usable mappings.")
    return mapping


def harmonise_strata(
    values: pd.Series,
    *,
    mapping_path: str | Path | None,
    column_name: str,
) -> pd.Series:
    """Apply an optional exact label map and reject absent/blank labels."""

    if values.isna().any():
        raise ValueError(f"{column_name!r} has missing stratum labels.")
    source = values.astype(str).str.strip()
    if (source == "").any() or source.str.lower().isin({"nan", "none", "unknown"}).any():
        raise ValueError(f"{column_name!r} has empty or unknown stratum labels.")
    if mapping_path is None:
        return source
    mapping = read_label_map(mapping_path)
    missing = sorted(set(source) - set(mapping))
    if missing:
        raise ValueError(
            f"{len(missing)} labels in {column_name!r} are absent from {mapping_path}, "
            f"e.g. {missing[:12]}. Extend the crosswalk explicitly."
        )
    target = source.map(mapping)
    if target.isna().any() or (target.astype(str).str.strip() == "").any():
        raise ValueError(f"{mapping_path} produced missing/empty target strata.")
    return target.astype(str)


def canonical_obs(
    adata,
    *,
    donor_source_col: str,
    donor_output_col: str,
    stratum_source_col: str,
    stratum_output_col: str,
    stratum_map: str | Path | None = None,
    disease_source_col: str | None = None,
    disease_output_col: str = "disease",
    constant_disease_label: str | None = None,
    require_unique_obs_names: bool = False,
) -> pd.DataFrame:
    """Keep input annotations and add canonical columns required by phase I."""

    columns = [donor_source_col, stratum_source_col]
    if disease_source_col is not None:
        columns.append(disease_source_col)
    require_obs_columns(adata, columns, dataset="input")
    if require_unique_obs_names and not adata.obs_names.is_unique:
        raise ValueError(
            "Input cell IDs are not unique. Do not make them unique implicitly: "
            "provide a stable cell identifier before preparation so signature alignment remains auditable."
        )
    if disease_source_col is not None and constant_disease_label is not None:
        raise ValueError("Choose either a disease source column or a constant disease label, not both.")

    obs = adata.obs.copy()
    donor = obs[donor_source_col].astype(str).str.strip()
    if (donor == "").any() or donor.str.lower().isin({"nan", "none"}).any():
        raise ValueError(f"{donor_source_col!r} contains blank/invalid donor labels.")
    obs[donor_output_col] = donor
    obs[stratum_output_col] = harmonise_strata(
        obs[stratum_source_col],
        mapping_path=stratum_map,
        column_name=stratum_source_col,
    ).to_numpy()
    if disease_source_col is not None:
        disease = obs[disease_source_col].astype(str).str.strip()
        if (disease == "").any() or disease.str.lower().isin({"nan", "none"}).any():
            raise ValueError(f"{disease_source_col!r} contains blank/invalid disease labels.")
        obs[disease_output_col] = disease
    elif constant_disease_label is not None:
        if not str(constant_disease_label).strip():
            raise ValueError("constant_disease_label may not be blank.")
        obs[disease_output_col] = str(constant_disease_label)
    return obs


def sample_reference_bank(
    obs: pd.DataFrame,
    *,
    donor_col: str,
    stratum_col: str,
    max_cells_per_donor: int | None,
    reserve_per_stratum: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Deterministically compact a reference without erasing rare strata.

    Every donor/stratum receives up to ``reserve_per_stratum`` protected cells,
    then remaining capacity is sampled uniformly within donor.  If the quota
    itself exceeds a donor cap we fail rather than quietly dropping strata.
    """

    missing = [column for column in (donor_col, stratum_col) if column not in obs]
    if missing:
        raise ValueError(f"reference is missing required columns: {missing}")
    for column in (donor_col, stratum_col):
        if obs[column].isna().any():
            raise ValueError(f"reference[{column!r}] contains missing values.")
    if max_cells_per_donor is not None and max_cells_per_donor < 1:
        raise ValueError("max_cells_per_donor must be positive or omitted.")
    if reserve_per_stratum < 0:
        raise ValueError("reserve_per_stratum cannot be negative.")

    rng = np.random.default_rng(seed)
    donor_values = obs[donor_col].astype(str).to_numpy()
    stratum_values = obs[stratum_col].astype(str).to_numpy()
    chosen: list[np.ndarray] = []
    support_rows: list[dict[str, object]] = []
    for donor in np.unique(donor_values):
        donor_pos = np.flatnonzero(donor_values == donor)
        by_stratum = {
            stratum: donor_pos[stratum_values[donor_pos] == stratum]
            for stratum in np.unique(stratum_values[donor_pos])
        }
        if max_cells_per_donor is None or donor_pos.size <= max_cells_per_donor:
            retained = donor_pos
        else:
            protected_parts = [
                rng.choice(positions, size=min(reserve_per_stratum, positions.size), replace=False)
                for positions in by_stratum.values()
                if reserve_per_stratum
            ]
            protected = (
                np.unique(np.concatenate(protected_parts))
                if protected_parts
                else np.empty(0, dtype=int)
            )
            if protected.size > max_cells_per_donor:
                raise ValueError(
                    f"Donor {donor!r} needs {protected.size} protected cells across strata, "
                    f"above --max-cells-per-donor={max_cells_per_donor}. Increase the cap "
                    "or lower --reserve-per-stratum explicitly."
                )
            remaining = np.setdiff1d(donor_pos, protected, assume_unique=False)
            n_extra = max_cells_per_donor - protected.size
            extra = rng.choice(remaining, size=min(n_extra, remaining.size), replace=False)
            retained = np.concatenate((protected, extra))
        retained = np.asarray(retained, dtype=int)
        chosen.append(retained)
        for stratum, positions in by_stratum.items():
            support_rows.append(
                {
                    "donor": donor,
                    "stratum": stratum,
                    "available_cells": int(positions.size),
                    "retained_cells": int(np.isin(positions, retained).sum()),
                }
            )
    positions = np.sort(np.concatenate(chosen)).astype(int, copy=False)
    report = pd.DataFrame(support_rows).sort_values(["stratum", "donor"], kind="stable")
    return positions, report


def build_model_panel_adata(
    adata,
    *,
    base_model: str | Path,
    source_counts_layer: str,
    input_panel_policy: str,
    output_obs: pd.DataFrame,
    provenance: dict[str, object],
):
    """Return a minimal exact-model-panel object and its count/audit metadata."""

    import anndata as ad

    setup = model_setup_args(base_model)
    expected_layer = setup.get("layer")
    expected_layer = None if expected_layer is None else str(expected_layer)
    source_audit = validate_raw_counts(
        adata,
        source_counts_layer,
        dataset="input",
        input_panel_policy=input_panel_policy,
    )
    aligned = align_to_model_genes(adata, model_var_names(base_model), dataset="input")
    matrix = count_matrix(aligned, source_counts_layer).copy()
    output = ad.AnnData(X=None, obs=output_obs.copy(), var=aligned.var.copy())
    if expected_layer is None:
        output.X = matrix
        output_count_location = "X"
    else:
        output.layers[expected_layer] = matrix
        output_count_location = f"layers/{expected_layer}"
    # Preserve exactly the declarations which the raw-count validator knows how
    # to audit; avoid carrying unrelated source preprocessing state forward.
    if "zero_padded_genes" in aligned.uns:
        # A source declaration can mention genes outside the model panel.  The
        # full source audit remains in provenance; the prepared artifact keeps
        # only declarations relevant to its actual matrix.
        model_panel = set(map(str, output.var_names))
        output.uns["zero_padded_genes"] = np.asarray(
            [
                str(gene)
                for gene in np.asarray(aligned.uns["zero_padded_genes"], dtype=str)
                if str(gene) in model_panel
            ],
            dtype=object,
        )
    if "malat1_fill" in aligned.uns:
        output.uns["malat1_fill"] = aligned.uns["malat1_fill"]
    output.uns["scgrip_preparation"] = json.dumps({
        **provenance,
        "source_counts_layer": source_counts_layer,
        "model_expected_layer": expected_layer,
        "output_count_location": output_count_location,
        "source_input_audit": source_audit,
    }, sort_keys=True, default=_json_default)
    validate_raw_counts(
        output,
        "X" if expected_layer is None else expected_layer,
        dataset="prepared output",
        input_panel_policy=input_panel_policy,
    )
    return output, output_count_location


def load_and_filter_input(
    path: str | Path,
    *,
    filter_column: str | None = None,
    filter_values: Iterable[str] | None = None,
):
    """Load an H5AD and apply one explicit, provenance-worthy filter."""

    import anndata as ad

    adata = ad.read_h5ad(path)
    return subset_obs_values(
        adata,
        column=filter_column,
        values=None if filter_values is None else tuple(filter_values),
        dataset="input",
    )
