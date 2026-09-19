"""Pre-specified expression positive controls, kept outside training.

The default IFN-18 set is inherited from the prior lupus protocol. It is a
declared positive control, not a hyperparameter-selection target. Custom
signatures can be supplied as a JSON list of stable gene IDs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats


CP10K = 10_000.0


@dataclass(frozen=True)
class SignatureSpec:
    key: str
    name: str
    genes: tuple[str, ...]
    role: str = "pre-specified positive control; never used for checkpoint selection"


# The exact IDs used by the previous lupus evaluator/paper encoder.
IFN18 = SignatureSpec(
    key="ifn18",
    name="type-I-interferon-response-18",
    genes=(
        "ENSG00000187608", "ENSG00000126709", "ENSG00000165949", "ENSG00000137959",
        "ENSG00000137965", "ENSG00000157601", "ENSG00000183486", "ENSG00000089127",
        "ENSG00000111335", "ENSG00000111331", "ENSG00000134321", "ENSG00000088827",
        "ENSG00000184979", "ENSG00000160932", "ENSG00000185745", "ENSG00000119917",
        "ENSG00000138646", "ENSG00000133106",
    ),
)
BUILTIN_SIGNATURES = {IFN18.key: IFN18}


def load_signature(spec: str | Path) -> SignatureSpec:
    """Resolve a built-in set or a JSON file ``{name, genes}`` / gene list."""

    raw = str(spec)
    if raw in BUILTIN_SIGNATURES:
        return BUILTIN_SIGNATURES[raw]
    path = Path(raw)
    if not path.exists():
        raise ValueError(f"Unknown built-in signature {raw!r} and no file exists at that path.")
    with path.open() as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        genes, name = payload, path.stem
    elif isinstance(payload, dict):
        genes = payload.get("genes")
        name = payload.get("name", path.stem)
    else:
        raise ValueError("Signature JSON must be a list of IDs or object with a genes field.")
    if not isinstance(genes, list) or len(genes) < 2:
        raise ValueError("A signature needs at least two stable gene IDs.")
    cleaned = tuple(dict.fromkeys(str(gene) for gene in genes if str(gene)))
    if len(cleaned) < 2:
        raise ValueError("A signature needs at least two distinct nonempty gene IDs.")
    return SignatureSpec(key=path.stem, name=str(name), genes=cleaned, role="user supplied")


def _decode(values) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O"}:
        return np.asarray(
            [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
            dtype=str,
        )
    return values.astype(str)


def _obs_values(handle: h5py.File, column: str | None) -> np.ndarray:
    obs = handle["obs"]
    index_name = obs.attrs.get("_index") if column is None else column
    if isinstance(index_name, bytes):
        index_name = index_name.decode()
    if index_name not in obs:
        raise ValueError(f"Signature h5ad lacks obs field {index_name!r}.")
    node = obs[index_name]
    if isinstance(node, h5py.Dataset):
        return _decode(node[:])
    if "codes" in node and "categories" in node:
        categories = _decode(node["categories"][:])
        codes = np.asarray(node["codes"][:], dtype=int)
        values = np.full(codes.size, "", dtype=object)
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        return values.astype(str)
    raise ValueError(f"Unsupported h5ad obs encoding for {index_name!r}.")


def _var_names(handle: h5py.File) -> np.ndarray:
    var = handle["var"]
    index_name = var.attrs["_index"]
    if isinstance(index_name, bytes):
        index_name = index_name.decode()
    return _decode(var[index_name][:])


def _matrix(handle: h5py.File, layer: str | None):
    path = f"layers/{layer}" if layer and f"layers/{layer}" in handle else "X"
    if path not in handle:
        raise ValueError(f"Signature h5ad has neither requested layer {layer!r} nor X.")
    return handle[path], path


def _library_sizes(handle: h5py.File, rows: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    for column in ("total_counts", "n_counts", "raw_sum"):
        node = handle.get(f"obs/{column}")
        if not isinstance(node, h5py.Dataset):
            continue
        values = np.asarray(node[:], dtype=float)[rows]
        if np.isfinite(values).all() and (values > 0).all():
            return values, column
    return None, None


def _source_positions(source_ids: np.ndarray, query_ids: np.ndarray, same_order: bool) -> np.ndarray:
    if same_order:
        if source_ids.size != query_ids.size:
            raise ValueError("Same-order signature source and query have different cell counts.")
        return np.arange(query_ids.size, dtype=int)
    source = pd.Index(source_ids)
    query = pd.Index(query_ids)
    if not source.is_unique or not query.is_unique:
        raise ValueError(
            "Cross-file signature alignment needs unique IDs; supply an explicit --signature-id-col."
        )
    positions = source.get_indexer(query)
    if (positions < 0).any():
        raise ValueError(f"{int((positions < 0).sum())} query cells are absent from signature source.")
    return positions


def _csr_expression(matrix, rows: np.ndarray, lookup: np.ndarray, width: int, library: np.ndarray | None) -> np.ndarray:
    indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
    start, stop = int(rows[0]), int(rows[-1])
    offset = indptr[start]
    data = np.asarray(matrix["data"][offset : indptr[stop + 1]], dtype=np.float32)
    columns = np.asarray(matrix["indices"][offset : indptr[stop + 1]], dtype=np.int64)
    result = np.zeros((rows.size, width), dtype=np.float32)
    lib = np.empty(rows.size, dtype=np.float64) if library is None else library.astype(np.float64, copy=False)
    for local, row in enumerate(rows):
        left, right = indptr[row] - offset, indptr[row + 1] - offset
        target = lookup[columns[left:right]]
        hit = target >= 0
        result[local, target[hit]] = data[left:right][hit]
        if library is None:
            lib[local] = data[left:right].sum(dtype=np.float64)
    return np.log1p(result * CP10K / np.maximum(lib[:, None], 1.0))


def _dense_expression(matrix, rows: np.ndarray, genes: np.ndarray, library: np.ndarray | None) -> np.ndarray:
    # h5py does not consistently support a two-axis Cartesian fancy index, so
    # pull the few signature columns one at a time.
    result = np.empty((rows.size, genes.size), dtype=np.float32)
    for output_column, gene in enumerate(genes):
        result[:, output_column] = np.asarray(matrix[rows, int(gene)], dtype=np.float32)
    if library is None:
        lib = np.asarray(matrix[rows, :].sum(axis=1)).ravel()
    else:
        lib = library
    return np.log1p(result * CP10K / np.maximum(lib[:, None], 1.0))


def _csc_expression(matrix, rows: np.ndarray, genes: np.ndarray, library: np.ndarray | None) -> np.ndarray:
    if library is None:
        raise ValueError("A CSC signature source needs a valid total-count obs column.")
    indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
    result = np.zeros((rows.size, genes.size), dtype=np.float32)
    for out_column, gene in enumerate(genes):
        left, right = indptr[gene], indptr[gene + 1]
        source_rows = np.asarray(matrix["indices"][left:right], dtype=np.int64)
        values = np.asarray(matrix["data"][left:right], dtype=np.float32)
        local = np.searchsorted(rows, source_rows)
        valid = local < rows.size
        candidates = np.flatnonzero(valid)
        valid[candidates] = rows[local[candidates]] == source_rows[candidates]
        result[local[valid], out_column] = values[valid]
    return np.log1p(result * CP10K / np.maximum(library[:, None], 1.0))


def _row_chunks(positions: np.ndarray, batch_size: int):
    order = np.argsort(positions)
    start = 0
    max_span = max(batch_size * 8, batch_size)
    while start < order.size:
        stop = min(start + batch_size, order.size)
        while stop > start + 1 and positions[order[stop - 1]] - positions[order[start]] > max_span:
            stop -= 1
        yield order[start:stop]
        start = stop


def score_signature_from_h5ad(
    source_path: str | Path,
    *,
    query_ids: np.ndarray,
    spec: SignatureSpec,
    layer: str | None = "counts",
    batch_size: int = 1024,
    id_column: str | None = None,
    same_order: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Stream a per-cell, gene-standardised log-CP10K module score.

    It works without loading a full 20k-gene signature source into RAM, which
    matters for the original full-transcriptome RA/Lupus files.
    """

    source_path = Path(source_path)
    if batch_size < 1:
        raise ValueError("signature batch_size must be positive.")
    with h5py.File(source_path, "r") as handle:
        source_ids = _obs_values(handle, id_column)
        positions = _source_positions(source_ids, np.asarray(query_ids, dtype=str), same_order)
        names = pd.Index(_var_names(handle))
        found = [gene for gene in spec.genes if gene in names]
        missing = [gene for gene in spec.genes if gene not in names]
        if len(found) < 2:
            raise ValueError(f"Only {len(found)} genes from signature {spec.name!r} are in {source_path}.")
        gene_positions = names.get_indexer(found)
        lookup = np.full(names.size, -1, dtype=np.int32)
        lookup[gene_positions] = np.arange(gene_positions.size, dtype=np.int32)
        matrix, matrix_path = _matrix(handle, layer)
        encoding = matrix.attrs.get("encoding-type", "array")
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        library, library_column = _library_sizes(handle, positions)
        expression = np.empty((positions.size, len(found)), dtype=np.float32)
        for target_rows in _row_chunks(positions, batch_size):
            source_rows = positions[target_rows]
            library_rows = None if library is None else library[target_rows]
            if encoding == "csr_matrix":
                values = _csr_expression(matrix, source_rows, lookup, len(found), library_rows)
            elif encoding == "csc_matrix":
                values = _csc_expression(matrix, source_rows, gene_positions, library_rows)
            else:
                values = _dense_expression(matrix, source_rows, gene_positions, library_rows)
            expression[target_rows] = values
    mean = expression.mean(axis=0, dtype=np.float64)
    std = expression.std(axis=0, dtype=np.float64)
    usable = std > 1e-8
    if usable.sum() < 2:
        raise ValueError("Fewer than two signature genes vary across this query.")
    scores = ((expression[:, usable] - mean[usable]) / std[usable]).mean(axis=1).astype(np.float32)
    return scores, {
        "signature": spec.name,
        "role": spec.role,
        "source_path": str(source_path),
        "source_matrix": matrix_path,
        "library_size": library_column or "computed_from_source_matrix",
        "n_requested_genes": len(spec.genes),
        "n_present_genes": len(found),
        "n_nonconstant_genes": int(usable.sum()),
        "present_genes": found,
        "missing_genes": missing,
        "score_definition": "mean z(log1p(CP10K)) across present nonconstant signature genes; z over evaluated query cells",
    }


def score_signature_from_cache(
    cache_path: str | Path,
    *,
    query_ids: np.ndarray,
    spec: SignatureSpec,
) -> tuple[np.ndarray, dict[str, object]]:
    """Score a signature from a compact full-transcriptome raw-count cache.

    A mapping model commonly retains only its frozen input panel, while a
    biological signature may include genes outside that panel.  The GEO 10x
    preparer writes this cache *before* panel subsetting: it contains only the
    requested signature genes, exact stable cell IDs, and each cell's original
    full-library count.  Thus this function avoids both a second full-gene
    H5AD and the invalid substitution of a partial panel score for a declared
    signature score.
    """

    cache_path = Path(cache_path)
    with np.load(cache_path, allow_pickle=False) as cache:
        required = {"cell_id", "library_size", "genes", "raw_counts"}
        if not required.issubset(cache.files):
            raise ValueError(f"{cache_path} misses cache arrays {sorted(required - set(cache.files))}.")
        source_ids = _decode(cache["cell_id"])
        library = np.asarray(cache["library_size"], dtype=np.float64)
        genes = pd.Index(_decode(cache["genes"]))
        raw = np.asarray(cache["raw_counts"], dtype=np.float32)
    if raw.ndim != 2 or raw.shape != (source_ids.size, genes.size):
        raise ValueError(f"{cache_path} raw_counts shape does not align with cell_id/genes arrays.")
    if library.shape != (source_ids.size,) or not np.isfinite(library).all() or (library <= 0).any():
        raise ValueError(f"{cache_path} has invalid full-library sizes.")
    if genes.has_duplicates:
        raise ValueError(f"{cache_path} contains duplicate signature gene IDs.")
    positions = _source_positions(source_ids, np.asarray(query_ids, dtype=str), same_order=False)
    found = [gene for gene in spec.genes if gene in genes]
    missing = [gene for gene in spec.genes if gene not in genes]
    if len(found) < 2:
        raise ValueError(
            f"Only {len(found)} requested signature genes are present in compact cache {cache_path}."
        )
    columns = genes.get_indexer(found)
    expression = np.log1p(
        raw[positions][:, columns] * CP10K / np.maximum(library[positions, None], 1.0)
    )
    mean = expression.mean(axis=0, dtype=np.float64)
    std = expression.std(axis=0, dtype=np.float64)
    usable = std > 1e-8
    if usable.sum() < 2:
        raise ValueError("Fewer than two cached signature genes vary across this query.")
    scores = ((expression[:, usable] - mean[usable]) / std[usable]).mean(axis=1).astype(np.float32)
    return scores, {
        "signature": spec.name,
        "role": spec.role,
        "source_path": str(cache_path),
        "source_matrix": "compact full-transcriptome signature raw-count cache",
        "library_size": "saved full-library size from source count matrix",
        "n_requested_genes": len(spec.genes),
        "n_present_genes": len(found),
        "n_nonconstant_genes": int(usable.sum()),
        "present_genes": found,
        "missing_genes": missing,
        "score_definition": "mean z(log1p(CP10K)) across present nonconstant signature genes; z over evaluated query cells",
    }


def donor_signature_table(
    *,
    report: pd.DataFrame,
    scores: np.ndarray,
    strata: np.ndarray,
    donors: np.ndarray,
    diseases: np.ndarray,
    min_cells: int,
    min_strata: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join donor/stratum signature means to donor/stratum latent distances."""

    frame = pd.DataFrame(
        {
            "donor": np.asarray(donors, dtype=str),
            "stratum": np.asarray(strata, dtype=str),
            "score": np.asarray(scores, dtype=float),
        }
    )
    grouped = frame.groupby(["donor", "stratum"], observed=True)["score"]
    means = grouped.mean().unstack("stratum")
    counts = grouped.size().unstack("stratum", fill_value=0)
    means = means.mask(counts < min_cells)
    disease_by_donor = pd.DataFrame({"donor": donors, "disease": diseases}).groupby("donor", observed=True)["disease"].agg(
        lambda values: ";".join(sorted(set(map(str, values))) )
    )
    component_rows: list[dict[str, object]] = []
    donor_rows: list[dict[str, object]] = []
    all_donors = report.index.union(means.index)
    for donor in all_donors:
        distance = report.loc[donor] if donor in report.index else pd.Series(dtype=float)
        score = means.loc[donor] if donor in means.index else pd.Series(dtype=float)
        shared = distance.index.intersection(score.index)
        shared = [stratum for stratum in shared if pd.notna(distance[stratum]) and pd.notna(score[stratum])]
        for stratum in shared:
            component_rows.append(
                {
                    "donor": donor,
                    "disease": disease_by_donor.get(donor, np.nan),
                    "stratum": stratum,
                    "latent_distance": float(distance[stratum]),
                    "signature_score": float(score[stratum]),
                    "signature_n_cells": int(counts.loc[donor, stratum]),
                }
            )
        if len(shared) >= min_strata:
            donor_rows.append(
                {
                    "donor": donor,
                    "disease": disease_by_donor.get(donor, np.nan),
                    "latent_distance": float(distance[shared].mean()),
                    "signature_score": float(score[shared].mean()),
                    "n_strata": len(shared),
                    "strata": ";".join(shared),
                }
            )
    return (
        pd.DataFrame(
            donor_rows,
            columns=["donor", "disease", "latent_distance", "signature_score", "n_strata", "strata"],
        ),
        pd.DataFrame(
            component_rows,
            columns=[
                "donor",
                "disease",
                "stratum",
                "latent_distance",
                "signature_score",
                "signature_n_cells",
            ],
        ),
    )


def signature_correlations(donor_table: pd.DataFrame, disease_label: str | None) -> dict[str, float | int]:
    """Spearman/Pearson table; disease grouping is reporting, never tuning."""

    out: dict[str, float | int] = {}
    masks = {"all": np.ones(len(donor_table), dtype=bool)}
    if disease_label is not None:
        masks["disease"] = donor_table["disease"].astype(str).to_numpy() == str(disease_label)
    for name, mask in masks.items():
        values = donor_table.loc[mask, ["latent_distance", "signature_score"]].dropna()
        out[f"signature_n_{name}_donors"] = int(len(values))
        if len(values) < 3 or values.nunique().min() < 2:
            out[f"signature_spearman_{name}"] = float("nan")
            out[f"signature_spearman_{name}_p"] = float("nan")
            out[f"signature_pearson_{name}"] = float("nan")
            out[f"signature_pearson_{name}_p"] = float("nan")
            continue
        spearman = stats.spearmanr(values["latent_distance"], values["signature_score"])
        pearson = stats.pearsonr(values["latent_distance"], values["signature_score"])
        out[f"signature_spearman_{name}"] = float(spearman.statistic)
        out[f"signature_spearman_{name}_p"] = float(spearman.pvalue)
        out[f"signature_pearson_{name}"] = float(pearson.statistic)
        out[f"signature_pearson_{name}_p"] = float(pearson.pvalue)
    return out
