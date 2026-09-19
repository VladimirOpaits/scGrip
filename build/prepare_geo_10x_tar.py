#!/usr/bin/env python3
"""Convert a GEO tar of per-sample 10x MTX files into a compact model panel.

Only raw counts for the frozen model panel are retained in the H5AD.  A small
raw-count cache for a pre-specified expression signature is saved separately,
so evaluation does not require storing a second full-transcriptome matrix.
This command does *not* invent cell types: the result is explicitly marked as
unannotated until an auditable annotation step supplies a stratum column.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io, sparse

from scgrip.reference import model_setup_args, model_var_names, validate_raw_counts
from scgrip.signatures import load_signature


SUFFIXES = {
    "_barcodes.tsv.gz": "barcodes",
    "_features.tsv.gz": "features",
    "_matrix.mtx.gz": "matrix",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", required=True)
    parser.add_argument("--sample-sheet", required=True, help="TSV with sample_id, donor_id, disease and optional metadata columns.")
    parser.add_argument("--accession", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--gene-id-map",
        default=None,
        help=(
            "Optional TSV with model_gene, source_gene and gene_symbol columns. "
            "Every row is validated against the source feature symbol; use only "
            "for a documented stable-ID migration."
        ),
    )
    parser.add_argument(
        "--zero-pad-gene",
        action="append",
        default=None,
        help=(
            "One explicitly approved frozen model gene absent from every source "
            "feature table. Repeat as needed. This creates an auditable zero-fill; "
            "unlisted missing genes remain a hard error."
        ),
    )
    parser.add_argument("--signature", default="ifn18", help="Built-in or JSON stable-gene-ID signature; use none to skip cache.")
    parser.add_argument("--temp-dir", default=None, help="Optional filesystem for one-sample temporary extraction.")
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke-test cap; never use for a final artifact.")
    return parser.parse_args()


def _json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _sample_members(archive: tarfile.TarFile) -> dict[str, dict[str, tarfile.TarInfo]]:
    groups: dict[str, dict[str, tarfile.TarInfo]] = {}
    for member in archive.getmembers():
        for suffix, kind in SUFFIXES.items():
            if member.name.endswith(suffix):
                prefix = Path(member.name).name[: -len(suffix)]
                sample_id = re.sub(r"^GSM\d+_", "", prefix)
                groups.setdefault(sample_id, {})[kind] = member
                break
    return groups


def _read_gene_id_map(path: str | None, model_genes: pd.Index) -> tuple[dict[str, str], dict[str, str]]:
    """Load a narrow, symbol-verified model-ID to source-ID crosswalk.

    Gene symbols are deliberately used only to verify an explicit mapping that
    was curated outside this command.  They are not used for heuristic global
    matching, which would silently merge paralogues/pseudogenes.
    """

    if path is None:
        return {}, {}
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"model_gene", "source_gene", "gene_symbol"}
    if not required.issubset(table.columns):
        raise ValueError(f"{path} must contain TSV columns {sorted(required)}.")
    table = table.loc[:, ["model_gene", "source_gene", "gene_symbol"]].copy()
    for column in table.columns:
        table[column] = table[column].astype(str).str.strip()
        if (table[column] == "").any():
            raise ValueError(f"{path} has blank values in {column!r}.")
    if table["model_gene"].duplicated().any() or table["source_gene"].duplicated().any():
        raise ValueError(f"{path} may not map one model/source gene more than once.")
    unknown = sorted(set(table["model_gene"]) - set(model_genes.astype(str)))
    if unknown:
        raise ValueError(f"{path} maps genes absent from the frozen model, e.g. {unknown[:8]}.")
    return (
        dict(zip(table["model_gene"], table["source_gene"], strict=True)),
        dict(zip(table["model_gene"], table["gene_symbol"], strict=True)),
    )


def _extract_one(archive: tarfile.TarFile, members: dict[str, tarfile.TarInfo], directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for kind, member in members.items():
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"Could not read archive member {member.name!r}.")
        destination = directory / f"{kind}{Path(member.name).suffixes[-2]}{Path(member.name).suffixes[-1]}"
        # The names above resolve to e.g. features.tsv.gz.  copyfileobj avoids
        # trusting archive paths and keeps extraction to a single sample.
        with source, destination.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        result[kind] = destination
    return result


def _read_one(
    paths: dict[str, Path],
    *,
    model_genes: pd.Index,
    signature_genes: tuple[str, ...] | None,
    gene_id_map: dict[str, str],
    gene_symbol_map: dict[str, str],
    zero_pad_genes: set[str],
):
    features = pd.read_csv(paths["features"], sep="\t", header=None, compression="gzip", dtype=str)
    if features.shape[1] < 2:
        raise ValueError(f"10x features file {paths['features']} has fewer than two columns.")
    feature_type = features.iloc[:, 2] if features.shape[1] >= 3 else pd.Series("Gene Expression", index=features.index)
    keep_feature = feature_type.astype(str).eq("Gene Expression").to_numpy()
    gene_ids = pd.Index(features.iloc[:, 0].astype(str))
    if gene_ids[keep_feature].has_duplicates:
        duplicates = gene_ids[keep_feature][gene_ids[keep_feature].duplicated()].unique().tolist()[:10]
        raise ValueError(f"Duplicate gene IDs in 10x feature table, e.g. {duplicates}")
    barcodes = pd.read_csv(paths["barcodes"], sep="\t", header=None, compression="gzip", dtype=str).iloc[:, 0].astype(str).to_numpy()
    matrix = sparse.csr_matrix(io.mmread(paths["matrix"]).T)
    if matrix.shape != (barcodes.size, features.shape[0]):
        raise ValueError(
            f"10x matrix shape {matrix.shape} does not match {barcodes.size} barcodes and {features.shape[0]} features."
        )
    full_library = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    source_index = pd.Index(gene_ids)
    source_ids_for_model = pd.Index(
        [gene_id_map.get(str(gene), str(gene)) for gene in model_genes],
        dtype=str,
    )
    if source_ids_for_model.has_duplicates:
        duplicates = source_ids_for_model[source_ids_for_model.duplicated()].unique().tolist()
        raise ValueError(
            "The explicit ID crosswalk makes multiple frozen features point to the same source feature, "
            f"e.g. {duplicates[:8]}."
        )
    model_index = source_index.get_indexer(source_ids_for_model)
    missing = model_genes[model_index < 0].tolist()
    if set(missing) != zero_pad_genes:
        unapproved = sorted(set(missing) - zero_pad_genes)
        unnecessary = sorted(zero_pad_genes - set(missing))
        details = []
        if unapproved:
            details.append(f"unapproved missing genes={unapproved[:12]}")
        if unnecessary:
            details.append(f"requested zero-pad genes that are present={unnecessary[:12]}")
        raise ValueError(
            "10x/model panel mismatch; " + "; ".join(details) + ". "
            "Add a symbol-verified --gene-id-map entry for an ID migration, or explicitly "
            "approve only genuinely absent genes with --zero-pad-gene."
        )
    for model_gene, source_gene in gene_id_map.items():
        source_position = source_index.get_loc(source_gene)
        if isinstance(source_position, slice) or np.ndim(source_position):
            raise AssertionError("Source feature IDs were already checked unique.")
        observed_symbol = str(features.iloc[int(source_position), 1]).strip()
        expected_symbol = gene_symbol_map[model_gene]
        if observed_symbol != expected_symbol:
            raise ValueError(
                f"Crosswalk for {model_gene} expects source {source_gene} to be {expected_symbol!r}, "
                f"but feature table says {observed_symbol!r}."
            )
    observed = model_index >= 0
    if not keep_feature[model_index[observed]].all():
        bad = model_genes[observed][~keep_feature[model_index[observed]]].tolist()[:12]
        raise ValueError(f"Frozen model genes are not marked Gene Expression, e.g. {bad}")
    # Preserve the exact frozen-model ordering while leaving explicitly
    # audited absent genes as structural zero columns.  Constructing this via
    # COO avoids a 3,000-column Python hstack for every sample.
    model_positions = np.flatnonzero(observed)
    selected = matrix[:, model_index[model_positions]].tocoo()
    panel = sparse.coo_matrix(
        (selected.data, (selected.row, model_positions[selected.col])),
        shape=(matrix.shape[0], model_genes.size),
    ).tocsr()
    signature = None
    signature_present: list[str] = []
    if signature_genes is not None:
        signature_index = source_index.get_indexer(pd.Index(signature_genes))
        signature = np.zeros((matrix.shape[0], len(signature_genes)), dtype=np.float32)
        present = signature_index >= 0
        if present.any():
            signature[:, present] = matrix[:, signature_index[present]].toarray().astype(np.float32, copy=False)
            signature_present = [gene for gene, available in zip(signature_genes, present, strict=True) if available]
    return panel, barcodes, full_library, signature, signature_present, int(matrix.nnz), missing


def main() -> None:
    args = parse_args()
    destination = Path(args.out_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    sample_sheet = pd.read_csv(args.sample_sheet, sep="\t", dtype=str)
    required = {"sample_id", "donor_id", "disease"}
    if not required.issubset(sample_sheet.columns):
        raise ValueError(f"Sample sheet misses columns {sorted(required - set(sample_sheet.columns))}")
    if sample_sheet["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique in sample sheet.")
    sample_sheet = sample_sheet.set_index("sample_id", drop=False)

    setup = model_setup_args(args.base_model)
    expected_layer = setup.get("layer")
    expected_layer = None if expected_layer is None else str(expected_layer)
    model_genes = model_var_names(args.base_model)
    gene_id_map, gene_symbol_map = _read_gene_id_map(args.gene_id_map, model_genes)
    zero_pad_genes = set(args.zero_pad_gene or ())
    unknown_zero_pad = sorted(zero_pad_genes - set(model_genes.astype(str)))
    if unknown_zero_pad:
        raise ValueError(f"--zero-pad-gene is not a frozen model gene, e.g. {unknown_zero_pad[:8]}")
    signature = None if str(args.signature).lower() == "none" else load_signature(args.signature)
    signature_genes = None if signature is None else signature.genes

    matrices: list[sparse.csr_matrix] = []
    obs_parts: list[pd.DataFrame] = []
    signature_parts: list[np.ndarray] = []
    library_parts: list[np.ndarray] = []
    reports: list[dict[str, object]] = []
    with tarfile.open(args.tar, "r") as archive:
        groups = _sample_members(archive)
        unknown = sorted(set(groups) - set(sample_sheet.index))
        missing = sorted(set(sample_sheet.index) - set(groups))
        if unknown or missing:
            raise ValueError(
                f"Sample sheet/archive mismatch: unknown archive samples={unknown[:8]}, missing archive samples={missing[:8]}"
            )
        sample_ids = list(sample_sheet.index)
        if args.max_samples is not None:
            sample_ids = sample_ids[: args.max_samples]
        for ordinal, sample_id in enumerate(sample_ids, start=1):
            members = groups[sample_id]
            if set(members) != set(SUFFIXES.values()):
                raise ValueError(f"Sample {sample_id!r} misses a 10x component: {sorted(set(SUFFIXES.values()) - set(members))}")
            with tempfile.TemporaryDirectory(dir=args.temp_dir) as temporary:
                paths = _extract_one(archive, members, Path(temporary))
                panel, barcodes, library, sig_values, sig_present, nnz, padded = _read_one(
                    paths,
                    model_genes=model_genes,
                    signature_genes=signature_genes,
                    gene_id_map=gene_id_map,
                    gene_symbol_map=gene_symbol_map,
                    zero_pad_genes=zero_pad_genes,
                )
            sample_row = sample_sheet.loc[sample_id]
            obs = pd.DataFrame(
                {column: np.repeat(str(sample_row[column]), panel.shape[0]) for column in sample_sheet.columns},
                index=[f"{args.accession}:{sample_id}:{barcode}" for barcode in barcodes],
            )
            obs["source_accession"] = args.accession
            obs["cell_barcode"] = barcodes
            obs["total_counts_full"] = library
            matrices.append(panel)
            obs_parts.append(obs)
            library_parts.append(library)
            if sig_values is not None:
                signature_parts.append(sig_values)
            reports.append(
                {
                    "sample_id": sample_id,
                    "donor_id": str(sample_row["donor_id"]),
                    "disease": str(sample_row["disease"]),
                    "n_cells": int(panel.shape[0]),
                    "n_model_genes": int(panel.shape[1]),
                    "panel_nnz": int(panel.nnz),
                    "full_matrix_nnz": nnz,
                    "signature_genes_present": ";".join(sig_present),
                    "zero_padded_model_genes": ";".join(padded),
                }
            )
            print(f"[{ordinal}/{len(sample_ids)}] {sample_id}: {panel.shape[0]} cells", flush=True)

    matrix = sparse.vstack(matrices, format="csr")
    obs = pd.concat(obs_parts, axis=0)
    # scVI records whether its raw count matrix lived in ``X`` or a named
    # layer.  Preserve that contract exactly: moving a count matrix between
    # the two changes neither values nor genes, but it does make loading the
    # frozen model error-prone and invites accidental use of normalised ``X``.
    output = ad.AnnData(X=None, obs=obs, var=pd.DataFrame(index=model_genes))
    if expected_layer is None:
        output.X = matrix
        count_location = "X"
        count_layer = "X"
    else:
        output.layers[expected_layer] = matrix
        count_location = f"layers/{expected_layer}"
        count_layer = expected_layer
    if zero_pad_genes:
        output.uns["zero_padded_genes"] = np.asarray(sorted(zero_pad_genes), dtype=str)
    input_panel_policy = "audited_imputation" if zero_pad_genes else "strict"
    validate_raw_counts(output, count_layer, dataset=args.accession, input_panel_policy=input_panel_policy)
    output.uns["scgrip_geo_10x"] = json.dumps(
        {
            "format": "scgrip_geo_10x_v1",
            "accession": args.accession,
            "source_tar": str(Path(args.tar).resolve()),
            "sample_sheet": str(Path(args.sample_sheet).resolve()),
            "base_model": str(Path(args.base_model).resolve()),
            "model_setup": setup,
            "output_count_location": count_location,
            "gene_id_map": None if args.gene_id_map is None else str(Path(args.gene_id_map).resolve()),
            "mapped_model_gene_count": len(gene_id_map),
            "zero_padded_model_genes": sorted(zero_pad_genes),
            "input_panel_policy": input_panel_policy,
            "status": "raw counts are model-panel aligned; cell types intentionally unannotated",
            "signature": None if signature is None else {"key": signature.key, "genes": list(signature.genes)},
        },
        sort_keys=True,
        default=_json_default,
    )
    output.write_h5ad(destination / "model_panel_unannotated.h5ad")
    pd.DataFrame(reports).to_csv(destination / "sample_qc.csv", index=False)
    if signature is not None:
        np.savez_compressed(
            destination / "signature_raw_cache.npz",
            # ``Index.to_numpy`` can retain object dtype even after astype(str),
            # which would require pickle to read.  Persist plain Unicode so
            # evaluators can reject pickled cache payloads entirely.
            cell_id=np.asarray(list(obs.index.astype(str)), dtype=np.str_),
            library_size=np.concatenate(library_parts).astype(np.float32),
            genes=np.asarray(signature.genes, dtype=str),
            raw_counts=np.vstack(signature_parts).astype(np.float32),
        )
    _write_json(
        destination / "manifest.json",
        {
            "format": "scgrip_geo_10x_v1",
            "accession": args.accession,
            "source_tar": str(Path(args.tar).resolve()),
            "sample_sheet": str(Path(args.sample_sheet).resolve()),
            "base_model": str(Path(args.base_model).resolve()),
            "model_n_genes": len(model_genes),
            "output_count_location": count_location,
            "gene_id_map": None if args.gene_id_map is None else str(Path(args.gene_id_map).resolve()),
            "mapped_model_gene_count": len(gene_id_map),
            "zero_padded_model_genes": sorted(zero_pad_genes),
            "input_panel_policy": input_panel_policy,
            "n_cells": output.n_obs,
            "n_samples": len(sample_ids),
            "signature": None if signature is None else signature.key,
            "annotation_status": "unannotated; do not use as phase-I query before a separate fixed stratum annotation step",
        },
    )
    print(f"wrote {destination}: {output.n_obs} cells × {output.n_vars} genes", flush=True)


if __name__ == "__main__":
    main()
