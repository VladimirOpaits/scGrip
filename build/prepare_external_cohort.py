#!/usr/bin/env python3
"""Make an exact-model-panel external cohort artifact for scGRIP phase I.

Disease cells are retained for post-fit encoding, but this preparation command
does not train, score, or select on them.  It only fixes the provenance and
input contract required for a later healthy-only phase-I run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scgrip.preparation import build_model_panel_adata, canonical_obs, load_and_filter_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-adata", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--out-adata", required=True)
    parser.add_argument("--source-counts-layer", default="counts", help="Raw count layer in input, or literal X.")
    parser.add_argument("--input-panel-policy", choices=("strict", "audited_imputation"), default="strict")

    labels = parser.add_argument_group("cohort annotations")
    labels.add_argument("--donor-col", required=True, help="Source donor column.")
    labels.add_argument("--stratum-col", required=True, help="Source cell-type column.")
    labels.add_argument("--disease-col", default=None, help="Source disease/condition column.")
    labels.add_argument("--constant-disease-label", default=None, help="Use only for a cohort whose condition is truly constant.")
    labels.add_argument(
        "--cell-id-col",
        default=None,
        help="Optional unique stable source cell ID to become output obs_names; needed if input barcodes repeat.",
    )
    labels.add_argument("--out-donor-col", default="donor_id")
    labels.add_argument("--out-stratum-col", default="cell_type")
    labels.add_argument("--out-disease-col", default="disease")
    labels.add_argument(
        "--stratum-map",
        default=None,
        help="Optional exact CSV source,target crosswalk. Unmapped labels are an error.",
    )

    filt = parser.add_argument_group("optional explicit subset")
    filt.add_argument("--filter-col", default=None)
    filt.add_argument("--filter-value", action="append", default=None, help="Repeat for multiple values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.disease_col is None) == (args.constant_disease_label is None):
        raise ValueError("Provide exactly one of --disease-col or --constant-disease-label.")
    destination = Path(args.out_adata)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing cohort artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    source = load_and_filter_input(
        args.input_adata,
        filter_column=args.filter_col,
        filter_values=args.filter_value,
    )
    if args.cell_id_col is not None:
        if args.cell_id_col not in source.obs:
            raise ValueError(f"Input .obs lacks --cell-id-col {args.cell_id_col!r}.")
        raw_cell_ids = source.obs[args.cell_id_col]
        cell_ids = raw_cell_ids.astype(str).str.strip()
        if raw_cell_ids.isna().any() or (cell_ids == "").any() or not cell_ids.is_unique:
            raise ValueError("--cell-id-col must be non-empty and unique after the requested filter.")
        source.obs_names = cell_ids.to_numpy()
    source.obs = canonical_obs(
        source,
        donor_source_col=args.donor_col,
        donor_output_col=args.out_donor_col,
        stratum_source_col=args.stratum_col,
        stratum_output_col=args.out_stratum_col,
        stratum_map=args.stratum_map,
        disease_source_col=args.disease_col,
        disease_output_col=args.out_disease_col,
        constant_disease_label=args.constant_disease_label,
        require_unique_obs_names=True,
    )
    output, count_location = build_model_panel_adata(
        source,
        base_model=args.base_model,
        source_counts_layer=args.source_counts_layer,
        input_panel_policy=args.input_panel_policy,
        output_obs=source.obs,
        provenance={
            "artifact_role": "external_query_cohort",
            "source_adata": str(Path(args.input_adata).resolve()),
            "filter": {"column": args.filter_col, "values": args.filter_value},
            "canonical_donor_column": args.out_donor_col,
            "canonical_stratum_column": args.out_stratum_col,
            "canonical_disease_column": args.out_disease_col,
            "stratum_map": None if args.stratum_map is None else str(Path(args.stratum_map).resolve()),
        },
    )
    output.write_h5ad(destination)
    print(
        f"wrote {destination}: {output.n_obs} cells × {output.n_vars} model genes; "
        f"counts at {count_location}",
        flush=True,
    )


if __name__ == "__main__":
    main()
