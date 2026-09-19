#!/usr/bin/env python3
"""Make a compact, exact-panel healthy reference bank for scGRIP phase I.

This command never edits its source H5AD.  It validates raw counts, applies an
explicit tissue/study filter, optionally compacts cells per donor while
protecting rare strata, and writes only the model panel/count matrix plus
provenance.  It is deliberately not a cell-type annotator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scgrip.preparation import (
    build_model_panel_adata,
    canonical_obs,
    load_and_filter_input,
    sample_reference_bank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-adata", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--out-adata", required=True)
    parser.add_argument("--source-counts-layer", default="counts", help="Raw count layer in input, or literal X.")
    parser.add_argument("--input-panel-policy", choices=("strict", "audited_imputation"), default="strict")

    labels = parser.add_argument_group("reference annotations")
    labels.add_argument("--donor-col", required=True, help="Source donor column.")
    labels.add_argument("--stratum-col", required=True, help="Source broad/fine cell-type column.")
    labels.add_argument("--out-donor-col", default="donor_id")
    labels.add_argument("--out-stratum-col", default="cell_type")
    labels.add_argument(
        "--stratum-map",
        default=None,
        help="Optional exact CSV source,target crosswalk. Unmapped labels are an error.",
    )

    filt = parser.add_argument_group("explicit healthy-tissue/study subset")
    filt.add_argument("--filter-col", default=None)
    filt.add_argument("--filter-value", action="append", default=None, help="Repeat for multiple values.")

    compact = parser.add_argument_group("optional deterministic compaction")
    compact.add_argument(
        "--max-cells-per-donor",
        type=int,
        default=None,
        help="Leave unset to retain all cells. Must cover later metric+replay+held-out needs.",
    )
    compact.add_argument(
        "--reserve-per-stratum",
        type=int,
        default=10,
        help="Protected cells per donor/stratum before uniform within-donor sampling.",
    )
    compact.add_argument("--seed", type=int, default=0)
    compact.add_argument("--support-report", default=None, help="Default: <out>_support.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = Path(args.out_adata)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing reference bank: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    source = load_and_filter_input(
        args.input_adata,
        filter_column=args.filter_col,
        filter_values=args.filter_value,
    )
    source.obs = canonical_obs(
        source,
        donor_source_col=args.donor_col,
        donor_output_col=args.out_donor_col,
        stratum_source_col=args.stratum_col,
        stratum_output_col=args.out_stratum_col,
        stratum_map=args.stratum_map,
    )
    positions, support = sample_reference_bank(
        source.obs,
        donor_col=args.out_donor_col,
        stratum_col=args.out_stratum_col,
        max_cells_per_donor=args.max_cells_per_donor,
        reserve_per_stratum=args.reserve_per_stratum,
        seed=args.seed,
    )
    source = source[positions].copy()
    output, count_location = build_model_panel_adata(
        source,
        base_model=args.base_model,
        source_counts_layer=args.source_counts_layer,
        input_panel_policy=args.input_panel_policy,
        output_obs=source.obs,
        provenance={
            "artifact_role": "healthy_reference_bank",
            "source_adata": str(Path(args.input_adata).resolve()),
            "filter": {"column": args.filter_col, "values": args.filter_value},
            "canonical_donor_column": args.out_donor_col,
            "canonical_stratum_column": args.out_stratum_col,
            "stratum_map": None if args.stratum_map is None else str(Path(args.stratum_map).resolve()),
            "compaction": {
                "max_cells_per_donor": args.max_cells_per_donor,
                "reserve_per_stratum": args.reserve_per_stratum,
                "seed": args.seed,
            },
        },
    )
    output.write_h5ad(destination)
    report_path = Path(args.support_report) if args.support_report else destination.with_name(destination.stem + "_support.csv")
    support.to_csv(report_path, index=False)
    print(
        f"wrote {destination}: {output.n_obs} cells × {output.n_vars} model genes; "
        f"counts at {count_location}; support report: {report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
