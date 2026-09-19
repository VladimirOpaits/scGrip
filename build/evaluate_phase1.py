#!/usr/bin/env python3
"""Evaluate adapter-only phase-I checkpoints on fixed reference and query roles.

No metric in this command feeds back into training or checkpoint selection. In
particular, optional disease signature correlations are post-hoc positive
controls and are written separately from the healthy-only selection record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scgrip.evaluation import (
    EvaluationPaths,
    evaluate_states,
    prepare_evaluation_context,
    save_core_evaluation,
    states_from_specs,
)
from scgrip.metrics import scib_embedding_metrics
from scgrip.signatures import (
    donor_signature_table,
    load_signature,
    score_signature_from_cache,
    score_signature_from_h5ad,
    signature_correlations,
)
from scgrip.strata import stratified_report
from scgrip.visualize import make_shared_umap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="A completed scgrip phase-I run directory.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Repeat for epoch_XXX.pt or a run directory; base is included automatically.",
    )
    parser.add_argument("--base-model", default=None, help="Optional relocated base model path.")
    parser.add_argument("--reference-adata", default=None, help="Optional relocated reference h5ad.")
    parser.add_argument("--query-adata", default=None, help="Optional relocated query h5ad.")
    parser.add_argument("--device", default=None)

    scib = parser.add_argument_group("optional scIB reporting")
    scib.add_argument("--technical-batch-col", default=None, help="e.g. Lane; never defaulted to donor ID.")
    scib.add_argument("--scib-max-cells", type=int, default=30_000)
    scib.add_argument("--scib-seed", type=int, default=0)
    scib.add_argument("--no-scib", action="store_true")

    signature = parser.add_argument_group("optional expression positive control")
    signature.add_argument(
        "--signature",
        default=None,
        help="Built-in ifn18 or path to JSON {name, genes}; never used for checkpoint selection.",
    )
    signature.add_argument("--signature-adata", default=None, help="Full-gene raw h5ad; defaults to query h5ad.")
    signature.add_argument(
        "--signature-cache",
        default=None,
        help=(
            "Compact full-transcriptome signature cache emitted by a preparer. "
            "Use this when a declared signature extends beyond the frozen model panel."
        ),
    )
    signature.add_argument("--signature-id-col", default=None, help="Unique shared ID for cross-file alignment.")
    signature.add_argument("--signature-layer", default="counts", help="Raw count layer; falls back to X.")
    signature.add_argument("--signature-batch-size", type=int, default=1024)
    signature.add_argument(
        "--signature-min-coverage",
        type=float,
        default=1.0,
        help=(
            "Minimum fraction of pre-specified signature genes observed before reporting; "
            "default 1.0 prevents a partial model-panel score being presented as the full signature."
        ),
    )
    signature.add_argument("--signature-min-cells", type=int, default=5)
    signature.add_argument("--signature-min-strata", type=int, default=2)
    signature.add_argument(
        "--signature-disease-label",
        default=None,
        help="Optional disease label for a separate within-disease correlation; all-donor correlation is always reported.",
    )

    umap = parser.add_argument_group("optional shared UMAP")
    umap.add_argument("--no-umap", action="store_true")
    umap.add_argument("--umap-cells", type=int, default=20_000)
    umap.add_argument("--umap-neighbors", type=int, default=20)
    umap.add_argument("--umap-min-dist", type=float, default=0.3)
    umap.add_argument("--umap-seed", type=int, default=0)
    return parser.parse_args()


def _signature_query_ids(context, id_column: str | None) -> np.ndarray:
    if id_column is None:
        return context.query_ids
    if id_column not in context.query_obs:
        raise ValueError(f"Query .obs lacks --signature-id-col {id_column!r}.")
    values = context.query_obs[id_column].astype(str).to_numpy()
    if pd.Index(values).has_duplicates:
        raise ValueError("--signature-id-col must be unique in the evaluated query.")
    return values


def _add_signature_reporting(args: argparse.Namespace, context, states) -> tuple[dict[str, np.ndarray] | None, dict[str, object] | None]:
    if args.signature is None:
        return None, None
    if not 0 < args.signature_min_coverage <= 1:
        raise ValueError("--signature-min-coverage must lie in (0, 1].")
    spec = load_signature(args.signature)
    query_ids = _signature_query_ids(context, args.signature_id_col)
    if args.signature_cache is not None:
        if args.signature_id_col is not None:
            raise ValueError("--signature-cache keys are prepared query obs_names; do not combine it with --signature-id-col.")
        scores, report = score_signature_from_cache(
            args.signature_cache,
            query_ids=query_ids,
            spec=spec,
        )
    else:
        source = Path(args.signature_adata) if args.signature_adata else context.query_path
        # Even when the source is the query h5ad, the evaluator may have applied a
        # manifest filter.  Match stable cell IDs rather than silently assuming the
        # retained rows are still a prefix/full copy of the source.
        scores, report = score_signature_from_h5ad(
            source,
            query_ids=query_ids,
            spec=spec,
            layer=args.signature_layer,
            batch_size=args.signature_batch_size,
            id_column=args.signature_id_col,
            same_order=False,
        )
    coverage = int(report["n_present_genes"]) / int(report["n_requested_genes"])
    if coverage < args.signature_min_coverage:
        raise ValueError(
            f"Signature coverage is only {coverage:.1%} ({report['n_present_genes']}/"
            f"{report['n_requested_genes']}) below --signature-min-coverage="
            f"{args.signature_min_coverage:.1%}. Supply a full-transcriptome source/cache or explicitly "
            "lower the threshold with scientific justification."
        )
    if scores.size != context.query_n_obs:
        raise AssertionError("Signature scores do not align with evaluator query rows.")
    destination = Path(args.out_dir)
    np.save(destination / "signature_scores.npy", scores)
    phase = context.manifest["config"]["phase1"]
    overlay: dict[str, np.ndarray] = {}
    for state in states:
        all_report = stratified_report(
            state.latent,
            context.query_strata,
            context.query_donors,
            context.geometry,
            min_cells_per_donor_stratum=int(phase["min_cells_per_donor_stratum"]),
        )
        donor, component = donor_signature_table(
            report=all_report.distances,
            scores=scores,
            strata=context.query_strata,
            donors=context.query_donors,
            diseases=context.query_disease,
            min_cells=args.signature_min_cells,
            min_strata=args.signature_min_strata,
        )
        donor.to_csv(destination / f"signature_donors_{state.label}.csv", index=False)
        component.to_csv(destination / f"signature_strata_{state.label}.csv", index=False)
        state.metrics.update(signature_correlations(donor, args.signature_disease_label))
        overlay[state.label] = scores
    report = {
        **report,
        "coverage_fraction": coverage,
        "minimum_coverage_required": args.signature_min_coverage,
        "disease_label_for_separate_correlation": args.signature_disease_label,
        "selection_role": "post-hoc positive control only; not used in phase-I checkpoint selection",
    }
    return overlay, report


def main() -> None:
    args = parse_args()
    destination = Path(args.out_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Evaluation out-dir is non-empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    context = prepare_evaluation_context(
        EvaluationPaths(
            run_dir=Path(args.run_dir),
            out_dir=destination,
            base_model=None if args.base_model is None else Path(args.base_model),
            reference_adata=None if args.reference_adata is None else Path(args.reference_adata),
            query_adata=None if args.query_adata is None else Path(args.query_adata),
        ),
        device=args.device,
    )
    states = evaluate_states(context, states_from_specs(args.checkpoint))

    if not args.no_scib:
        if args.technical_batch_col is None:
            technical = None
        else:
            if args.technical_batch_col not in context.query_obs:
                raise ValueError(f"Query .obs lacks technical batch column {args.technical_batch_col!r}.")
            technical = context.query_obs[args.technical_batch_col].astype(str).to_numpy()
        for state in states:
            state.metrics.update(
                scib_embedding_metrics(
                    query_latent=state.latent,
                    query_strata=context.query_strata,
                    healthy_a=context.groups_a,
                    healthy_b=context.groups_b,
                    reference_metric_latent=context.metric_latents,
                    reference_metric_strata=context.metric_strata,
                    technical_batch=technical,
                    max_cells=args.scib_max_cells,
                    seed=args.scib_seed,
                )
            )

    overlay, signature_report = _add_signature_reporting(args, context, states)
    metrics = save_core_evaluation(context, states, destination)
    if not args.no_umap:
        metadata = pd.read_csv(destination / "query_metadata.csv")
        points = make_shared_umap(
            {state.label: state.latent for state in states},
            metadata,
            destination / "umap",
            max_fit_cells=args.umap_cells,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            seed=args.umap_seed,
            continuous=overlay,
        )
        del points

    with (destination / "evaluation_options.json").open("w") as handle:
        json.dump(
            {
                "technical_batch_col": args.technical_batch_col,
                "scib_enabled": not args.no_scib,
                "signature": signature_report,
                "umap_enabled": not args.no_umap,
                "note": "Disease/signature readouts are reporting only and never participate in phase-I checkpoint selection.",
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print("\n" + metrics.round(4).to_string(), flush=True)
    print(f"wrote phase-I evaluation to {destination}", flush=True)


if __name__ == "__main__":
    main()
