#!/usr/bin/env python3
"""Train the strict healthy-only W0 reference-mapping phase.

The command intentionally has no disease-loss flag. Disease cells may be in
``--query-adata`` for later encoding, but never enter a gradient loader here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scgrip.config import AdapterConfig, DataConfig, ExperimentConfig, Phase1Config, ReferenceSplitConfig
from scgrip.phase1 import DatasetFilter, Phase1Paths, run_phase1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Saved frozen scVI model directory.")
    parser.add_argument("--reference-adata", required=True)
    parser.add_argument("--query-adata", required=True)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu; default auto.")

    data = parser.add_argument_group("annotations and raw counts")
    data.add_argument("--reference-donor-col", default="donor_id")
    data.add_argument("--reference-stratum-col", default="cell_type")
    data.add_argument("--query-donor-col", default="donor_id")
    data.add_argument("--query-disease-col", default="disease")
    data.add_argument("--query-stratum-col", default="cell_type")
    data.add_argument(
        "--healthy-label",
        action="append",
        default=None,
        help="Healthy value in query disease column; repeat for multiple values. Default: normal.",
    )
    data.add_argument("--counts-layer", default="counts", help="Explicit raw-count layer, or literal X.")
    data.add_argument(
        "--input-panel-policy",
        choices=("strict", "audited_imputation"),
        default="strict",
        help="strict rejects padded/imputed genes; audited_imputation records declared legacy fills explicitly.",
    )

    filters = parser.add_argument_group("optional explicit input filters")
    filters.add_argument("--reference-filter-col", default=None)
    filters.add_argument("--reference-filter-value", action="append", default=None)
    filters.add_argument("--query-filter-col", default=None)
    filters.add_argument("--query-filter-value", action="append", default=None)

    reference = parser.add_argument_group("disjoint reference roles")
    reference.add_argument("--metric-per-donor", type=int, default=50)
    reference.add_argument("--replay-per-donor", type=int, default=25)
    reference.add_argument("--heldout-per-donor", type=int, default=25)
    reference.add_argument("--metric-seed", type=int, default=12345)
    reference.add_argument("--reference-split-seed", type=int, default=0)

    adapter = parser.add_argument_group("the only trainable layer")
    adapter.add_argument(
        "--adapter",
        choices=("full_w0", "affine_w0"),
        default="full_w0",
        help="full_w0 is the main method; affine_w0 is a labeled ablation.",
    )
    adapter.add_argument("--freeze-bias", action="store_true", help="Do not fit the first-layer bias delta.")
    adapter.add_argument("--adapter-l2", type=float, default=0.0)
    adapter.add_argument(
        "--adapter-target-path",
        default=None,
        help="Advanced override for an unusual scVI encoder; normally inferred and audited.",
    )

    train = parser.add_argument_group("phase-I objective")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--split-seed", type=int, default=0, help="Healthy donor A/B split seed.")
    train.add_argument("--lambda-anchor", type=float, default=20.0)
    train.add_argument("--lambda-mmd", type=float, default=20.0)
    train.add_argument("--healthy-holdout-fraction", type=float, default=0.35)
    train.add_argument("--mmd-k-per-stratum", type=int, default=64)
    train.add_argument("--mmd-min-per-side", type=int, default=32)
    train.add_argument("--mmd-bandwidth-points", type=int, default=256)
    train.add_argument("--mmd-scales", type=float, nargs="+", default=(0.25, 0.5, 1.0, 2.0, 4.0))
    train.add_argument("--min-cells-per-donor-stratum", type=int, default=3)
    train.add_argument("--min-cells-per-reference-stratum", type=int, default=20)
    train.add_argument("--latent-scale-floor", type=float, default=1e-6)
    train.add_argument("--save-epochs", type=int, nargs="+", default=(1, 2, 5, 10, 20))
    train.add_argument(
        "--selection-anchor-budget",
        type=float,
        default=None,
        help="Predeclare max held-out reference anchor for automatic healthy-B checkpoint selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        data=DataConfig(
            reference_donor_col=args.reference_donor_col,
            reference_stratum_col=args.reference_stratum_col,
            query_donor_col=args.query_donor_col,
            query_disease_col=args.query_disease_col,
            query_stratum_col=args.query_stratum_col,
            healthy_labels=tuple(args.healthy_label or ("normal",)),
            counts_layer=args.counts_layer,
            input_panel_policy=args.input_panel_policy,
        ),
        reference=ReferenceSplitConfig(
            metric_per_donor=args.metric_per_donor,
            replay_per_donor=args.replay_per_donor,
            heldout_per_donor=args.heldout_per_donor,
            metric_seed=args.metric_seed,
            split_seed=args.reference_split_seed,
        ),
        adapter=AdapterConfig(
            kind=args.adapter,
            train_bias=not args.freeze_bias,
            l2_weight=args.adapter_l2,
            target_path=args.adapter_target_path,
        ),
        phase1=Phase1Config(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
            lambda_anchor=args.lambda_anchor,
            lambda_mmd=args.lambda_mmd,
            healthy_holdout_fraction=args.healthy_holdout_fraction,
            split_seed=args.split_seed,
            mmd_k_per_stratum=args.mmd_k_per_stratum,
            mmd_min_per_side=args.mmd_min_per_side,
            mmd_bandwidth_points=args.mmd_bandwidth_points,
            mmd_scales=tuple(args.mmd_scales),
            min_cells_per_donor_stratum=args.min_cells_per_donor_stratum,
            min_cells_per_reference_stratum=args.min_cells_per_reference_stratum,
            latent_scale_floor=args.latent_scale_floor,
            save_epochs=tuple(args.save_epochs),
            max_heldout_anchor_for_selection=args.selection_anchor_budget,
            device=args.device,
        ),
    )
    paths = Phase1Paths(
        base_model=Path(args.base_model),
        reference_adata=Path(args.reference_adata),
        query_adata=Path(args.query_adata),
        run_dir=Path(args.out_dir) / args.run_name,
    )
    print(
        "phase-I invariant: ELBO=A+B healthy only | anchor=replay only | "
        "MMD=healthy A vs independent metric reference | disease=no gradients",
        flush=True,
    )
    run_phase1(
        paths,
        config,
        reference_filter=DatasetFilter(
            column=args.reference_filter_col,
            values=None if args.reference_filter_value is None else tuple(args.reference_filter_value),
        ),
        query_filter=DatasetFilter(
            column=args.query_filter_col,
            values=None if args.query_filter_value is None else tuple(args.query_filter_value),
        ),
    )


if __name__ == "__main__":
    main()
