from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from .adapters import ExpressionLayerAdapter, attach_expression_adapter
from .config import ExperimentConfig
from .mmd import (
    MMDDraw,
    StratifiedMMDPlan,          # план, как считать MMD по стратам
    build_stratified_mmd_plan,
    mmd_loss_from_draw,
    target_tensors_on_device,
)
from .reference import (
    QueryGroups,
    ReferenceRoles,
    align_to_model_genes,
    combine_replay_query,
    fingerprint_adata,
    model_setup_args,
    model_var_names,
    prepare_reference_batch_key,
    require_obs_columns,
    sample_reference_roles,
    split_query_groups,
    subset_obs_values,
    validate_raw_counts,
)
from .strata import (
    ReferenceGeometry,
    StratifiedReport,
    compare_stratified_groups,
    format_stratified_reports,
    normalise_strata,
    reference_geometry,
    stratified_report,
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
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}.")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _device_from_config(value: str | None) -> torch.device:
    if value is not None:
        device = torch.device(value)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return device


def _to_device(tensors: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in tensors.items()
    }


def _tensors_for_positions(adata_manager, positions: np.ndarray, device: torch.device) -> dict[str, Any]:
    """Load only a requested set of cells; MMD never materialises all healthy A."""

    from scvi.dataloaders import AnnDataLoader

    positions = np.asarray(positions, dtype=int)
    if positions.ndim != 1 or not positions.size:
        raise ValueError("A non-empty one-dimensional position array is required.")
    loader = AnnDataLoader(
        adata_manager,
        indices=positions,
        batch_size=positions.size,
        shuffle=False,
    )
    return _to_device(next(iter(loader)), device)


def _qz_loc(module, tensors: dict[str, Any]) -> torch.Tensor:
    """Deterministic encoder mean used for every geometry-bearing loss/metric."""

    return module.inference(**module._get_inference_input(tensors))["qz"].loc


def _elbo(module, tensors: dict[str, Any]) -> torch.Tensor:
    output = module(tensors, loss_kwargs={"kl_weight": 1.0})
    try:
        return output[2].loss
    except (IndexError, AttributeError) as error:
        raise RuntimeError("Unexpected scVI module forward output; cannot obtain ELBO.") from error


def _encode(q_model, adata, indices: np.ndarray, batch_size: int) -> np.ndarray:
    values = q_model.get_latent_representation(
        adata,
        indices=np.asarray(indices, dtype=int),
        batch_size=batch_size,
    )
    return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True)
class Phase1Paths:
    base_model: Path
    reference_adata: Path
    query_adata: Path
    run_dir: Path


@dataclass(frozen=True)
class DatasetFilter:
    """Optional explicit subset filter, normally used to select healthy tissue."""

    column: str | None = None
    values: tuple[str, ...] | None = None


@dataclass
class PreparedPhase1:
    """All fitting state, with disease labels retained only for provenance."""

    q_model: Any
    module: torch.nn.Module
    adapter: ExpressionLayerAdapter
    combined: Any
    heldout_reference: Any
    replay_latents: np.ndarray
    heldout_latents: np.ndarray
    anchor_sigma: np.ndarray
    metric_latents: np.ndarray
    metric_strata: np.ndarray
    geometry: ReferenceGeometry
    roles: ReferenceRoles
    groups: QueryGroups
    pos_a: np.ndarray
    pos_b: np.ndarray
    pos_disease: np.ndarray
    train_positions: np.ndarray
    query_donors: np.ndarray
    query_strata: np.ndarray
    plan: StratifiedMMDPlan
    target_tensors: dict[str, torch.Tensor]
    eval_mmd_draw: MMDDraw | None
    data_fingerprints: dict[str, object]
    input_audit: dict[str, object]
    base_setup: dict[str, object]


class Phase1Trainer:
    """Orchestrates one auditable phase-I run without exposing disease gradients."""

    def __init__(
        self,
        paths: Phase1Paths,
        config: ExperimentConfig,
        *,
        reference_filter: DatasetFilter | None = None,
        query_filter: DatasetFilter | None = None,
    ) -> None:
        config.validate()
        self.paths = paths
        self.config = config
        self.reference_filter = reference_filter or DatasetFilter()
        self.query_filter = query_filter or DatasetFilter()
        self.device = _device_from_config(config.phase1.device)
        self.prepared: PreparedPhase1 | None = None
        self._history: list[dict[str, object]] = []
        self._stratified_history: list[pd.DataFrame] = []

        self._train_rng = np.random.default_rng(config.phase1.seed + 10_001)
        self._mmd_rng = np.random.default_rng(config.phase1.seed + 20_001)
        torch.manual_seed(config.phase1.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.phase1.seed)

    def _create_run_dir(self) -> None:
        if self.paths.run_dir.exists():
            contents = list(self.paths.run_dir.iterdir())
            if contents:
                raise FileExistsError(
                    f"Run directory already exists and is non-empty: {self.paths.run_dir}. "
                    "Choose a fresh --run-name; phase-I never overwrites a prior run."
                )
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        (self.paths.run_dir / "checkpoints").mkdir(exist_ok=True)

    def _validate_base_setup(self, setup: dict[str, object]) -> tuple[str | None, str | None]:
        expected_layer = setup.get("layer")
        expected_layer = None if expected_layer is None else str(expected_layer)
        requested_layer = None if self.config.data.counts_layer == "X" else self.config.data.counts_layer
        if expected_layer != requested_layer:
            raise ValueError(
                "The model was trained with layer="
                f"{expected_layer!r}, but phase I was given counts_layer="
                f"{self.config.data.counts_layer!r}. Use the exact raw-count layer used by the base model."
            )
        batch_key = setup.get("batch_key")
        return expected_layer, None if batch_key is None else str(batch_key)

    def prepare(self) -> PreparedPhase1:
        """Load/validate data, build frozen roles, and attach an identity W0 update."""

        import anndata as ad
        import scvi

        self._create_run_dir()
        paths = self.paths
        data = self.config.data
        phase = self.config.phase1
        setup = model_setup_args(paths.base_model)
        setup_layer, model_batch_key = self._validate_base_setup(setup)
        model_genes = model_var_names(paths.base_model)

        reference = ad.read_h5ad(paths.reference_adata)
        query = ad.read_h5ad(paths.query_adata)
        reference = subset_obs_values(
            reference,
            column=self.reference_filter.column,
            values=self.reference_filter.values,
            dataset="reference",
        )
        query = subset_obs_values(
            query,
            column=self.query_filter.column,
            values=self.query_filter.values,
            dataset="query",
        )
        require_obs_columns(
            reference,
            (data.reference_donor_col, data.reference_stratum_col),
            dataset="reference",
        )
        require_obs_columns(
            query,
            (data.query_donor_col, data.query_disease_col, data.query_stratum_col),
            dataset="query",
        )
        reference_audit = validate_raw_counts(
            reference,
            data.counts_layer,
            dataset="reference",
            input_panel_policy=data.input_panel_policy,
        )
        query_audit = validate_raw_counts(
            query,
            data.counts_layer,
            dataset="query",
            input_panel_policy=data.input_panel_policy,
        )
        reference = align_to_model_genes(reference, model_genes, dataset="reference")
        query = align_to_model_genes(query, model_genes, dataset="query")
        reference, query, batch_key = prepare_reference_batch_key(
            reference,
            query,
            reference_donor_col=data.reference_donor_col,
            query_donor_col=data.query_donor_col,
            model_batch_key=model_batch_key,
        )
        fingerprints: dict[str, object] = {
            "reference": fingerprint_adata(reference),
            "query": fingerprint_adata(query),
            "model_panel_sha256": str(fingerprint_adata(reference)["var_names_sha256"]),
        }

        roles = sample_reference_roles(reference.obs, data.reference_donor_col, self.config.reference)
        groups = split_query_groups(
            query.obs,
            donor_col=data.query_donor_col,
            disease_col=data.query_disease_col,
            healthy_labels=data.healthy_labels,
            holdout_fraction=phase.healthy_holdout_fraction,
            seed=phase.split_seed,
        )
        query_donors = query.obs[data.query_donor_col].astype(str).to_numpy()
        query_strata = normalise_strata(query.obs[data.query_stratum_col], name="query stratum")
        metric_strata = normalise_strata(
            reference.obs.iloc[roles.metric][data.reference_stratum_col],
            name="reference metric stratum",
        )

        # The model itself is frozen. It is loaded without an attached full
        # AnnData to keep memory bounded; every encoding call receives only the
        # explicit, validated AnnData above.
        model = scvi.model.SCVI.load(str(paths.base_model))
        model.module.to(self.device)
        model.module.eval()
        all_reference_positions = np.concatenate((roles.metric, roles.replay, roles.heldout))
        all_reference_latents = _encode(
            model,
            reference,
            all_reference_positions,
            phase.batch_size,
        )
        metric_end = roles.metric.size
        replay_end = metric_end + roles.replay.size
        metric_latents = all_reference_latents[:metric_end]
        replay_latents = all_reference_latents[metric_end:replay_end]
        heldout_latents = all_reference_latents[replay_end:]
        del all_reference_latents
        anchor_sigma = np.maximum(replay_latents.std(axis=0, ddof=0), phase.latent_scale_floor)
        geometry = reference_geometry(
            metric_latents,
            metric_strata,
            min_cells_per_stratum=phase.min_cells_per_reference_stratum,
            scale_floor=phase.latent_scale_floor,
        )
        replay = reference[roles.replay].copy()
        heldout_reference = reference[roles.heldout].copy()
        # No generic X matrix is needed after this point when an explicit count
        # layer is used. This commonly releases a second large copy of counts.
        if data.counts_layer != "X":
            reference.X = None
            query.X = None
            replay.X = None
            heldout_reference.X = None

        combined, query_positions = combine_replay_query(replay, query)
        del reference, replay
        gc.collect()
        pos_a = query_positions[groups.healthy_a]
        pos_b = query_positions[groups.healthy_b]
        pos_disease = query_positions[groups.disease]
        train_positions = np.concatenate((pos_a, pos_b)).astype(int, copy=False)
        if not train_positions.size or np.intersect1d(train_positions, pos_disease).size:
            raise AssertionError("Disease cells reached the phase-I ELBO position set.")

        # ``setup_anndata`` reconstructs the same registry contract as the
        # saved base. It must use the base model's expected batch-key spelling.
        scvi.model.SCVI.setup_anndata(combined, batch_key=batch_key, layer=setup_layer)
        q_model = scvi.model.SCVI.load_query_data(combined, model)
        q_model.is_trained_ = True
        module = q_model.module.to(self.device)
        module.eval()  # freeze BatchNorm/dropout state for every phase-I loss
        adapter = attach_expression_adapter(
            module,
            n_expression=combined.n_vars,
            config=self.config.adapter,
        )
        module.eval()
        if not adapter.is_identity():
            raise AssertionError("New W0 adapter is not algebraically identity at initialization.")
        n_check = min(512, roles.replay.size)
        check = _encode(q_model, combined, np.arange(n_check), phase.batch_size)
        initial_max_diff = float(np.abs(check - replay_latents[:n_check]).max())
        if initial_max_diff > 1e-4:
            raise AssertionError(
                "Replay encoding changed before phase-I training "
                f"(max |Δz|={initial_max_diff:.3e}). Check genes/count layer/batch registry."
            )

        # Fixed MMD bandwidths are based on the zero-update/base encoding of
        # healthy A. The input adapter cannot later change this scale.
        base_a_latents = _encode(q_model, combined, pos_a, phase.batch_size)
        plan = build_stratified_mmd_plan(
            metric_latents=metric_latents,
            metric_strata=metric_strata,
            query_base_latents=base_a_latents,
            query_strata=query_strata[groups.healthy_a],
            query_positions=pos_a,
            min_per_side=phase.mmd_min_per_side,
            k_per_stratum=phase.mmd_k_per_stratum,
            scales=phase.mmd_scales,
            bandwidth_points=phase.mmd_bandwidth_points,
        )
        if phase.lambda_mmd > 0 and not plan.strata:
            raise ValueError(
                "No shared stratum has enough healthy-A and metric-reference cells for MMD. "
                "Inspect labels/support or lower --mmd-min-per-side deliberately."
            )
        target_tensors = target_tensors_on_device(plan, self.device) if plan.strata else {}
        eval_mmd_draw = (
            plan.draw(np.random.default_rng(913)) if plan.strata else None
        )

        prepared = PreparedPhase1(
            q_model=q_model,
            module=module,
            adapter=adapter,
            combined=combined,
            heldout_reference=heldout_reference,
            replay_latents=replay_latents,
            heldout_latents=heldout_latents,
            anchor_sigma=anchor_sigma,
            metric_latents=metric_latents,
            metric_strata=metric_strata,
            geometry=geometry,
            roles=roles,
            groups=groups,
            pos_a=pos_a,
            pos_b=pos_b,
            pos_disease=pos_disease,
            train_positions=train_positions,
            query_donors=query_donors,
            query_strata=query_strata,
            plan=plan,
            target_tensors=target_tensors,
            eval_mmd_draw=eval_mmd_draw,
            data_fingerprints=fingerprints,
            input_audit={"reference": reference_audit, "query": query_audit},
            base_setup=setup,
        )
        self.prepared = prepared
        self._write_initial_provenance(initial_max_diff)
        return prepared

    def _write_initial_provenance(self, initial_max_diff: float) -> None:
        if self.prepared is None:
            raise RuntimeError("prepare() must run before writing provenance.")
        state = self.prepared
        np.savez_compressed(
            self.paths.run_dir / "role_indices.npz",
            reference_metric=state.roles.metric,
            reference_replay=state.roles.replay,
            reference_heldout=state.roles.heldout,
            query_healthy_A=state.groups.healthy_a,
            query_healthy_B=state.groups.healthy_b,
            query_disease_excluded=state.groups.disease,
        )
        state.roles.per_donor.to_csv(self.paths.run_dir / "reference_role_support.csv")
        query_groups = pd.DataFrame(
            {
                "obs_name": state.combined.obs_names[state.pos_a.tolist() + state.pos_b.tolist() + state.pos_disease.tolist()],
                "phase1_group": (
                    ["A"] * state.pos_a.size
                    + ["B"] * state.pos_b.size
                    + ["disease_excluded"] * state.pos_disease.size
                ),
            }
        )
        query_groups.to_csv(self.paths.run_dir / "query_groups.csv", index=False)
        _write_json(
            self.paths.run_dir / "run_manifest.json",
            {
                "format": "scgrip_phase1_run_v1",
                "base_model": str(self.paths.base_model),
                "reference_adata": str(self.paths.reference_adata),
                "query_adata": str(self.paths.query_adata),
                "device": str(self.device),
                "input_filters": {
                    "reference": {
                        "column": self.reference_filter.column,
                        "values": self.reference_filter.values,
                    },
                    "query": {
                        "column": self.query_filter.column,
                        "values": self.query_filter.values,
                    },
                },
                "config": self.config.to_dict(),
                "base_scvi_setup_args": state.base_setup,
                "data_fingerprints": state.data_fingerprints,
                "input_panel_audit": state.input_audit,
                "reference_roles": state.roles.summary(),
                "query_groups": state.groups.summary(),
                "reference_global_stratum_cell_support": state.geometry.cell_support.to_dict(),
                "mmd": state.plan.manifest(),
                "adapter": state.adapter.metadata.to_dict(),
                "initial_replay_identity_max_abs_diff": initial_max_diff,
                "loss_roles": {
                    "elbo": "healthy_query_A_and_B_only",
                    "functional_anchor": "reference_replay_only",
                    "mmd": "healthy_query_A_to_independent_reference_metric_only",
                    "disease_gradients": False,
                },
            },
        )

    def _mmd_from_draw(self, draw: MMDDraw) -> torch.Tensor:
        assert self.prepared is not None
        state = self.prepared
        tensors = _tensors_for_positions(state.q_model.adata_manager, draw.source_positions, self.device)
        z = _qz_loc(state.module, tensors)
        return mmd_loss_from_draw(
            z,
            draw,
            state.plan,
            target_tensors=state.target_tensors,
        )

    @torch.no_grad()
    def evaluate(self, epoch: int) -> tuple[dict[str, float | int], dict[str, StratifiedReport], list[str]]:
        """Evaluate only frozen reference and healthy A/B criteria."""

        if self.prepared is None:
            raise RuntimeError("prepare() must run before evaluate().")
        state = self.prepared
        phase = self.config.phase1
        state.module.eval()
        heldout = _encode(
            state.q_model,
            state.heldout_reference,
            np.arange(state.heldout_reference.n_obs),
            phase.batch_size,
        )
        difference = heldout - state.heldout_latents
        euclidean = np.linalg.norm(difference, axis=1)
        metrics: dict[str, float | int] = {
            "anchor_held_out": float(np.mean((difference / state.anchor_sigma) ** 2)),
            "drift_max": float(euclidean.max()),
            "drift_p95": float(np.quantile(euclidean, 0.95)),
        }
        z_a = _encode(state.q_model, state.combined, state.pos_a, phase.batch_size)
        z_b = _encode(state.q_model, state.combined, state.pos_b, phase.batch_size)
        report_a = stratified_report(
            z_a,
            state.query_strata[state.groups.healthy_a],
            state.query_donors[state.groups.healthy_a],
            state.geometry,
            min_cells_per_donor_stratum=phase.min_cells_per_donor_stratum,
        )
        report_b = stratified_report(
            z_b,
            state.query_strata[state.groups.healthy_b],
            state.query_donors[state.groups.healthy_b],
            state.geometry,
            min_cells_per_donor_stratum=phase.min_cells_per_donor_stratum,
        )
        comparison, common = compare_stratified_groups(report_a, report_b)
        metrics.update(comparison)
        if state.eval_mmd_draw is not None:
            metrics["mmd_A_eval"] = float(self._mmd_from_draw(state.eval_mmd_draw).cpu())
        self._record_stratified(epoch, {"A": report_a, "B": report_b})
        return metrics, {"A": report_a, "B": report_b}, common

    def _record_stratified(self, epoch: int, reports: dict[str, StratifiedReport]) -> None:
        frames: list[pd.DataFrame] = []
        for group, report in reports.items():
            table = report.by_stratum.reset_index(names="stratum").copy()
            table.insert(0, "epoch", epoch)
            table.insert(1, "group", group)
            frames.append(table)
        self._stratified_history.extend(frames)
        pd.concat(self._stratified_history, ignore_index=True).to_csv(
            self.paths.run_dir / "stratified_history.csv", index=False
        )

    def _checkpoint_payload(
        self,
        *,
        epoch: int,
        optimizer: torch.optim.Optimizer | None,
        include_optimizer: bool,
    ) -> dict[str, object]:
        if self.prepared is None:
            raise RuntimeError("prepare() must run before checkpointing.")
        payload: dict[str, object] = {
            "format": "scgrip_phase1_checkpoint_v1",
            "epoch": int(epoch),
            "adapter_metadata": self.prepared.adapter.metadata.to_dict(),
            "adapter_state_dict": self.prepared.adapter.trainable_state_dict(),
            "model_panel_sha256": self.prepared.data_fingerprints["model_panel_sha256"],
            "base_model": str(self.paths.base_model),
            "config": self.config.to_dict(),
        }
        if include_optimizer and optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
            payload["train_rng_state"] = self._train_rng.bit_generator.state
            payload["mmd_rng_state"] = self._mmd_rng.bit_generator.state
            payload["torch_rng_state"] = torch.get_rng_state()
            if torch.cuda.is_available():
                payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return payload

    def _save_checkpoint(self, epoch: int, optimizer: torch.optim.Optimizer) -> Path:
        path = self.paths.run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
        torch.save(
            self._checkpoint_payload(epoch=epoch, optimizer=optimizer, include_optimizer=True),
            path,
        )
        return path

    def _write_history(self) -> None:
        pd.DataFrame(self._history).sort_values("epoch").to_csv(
            self.paths.run_dir / "history.csv", index=False
        )

    def _select_checkpoint(self) -> None:
        budget = self.config.phase1.max_heldout_anchor_for_selection
        payload: dict[str, object] = {
            "criterion": "minimise_strat_B_subject_to_anchor_held_out_budget",
            "anchor_held_out_budget": budget,
            "selected_epoch": None,
            "selected_checkpoint": None,
        }
        if budget is None:
            payload["reason"] = (
                "No checkpoint selected automatically: set a predeclared "
                "--selection-anchor-budget to enable constrained selection."
            )
        else:
            candidates = [
                row
                for row in self._history
                if int(row["epoch"]) > 0
                and np.isfinite(float(row.get("strat_B", np.nan)))
                and float(row.get("anchor_held_out", np.inf)) <= budget
            ]
            if not candidates:
                payload["reason"] = "No epoch satisfied the held-out reference anchor budget."
            else:
                best = min(candidates, key=lambda row: float(row["strat_B"]))
                epoch = int(best["epoch"])
                payload.update(
                    {
                        "selected_epoch": epoch,
                        "selected_checkpoint": f"checkpoints/epoch_{epoch:03d}.pt",
                        "selected_metrics": best,
                    }
                )
        _write_json(self.paths.run_dir / "selection.json", payload)

    def fit(self) -> Path:
        """Run phase I and return the directory holding adapter-only checkpoints."""

        state = self.prepared or self.prepare()
        phase = self.config.phase1
        optimizer = torch.optim.Adam(
            list(state.adapter.trainable_parameters()),
            lr=phase.learning_rate,
        )
        print(
            f"phase I device={self.device} | adapter={state.adapter.kind} | "
            f"trainable={sum(parameter.numel() for parameter in state.adapter.trainable_parameters())} | "
            f"healthy A/B cells={state.pos_a.size}/{state.pos_b.size} | "
            f"disease excluded={state.pos_disease.size} | MMD strata={list(state.plan.names)}",
            flush=True,
        )

        # Epoch zero is a real adapter checkpoint (the base/identity state), so
        # UMAPs and metrics can be reconstructed without special cases.
        metrics, reports, common = self.evaluate(epoch=0)
        initial_row: dict[str, object] = {"epoch": 0, "elbo": np.nan, "anchor": np.nan, "mmd": np.nan, "adapter_l2": np.nan, **metrics}
        self._history.append(initial_row)
        self._write_history()
        self._save_checkpoint(0, optimizer)
        print(
            "epoch 0 (identity): "
            + " ".join(f"{key}={value:.4f}" for key, value in metrics.items() if isinstance(value, float))
            + "\n  "
            + format_stratified_reports(reports, common),
            flush=True,
        )

        replay_count = state.replay_latents.shape[0]
        replay_latents = torch.as_tensor(state.replay_latents, dtype=torch.float32, device=self.device)
        anchor_sigma = torch.as_tensor(state.anchor_sigma, dtype=torch.float32, device=self.device)
        started = time.time()
        for epoch in range(1, phase.epochs + 1):
            state.module.eval()  # no BatchNorm running-stat or dropout movement
            train_order = self._train_rng.permutation(state.train_positions)
            replay_order = self._train_rng.permutation(replay_count)
            from scvi.dataloaders import AnnDataLoader

            healthy_loader = AnnDataLoader(
                state.q_model.adata_manager,
                indices=train_order,
                batch_size=phase.batch_size,
                shuffle=False,
            )
            replay_loader = AnnDataLoader(
                state.q_model.adata_manager,
                indices=replay_order,
                batch_size=phase.batch_size,
                shuffle=False,
            )
            replay_iter = iter(replay_loader)
            replay_cursor = 0
            sums = {"elbo": 0.0, "anchor": 0.0, "mmd": 0.0, "adapter_l2": 0.0}
            steps = 0
            for healthy_tensors in healthy_loader:
                try:
                    replay_tensors = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    replay_cursor = 0
                    replay_tensors = next(replay_iter)
                healthy_tensors = _to_device(healthy_tensors, self.device)
                replay_tensors = _to_device(replay_tensors, self.device)
                replay_batch_n = int(replay_tensors["X"].shape[0])
                target = replay_latents[replay_order[replay_cursor : replay_cursor + replay_batch_n]]
                replay_cursor += replay_batch_n

                optimizer.zero_grad(set_to_none=True)
                state.module.eval()
                elbo = _elbo(state.module, healthy_tensors)
                state.module.eval()
                replay_z = _qz_loc(state.module, replay_tensors)
                anchor = ((replay_z - target) / anchor_sigma).square().mean()
                mmd = torch.zeros((), device=self.device)
                if phase.lambda_mmd > 0:
                    draw = state.plan.draw(self._mmd_rng)
                    mmd = self._mmd_from_draw(draw)
                adapter_l2 = state.adapter.l2_penalty()
                loss = (
                    elbo
                    + phase.lambda_anchor * anchor
                    + phase.lambda_mmd * mmd
                    + self.config.adapter.l2_weight * adapter_l2
                )
                loss.backward()
                trainable = list(state.adapter.trainable_parameters())
                if any(parameter.grad is None for parameter in trainable):
                    missing = [name for name, parameter in state.adapter.named_parameters() if parameter.grad is None]
                    raise RuntimeError(f"Adapter parameters without gradient: {missing}")
                optimizer.step()

                sums["elbo"] += float(elbo.detach())
                sums["anchor"] += float(anchor.detach())
                sums["mmd"] += float(mmd.detach())
                sums["adapter_l2"] += float(adapter_l2.detach())
                steps += 1
            if not steps:
                raise RuntimeError("Healthy A+B loader yielded no training batches.")

            metrics, reports, common = self.evaluate(epoch=epoch)
            row = {
                "epoch": epoch,
                **{name: value / steps for name, value in sums.items()},
                **metrics,
            }
            self._history.append(row)
            self._write_history()
            print(
                f"epoch {epoch}: "
                + " ".join(f"{name}={value / steps:.4f}" for name, value in sums.items())
                + " | "
                + " ".join(f"{key}={value:.4f}" for key, value in metrics.items() if isinstance(value, float))
                + f" | t={time.time() - started:.0f}s"
                + "\n  "
                + format_stratified_reports(reports, common),
                flush=True,
            )
            if epoch in phase.save_epochs or epoch == phase.epochs:
                path = self._save_checkpoint(epoch, optimizer)
                print(f"saved phase-I checkpoint: {path}", flush=True)

        torch.save(
            self._checkpoint_payload(epoch=phase.epochs, optimizer=None, include_optimizer=False),
            self.paths.run_dir / "adapter.pt",
        )
        self._select_checkpoint()
        print(f"phase I complete: {self.paths.run_dir}", flush=True)
        return self.paths.run_dir


def run_phase1(
    paths: Phase1Paths,
    config: ExperimentConfig,
    *,
    reference_filter: DatasetFilter | None = None,
    query_filter: DatasetFilter | None = None,
) -> Path:
    """Convenience API for CLI and programmatic phase-I experiments."""

    return Phase1Trainer(
        paths,
        config,
        reference_filter=reference_filter,
        query_filter=query_filter,
    ).fit()
