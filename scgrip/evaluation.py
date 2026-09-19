"""Reconstruct adapter-only phase-I states and evaluate frozen criteria.

This module deliberately rebuilds a fresh query model for every state. It
never trusts a pickled mutable scVI module, which makes epoch checkpoints
portable and makes the base-vs-adapted comparison unambiguous.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .adapters import attach_from_metadata
from .mmd import (
    MMDDraw,
    StratifiedMMDPlan,
    build_stratified_mmd_plan,
    mmd_loss_from_draw,
    target_tensors_on_device,
)
from .phase1 import _device_from_config, _encode
from .reference import (
    align_to_model_genes,
    combine_replay_query,
    model_setup_args,
    model_var_names,
    prepare_reference_batch_key,
    subset_obs_values,
    validate_raw_counts,
)
from .strata import (
    ReferenceGeometry,
    StratifiedReport,
    compare_stratified_groups,
    normalise_strata,
    reference_geometry,
    stratified_report,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    def default(value: object):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=default)
        handle.write("\n")


@dataclass(frozen=True)
class EvaluationPaths:
    """Run provenance and optional relocatable input overrides."""

    run_dir: Path
    out_dir: Path
    base_model: Path | None = None
    reference_adata: Path | None = None
    query_adata: Path | None = None


@dataclass(frozen=True)
class AdapterState:
    label: str
    path: Path | None
    payload: dict[str, Any] | None


@dataclass
class EvaluationContext:
    manifest: dict[str, Any]
    base_model_path: Path
    reference_path: Path
    query_path: Path
    device: torch.device
    batch_size: int
    setup_layer: str | None
    batch_key: str | None
    combined: Any
    heldout_reference: Any
    n_replay: int
    query_positions: np.ndarray
    query_ids: np.ndarray
    query_obs: pd.DataFrame
    query_donors: np.ndarray
    query_strata: np.ndarray
    query_disease: np.ndarray
    groups_a: np.ndarray
    groups_b: np.ndarray
    groups_disease: np.ndarray
    heldout_latents: np.ndarray
    anchor_sigma: np.ndarray
    metric_latents: np.ndarray
    metric_strata: np.ndarray
    geometry: ReferenceGeometry
    mmd_plan: StratifiedMMDPlan
    mmd_eval_draw: MMDDraw | None
    mmd_target_cpu: dict[str, torch.Tensor]
    model_panel_sha256: str

    @property
    def query_n_obs(self) -> int:
        return int(self.query_positions.size)


@dataclass
class StateEvaluation:
    label: str
    latent: np.ndarray
    metrics: dict[str, float | int]
    reports: dict[str, StratifiedReport]
    checkpoint: str


def read_adapter_state(label: str, path: str | Path | None) -> AdapterState:
    """Read a phase-I adapter checkpoint; ``None`` denotes the unmodified base."""

    if path is None:
        return AdapterState(label=label, path=None, payload=None)
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / "adapter.pt"
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if payload.get("format") != "scgrip_phase1_checkpoint_v1":
        raise ValueError(f"{resolved} is not a scgrip phase-I adapter checkpoint.")
    for key in ("adapter_metadata", "adapter_state_dict", "model_panel_sha256"):
        if key not in payload:
            raise ValueError(f"{resolved} misses required key {key!r}.")
    return AdapterState(label=label, path=resolved, payload=payload)


def states_from_specs(specs: list[str]) -> list[AdapterState]:
    """Parse repeatable ``label=checkpoint`` CLI values plus the base state."""

    states = [read_adapter_state("base", None)]
    labels = {"base"}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"State must have LABEL=PATH form, got {spec!r}.")
        label, raw_path = spec.split("=", 1)
        if not label or label in labels:
            raise ValueError(f"State label {label!r} is empty or duplicated.")
        labels.add(label)
        states.append(read_adapter_state(label, raw_path))
    return states


def _manifest_filters(manifest: dict[str, Any], key: str) -> tuple[str | None, tuple[str, ...] | None]:
    filters = manifest.get("input_filters", {})
    payload = filters.get(key, {}) if isinstance(filters, dict) else {}
    column = payload.get("column") if isinstance(payload, dict) else None
    values = payload.get("values") if isinstance(payload, dict) else None
    return column, None if values is None else tuple(str(value) for value in values)


def prepare_evaluation_context(paths: EvaluationPaths, *, device: str | None = None) -> EvaluationContext:
    """Load exactly the train-time roles and freeze all reference geometry."""

    import anndata as ad
    import scvi

    manifest_path = paths.run_dir / "run_manifest.json"
    roles_path = paths.run_dir / "role_indices.npz"
    if not manifest_path.exists() or not roles_path.exists():
        raise FileNotFoundError(
            "A phase-I run needs run_manifest.json and role_indices.npz for faithful evaluation."
        )
    manifest = _read_json(manifest_path)
    if manifest.get("format") != "scgrip_phase1_run_v1":
        raise ValueError(f"Unsupported run format: {manifest.get('format')!r}")
    config = manifest["config"]
    data = config["data"]
    phase = config["phase1"]
    base_path = paths.base_model or Path(manifest["base_model"])
    reference_path = paths.reference_adata or Path(manifest["reference_adata"])
    query_path = paths.query_adata or Path(manifest["query_adata"])
    run_device = _device_from_config(device or phase.get("device"))
    setup = model_setup_args(base_path)
    setup_layer = setup.get("layer")
    setup_layer = None if setup_layer is None else str(setup_layer)
    requested_layer = None if data["counts_layer"] == "X" else data["counts_layer"]
    if setup_layer != requested_layer:
        raise ValueError("Evaluation counts layer differs from the saved base model registry.")
    batch_key = setup.get("batch_key")
    batch_key = None if batch_key is None else str(batch_key)
    model_genes = model_var_names(base_path)

    reference = ad.read_h5ad(reference_path)
    query = ad.read_h5ad(query_path)
    ref_column, ref_values = _manifest_filters(manifest, "reference")
    query_column, query_values = _manifest_filters(manifest, "query")
    reference = subset_obs_values(reference, column=ref_column, values=ref_values, dataset="reference")
    query = subset_obs_values(query, column=query_column, values=query_values, dataset="query")
    policy = str(data.get("input_panel_policy", "strict"))
    validate_raw_counts(reference, data["counts_layer"], dataset="reference", input_panel_policy=policy)
    validate_raw_counts(query, data["counts_layer"], dataset="query", input_panel_policy=policy)
    reference = align_to_model_genes(reference, model_genes, dataset="reference")
    query = align_to_model_genes(query, model_genes, dataset="query")
    reference, query, batch_key = prepare_reference_batch_key(
        reference,
        query,
        reference_donor_col=data["reference_donor_col"],
        query_donor_col=data["query_donor_col"],
        model_batch_key=batch_key,
    )
    role_values = np.load(roles_path)
    metric_idx = np.asarray(role_values["reference_metric"], dtype=int)
    replay_idx = np.asarray(role_values["reference_replay"], dtype=int)
    heldout_idx = np.asarray(role_values["reference_heldout"], dtype=int)
    groups_a = np.asarray(role_values["query_healthy_A"], dtype=int)
    groups_b = np.asarray(role_values["query_healthy_B"], dtype=int)
    groups_disease = np.asarray(role_values["query_disease_excluded"], dtype=int)
    for name, indices, upper in (
        ("metric", metric_idx, reference.n_obs),
        ("replay", replay_idx, reference.n_obs),
        ("heldout", heldout_idx, reference.n_obs),
        ("healthy A", groups_a, query.n_obs),
        ("healthy B", groups_b, query.n_obs),
        ("disease", groups_disease, query.n_obs),
    ):
        if indices.ndim != 1 or (indices < 0).any() or (indices >= upper).any():
            raise ValueError(f"Saved {name} role indices are incompatible with current input data.")
    if np.intersect1d(replay_idx, metric_idx).size or np.intersect1d(replay_idx, heldout_idx).size:
        raise ValueError("Saved reference roles overlap; run provenance is corrupt.")

    query_donors = query.obs[data["query_donor_col"]].astype(str).to_numpy()
    query_strata = normalise_strata(query.obs[data["query_stratum_col"]], name="query stratum")
    query_disease = query.obs[data["query_disease_col"]].astype(str).to_numpy()
    query_ids = query.obs_names.astype(str).to_numpy()
    query_obs = query.obs.copy()
    metric_strata = normalise_strata(
        reference.obs.iloc[metric_idx][data["reference_stratum_col"]],
        name="reference metric stratum",
    )
    base = scvi.model.SCVI.load(str(base_path))
    base.module.to(run_device)
    base.module.eval()
    all_reference_idx = np.concatenate((metric_idx, replay_idx, heldout_idx))
    all_reference_latent = _encode(base, reference, all_reference_idx, int(phase["batch_size"]))
    first = metric_idx.size
    second = first + replay_idx.size
    metric_latent = all_reference_latent[:first]
    replay_latent = all_reference_latent[first:second]
    heldout_latent = all_reference_latent[second:]
    anchor_sigma = np.maximum(
        replay_latent.std(axis=0, ddof=0), float(phase.get("latent_scale_floor", 1e-6))
    )
    geometry = reference_geometry(
        metric_latent,
        metric_strata,
        min_cells_per_stratum=int(phase["min_cells_per_reference_stratum"]),
        scale_floor=float(phase.get("latent_scale_floor", 1e-6)),
    )
    replay = reference[replay_idx].copy()
    heldout_reference = reference[heldout_idx].copy()
    if data["counts_layer"] != "X":
        reference.X = None
        replay.X = None
        heldout_reference.X = None
        query.X = None
    combined, query_positions = combine_replay_query(replay, query)
    del replay, reference
    gc.collect()

    # The base encoding of healthy A establishes the immutable MMD bandwidths.
    # Register exactly the saved-model fields before asking the base to encode
    # this newly concatenated AnnData.
    scvi.model.SCVI.setup_anndata(combined, batch_key=batch_key, layer=setup_layer)

    # A plain reference model cannot validate unseen query donor categories.
    # ``load_query_data`` creates the same frozen expanded registry used at
    # training time; before an adapter is attached this is the base state.
    base_query_model = scvi.model.SCVI.load_query_data(combined, base)
    base_query_model.is_trained_ = True
    base_query_model.module.to(run_device)
    base_query_model.module.eval()
    base_a_latent = _encode(
        base_query_model,
        combined,
        query_positions[groups_a],
        int(phase["batch_size"]),
    )
    del base_query_model
    gc.collect()
    plan = build_stratified_mmd_plan(
        metric_latents=metric_latent,
        metric_strata=metric_strata,
        query_base_latents=base_a_latent,
        query_strata=query_strata[groups_a],
        query_positions=query_positions[groups_a],
        min_per_side=int(phase["mmd_min_per_side"]),
        k_per_stratum=int(phase["mmd_k_per_stratum"]),
        scales=tuple(float(value) for value in phase["mmd_scales"]),
        bandwidth_points=int(phase["mmd_bandwidth_points"]),
    )
    expected_mmd = manifest.get("mmd", {}).get("strata", {})
    if sorted(expected_mmd) != sorted(plan.names):
        raise ValueError(
            "Reconstructed MMD strata differ from training provenance; input labels or role order changed."
        )
    eval_draw = plan.draw(np.random.default_rng(913)) if plan.strata else None
    panel_hash = str(manifest["data_fingerprints"]["model_panel_sha256"])
    current_panel_hash = __import__("hashlib").sha256(
        "\0".join(model_genes.astype(str)).encode("utf-8") + b"\0"
    ).hexdigest()
    if current_panel_hash != panel_hash:
        raise ValueError("Base model gene panel differs from the saved training run.")
    return EvaluationContext(
        manifest=manifest,
        base_model_path=base_path,
        reference_path=reference_path,
        query_path=query_path,
        device=run_device,
        batch_size=int(phase["batch_size"]),
        setup_layer=setup_layer,
        batch_key=batch_key,
        combined=combined,
        heldout_reference=heldout_reference,
        n_replay=int(replay_idx.size),
        query_positions=query_positions,
        query_ids=query_ids,
        query_obs=query_obs,
        query_donors=query_donors,
        query_strata=query_strata,
        query_disease=query_disease,
        groups_a=groups_a,
        groups_b=groups_b,
        groups_disease=groups_disease,
        heldout_latents=heldout_latent,
        anchor_sigma=anchor_sigma,
        metric_latents=metric_latent,
        metric_strata=metric_strata,
        geometry=geometry,
        mmd_plan=plan,
        mmd_eval_draw=eval_draw,
        mmd_target_cpu=target_tensors_on_device(plan, "cpu") if plan.strata else {},
        model_panel_sha256=panel_hash,
    )


def _fresh_query_model(context: EvaluationContext, state: AdapterState):
    import scvi

    base = scvi.model.SCVI.load(str(context.base_model_path))
    base.module.to(context.device)
    base.module.eval()
    q_model = scvi.model.SCVI.load_query_data(context.combined, base)
    q_model.is_trained_ = True
    module = q_model.module.to(context.device)
    module.eval()
    if state.payload is not None:
        if str(state.payload["model_panel_sha256"]) != context.model_panel_sha256:
            raise ValueError(f"State {state.label!r} was trained on a different model gene panel.")
        adapter = attach_from_metadata(module, state.payload["adapter_metadata"])
        adapter.load_trainable_state_dict(state.payload["adapter_state_dict"])
        module.eval()
    return q_model


@torch.no_grad()
def evaluate_state(context: EvaluationContext, state: AdapterState) -> StateEvaluation:
    """Encode a named state and calculate only fixed, non-disease-tuned metrics."""

    q_model = _fresh_query_model(context, state)
    latent = _encode(
        q_model,
        context.combined,
        context.query_positions,
        context.batch_size,
    )
    heldout = _encode(
        q_model,
        context.heldout_reference,
        np.arange(context.heldout_reference.n_obs),
        context.batch_size,
    )
    difference = heldout - context.heldout_latents
    euclidean = np.linalg.norm(difference, axis=1)
    report_a = stratified_report(
        latent[context.groups_a],
        context.query_strata[context.groups_a],
        context.query_donors[context.groups_a],
        context.geometry,
        min_cells_per_donor_stratum=int(context.manifest["config"]["phase1"]["min_cells_per_donor_stratum"]),
    )
    report_b = stratified_report(
        latent[context.groups_b],
        context.query_strata[context.groups_b],
        context.query_donors[context.groups_b],
        context.geometry,
        min_cells_per_donor_stratum=int(context.manifest["config"]["phase1"]["min_cells_per_donor_stratum"]),
    )
    comparison, _ = compare_stratified_groups(report_a, report_b)
    metrics: dict[str, float | int] = {
        "anchor_held_out": float(np.mean((difference / context.anchor_sigma) ** 2)),
        "drift_max": float(euclidean.max()),
        "drift_p95": float(np.quantile(euclidean, 0.95)),
        **comparison,
    }
    if context.mmd_eval_draw is not None:
        draw = context.mmd_eval_draw
        local = draw.source_positions - context.n_replay
        if (local < 0).any() or (local >= latent.shape[0]).any():
            raise AssertionError("MMD evaluation positions do not refer to query cells.")
        source = torch.as_tensor(latent[local], dtype=torch.float32)
        metrics["mmd_A_eval"] = float(
            mmd_loss_from_draw(
                source,
                draw,
                context.mmd_plan,
                target_tensors=context.mmd_target_cpu,
            )
        )
    checkpoint = "base" if state.path is None else str(state.path)
    del q_model
    gc.collect()
    if context.device.type == "cuda":
        torch.cuda.empty_cache()
    return StateEvaluation(
        label=state.label,
        latent=latent,
        metrics=metrics,
        reports={"A": report_a, "B": report_b},
        checkpoint=checkpoint,
    )


def evaluate_states(context: EvaluationContext, states: list[AdapterState]) -> list[StateEvaluation]:
    return [evaluate_state(context, state) for state in states]


def save_core_evaluation(
    context: EvaluationContext,
    states: list[StateEvaluation],
    out_dir: str | Path,
) -> pd.DataFrame:
    """Save embeddings, fixed metrics, and support tables in a simple portable layout."""

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    embedding_dir = destination / "latents"
    embedding_dir.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    stratum_rows: list[pd.DataFrame] = []
    for state in states:
        np.save(embedding_dir / f"{state.label}.npy", state.latent)
        rows.append({"state": state.label, "checkpoint": state.checkpoint, **state.metrics})
        for group, report in state.reports.items():
            table = report.by_stratum.reset_index(names="stratum").copy()
            table.insert(0, "state", state.label)
            table.insert(1, "group", group)
            stratum_rows.append(table)
    query_group = np.full(context.query_n_obs, "disease_excluded", dtype=object)
    query_group[context.groups_a] = "A"
    query_group[context.groups_b] = "B"
    metadata = pd.DataFrame(
        {
            "query_index": np.arange(context.query_n_obs),
            "donor": context.query_donors,
            "stratum": context.query_strata,
            "disease": context.query_disease,
            "phase1_group": query_group,
        }
    )
    # Preserve original query annotations for downstream plotting and technical
    # batch metrics, without duplicating the count matrix.
    for column in context.query_obs.columns:
        if column not in metadata:
            metadata[column] = context.query_obs[column].to_numpy()
    metadata.to_csv(destination / "query_metadata.csv", index=False)
    metrics = pd.DataFrame(rows).set_index("state")
    metrics.to_csv(destination / "reference_mapping_metrics.csv")
    pd.concat(stratum_rows, ignore_index=True).to_csv(
        destination / "reference_mapping_strata.csv", index=False
    )
    _write_json(
        destination / "evaluation_manifest.json",
        {
            "format": "scgrip_phase1_evaluation_v1",
            "source_run": str(context.manifest.get("base_model")),
            "base_model": str(context.base_model_path),
            "reference_adata": str(context.reference_path),
            "query_adata": str(context.query_path),
            "states": {state.label: state.checkpoint for state in states},
            "fixed_metrics": {
                "anchor_held_out": "Normalized qz-mean movement on disjoint reference held-out cells; lower is better.",
                "strat_A/B": "Donor-balanced distance to global all-cell metric-reference stratum centres; lower is closer.",
                "mmd_A_eval": "Full biased MMD² on a fixed healthy-A draw and independent metric-reference target; lower is closer.",
            },
        },
    )
    return metrics
