"""Explicit configuration objects for the phase-I reference-mapping pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


AdapterKind = Literal["full_w0", "affine_w0"]
InputPanelPolicy = Literal["strict", "audited_imputation"]


@dataclass(frozen=True)
class DataConfig:
    """Names of columns/layers required in a reference bank and query cohort."""

    reference_donor_col: str = "donor_id"
    reference_stratum_col: str = "cell_type"
    query_donor_col: str = "donor_id"
    query_disease_col: str = "disease"
    query_stratum_col: str = "cell_type"
    healthy_labels: tuple[str, ...] = ("normal",)
    counts_layer: str = "counts"
    input_panel_policy: InputPanelPolicy = "strict"

    def validate(self) -> None:
        if not self.healthy_labels:
            raise ValueError("At least one healthy label is required.")
        if not self.counts_layer:
            raise ValueError("counts_layer must name an explicit raw-count layer.")
        if self.input_panel_policy not in {"strict", "audited_imputation"}:
            raise ValueError(
                "input_panel_policy must be 'strict' or 'audited_imputation'."
            )


@dataclass(frozen=True)
class ReferenceSplitConfig:
    """Disjoint roles sampled within each reference donor."""

    metric_per_donor: int = 50
    replay_per_donor: int = 25
    heldout_per_donor: int = 25
    metric_seed: int = 12_345
    split_seed: int = 0

    def validate(self) -> None:
        for name, value in (
            ("metric_per_donor", self.metric_per_donor),
            ("replay_per_donor", self.replay_per_donor),
            ("heldout_per_donor", self.heldout_per_donor),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least one.")


@dataclass(frozen=True)
class AdapterConfig:
    """Parameterisation of the one allowed phase-I encoder update."""

    kind: AdapterKind = "full_w0"
    train_bias: bool = True
    l2_weight: float = 0.0
    target_path: str | None = None

    def validate(self) -> None:
        if self.kind not in {"full_w0", "affine_w0"}:
            raise ValueError(f"Unknown adapter kind: {self.kind}")
        if self.l2_weight < 0:
            raise ValueError("Adapter L2 weight cannot be negative.")


@dataclass(frozen=True)
class Phase1Config:
    """Training and validation settings for disease-free phase I."""

    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 1e-4
    seed: int = 0
    lambda_anchor: float = 20.0
    lambda_mmd: float = 20.0
    healthy_holdout_fraction: float = 0.35
    split_seed: int = 0
    mmd_k_per_stratum: int = 64
    mmd_min_per_side: int = 32
    mmd_bandwidth_points: int = 256
    mmd_scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    min_cells_per_donor_stratum: int = 3
    min_cells_per_reference_stratum: int = 20
    latent_scale_floor: float = 1e-6
    save_epochs: tuple[int, ...] = (1, 2, 5, 10, 20)
    max_heldout_anchor_for_selection: float | None = None
    device: str | None = None

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.lambda_anchor < 0 or self.lambda_mmd < 0:
            raise ValueError("Loss weights cannot be negative.")
        if not 0 < self.healthy_holdout_fraction < 1:
            raise ValueError("healthy_holdout_fraction must lie strictly between zero and one.")
        if self.mmd_k_per_stratum < 2 or self.mmd_min_per_side < 2:
            raise ValueError("MMD requires at least two cells per participating stratum.")
        if self.min_cells_per_donor_stratum < 1:
            raise ValueError("min_cells_per_donor_stratum must be positive.")
        if self.min_cells_per_reference_stratum < 1:
            raise ValueError("min_cells_per_reference_stratum must be positive.")
        if self.latent_scale_floor <= 0:
            raise ValueError("latent_scale_floor must be positive.")
        if not self.mmd_scales or any(scale <= 0 for scale in self.mmd_scales):
            raise ValueError("mmd_scales must be non-empty and positive.")
        if self.max_heldout_anchor_for_selection is not None and self.max_heldout_anchor_for_selection < 0:
            raise ValueError("max_heldout_anchor_for_selection cannot be negative.")


@dataclass(frozen=True)
class ExperimentConfig:
    """Bundle serialised beside every run."""

    data: DataConfig = field(default_factory=DataConfig)
    reference: ReferenceSplitConfig = field(default_factory=ReferenceSplitConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)

    def validate(self) -> None:
        self.data.validate()
        self.reference.validate()
        self.adapter.validate()
        self.phase1.validate()

    def to_dict(self) -> dict:
        return asdict(self)
