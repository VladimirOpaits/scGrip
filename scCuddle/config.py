from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

BatchEmbeddingPolicy = Literal["frozen_zero", "trained", "trained_penalized"]
AdapterMode = Literal["w0", "scarches_new_batch", "both"]


@dataclass(frozen=True)  # иммутабельность
class DataConfig:
    bank_path: Path
    bank_donor_col: str = "donor_id"
    bank_stratum_col: str = "cell_type"
    bank_batch_col: str = "dataset_id"
    counts_layer: str | None = None  # None -> counts in .X

    holdout_fraction: float = 0.2  # A/B клекти 
    split_seed: int = 0

    def validate(self) -> None:
        if not self.bank_path.exists():
            raise FileNotFoundError(self.bank_path)
        if not 0 < self.holdout_fraction < 1:
            raise ValueError("holdout_fraction must be in (0, 1).")


@dataclass(frozen=True)
class SplitConfig:
    holdout_fraction: float = 0.2
    seed: int = 0

    def validate(self) -> None:
        if not 0 < self.holdout_fraction < 1:
            raise ValueError("holdout_fraction must be in (0, 1).")

@dataclass(frozen=True)
class MMDConfig:
    n_donors_per_type: int = 16
    k_cells_per_donor: int = 64
    min_cells_per_donor_type: int = 30
    min_donors_per_type: int = 4

    scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)  # используем иммутабельность туплы как страховку
    bandwidth_points: int = 512       # ограничиваем количество точек для выбора, стандарт в литературе
    bandwidth_floor: float = 1e-6

    def validate(self) -> None:
        if self.n_donors_per_type < self.min_donors_per_type:
            raise ValueError("n_donors_per_type below min_donors_per_type.")
        if self.k_cells_per_donor < 2:
            raise ValueError("k_cells_per_donor must be at least 2 for MMD.")
        if self.min_cells_per_donor_type < 1:
            raise ValueError("min_cells_per_donor_type must be positive.")
        if not self.scales or any(s <= 0 for s in self.scales):
            raise ValueError("scales must be non-empty and positive.")
        if self.bandwidth_points < 8:
            raise ValueError("bandwidth_points must be at least 8.")
        if self.bandwidth_floor <= 0:
            raise ValueError("bandwidth_floor must be positive.")

@dataclass(frozen=True)
class TrainConfig:
    warmup_epochs: int = 15
    mmd_epochs: int = 10
    elbo_batch_size: int = 512
    combined_batch: bool = True  # один форвард для ELBO и MMD
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 0
    lambda_mmd: float = 100.0
    save_at: tuple[int, ...] = ()  # in addition to warmup end and final
    save_warmup_checkpoint: bool = True
    device: str | None = None

    @property  # превращаем метод в атрибут
    def total_epochs(self) -> int:
        return self.warmup_epochs + self.mmd_epochs

    def validate(self) -> None:
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative.")
        if self.mmd_epochs < 1:
            raise ValueError("mmd_epochs must be positive.")
        if self.elbo_batch_size < 1:
            raise ValueError("elbo_batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.lambda_mmd < 0:
            raise ValueError("lambda_mmd cannot be negative.")

