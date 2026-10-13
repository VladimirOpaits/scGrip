import numpy as np
import pandas as pd
import torch
from typing import Sequence
from dataclasses import dataclass
from .data import BankData

def multiscale_rbf(sigmas: Sequence[float], coords_1: torch.Tensor, coords_2: torch.Tensor) -> torch.Tensor:
    if coords_1.ndim != 2 or coords_2.ndim != 2 or coords_1.shape[1] != coords_2.shape[1]: raise ValueError("Wrong coords dimetions")
    if not sigmas: raise ValueError("Wrong coords dimentions")
    
    x2 = coords_1.square().sum(dim=1, keepdim=True)
    y2 = coords_2.square().sum(dim=1, keepdim=True)
    sq_dist = (x2 + y2.T - 2.0 * coords_1 @ coords_2.T).clamp_min(0.0)
    kernel_sum = torch.zeros_like(sq_dist)
    for sigma in sigmas:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        kernel_sum = kernel_sum + torch.exp(-sq_dist / (2.0 * float(sigma) ** 2))
    return kernel_sum / len(sigmas)

def mmd2_biased(x: torch.Tensor, y: torch.Tensor, sigmas: Sequence[float]) -> torch.Tensor:  # biased, потому что небольшой сдвиг не влияет на тенденцию обучения, но позволяет нам иметь стабильно loss >= 0
    k_xx = multiscale_rbf(sigmas, x, x, ).mean()
    k_yy = multiscale_rbf(sigmas, y, y, ).mean()
    k_xy = multiscale_rbf(sigmas, x, y, ).mean()
    return k_xx + k_yy - 2.0 * k_xy

def _median_heuristic(sample: torch.Tensor, max_points: int, floor: float) -> float:
    if sample.shape[0] > max_points:
        idx = torch.randperm(sample.shape[0], device=sample.device)[:max_points]
        sample = sample[idx]
    if sample.shape[0] < 2:
        return floor
    dist = torch.pdist(sample, p=2)
    positive = dist[dist > 0]
    if positive.numel() == 0:
        return floor
    return float(positive.median().clamp_min(floor).item())

def compute_bandwidths_per_type(latents_by_type: dict[str, torch.Tensor],*,scales: Sequence[float],max_points: int,floor: float) -> dict[str, tuple[float, ...]]:
    # разные сигмы для разных типов клеток, потому что они в разных масштабах
    bandwidths: dict[str, tuple[float, ...]] = {}
    for cell_type, latents in latents_by_type.items():
        base = _median_heuristic(latents, max_points=max_points, floor=floor)
        bandwidths[cell_type] = tuple(base * float(s) for s in scales)
    return bandwidths    



@dataclass(frozen=True)
class MMDPlan:
    A_donors_by_type: dict[str, np.ndarray]
    bandwidths: dict[str, tuple[float, ...]]
    n_donors_per_type: int
    k_cells_per_donor: int
    
    @property
    def cell_types(self) -> tuple[str, ...]:
        return tuple(self.A_donors_by_type.keys())
    
    
@dataclass(frozen = False)
class MMDBatch:
    donor_cells: dict[str, dict[str, np.ndarray]]
    donors_by_type: dict[str, tuple[str, ...]]
    bandwidths_by_type: dict[str, tuple[float, ...]]
    k_cells: int
    
def _balanced_rest_sample(donor_cells: dict[str, np.ndarray], exclude_donor: str, k_cells: int, rng: np.random.Generator) -> np.ndarray:  # для честного MMD сэмплируем одинаковое количество клеток
    rest_donors = [d for d in donor_cells if d != exclude_donor]
    if not rest_donors:
        raise ValueError("no donors in rest")

    cells_per_donor = max(1, k_cells // len(rest_donors))
    parts: list[np.ndarray] = []
    for donor in rest_donors:
        donor_indices = donor_cells[donor]
        take = min(cells_per_donor, len(donor_indices))
        if take > 0:
            parts.append(rng.choice(donor_indices, size=take, replace=False))

    selected = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    if selected.size < k_cells:
        all_rest = np.concatenate([donor_cells[d] for d in rest_donors])
        used = np.isin(all_rest, selected)
        free = all_rest[~used]
        n_extra = min(k_cells - selected.size, free.size)
        if n_extra > 0:
            extra = rng.choice(free, size=n_extra, replace=False)
            selected = np.concatenate([selected, extra])

    return selected

def sample_mmd_batch(plan: MMDPlan, bank: BankData, rng) -> MMDBatch | None:
    donor_cells_by_type: dict[str, dict[str, np.ndarray]] = {}
    donors_by_type: dict[str, tuple[str, ...]] = {}   # порядок важен
    bandwidths_by_type: dict[str, tuple[float, ...]] = {}

    for cell_type in plan.cell_types:
        a_donors = plan.A_donors_by_type[cell_type]
        if a_donors.size < 2:  # хотя бы 2, что бы сравнивать
            continue

        n_take = min(plan.n_donors_per_type, a_donors.size)
        selected_donors = rng.choice(a_donors, size=n_take, replace=False)

        donor_cells: dict[str, np.ndarray] = {}
        for donor in selected_donors:
            donor = str(donor)
            all_cells = bank.cells_for(cell_type, donor)
            k = min(plan.k_cells_per_donor, all_cells.size)
            if k < 2:
                continue
            donor_cells[donor] = rng.choice(all_cells, size=k, replace=False)

        if len(donor_cells) < 2:
            continue

        donor_cells_by_type[cell_type] = donor_cells
        donors_by_type[cell_type] = tuple(donor_cells.keys())
        bandwidths_by_type[cell_type] = plan.bandwidths[cell_type]

    if not donor_cells_by_type:
        return None

    return MMDBatch(donor_cells=donor_cells_by_type, donors_by_type=donors_by_type, bandwidths_by_type=bandwidths_by_type, k_cells=plan.k_cells_per_donor)

def _each_vs_rest_mmd(donor_cells: dict[str, np.ndarray], latents: torch.Tensor, bandwidths: tuple[float, ...], k_cells: int, rng: np.random.Generator) -> torch.Tensor:
    donors = list(donor_cells.keys())
    terms: list[torch.Tensor] = []
    for donor in donors:
        donor_latents = latents[donor_cells[donor]]
        rest_indices = _balanced_rest_sample(donor_cells, donor, k_cells, rng)
        rest_latents = latents[rest_indices]
        terms.append(mmd2_biased(donor_latents, rest_latents, bandwidths))
    return torch.stack(terms).mean()

def mmd_loss_from_batch(latents, batch, bank, rng):
    del bank
    per_type_terms = []

    offset = 0
    for cell_type in batch.donor_cells:
        donor_cells = batch.donor_cells[cell_type]
        n_cells = sum(len(idx) for idx in donor_cells.values())
        local_latents = latents[offset : offset + n_cells]

        local_index = {}
        cursor = 0
        for donor, idx in donor_cells.items():
            local_index[donor] = np.arange(cursor, cursor + len(idx), dtype=np.int64)
            cursor += len(idx)

        bandwidths = batch.bandwidths_by_type[cell_type]

        term = _each_vs_rest_mmd(
            local_index, local_latents, bandwidths,
            k_cells=min(64, max(len(i) for i in local_index.values())),
            rng=rng,
        )

        per_type_terms.append(term)
        offset += n_cells

    if offset != latents.shape[0]:
        raise AssertionError("latents size does not match batch size")

    return torch.stack(per_type_terms).mean()