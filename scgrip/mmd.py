"""Fixed-bandwidth, interpretable MMD² utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch


def fixed_multiscale_bandwidths(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    scales: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    max_points: int = 256,
) -> tuple[float, ...]:
    """Estimate a median distance once on frozen base latents.

    The returned values are ordinary Python floats.  They must be saved in the
    run manifest and reused for every training step; recalculating bandwidths
    from moving query latents creates an avoidable scale loophole.
    """

    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be two-dimensional latent matrices with equal width.")
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("scales must be non-empty and positive.")
    with torch.no_grad():
        pooled = torch.cat((x.detach(), y.detach()), dim=0)
        if pooled.shape[0] > max_points:
            generator = torch.Generator(device=pooled.device)
            generator.manual_seed(0)
            keep = torch.randperm(pooled.shape[0], generator=generator, device=pooled.device)[:max_points]
            pooled = pooled[keep]
        distances = torch.pdist(pooled, p=2)
        positive = distances[distances > 0]
        base = positive.median() if positive.numel() else torch.tensor(1.0, device=pooled.device)
        base = float(base.clamp_min(1e-6).cpu())
    return tuple(base * float(scale) for scale in scales)


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigmas: Sequence[float]) -> torch.Tensor:
    """Multi-scale RBF kernel matrix without silently changing its bandwidth."""

    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be two-dimensional latent matrices with equal width.")
    if not sigmas:
        raise ValueError("At least one RBF bandwidth is required.")
    x2 = (x.square()).sum(dim=1, keepdim=True)
    y2 = (y.square()).sum(dim=1, keepdim=True)
    distance2 = (x2 + y2.T - 2 * x @ y.T).clamp_min(0)
    result = torch.zeros_like(distance2)
    for sigma in sigmas:
        if sigma <= 0:
            raise ValueError("RBF bandwidths must be positive.")
        result = result + torch.exp(-distance2 / (2.0 * float(sigma) ** 2))
    return result / len(sigmas)


def mmd2_biased(x: torch.Tensor, y: torch.Tensor, sigmas: Sequence[float]) -> torch.Tensor:
    """Full, biased MMD²: Kxx + Kyy - 2Kxy.

    Kyy is constant with respect to a frozen reference target, but retaining it
    makes logs non-negative/interpretable and avoids calling a surrogate MMD².
    """

    k_xx = rbf_kernel(x, x, sigmas).mean()
    k_yy = rbf_kernel(y, y, sigmas).mean()
    k_xy = rbf_kernel(x, y, sigmas).mean()
    return k_xx + k_yy - 2.0 * k_xy


def mmd2_unbiased(x: torch.Tensor, y: torch.Tensor, sigmas: Sequence[float]) -> torch.Tensor:
    """Unbiased MMD² estimator for reporting; needs two cells on each side."""

    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError("Unbiased MMD² needs at least two cells on both sides.")
    k_xx = rbf_kernel(x, x, sigmas)
    k_yy = rbf_kernel(y, y, sigmas)
    k_xy = rbf_kernel(x, y, sigmas)
    x_term = (k_xx.sum() - k_xx.diagonal().sum()) / (x.shape[0] * (x.shape[0] - 1))
    y_term = (k_yy.sum() - k_yy.diagonal().sum()) / (y.shape[0] * (y.shape[0] - 1))
    return x_term + y_term - 2.0 * k_xy.mean()


@dataclass(frozen=True)
class MMDStratum:
    """One balanced source/target stratum in a phase-I MMD objective."""

    name: str
    source_positions: np.ndarray
    target_latents: np.ndarray
    bandwidths: tuple[float, ...]

    @property
    def source_size(self) -> int:
        return int(self.source_positions.size)

    @property
    def target_size(self) -> int:
        return int(self.target_latents.shape[0])


@dataclass(frozen=True)
class MMDDraw:
    """A single balanced multi-stratum MMD minibatch."""

    source_positions: np.ndarray
    target_indices: dict[str, np.ndarray]
    sizes: dict[str, int]


@dataclass(frozen=True)
class StratifiedMMDPlan:
    """Frozen target bank, support, and fixed bandwidths for MMD training."""

    strata: tuple[MMDStratum, ...]
    k_per_stratum: int

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.strata)

    def manifest(self) -> dict[str, object]:
        return {
            "objective": "balanced_stratum_mean_of_full_biased_mmd2",
            "k_per_stratum": self.k_per_stratum,
            "strata": {
                item.name: {
                    "n_query_healthy_A": item.source_size,
                    "n_reference_metric": item.target_size,
                    "fixed_bandwidths": list(item.bandwidths),
                }
                for item in self.strata
            },
        }

    def draw(self, rng: np.random.Generator) -> MMDDraw:
        """Choose equal-weight per-stratum MMD samples without replacement."""

        source_parts: list[np.ndarray] = []
        target_indices: dict[str, np.ndarray] = {}
        sizes: dict[str, int] = {}
        for item in self.strata:
            n = min(self.k_per_stratum, item.source_size, item.target_size)
            if n < 2:
                raise RuntimeError(f"MMD stratum {item.name!r} no longer has two cells per side.")
            source_parts.append(rng.choice(item.source_positions, size=n, replace=False))
            target_indices[item.name] = rng.choice(item.target_size, size=n, replace=False)
            sizes[item.name] = n
        if not source_parts:
            raise RuntimeError("Cannot draw MMD batch from an empty plan.")
        return MMDDraw(
            source_positions=np.concatenate(source_parts).astype(int, copy=False),
            target_indices=target_indices,
            sizes=sizes,
        )


def build_stratified_mmd_plan(
    *,
    metric_latents: np.ndarray,
    metric_strata,
    query_base_latents: np.ndarray,
    query_strata,
    query_positions: np.ndarray,
    min_per_side: int,
    k_per_stratum: int,
    scales: Sequence[float],
    bandwidth_points: int,
) -> StratifiedMMDPlan:
    """Build a plan whose target and kernel scales cannot move during fitting.

    ``metric_latents`` are from the independent frozen reference bank. Query
    base latents are measured exactly once before any update. The plan contains
    **only healthy A** positions; replay is intentionally not an MMD side.
    """

    metric = np.asarray(metric_latents, dtype=np.float32)
    query = np.asarray(query_base_latents, dtype=np.float32)
    metric_labels = np.asarray(metric_strata, dtype=str)
    query_labels = np.asarray(query_strata, dtype=str)
    query_positions = np.asarray(query_positions, dtype=int)
    if metric.ndim != 2 or query.ndim != 2 or metric.shape[1] != query.shape[1]:
        raise ValueError("MMD metric/query latent matrices must be 2D with equal latent width.")
    if metric.shape[0] != metric_labels.size or query.shape[0] != query_labels.size:
        raise ValueError("MMD labels do not align with latent rows.")
    if query.shape[0] != query_positions.size:
        raise ValueError("MMD query positions do not align with query latent rows.")
    if min_per_side < 2 or k_per_stratum < 2:
        raise ValueError("MMD requires at least two cells per side and per draw.")

    strata: list[MMDStratum] = []
    shared = sorted(set(metric_labels).intersection(query_labels))
    for name in shared:
        metric_pos = np.flatnonzero(metric_labels == name)
        query_pos = np.flatnonzero(query_labels == name)
        if metric_pos.size < min_per_side or query_pos.size < min_per_side:
            continue
        bandwidths = fixed_multiscale_bandwidths(
            torch.as_tensor(query[query_pos]),
            torch.as_tensor(metric[metric_pos]),
            scales=scales,
            max_points=bandwidth_points,
        )
        strata.append(
            MMDStratum(
                name=str(name),
                source_positions=query_positions[query_pos],
                target_latents=metric[metric_pos].copy(),
                bandwidths=bandwidths,
            )
        )
    return StratifiedMMDPlan(strata=tuple(strata), k_per_stratum=k_per_stratum)


def mmd_loss_from_draw(
    source_latents: torch.Tensor,
    draw: MMDDraw,
    plan: StratifiedMMDPlan,
    *,
    target_tensors: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Calculate equal-stratum full MMD² for a latent batch in draw order."""

    if source_latents.ndim != 2:
        raise ValueError("MMD source latents must be a two-dimensional matrix.")
    terms: list[torch.Tensor] = []
    offset = 0
    for item in plan.strata:
        n = draw.sizes[item.name]
        source = source_latents[offset : offset + n]
        target = target_tensors[item.name][
            torch.as_tensor(draw.target_indices[item.name], device=source_latents.device)
        ]
        terms.append(mmd2_biased(source, target, item.bandwidths))
        offset += n
    if offset != source_latents.shape[0]:
        raise AssertionError("MMD draw/source latent ordering mismatch.")
    return torch.stack(terms).mean()


def target_tensors_on_device(
    plan: StratifiedMMDPlan,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Transfer only the frozen MMD metric bank once, never every training step."""

    return {
        item.name: torch.as_tensor(item.target_latents, dtype=torch.float32, device=device)
        for item in plan.strata
    }
