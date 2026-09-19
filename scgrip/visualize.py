"""Shared-coordinate visualisation for phase-I checkpoint comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _balanced_indices(metadata: pd.DataFrame, max_cells: int, seed: int) -> np.ndarray:
    if len(metadata) <= max_cells:
        return np.arange(len(metadata), dtype=int)
    groups = metadata.groupby(["phase1_group", "stratum"], observed=True, sort=True).indices
    rng = np.random.default_rng(seed)
    per_group = max(1, max_cells // len(groups))
    selected = [rng.choice(rows, size=min(per_group, len(rows)), replace=False) for rows in groups.values()]
    selected = np.concatenate(selected)
    if selected.size < max_cells:
        unselected = np.ones(len(metadata), dtype=bool)
        unselected[selected] = False
        remaining = np.flatnonzero(unselected)
        selected = np.concatenate(
            (selected, rng.choice(remaining, size=min(max_cells - selected.size, remaining.size), replace=False))
        )
    return rng.permutation(selected)


def _categorical_plot(points: pd.DataFrame, category: str, states: list[str], path: Path) -> None:
    labels = sorted(points[category].astype(str).unique())
    color_map = dict(zip(labels, plt.get_cmap("tab20")(np.arange(len(labels)) % 20)))
    figure, axes = plt.subplots(1, len(states), figsize=(5 * len(states), 4), squeeze=False)
    handles = []
    xlim = (points["umap1"].min(), points["umap1"].max())
    ylim = (points["umap2"].min(), points["umap2"].max())
    for axis, state in zip(axes.flat, states):
        subset = points[points["state"] == state]
        for label in labels:
            data = subset[subset[category].astype(str) == label]
            handle = axis.scatter(
                data["umap1"], data["umap2"], s=2, alpha=0.55,
                color=color_map[label], rasterized=True, label=label,
            )
            if state == states[0]:
                handles.append(handle)
        axis.set_title(state)
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), markerscale=3)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _continuous_plot(points: pd.DataFrame, value: str, states: list[str], path: Path) -> None:
    lower, upper = points[value].quantile([0.01, 0.99])
    figure, axes = plt.subplots(1, len(states), figsize=(5 * len(states), 4), squeeze=False)
    scatter = None
    for axis, state in zip(axes.flat, states):
        subset = points[points["state"] == state]
        scatter = axis.scatter(
            subset["umap1"], subset["umap2"], c=subset[value], cmap="viridis",
            vmin=lower, vmax=upper, s=2, alpha=0.6, rasterized=True,
        )
        axis.set_title(state)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(scatter, ax=axes.ravel().tolist(), label=value)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_shared_umap(
    embeddings: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    out_dir: str | Path,
    *,
    max_fit_cells: int = 20_000,
    n_neighbors: int = 20,
    min_dist: float = 0.3,
    seed: int = 0,
    continuous: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Fit one UMAP mapper to an equal state sample and transform every state.

    Coordinates are a visual diagnostic, not a numerical selection criterion.
    The same mapper is used for all states, so a movement in a panel is not an
    artefact of independently refitting UMAP.
    """

    import umap

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    states = list(embeddings)
    if not states:
        raise ValueError("At least one embedding state is required for UMAP.")
    n_obs = len(metadata)
    if any(np.asarray(value).shape[0] != n_obs for value in embeddings.values()):
        raise ValueError("Embedding rows must align with metadata rows.")
    sample = _balanced_indices(metadata, max_fit_cells, seed)
    fit_values = np.vstack([np.asarray(embeddings[state])[sample] for state in states])
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
        transform_seed=seed,
    )
    reducer.fit(fit_values)
    frames: list[pd.DataFrame] = []
    for state in states:
        xy = reducer.transform(np.asarray(embeddings[state]))
        frame = metadata.copy().reset_index(drop=True)
        frame.insert(0, "state", state)
        frame["umap1"] = xy[:, 0]
        frame["umap2"] = xy[:, 1]
        if continuous and state in continuous:
            frame["signature_score"] = np.asarray(continuous[state])
        frames.append(frame)
    points = pd.concat(frames, ignore_index=True)
    points.to_csv(destination / "shared_umap_coordinates.csv", index=False)
    for category in ("stratum", "disease", "phase1_group"):
        if category in points and points[category].nunique() > 1:
            _categorical_plot(points, category, states, destination / f"umap_{category}.png")
    if "signature_score" in points:
        _continuous_plot(points, "signature_score", states, destination / "umap_signature_score.png")
    return points
