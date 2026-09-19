"""Optional, support-aware scIB readouts for phase-I embeddings.

These are reporting metrics only. They are intentionally separate from
``phase1.py`` so no disease or visual metric can leak into checkpoint selection.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def balanced_sample(
    frame: pd.DataFrame,
    columns: list[str],
    max_cells: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fixed-stratum sample that does not let abundant types dominate scIB."""

    if len(frame) <= max_cells:
        return np.arange(len(frame), dtype=int)
    groups = frame.groupby(columns, observed=True, sort=True).indices
    per_group = max(1, max_cells // len(groups))
    selected = [rng.choice(indices, size=min(per_group, len(indices)), replace=False) for indices in groups.values()]
    chosen = np.concatenate(selected)
    if chosen.size < max_cells:
        unseen = np.ones(len(frame), dtype=bool)
        unseen[chosen] = False
        remaining = np.flatnonzero(unseen)
        chosen = np.concatenate(
            (chosen, rng.choice(remaining, size=min(max_cells - chosen.size, remaining.size), replace=False))
        )
    return rng.permutation(chosen)


def _labels_with_support(
    obs: pd.DataFrame,
    label_key: str,
    batch_key: str | None = None,
    min_cells: int = 10,
) -> np.ndarray:
    counts = obs[label_key].astype(str).value_counts()
    labels = counts.index[counts >= min_cells]
    if batch_key is not None:
        by_label = obs.assign(**{label_key: obs[label_key].astype(str)}).groupby(label_key, observed=True)[batch_key].nunique()
        labels = labels.intersection(by_label.index[by_label >= 2])
    return obs[label_key].astype(str).isin(labels).to_numpy()


def _metric_adata(embedding: np.ndarray, obs: pd.DataFrame):
    import anndata as ad

    result = ad.AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs.reset_index(drop=True).copy())
    result.obsm["X_latent"] = np.asarray(embedding, dtype=np.float32)
    return result


def _safe(name: str, callback) -> float:
    try:
        return float(callback())
    except Exception as error:  # scIB has legitimate support-dependent failures
        warnings.warn(f"scIB {name} unavailable: {error}", stacklevel=2)
        return float("nan")


def scib_embedding_metrics(
    *,
    query_latent: np.ndarray,
    query_strata: np.ndarray,
    healthy_a: np.ndarray,
    healthy_b: np.ndarray,
    reference_metric_latent: np.ndarray,
    reference_metric_strata: np.ndarray,
    technical_batch: np.ndarray | None,
    max_cells: int = 30_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Compute comparable biology/healthy-mixing ASW metrics.

    ``technical_batch`` is deliberately optional: a donor ID is not a useful
    technical batch proxy. For RA, pass ``Lane`` explicitly; for a study with
    no trustworthy technical variable, the batch ASW stays NaN instead of
    silently answering a different question.
    """

    import scib

    z = np.asarray(query_latent, dtype=np.float32)
    strata = np.asarray(query_strata, dtype=str)
    if z.shape[0] != strata.size:
        raise ValueError("query_latent and query_strata do not align.")
    rng = np.random.default_rng(seed)
    query_obs = pd.DataFrame({"stratum": strata})
    all_sample = balanced_sample(query_obs, ["stratum"], max_cells, rng)
    all_obs = query_obs.iloc[all_sample].reset_index(drop=True)
    all_keep = _labels_with_support(all_obs, "stratum")
    out: dict[str, float | int] = {
        "scib_n_query_cells": int(all_sample.size),
        "scib_stratum_asw": _safe(
            "stratum ASW",
            lambda: scib.metrics.silhouette(
                _metric_adata(z[all_sample][all_keep], all_obs.iloc[np.flatnonzero(all_keep)]),
                "stratum",
                "X_latent",
            ),
        ),
    }

    if technical_batch is None:
        out["scib_batch_healthy_A_asw"] = float("nan")
        out["scib_batch_healthy_B_asw"] = float("nan")
    else:
        batches = np.asarray(technical_batch, dtype=str)
        if batches.size != z.shape[0]:
            raise ValueError("technical_batch and query_latent do not align.")
        for name, positions in (("A", healthy_a), ("B", healthy_b)):
            obs = pd.DataFrame({"stratum": strata[positions], "batch": batches[positions]})
            sample = balanced_sample(obs, ["stratum", "batch"], max_cells, rng)
            obs = obs.iloc[sample].reset_index(drop=True)
            keep = _labels_with_support(obs, "stratum", "batch")
            out[f"scib_batch_healthy_{name}_asw"] = _safe(
                f"healthy {name} batch ASW",
                lambda obs=obs, keep=keep, sample=sample, positions=positions: scib.metrics.silhouette_batch(
                    _metric_adata(z[positions][sample][keep], obs.iloc[np.flatnonzero(keep)]),
                    batch_key="batch",
                    label_key="stratum",
                    embed="X_latent",
                    verbose=False,
                ),
            )

    ref_z = np.asarray(reference_metric_latent, dtype=np.float32)
    ref_strata = np.asarray(reference_metric_strata, dtype=str)
    for name, positions in (("A", healthy_a), ("B", healthy_b)):
        block_obs = pd.DataFrame(
            {
                "stratum": np.concatenate((ref_strata, strata[positions])),
                "block": np.concatenate(
                    (np.repeat("reference", ref_z.shape[0]), np.repeat("query", positions.size))
                ),
            }
        )
        block_z = np.vstack((ref_z, z[positions]))
        sample = balanced_sample(block_obs, ["block", "stratum"], max_cells, rng)
        block_obs = block_obs.iloc[sample].reset_index(drop=True)
        block_z = block_z[sample]
        keep = _labels_with_support(block_obs, "stratum", "block")
        out[f"scib_block_healthy_{name}_asw"] = _safe(
            f"reference/query healthy {name} block ASW",
            lambda block_z=block_z, block_obs=block_obs, keep=keep: scib.metrics.silhouette_batch(
                _metric_adata(block_z[keep], block_obs.iloc[np.flatnonzero(keep)]),
                batch_key="block",
                label_key="stratum",
                embed="X_latent",
                verbose=False,
            ),
        )
    return out
