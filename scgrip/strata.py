"""Tissue-agnostic stratified geometry for reference-mapping diagnostics.

There is intentionally no PBMC keyword map here. A caller supplies a stable
stratum label (ideally a Cell Ontology ID or harmonised cell-type label), and
the metric reports support instead of pretending that absent strata are zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def normalise_strata(values, *, name: str = "stratum") -> np.ndarray:
    """Convert labels to strings while rejecting missing/empty annotations."""

    series = pd.Series(values)
    if series.isna().any():
        raise ValueError(f"{name} contains missing labels.")
    out = series.astype(str).str.strip().to_numpy()
    bad = np.isin(np.char.lower(out.astype(str)), ["", "nan", "none", "unknown"])
    if bad.any():
        examples = np.unique(out[bad]).tolist()[:10]
        raise ValueError(f"{name} contains empty/unknown labels: {examples}")
    return out


def _latent_array(latents: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(latents, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional latent matrix.")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return value


@dataclass(frozen=True)
class ReferenceGeometry:
    """Global all-cell reference centres and fixed per-latent-axis scales."""

    centroids: pd.DataFrame
    sigma: np.ndarray
    cell_support: pd.Series

    def validate(self) -> None:
        if self.centroids.empty:
            raise ValueError("Reference geometry has no supported strata.")
        if self.centroids.shape[1] != self.sigma.size:
            raise ValueError("Reference centroid width and sigma length differ.")
        if not np.isfinite(self.centroids.to_numpy()).all() or not np.isfinite(self.sigma).all():
            raise ValueError("Reference geometry contains non-finite values.")
        if (self.sigma <= 0).any():
            raise ValueError("Reference latent-axis scales must be strictly positive.")


def reference_geometry(
    latents: np.ndarray,
    strata,
    *,
    min_cells_per_stratum: int,
    scale_floor: float = 1e-6,
) -> ReferenceGeometry:
    """Build one all-cell centre per stratum, never a donor-of-donor average.

    The global centre answers the desired question directly: where are all
    healthy monocytes (or any supplied stratum) in the independent reference
    bank? Query donor centres are then compared to that fixed centre. Donor
    counts remain visible in the query report rather than altering the target.
    """

    z = _latent_array(latents, name="reference latents")
    labels = normalise_strata(strata, name="reference stratum")
    if labels.size != z.shape[0]:
        raise ValueError("reference strata and latent rows differ.")
    if min_cells_per_stratum < 1:
        raise ValueError("min_cells_per_stratum must be positive.")
    if scale_floor <= 0:
        raise ValueError("scale_floor must be positive.")

    dimensions = [f"z{i}" for i in range(z.shape[1])]
    frame = pd.DataFrame(z, columns=dimensions)
    frame["stratum"] = labels
    support = frame.groupby("stratum", observed=True).size().astype(int).sort_index()
    keep = support.index[support >= min_cells_per_stratum]
    centroids = frame.groupby("stratum", observed=True)[dimensions].mean().loc[keep]
    sigma = np.maximum(z.std(axis=0, ddof=0), scale_floor)
    result = ReferenceGeometry(
        centroids=centroids,
        sigma=sigma,
        cell_support=support.loc[keep],
    )
    result.validate()
    return result


@dataclass(frozen=True)
class StratifiedReport:
    """Donor-to-reference distances plus decomposed per-stratum diagnostics."""

    distances: pd.DataFrame
    by_stratum: pd.DataFrame

    @property
    def supported_strata(self) -> pd.Index:
        return self.by_stratum.index[self.by_stratum["n_donors"] > 0]


def stratified_report(
    latents: np.ndarray,
    strata,
    donors,
    geometry: ReferenceGeometry,
    *,
    min_cells_per_donor_stratum: int,
) -> StratifiedReport:
    """Compare each eligible query donor/stratum centre to a global reference.

    ``mean_distance`` is donor-balanced. ``block_offset`` tells whether a group
    mean moved toward the reference; ``donor_spread`` distinguishes real
    centring from merely shrinking the donor cloud. This prevents a single
    scalar from hiding a degenerate integration result.
    """

    geometry.validate()
    z = _latent_array(latents, name="query latents")
    labels = normalise_strata(strata, name="query stratum")
    donor_values = pd.Series(donors)
    if donor_values.isna().any():
        raise ValueError("query donor contains missing labels.")
    donor_values = donor_values.astype(str).to_numpy()
    if labels.size != z.shape[0] or donor_values.size != z.shape[0]:
        raise ValueError("query latents, strata, and donors must have equal length.")
    if min_cells_per_donor_stratum < 1:
        raise ValueError("min_cells_per_donor_stratum must be positive.")

    ref_strata = geometry.centroids.index.astype(str)
    dimensions = geometry.centroids.columns.tolist()
    frame = pd.DataFrame(z, columns=dimensions)
    frame["stratum"] = labels
    frame["donor"] = donor_values
    frame = frame[frame["stratum"].isin(ref_strata)]
    donor_index = pd.Index(np.sort(np.unique(donor_values)), name="donor")
    distances = pd.DataFrame(np.nan, index=donor_index, columns=ref_strata, dtype=float)
    summary = pd.DataFrame(
        {
            "mean_distance": np.nan,
            "block_offset": np.nan,
            "donor_spread": np.nan,
            "n_donors": 0,
            "n_cells": 0,
        },
        index=ref_strata,
    )
    if frame.empty:
        return StratifiedReport(distances=distances, by_stratum=summary)

    grouped = frame.groupby(["donor", "stratum"], observed=True, sort=True)
    counts = grouped.size()
    centres = grouped[dimensions].mean()

    for stratum in ref_strata:
        try:
            count_for_stratum = counts.xs(stratum, level="stratum")
            centre_for_stratum = centres.xs(stratum, level="stratum")
        except KeyError:
            continue
        valid = count_for_stratum >= min_cells_per_donor_stratum
        if not valid.any():
            continue
        centre_for_stratum = centre_for_stratum.loc[valid]
        counts_for_valid = count_for_stratum.loc[valid]
        reference = geometry.centroids.loc[stratum].to_numpy(dtype=float)
        scaled = (centre_for_stratum.to_numpy(dtype=float) - reference) / geometry.sigma
        per_donor = np.linalg.norm(scaled, axis=1)
        distances.loc[centre_for_stratum.index, stratum] = per_donor
        group_centre = centre_for_stratum.to_numpy(dtype=float).mean(axis=0)
        donor_spread = np.linalg.norm(
            (centre_for_stratum.to_numpy(dtype=float) - group_centre) / geometry.sigma,
            axis=1,
        )
        summary.loc[stratum] = {
            "mean_distance": float(per_donor.mean()),
            "block_offset": float(np.linalg.norm((group_centre - reference) / geometry.sigma)),
            "donor_spread": float(donor_spread.mean()),
            "n_donors": int(valid.sum()),
            "n_cells": int(counts_for_valid.sum()),
        }
    return StratifiedReport(distances=distances, by_stratum=summary)


def compare_stratified_groups(
    group_a: StratifiedReport,
    group_b: StratifiedReport,
) -> tuple[dict[str, float | int], list[str]]:
    """Produce A/B scalar summaries over exactly the same supported strata."""

    a = group_a.by_stratum
    b = group_b.by_stratum
    if not a.index.equals(b.index):
        raise ValueError("A and B reports do not use the same reference strata.")
    common = a.index[(a["n_donors"] > 0) & (b["n_donors"] > 0)].tolist()
    if not common:
        return {
            "strat_A": float("nan"),
            "strat_B": float("nan"),
            "strat_A_block_offset": float("nan"),
            "strat_B_block_offset": float("nan"),
            "strat_common_n": 0,
        }, []
    return (
        {
            "strat_A": float(a.loc[common, "mean_distance"].mean()),
            "strat_B": float(b.loc[common, "mean_distance"].mean()),
            "strat_A_block_offset": float(a.loc[common, "block_offset"].mean()),
            "strat_B_block_offset": float(b.loc[common, "block_offset"].mean()),
            "strat_common_n": int(len(common)),
        },
        [stratum for stratum in common],
    )


def format_stratified_reports(
    reports: dict[str, StratifiedReport],
    common_strata: list[str],
) -> str:
    """Human-readable support-aware one-line logging for the training loop."""

    pieces = [f"strat common={common_strata}"]
    for name, report in reports.items():
        values = []
        for stratum, row in report.by_stratum.iterrows():
            if int(row["n_donors"]) > 0:
                values.append(
                    f"{stratum}={row['mean_distance']:.3f}[donors={int(row['n_donors'])},cells={int(row['n_cells'])}]"
                )
        pieces.append(f"strat_{name}_by_stratum: " + (", ".join(values) or "none"))
    return " | ".join(pieces)
