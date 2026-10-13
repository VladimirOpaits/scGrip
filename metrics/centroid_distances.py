"""
Сравнение сжатия кластеров: warmup vs MMD.

Считает для каждого (cell_type):
  - силуэт по cell_type в латентах
  - среднее внутритиповое расстояние (компактность)
  - среднее расстояние до ближайшего другого типа (разделение)
  - отношение (разделение / компактность)
  - kNN accuracy по cell_type (насколько типы разделимы)

Сравнивает warmup vs MMD.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scvi
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


BASE = Path('/home/vlad/scGRIP')


def compute_latents(model_dir: Path, adata_path: Path) -> tuple[ad.AnnData, np.ndarray]:
    adata = ad.read_h5ad(adata_path)
    scvi.model.SCVI.setup_anndata(adata, batch_key='dataset_id', layer='counts')
    model = scvi.model.SCVI.load(str(model_dir), adata=adata)
    latents = model.get_latent_representation(adata)
    return adata, latents


def per_type_distances(latents: np.ndarray, labels: np.ndarray, cell_types: np.ndarray) -> pd.DataFrame:
    """Для каждого типа: compactness, separation, ratio."""
    rows = []
    for ct in np.unique(cell_types):
        mask = cell_types == ct
        z_ct = latents[mask]
        n = z_ct.shape[0]

        # Компактность: среднее попарное расстояние внутри типа
        if n > 1:
            # подвыборка если много клеток
            n_sub = min(n, 2000)
            idx = np.random.default_rng(0).choice(n, size=n_sub, replace=False)
            z_sub = z_ct[idx]
            # попарные расстояния
            diff = z_sub[:, None, :] - z_sub[None, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=-1))
            # исключить диагональ
            compactness = dists[np.triu_indices(n_sub, k=1)].mean()
        else:
            compactness = 0.0

        # Разделение: среднее расстояние от клеток этого типа до ближайших клеток другого типа
        other_mask = ~mask
        if other_mask.sum() > 0 and n > 0:
            n_sub = min(n, 1000)
            idx = np.random.default_rng(1).choice(n, size=n_sub, replace=False)
            z_self = z_ct[idx]

            n_other_sub = min(other_mask.sum(), 2000)
            idx_o = np.random.default_rng(2).choice(other_mask.sum(), size=n_other_sub, replace=False)
            z_other = latents[other_mask][idx_o]

            # для каждой z_self — расстояние до ближайшего z_other
            diff = z_self[:, None, :] - z_other[None, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=-1))
            separation = dists.min(axis=1).mean()
        else:
            separation = 0.0

        ratio = separation / compactness if compactness > 0 else 0.0

        rows.append({
            'cell_type': ct,
            'n_cells': int(n),
            'compactness': compactness,
            'separation': separation,
            'ratio': ratio,
        })
    return pd.DataFrame(rows)


def global_metrics(latents: np.ndarray, cell_types: np.ndarray) -> dict:
    """Global: silhouette, kNN accuracy по cell_type."""
    # silhouette (на подвыборке если много)
    n = latents.shape[0]
    if n > 10000:
        idx = np.random.default_rng(42).choice(n, size=10000, replace=False)
        z_sub = latents[idx]
        ct_sub = cell_types[idx]
    else:
        z_sub = latents
        ct_sub = cell_types

    sil = silhouette_score(z_sub, ct_sub)

    # kNN accuracy: 5-fold CV
    knn = KNeighborsClassifier(n_neighbors=15)
    accs = cross_val_score(knn, z_sub, ct_sub, cv=5, scoring='accuracy', n_jobs=1)

    return {
        'silhouette_celltype': float(sil),
        'knn_celltype_acc_mean': float(accs.mean()),
        'knn_celltype_acc_std': float(accs.std()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup-dir', type=Path, default=BASE / 'runs/warmup')
    parser.add_argument('--mmd-dir', type=Path, default=BASE / 'runs/mmd/best')
    parser.add_argument('--adata', type=Path, default=BASE / 'data/bank/healthy_only.h5ad')
    parser.add_argument('--out', type=Path, default=BASE / 'runs/analysis/compare_warmup_mmd')
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading warmup latents...")
    print("=" * 60)
    adata_w, lat_w = compute_latents(args.warmup_dir, args.adata)
    print(f"  Latents: {lat_w.shape}")

    print("\n" + "=" * 60)
    print("Loading MMD latents...")
    print("=" * 60)
    adata_m, lat_m = compute_latents(args.mmd_dir, args.adata)
    print(f"  Latents: {lat_m.shape}")

    cell_types = adata_w.obs['cell_type'].values
    assert (adata_w.obs['cell_type'].values == adata_m.obs['cell_type'].values).all()

    # ===== Per-type =====
    print("\n" + "=" * 60)
    print("Per-type distances (WARMUP)")
    print("=" * 60)
    df_w = per_type_distances(lat_w, None, cell_types)
    print(df_w.round(3).to_string(index=False))

    print("\n" + "=" * 60)
    print("Per-type distances (MMD)")
    print("=" * 60)
    df_m = per_type_distances(lat_m, None, cell_types)
    print(df_m.round(3).to_string(index=False))

    # ===== Merge =====
    df_cmp = df_w.merge(df_m, on='cell_type', suffixes=('_warmup', '_mmd'))
    df_cmp['compactness_ratio'] = df_cmp['compactness_mmd'] / df_cmp['compactness_warmup']
    df_cmp['separation_ratio'] = df_cmp['separation_mmd'] / df_cmp['separation_warmup']
    df_cmp['ratio_improvement'] = df_cmp['ratio_mmd'] / df_cmp['ratio_warmup']

    print("\n" + "=" * 60)
    print("COMPARISON (warmup vs MMD)")
    print("=" * 60)
    print(df_cmp[[
        'cell_type',
        'compactness_warmup', 'compactness_mmd', 'compactness_ratio',
        'separation_warmup', 'separation_mmd', 'separation_ratio',
        'ratio_warmup', 'ratio_mmd', 'ratio_improvement',
    ]].round(3).to_string(index=False))

    # ===== Global =====
    print("\n" + "=" * 60)
    print("Global metrics")
    print("=" * 60)
    g_w = global_metrics(lat_w, cell_types)
    g_m = global_metrics(lat_m, cell_types)
    print(f"{'metric':<30} {'warmup':>10} {'mmd':>10} {'Δ':>10}")
    print("-" * 62)
    for key in g_w:
        w, m = g_w[key], g_m[key]
        print(f"{key:<30} {w:>10.4f} {m:>10.4f} {m - w:>+10.4f}")

    # ===== Save =====
    df_cmp.to_csv(args.out / 'per_type_comparison.csv', index=False)
    pd.Series({**{f'warmup_{k}': v for k, v in g_w.items()},
               **{f'mmd_{k}': v for k, v in g_m.items()}}).to_csv(args.out / 'global_comparison.csv')
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()