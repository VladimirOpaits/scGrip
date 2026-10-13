import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


MODELS = {
    'warmup':      Path('/home/vlad/scGRIP/runs/warmup_clean'),
    'mmd_best06':  Path('/home/vlad/scGRIP/runs/mmd_clean_lambda10_v2/exported'),
    'mmd_last':    Path('/home/vlad/scGRIP/runs/mmd_clean_lambda10_v2/exported_last'),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--synthetic-dir', type=Path, default=Path('/bank/synthetic'))
    p.add_argument('--out-dir', type=Path, default=Path('/bank/synthetic/eval'))
    p.add_argument('--k', type=int, default=30)
    p.add_argument('--ref-alpha', type=float, default=0.0)
    return p.parse_args()


def load_synthetic(synth_dir, ref_alpha):
    files = sorted(synth_dir.glob('synthetic_alpha_*.h5ad'))
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  {f.name}")

    parts = []
    for f in files:
        a = sc.read_h5ad(f)
        alpha = float(f.stem.replace('synthetic_alpha_', ''))
        a.obs['alpha'] = alpha
        if 'counts' not in a.layers:
            a.layers['counts'] = a.X.copy()
        parts.append(a)

    adata = ad.concat(parts, join='inner', index_unique='_')
    adata.obs['alpha'] = adata.obs['alpha'].astype(float)
    adata.obs['is_reference'] = (adata.obs['alpha'] == ref_alpha).values

    print(f"Combined: {adata.n_obs:,} cells")
    print(f"Alphas: {sorted(adata.obs['alpha'].unique())}")
    print(f"Reference (alpha={ref_alpha}): {adata.obs['is_reference'].sum():,} cells")
    return adata


def get_latents(model_dir, adata, model_name):
    a = adata.copy()

    scvi.model.SCVI.setup_anndata(a, batch_key='dataset_id', layer='counts')

    try:
        model = scvi.model.SCVI.load(str(model_dir), adata=a)
    except Exception as e:
        print(f"  load failed: {e}")
        print(f"  trying to align dataset_id categories...")
        model_tmp = scvi.model.SCVI.load(str(model_dir))
        cats = model_tmp.adata_manager.get_state_registry('batch').categorical_mapping
        print(f"  model categories: {[str(c) for c in cats]}")
        print(f"  synthetic categories: {sorted(a.obs['dataset_id'].unique().tolist())}")

        a.obs['dataset_id'] = pd.Categorical(
            a.obs['dataset_id'].astype(str),
            categories=[str(c) for c in cats]
        )
        n_missing = int(a.obs['dataset_id'].isna().sum())
        if n_missing > 0:
            print(f"  WARNING: {n_missing} cells have unseen dataset_id")

        scvi.model.SCVI.setup_anndata(a, batch_key='dataset_id', layer='counts')
        model = scvi.model.SCVI.load(str(model_dir), adata=a)

    latents = model.get_latent_representation(a)
    return latents


def knn_kth_distance(h_lat, q_lat, k):
    import faiss
    h_lat = h_lat.astype(np.float32)
    q_lat = q_lat.astype(np.float32)
    index = faiss.IndexFlatL2(h_lat.shape[1])
    index.add(h_lat)
    D_q, _ = index.search(q_lat, k)
    query_delta = np.sqrt(D_q[:, k - 1])
    D_h, _ = index.search(h_lat, k + 1)
    null_delta = np.sqrt(D_h[:, k])
    return query_delta, null_delta


def compute_scpss(latents, cell_types, reference_mask, k=30):
    p_values = np.full(len(latents), np.nan, dtype=np.float32)

    for ct in tqdm(np.unique(cell_types), desc='  scPSS', leave=False):
        ct_mask = cell_types == ct
        ref_mask = ct_mask & reference_mask

        if ref_mask.sum() < k + 1:
            continue

        h_lat = latents[ref_mask]
        all_lat = latents[ct_mask]

        query_delta, null_delta = knn_kth_distance(h_lat, all_lat, k=k)

        sorted_null = np.sort(null_delta)
        n_null = len(sorted_null)
        counts = n_null - np.searchsorted(sorted_null, query_delta, side='left')
        p = counts / n_null
        p = np.clip(p, 1.0 / (n_null + 1), 1.0)
        p_values[ct_mask] = p

    return p_values


def compute_auc(p_vals, cell_types, alphas, ref_alpha=0.0):
    rows = []
    for ct in np.unique(cell_types):
        ct_mask = cell_types == ct
        ref_mask = ct_mask & (alphas == ref_alpha)

        for a in np.unique(alphas):
            if a == ref_alpha:
                continue
            stim_mask = ct_mask & (alphas == a)

            n_ref = int(ref_mask.sum())
            n_stim = int(stim_mask.sum())
            if n_ref < 10 or n_stim < 10:
                continue

            labels = np.concatenate([np.zeros(n_ref), np.ones(n_stim)])
            scores = np.concatenate([-p_vals[ref_mask], -p_vals[stim_mask]])

            valid = ~np.isnan(scores)
            if valid.sum() < 20 or len(np.unique(labels[valid])) < 2:
                continue

            auc = roc_auc_score(labels[valid], scores[valid])
            rows.append({
                'cell_type': ct,
                'alpha': a,
                'auc': auc,
                'n_ref': n_ref,
                'n_stim': n_stim,
            })

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    adata = load_synthetic(args.synthetic_dir, ref_alpha=args.ref_alpha)

    cell_types = adata.obs['cell_type'].values
    alphas = adata.obs['alpha'].values
    reference_mask = adata.obs['is_reference'].values

    all_results = []

    for model_name, model_dir in MODELS.items():
        if not model_dir.exists():
            print(f"\nSkip {model_name}: {model_dir} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_name}  ({model_dir})")
        print(f"{'='*60}")

        try:
            latents = get_latents(model_dir, adata, model_name)
            print(f"Latents: {latents.shape}")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

        p_vals = compute_scpss(latents, cell_types, reference_mask, k=args.k)
        valid = ~np.isnan(p_vals)
        print(f"Valid p-values: {valid.sum():,} / {len(p_vals):,}")

        # sanity: mean p-value for reference should be ~0.5
        ref_p = p_vals[reference_mask & valid]
        print(f"Mean p-value @ alpha=0 (should be ~0.5): {ref_p.mean():.4f}")

        df = compute_auc(p_vals, cell_types, alphas, args.ref_alpha)
        df['model'] = model_name
        all_results.append(df)

        print(f"\nMean AUC by alpha:")
        for a in sorted(df['alpha'].unique()):
            m = df['alpha'] == a
            print(f"  alpha={a:.2f}: mean={df.loc[m, 'auc'].mean():.4f} (n_types={m.sum()})")

        df.to_csv(args.out_dir / f'auc_{model_name}.csv', index=False)
        del latents, p_vals

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(args.out_dir / 'auc_all_models.csv', index=False)

        print(f"\n{'='*60}")
        print("SUMMARY: mean AUC per model per alpha")
        print(f"{'='*60}")
        pivot = combined.pivot_table(index='model', columns='alpha', values='auc', aggfunc='mean')
        print(pivot.round(4))

        max_alpha = combined['alpha'].max()
        print(f"\n{'='*60}")
        print(f"Per cell type @ alpha={max_alpha}")
        print(f"{'='*60}")
        sub = combined[combined['alpha'] == max_alpha]
        pivot2 = sub.pivot_table(index='cell_type', columns='model', values='auc')
        print(pivot2.round(4))

        print(f"\nSaved: {args.out_dir}")
    else:
        print("\nNo results.")


if __name__ == '__main__':
    main()