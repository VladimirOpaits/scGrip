from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse
import re
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scvi
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


BASE = Path('/home/vlad/scGRIP')
HEALTHY_TRAIN = BASE / 'data/bank/healthy_train.h5ad'
SYNTH_DIR = BASE / 'data/bank/synthetic'
OUT_DIR = BASE / 'runs/analysis/scpss_synthetic'

IFN_GENES = [
    'ADAR', 'BST2', 'CMPK2', 'DDX58', 'DDX60', 'EIF2AK2', 'HERC5', 'HERC6',
    'IFI27', 'IFI35', 'IFI44', 'IFI44L', 'IFIH1', 'IFIT1', 'IFIT2', 'IFIT3',
    'IFITM1', 'IFITM2', 'IFITM3', 'IRF7', 'ISG15', 'ISG20', 'LGALS3BP',
    'LY6E', 'MX1', 'MX2', 'OAS1', 'OAS2', 'OAS3', 'OASL', 'PARP9', 'PLSCR1',
    'RSAD2', 'RTP4', 'SAMD9', 'SAMD9L', 'SP110', 'STAT1', 'STAT2', 'TRIM22',
    'USP18', 'XAF1',
]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--healthy-train', type=Path, default=HEALTHY_TRAIN)
    p.add_argument('--synth-dir', type=Path, default=SYNTH_DIR)
    p.add_argument('--out-dir', type=Path, default=OUT_DIR)
    p.add_argument('--models', nargs='+', default=None, metavar='NAME=PATH')
    p.add_argument('--warmup-dir', type=Path, default=BASE / 'runs/warmup_clean')
    p.add_argument('--extra-dir', type=Path, default=BASE / 'runs/warmup_extra_clean')
    p.add_argument('--train-extra', action='store_true')
    p.add_argument('--extra-epochs', type=int, default=4)
    p.add_argument('--export-best', nargs='+', type=Path, default=None)
    p.add_argument('--skip-analysis', action='store_true')
    p.add_argument('--k', type=int, default=30)
    p.add_argument('--p-threshold', type=float, default=0.05)
    p.add_argument('--min-healthy', type=int, default=100)
    p.add_argument('--max-healthy', type=int, default=10_000)
    p.add_argument('--min-cells-type', type=int, default=50)
    p.add_argument('--round-counts', action='store_true')
    p.add_argument('--allow-donor-overlap', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def resolve_models(args):
    if args.models:
        out = {}
        for item in args.models:
            name, _, path = item.partition('=')
            out[name] = Path(path)
        return out
    return {
        'warmup': args.warmup_dir,
        'extra': args.extra_dir,
        'mmd_l10': BASE / 'runs/mmd_clean_lambda10/best',
        'mmd_l100': BASE / 'runs/mmd_clean_lambda100/best',
    }


def train_extra(args):
    if args.extra_dir.exists():
        print(f"extra exists: {args.extra_dir}")
        return
    print("=" * 60)
    print(f"Training extra ({args.extra_epochs} epochs, no MMD) from {args.warmup_dir}")
    print("=" * 60)
    adata = ad.read_h5ad(args.healthy_train)
    scvi.model.SCVI.setup_anndata(adata, batch_key='dataset_id', layer='counts')
    model = scvi.model.SCVI.load(str(args.warmup_dir), adata=adata)
    model.train(max_epochs=args.extra_epochs, batch_size=512, early_stopping=False)
    model.save(str(args.extra_dir), overwrite=True)
    print(f"Saved: {args.extra_dir}")


def export_best(run_dirs, args):
    adata = ad.read_h5ad(args.healthy_train)
    scvi.model.SCVI.setup_anndata(adata, batch_key='dataset_id', layer='counts')
    for run_dir in run_dirs:
        ckpts = sorted((run_dir / 'checkpoints').glob('best-*.ckpt'))
        if not ckpts:
            raise FileNotFoundError(f"No best-*.ckpt in {run_dir / 'checkpoints'}")
        ckpt = ckpts[-1]
        print(f"Exporting {ckpt}")
        model = scvi.model.SCVI.load(str(args.warmup_dir), adata=adata)
        state = torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict']
        state = {k[len('module.'):]: v for k, v in state.items() if k.startswith('module.')}
        res = model.module.load_state_dict(state, strict=False)
        print(f"  missing keys: {list(res.missing_keys)}")
        print(f"  unexpected keys: {list(res.unexpected_keys)}")
        if res.missing_keys:
            raise RuntimeError("Checkpoint does not match module, export aborted")
        model.save(str(run_dir / 'best'), overwrite=True)
        print(f"  saved: {run_dir / 'best'}")


def load_synthetic(synth_dir, var_names, round_counts):
    files = {}
    for p in synth_dir.glob('synthetic_alpha_*.h5ad'):
        m = re.fullmatch(r'synthetic_alpha_([0-9.]+)\.h5ad', p.name)
        if m:
            files[float(m.group(1))] = p
    if 0.0 not in files:
        raise FileNotFoundError(f"synthetic_alpha_0.00.h5ad not found in {synth_dir}")
    alphas = sorted(files)
    print(f"Synthetic alphas: {alphas}")

    out = {}
    for a in alphas:
        s = ad.read_h5ad(files[a])
        missing = set(var_names) - set(s.var_names)
        if missing:
            raise ValueError(f"alpha={a}: {len(missing)} genes of the reference are missing in synthetic")
        s = s[:, var_names].copy()
        X = s.X.toarray() if hasattr(s.X, 'toarray') else np.asarray(s.X)
        X = np.asarray(X, dtype=np.float32)
        n_neg = int((X < 0).sum())
        if n_neg:
            print(f"  alpha={a}: clipping {n_neg:,} negative values")
            X = np.clip(X, 0, None)
        if round_counts:
            X = np.rint(X)
        s.X = X
        s.layers['counts'] = X
        out[a] = s
        print(f"  alpha={a}: {s.n_obs:,} × {s.n_vars:,}")

    base_names = out[alphas[0]].obs_names
    for a in alphas[1:]:
        if not out[a].obs_names.equals(base_names):
            raise ValueError(f"alpha={a}: cells differ from alpha={alphas[0]}, paired comparison impossible")
    return alphas, out


def check_donor_overlap(ref, synth, allow):
    if 'donor_id' not in ref.obs or 'donor_id' not in synth.obs:
        print("[warn] donor_id missing in reference or synthetic, leakage check skipped")
        return
    overlap = set(ref.obs['donor_id'].astype(str)) & set(synth.obs['donor_id'].astype(str))
    print(f"Donor overlap reference/synthetic: {len(overlap)}")
    if overlap and not allow:
        raise RuntimeError(f"Reference contains held-out donors: {sorted(overlap)[:10]}")


def ifn_logcpm(X, var_names):
    idx = np.flatnonzero(np.isin(np.asarray(var_names), IFN_GENES))
    if len(idx) < 5:
        raise ValueError(f"Too few IFN genes: {len(idx)}")
    lib = np.maximum(X.sum(axis=1), 1.0)
    return np.log1p(X[:, idx] / lib[:, None] * 1e6).astype(np.float32)


def pooled_ifn_scores(logcpm_by_alpha, cell_types):
    alphas = list(logcpm_by_alpha)
    n = len(cell_types)
    stack = np.vstack([logcpm_by_alpha[a] for a in alphas])
    z = (stack - stack.mean(axis=0, keepdims=True)) / (stack.std(axis=0, keepdims=True) + 1e-8)
    score = z.mean(axis=1)
    ct_all = np.tile(cell_types, len(alphas))
    for ct in np.unique(ct_all):
        m = ct_all == ct
        if m.sum() < 2:
            continue
        score[m] = (score[m] - score[m].mean()) / (score[m].std() + 1e-8)
    return {a: score[i * n:(i + 1) * n] for i, a in enumerate(alphas)}


def align_batches(s, mapping):
    if 'dataset_id' not in s.obs:
        raise KeyError("synthetic obs has no 'dataset_id' column")
    str_map = {str(c): c for c in mapping}
    ids = s.obs['dataset_id'].astype(str)
    unseen = sorted(set(ids) - set(str_map))
    if unseen:
        print(f"  [warn] dataset_id not seen in training: {unseen}, replaced by {mapping[0]}")
    vals = [str_map.get(i, mapping[0]) for i in ids]
    s.obs['dataset_id'] = pd.Categorical(vals, categories=list(mapping))


def kth_distances(h_lat, q_lats, k):
    h = np.ascontiguousarray(h_lat, dtype=np.float32)
    try:
        import faiss
        index = faiss.IndexFlatL2(h.shape[1])
        index.add(h)

        def search(q, kk):
            D, _ = index.search(np.ascontiguousarray(q, dtype=np.float32), kk)
            return np.sqrt(np.maximum(D, 0))
    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', n_jobs=-1).fit(h)

        def search(q, kk):
            D, _ = nn.kneighbors(q, n_neighbors=kk)
            return D

    null = search(h, k + 1)[:, k]
    return null, {a: search(q, k)[:, k - 1] for a, q in q_lats.items()}


def scpss_scores(ref_lat, ref_ct, qry_lats, qry_ct, k, min_healthy):
    n = len(qry_ct)
    p = {a: np.full(n, np.nan, dtype=np.float32) for a in qry_lats}
    ratio = {a: np.full(n, np.nan, dtype=np.float32) for a in qry_lats}
    for ct in tqdm(np.unique(qry_ct), desc='  scPSS'):
        h = ref_lat[ref_ct == ct]
        if len(h) < min_healthy:
            continue
        qm = qry_ct == ct
        null, qd = kth_distances(h, {a: l[qm] for a, l in qry_lats.items()}, k)
        sorted_null = np.sort(null)
        n_null = len(sorted_null)
        med = max(float(np.median(null)), 1e-8)
        for a, d in qd.items():
            counts = n_null - np.searchsorted(sorted_null, d, side='left')
            p[a][qm] = np.clip(counts / n_null, 1.0 / (n_null + 1), 1.0)
            ratio[a][qm] = d / med
    return p, ratio


def auc(neg, pos):
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    return float(roc_auc_score(y, np.r_[neg, pos]))


def evaluate(name, alphas, p, ratio, ifn, cell_types, p_threshold, min_n):
    valid = ~np.isnan(ratio[0.0])
    r0 = ratio[0.0]
    rows = []
    for a in alphas:
        ra = ratio[a]
        row = {
            'model': name,
            'alpha': a,
            'n_cells': int(valid.sum()),
            'frac_anomalous': float((p[a][valid] < p_threshold).mean()),
            'mean_ratio': float(ra[valid].mean()),
            'auc_pooled': np.nan,
            'auc_per_type': np.nan,
            'rho_ifn_per_type': np.nan,
            'frac_increased': np.nan,
        }
        if a != 0.0:
            row['auc_pooled'] = auc(r0[valid], ra[valid])
            row['frac_increased'] = float((ra[valid] > r0[valid]).mean())
            aucs, rhos = [], []
            for ct in np.unique(cell_types[valid]):
                m = valid & (cell_types == ct)
                if m.sum() < min_n:
                    continue
                aucs.append(auc(r0[m], ra[m]))
                rho, _ = spearmanr(np.r_[r0[m], ra[m]], np.r_[ifn[0.0][m], ifn[a][m]])
                rhos.append(rho)
            row['auc_per_type'] = float(np.nanmean(aucs)) if aucs else np.nan
            row['rho_ifn_per_type'] = float(np.nanmean(rhos)) if rhos else np.nan
        rows.append(row)
    return rows


def analyze_model(name, model_dir, ref, ref_idx, ref_ct, synth, alphas, ifn, cell_types, args):
    print("\n" + "=" * 60)
    print(f"Model: {name}  ({model_dir})")
    print("=" * 60)

    t0 = time.time()
    model = scvi.model.SCVI.load(str(model_dir), adata=ref)
    mapping = list(model.adata_manager.get_state_registry('batch').categorical_mapping)
    ref_lat = model.get_latent_representation(ref, indices=ref_idx, batch_size=1024)

    qry_lats = {}
    for a in alphas:
        align_batches(synth[a], mapping)
        qry_lats[a] = model.get_latent_representation(synth[a], batch_size=1024)
    print(f"Latents: ref {ref_lat.shape}, synthetic {qry_lats[alphas[0]].shape} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    p, ratio = scpss_scores(ref_lat, ref_ct, qry_lats, cell_types, args.k, args.min_healthy)
    print(f"scPSS done in {time.time() - t0:.1f}s")

    valid = ~np.isnan(ratio[alphas[0]])
    print(f"Valid cells: {valid.sum():,} / {len(valid):,}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / f'scores_{name}.npz',
        cell_type=cell_types,
        **{f'ratio_{a:.2f}': ratio[a] for a in alphas},
        **{f'p_{a:.2f}': p[a] for a in alphas},
    )
    return evaluate(name, alphas, p, ratio, ifn, cell_types, args.p_threshold, args.min_cells_type)


def main():
    args = parse_args()

    if args.train_extra:
        train_extra(args)
    if args.export_best:
        export_best(args.export_best, args)
    if args.skip_analysis:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading data")
    print("=" * 60)
    ref = ad.read_h5ad(args.healthy_train)
    ref_vars = list(ref.var_names)
    print(f"Reference: {ref.n_obs:,} × {ref.n_vars:,}")
    alphas, synth = load_synthetic(args.synth_dir, ref_vars, args.round_counts)
    first = synth[alphas[0]]
    check_donor_overlap(ref, first, args.allow_donor_overlap)

    cell_types = first.obs['cell_type'].astype(str).values
    ref_ct_all = ref.obs['cell_type'].astype(str).values
    unknown = sorted(set(cell_types) - set(ref_ct_all))
    if unknown:
        print(f"[warn] synthetic cell types absent from reference (will be skipped): {unknown}")

    rng = np.random.default_rng(args.seed)
    sel = []
    for ct in np.unique(ref_ct_all):
        idx = np.flatnonzero(ref_ct_all == ct)
        if len(idx) < args.min_healthy:
            continue
        if len(idx) > args.max_healthy:
            idx = rng.choice(idx, size=args.max_healthy, replace=False)
        sel.append(idx)
    ref_idx = np.sort(np.concatenate(sel))
    ref_ct = ref_ct_all[ref_idx]
    print(f"Reference subsample: {len(ref_idx):,} cells, {len(sel)} cell types")

    ifn = pooled_ifn_scores(
        {a: ifn_logcpm(synth[a].X, ref_vars) for a in alphas}, cell_types)
    for a in alphas:
        print(f"IFN score (pooled z) alpha={a}: mean={ifn[a].mean():+.3f}")

    scvi.model.SCVI.setup_anndata(ref, batch_key='dataset_id', layer='counts')

    rows = []
    for name, model_dir in resolve_models(args).items():
        if not Path(model_dir).exists():
            print(f"\nSkip {name}: {model_dir} does not exist")
            continue
        try:
            rows += analyze_model(name, Path(model_dir), ref, ref_idx, ref_ct,
                                  synth, alphas, ifn, cell_types, args)
        except Exception as e:
            import traceback
            print(f"\nFailed {name}: {type(e).__name__}: {e}")
            traceback.print_exc()

    if rows:
        df = pd.DataFrame(rows)
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(df.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
        out = args.out_dir / 'synthetic_comparison.csv'
        df.to_csv(out, index=False)
        print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()