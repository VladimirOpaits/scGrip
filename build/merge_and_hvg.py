"""
Merge 3 datasets and select HVG (robust version).

Strategy:
  1. Load 3 files, align by common genes.
  2. Concatenate into one raw-counts AnnData.
  3. Save raw combined to disk (backup).
  4. HVG via seurat_v3 on STRATIFIED subsample (50k cells).
     - No batch_key (avoids LOESS singularities).
  5. Apply HVG mask to full combined, remove artifacts.
  6. Save final combined_raw_hvg.h5ad.
"""
import anndata as ad
import scanpy as sc
import numpy as np
from pathlib import Path
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split

# ===== Paths =====
BASE = Path('/home/vlad/scGRIP')

LUPUS_DISEASE = BASE / 'data/lupus_raw/lupus_c55dc602_disease_ct_rawcounts.h5ad'
LUPUS_HEALTHY = BASE / 'data/bank/lupus_c55dc602_healthy_ct_rawcounts.h5ad'
CENSUS_HEALTHY = BASE / 'data/census/healthy_pbmc_celltypist_rawcounts.h5ad'

OUTPUT_RAW = BASE / 'data/bank/combined_raw.h5ad'
OUTPUT_HVG = BASE / 'data/bank/combined_raw_hvg.h5ad'

# HVG params
N_TOP_GENES = 8000
N_SUB = 50000
SEED = 42

# ===== 1. Load =====
print("=" * 60)
print("STEP 1: Loading files")
print("=" * 60)

lupus_disease = ad.read_h5ad(LUPUS_DISEASE)
lupus_healthy = ad.read_h5ad(LUPUS_HEALTHY)
census_healthy = ad.read_h5ad(CENSUS_HEALTHY)

print(f"Lupus disease:  {lupus_disease.n_obs:,} × {lupus_disease.n_vars:,}")
print(f"Lupus healthy:  {lupus_healthy.n_obs:,} × {lupus_healthy.n_vars:,}")
print(f"Census healthy: {census_healthy.n_obs:,} × {census_healthy.n_vars:,}")

# ===== 2. Unique obs_names =====
for a in (lupus_disease, lupus_healthy, census_healthy):
    a.obs_names_make_unique()

# ===== 3. Common genes =====
print("\n" + "=" * 60)
print("STEP 2: Finding common genes")
print("=" * 60)

common_genes = sorted(
    set(lupus_disease.var_names)
    & set(lupus_healthy.var_names)
    & set(census_healthy.var_names)
)
print(f"Common genes: {len(common_genes)}")

# ===== 4. Subset =====
lupus_disease = lupus_disease[:, common_genes].copy()
lupus_healthy = lupus_healthy[:, common_genes].copy()
census_healthy = census_healthy[:, common_genes].copy()

# ===== 5. Labels =====
print("\n" + "=" * 60)
print("STEP 3: Adding labels")
print("=" * 60)

lupus_disease.obs['group'] = 'lupus_disease'
lupus_healthy.obs['group'] = 'lupus_healthy'
census_healthy.obs['group'] = 'census_healthy'

lupus_disease.obs['dataset_id'] = 'lupus_c55dc602'
lupus_healthy.obs['dataset_id'] = 'lupus_c55dc602'

for name, a in [
    ('lupus_disease', lupus_disease),
    ('lupus_healthy', lupus_healthy),
    ('census_healthy', census_healthy),
]:
    if 'cell_type' not in a.obs.columns:
        if 'celltypist' in a.obs.columns:
            a.obs['cell_type'] = a.obs['celltypist']
    print(f"{name}: cell_type unique = {a.obs['cell_type'].nunique()}")

# ===== 6. Concatenate =====
print("\n" + "=" * 60)
print("STEP 4: Concatenating")
print("=" * 60)

combined = ad.concat(
    [lupus_disease, lupus_healthy, census_healthy],
    join='inner',
    label='source',
    keys=['lupus_disease', 'lupus_healthy', 'census_healthy'],
    index_unique='_',
)
print(f"Combined: {combined.n_obs:,} × {combined.n_vars:,}")

# ===== 7. Save raw =====
print("\n" + "=" * 60)
print("STEP 5: Saving raw combined")
print("=" * 60)

combined.layers['counts'] = combined.X.copy()
combined.write_h5ad(OUTPUT_RAW)
print(f"Saved: {OUTPUT_RAW}")

# ===== 8. Stratified subsample =====
print("\n" + "=" * 60)
print(f"STEP 6: Stratified subsample ({N_SUB:,} cells)")
print("=" * 60)

strat_key = (
    combined.obs['dataset_id'].astype(str)
    + '|'
    + combined.obs['cell_type'].astype(str)
)

vc = strat_key.value_counts()
rare = vc[vc < 2].index
if len(rare) > 0:
    print(f"Dropping {len(rare)} rare strata (< 2 cells)")
    keep_mask = ~strat_key.isin(rare).to_numpy()
    strat_key_use = strat_key[keep_mask]
    idx_pool = np.arange(combined.n_obs)[keep_mask]
else:
    strat_key_use = strat_key
    idx_pool = np.arange(combined.n_obs)

n_sub_eff = min(N_SUB, len(idx_pool))

idx_train, _ = train_test_split(
    idx_pool,
    train_size=n_sub_eff,
    stratify=strat_key_use,
    random_state=SEED,
)

adata_sub = combined[idx_train].copy()
print(f"Subsample: {adata_sub.n_obs:,} × {adata_sub.n_vars:,}")

# ===== 9. HVG via seurat_v3 (no batch_key) =====
print("\n" + "=" * 60)
print("STEP 7: HVG selection (seurat_v3, no batch_key)")
print("=" * 60)

try:
    sc.pp.highly_variable_genes(
        adata_sub,
        n_top_genes=N_TOP_GENES,
        flavor='seurat_v3',
    )
    print("  seurat_v3 succeeded")
except Exception as e:
    print(f"  seurat_v3 failed: {e}")
    print("  Falling back to seurat (log1p)...")

    # Normalize + log1p
    sc.pp.normalize_total(adata_sub, target_sum=1e4)
    sc.pp.log1p(adata_sub)

    sc.pp.highly_variable_genes(
        adata_sub,
        n_top_genes=N_TOP_GENES,
        flavor='seurat',
    )
    print("  seurat succeeded")

# Transfer mask
combined.var['highly_variable'] = adata_sub.var['highly_variable'].values
n_hvg = int(combined.var['highly_variable'].sum())
print(f"HVG selected: {n_hvg}")

combined = combined[:, combined.var['highly_variable']].copy()
print(f"After HVG: {combined.n_obs:,} × {combined.n_vars:,}")

del adata_sub

# ===== 10. Artifacts =====
print("\n" + "=" * 60)
print("STEP 8: Removing artifacts")
print("=" * 60)

var = combined.var_names
mt = var.str.startswith(('MT-', 'mt-'))
ribo = var.str.startswith(('RPS', 'RPL', 'Rps', 'Rpl'))
hb = var.str.match(r'^HB[ABDEGMQZ]\d?$')

print(f"MT: {mt.sum()}, Ribo: {ribo.sum()}, Hb: {hb.sum()}")

keep = ~(mt | ribo | hb)
combined = combined[:, keep].copy()
print(f"After artifacts: {combined.n_vars:,} genes")

# ===== 11. Verify X =====
print("\n" + "=" * 60)
print("STEP 9: Verify X")
print("=" * 60)

X = combined.X[:500]
if issparse(X):
    X = X.toarray()
print(f"Max: {X.max():.2f}")
print(f"Type: {'RAW' if X.max() > 20 else 'LOG1P'}")

# ===== 12. Save =====
print("\n" + "=" * 60)
print("STEP 10: Saving final")
print("=" * 60)

combined.write_h5ad(OUTPUT_HVG)
print(f"Saved: {OUTPUT_HVG}")

# ===== 13. Summary =====
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Cells:      {combined.n_obs:,}")
print(f"Genes:      {combined.n_vars:,}")
print(f"Donors:     {combined.obs['donor_id'].nunique():,}")
print(f"Cell types: {combined.obs['cell_type'].nunique()}")
print(f"Datasets:   {combined.obs['dataset_id'].nunique()}")
print()
print("Group distribution:")
print(combined.obs['group'].value_counts())
print()
print("Layers:", list(combined.layers.keys()))