"""
Split combined dataset into:
  - healthy_only.h5ad           (census_healthy + lupus_healthy)
  - disease_cohort_{2,3,4}.h5ad (lupus_disease by Processing_Cohort)

Cohorts from Processing_Cohort in raw lupus file.
Donor overlap between cohorts is OK — cohorts are evaluated independently.
"""
import anndata as ad
from pathlib import Path

BASE = Path('/home/vlad/scGRIP')

COMBINED = BASE / 'data/bank/combined_raw_hvg.h5ad'
LUPUS_RAW = BASE / 'data/lupus_raw/lupus_c55dc602_disease_ct_rawcounts.h5ad'
OUT_HEALTHY = BASE / 'data/bank/healthy_only.h5ad'
OUT_DISEASE_DIR = BASE / 'data/bank'


def main():
    # ===== 1. Load combined =====
    print("=" * 60)
    print("STEP 1: Load combined")
    print("=" * 60)
    combined = ad.read_h5ad(COMBINED)
    print(f"Combined: {combined.n_obs:,} × {combined.n_vars:,}")
    print(combined.obs['group'].value_counts())

    # ===== 2. Healthy =====
    print("\n" + "=" * 60)
    print("STEP 2: Split healthy")
    print("=" * 60)
    healthy = combined[
        combined.obs['group'].isin(['census_healthy', 'lupus_healthy'])
    ].copy()
    print(f"Healthy: {healthy.n_obs:,} cells, {healthy.obs['donor_id'].nunique()} donors")
    print(healthy.obs['group'].value_counts())
    print(f"Cell types: {healthy.obs['cell_type'].nunique()}")

    # A/B eligibility
    pairs = healthy.obs.groupby(['cell_type', 'donor_id']).size()
    print(f"\n(donor, type) pairs >= 30 cells: {(pairs >= 30).sum()}")
    print(f"(donor, type) pairs >= 10 cells: {(pairs >= 10).sum()}")
    print(f"(donor, type) pairs >= 5 cells:  {(pairs >= 5).sum()}")

    healthy.write_h5ad(OUT_HEALTHY)
    print(f"\nSaved: {OUT_HEALTHY}")

    # ===== 3. Disease by cohort =====
    print("\n" + "=" * 60)
    print("STEP 3: Split disease by Processing_Cohort")
    print("=" * 60)
    disease = combined[combined.obs['group'] == 'lupus_disease'].copy()
    print(f"Disease: {disease.n_obs:,} cells, {disease.obs['donor_id'].nunique()} donors")

    # Load raw lupus
    print(f"\nLoading raw lupus: {LUPUS_RAW}")
    lupus_raw = ad.read_h5ad(LUPUS_RAW)

    if 'Processing_Cohort' not in lupus_raw.obs.columns:
        print("ERROR: Processing_Cohort not in raw lupus")
        return

    # cohort → set of donor_id
    cohort_map = (
        lupus_raw.obs
        .groupby('Processing_Cohort', observed=True)['donor_id']
        .apply(lambda x: set(map(str, x.unique())))
        .to_dict()
    )

    print(f"\nCohorts: {len(cohort_map)}")
    for name, donors in sorted(cohort_map.items()):
        n_cells = (lupus_raw.obs['Processing_Cohort'] == name).sum()
        print(f"  Cohort {name}: {len(donors)} donors, {n_cells:,} cells (raw)")

    # Split combined disease
    print()
    for cohort_name, donors in sorted(cohort_map.items()):
        mask = disease.obs['donor_id'].astype(str).isin(donors)
        sub = disease[mask].copy()

        if sub.n_obs == 0:
            print(f"  Cohort {cohort_name}: SKIP (0 cells)")
            continue

        # Clean filename (2.0 → 2)
        name_clean = str(int(float(cohort_name))) if str(cohort_name).replace('.', '').isdigit() else str(cohort_name)
        out = OUT_DISEASE_DIR / f'disease_cohort_{name_clean}.h5ad'
        sub.write_h5ad(out)
        print(f"  Cohort {name_clean}: {sub.n_obs:,} cells, {sub.obs['donor_id'].nunique()} donors → {out.name}")

    # ===== 4. Summary =====
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Healthy: {healthy.n_obs:,} × {healthy.n_vars:,}")
    print(f"Disease total: {disease.n_obs:,} × {disease.n_vars:,}")
    print(f"\nSaved files:")
    for f in sorted(OUT_DISEASE_DIR.glob('*.h5ad')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()