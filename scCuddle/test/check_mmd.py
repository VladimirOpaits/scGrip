"""
Проверка гиперпараметров MMD перед запуском обучения.

Что показывает:
  - Сколько типов клеток пройдут фильтр (>= 1000 клеток).
  - Сколько eligible доноров на тип (после min_cells_per_donor_type).
  - Сколько A-eligible / B-eligible на тип.
  - Сколько типов пройдут min_A_eligible и min_B_eligible.
  - Сколько (donor, type) пар с >= 30 клеток.
  - Сколько A-клеток доступно (для n_batches).
  - Оценку: сколько MMD-батчей получится за эпоху.
"""
import numpy as np
from pathlib import Path

from scCuddle.data import load_bank


# ===== Параметры (те же, что в train.py) =====
BASE = Path('/home/vlad/scGRIP')
HEALTHY = BASE / 'data/bank/healthy_only.h5ad'

N_DONORS_PER_TYPE = 12
K_CELLS_PER_DONOR = 32
MIN_CELLS_PER_DONOR_TYPE = 30
MIN_A_ELIGIBLE = 4
MIN_B_ELIGIBLE = 1
ELBO_BATCH_SIZE = 512


# ===== 1. Загрузка adata (только obs) =====
import anndata as ad
adata = ad.read_h5ad(HEALTHY, backed='r')
print("=" * 70)
print("DATASET")
print("=" * 70)
print(f"Cells: {adata.n_obs:,}")
print(f"Genes: {adata.n_vars:,}")
print(f"Donors (total): {adata.obs['donor_id'].nunique()}")
print(f"Cell types (total): {adata.obs['cell_type'].nunique()}")
print(f"Datasets: {adata.obs['dataset_id'].nunique()}")

# ===== 2. Фильтр типов =====
print("\n" + "=" * 70)
print("CELL TYPE FILTER (>= 1000 cells)")
print("=" * 70)
ct_counts = adata.obs['cell_type'].value_counts()
mmd_types = tuple(ct_counts[ct_counts >= 1000].index)
print(f"All types: {len(ct_counts)}")
print(f"Types >= 1000 cells: {len(mmd_types)}")
print("\nTypes for MMD:")
for ct in mmd_types:
    print(f"  {ct}: {ct_counts[ct]:,}")

# ===== 3. (donor, type) pairs =====
print("\n" + "=" * 70)
print("(DONOR, TYPE) PAIRS")
print("=" * 70)

# Все пары по типам MMD
subset = adata.obs[adata.obs['cell_type'].isin(mmd_types)]
pairs = subset.groupby(['cell_type', 'donor_id'], observed=True).size()
print(f"Total (donor, type) pairs: {len(pairs)}")
print(f"Pairs >= {MIN_CELLS_PER_DONOR_TYPE} cells: {(pairs >= MIN_CELLS_PER_DONOR_TYPE).sum()}")
print(f"Pairs >= 10 cells: {(pairs >= 10).sum()}")
print(f"Pairs >= 5 cells: {(pairs >= 5).sum()}")

# ===== 4. load_bank =====
print("\n" + "=" * 70)
print("LOAD_BANK")
print("=" * 70)

bank = load_bank(
    path=HEALTHY,
    donor_col='donor_id',
    cell_type_col='cell_type',
    batch_col='dataset_id',
    holdout_frac=0.2,
    split_seed=0,
    min_cells_per_donor_type=MIN_CELLS_PER_DONOR_TYPE,
    min_A_eligible=MIN_A_ELIGIBLE,
    min_B_eligible=MIN_B_ELIGIBLE,
    mmd_types=mmd_types,
)

print(f"\nA donors total: {bank.A_donors.size}")
print(f"B donors total: {bank.B_donors.size}")
print(f"Eligible types: {len(bank.eligible_donors)}")

# ===== 5. Детали по типам =====
print("\n" + "=" * 70)
print("PER-TYPE DETAILS")
print("=" * 70)
print(f"{'cell_type':<55} {'elig':>5} {'A':>5} {'B':>5} {'cells_A':>10}")
print("-" * 85)

total_a_cells = 0
types_ok = 0

for ct in sorted(bank.eligible_donors.keys()):
    eligible = bank.eligible_donors[ct]
    a_elig = bank.A_eligible_donors(ct)
    a_set = bank.A_donor_set
    b_elig = [d for d in eligible if str(d) not in a_set]

    # Сколько A-клеток доступно
    cells_a = sum(len(bank.cells_for(ct, str(d))) for d in a_elig)
    total_a_cells += cells_a

    ok = "✓" if (len(a_elig) >= MIN_A_ELIGIBLE and len(b_elig) >= MIN_B_ELIGIBLE) else "✗"
    types_ok += 1 if ok == "✓" else 0

    ct_short = ct[:53] + ".." if len(ct) > 53 else ct
    print(f"{ct_short:<55} {len(eligible):>5} {len(a_elig):>5} {len(b_elig):>5} {cells_a:>10,}  {ok}")

print("-" * 85)
print(f"Types OK: {types_ok} / {len(bank.eligible_donors)}")
print(f"Total A cells: {total_a_cells:,}")

# ===== 6. MMDPlan stats =====
print("\n" + "=" * 70)
print("MMDPLAN (после min_A_eligible и min_B_eligible)")
print("=" * 70)

bandwidths_types = {}
for ct in bank.eligible_donors:
    a_elig = bank.A_eligible_donors(ct)
    a_set = bank.A_donor_set
    b_elig = [d for d in bank.eligible_donors[ct] if str(d) not in a_set]
    if len(a_elig) >= MIN_A_ELIGIBLE and len(b_elig) >= MIN_B_ELIGIBLE:
        bandwidths_types[ct] = (0.5, 1.0, 2.0)  # placeholder

print(f"Types in MMDPlan: {len(bandwidths_types)}")
for ct in sorted(bandwidths_types.keys()):
    a_elig = bank.A_eligible_donors(ct)
    print(f"  {ct}: {len(a_elig)} A-donors")

# ===== 7. Оценка MMD-батча =====
print("\n" + "=" * 70)
print("MMD BATCH ESTIMATE")
print("=" * 70)
print(f"n_donors_per_type = {N_DONORS_PER_TYPE}")
print(f"k_cells_per_donor = {K_CELLS_PER_DONOR}")
print(f"MMD cells per type (max): {N_DONORS_PER_TYPE} × {K_CELLS_PER_DONOR} = {N_DONORS_PER_TYPE * K_CELLS_PER_DONOR:,}")
print(f"MMD cells per batch (all types): {len(bandwidths_types)} × {N_DONORS_PER_TYPE * K_CELLS_PER_DONOR:,} = {len(bandwidths_types) * N_DONORS_PER_TYPE * K_CELLS_PER_DONOR:,}")

# Реальный размер (с учётом min с количеством клеток)
real_mmd_cells = 0
for ct in bandwidths_types:
    a_elig = bank.A_eligible_donors(ct)
    n_take = min(N_DONORS_PER_TYPE, len(a_elig))
    # сколько клеток реально можно взять
    for d in a_elig[:n_take]:
        cells = bank.cells_for(ct, str(d))
        real_mmd_cells += min(K_CELLS_PER_DONOR, len(cells))

print(f"Real MMD cells per batch (with min): ~{real_mmd_cells:,}")

# ===== 8. n_batches per epoch =====
print("\n" + "=" * 70)
print("N_BATCHES PER EPOCH")
print("=" * 70)
n_batches = max(1, total_a_cells // ELBO_BATCH_SIZE)
print(f"Total A cells: {total_a_cells:,}")
print(f"ELBO batch size: {ELBO_BATCH_SIZE}")
print(f"n_batches/epoch = {total_a_cells:,} / {ELBO_BATCH_SIZE} = {n_batches}")
print(f"\nCombined batch size (ELBO + MMD): {ELBO_BATCH_SIZE + real_mmd_cells:,}")

# ===== 9. Проверки =====
print("\n" + "=" * 70)
print("CHECKS")
print("=" * 70)

checks = [
    ("At least 1 type in MMDPlan", len(bandwidths_types) >= 1),
    ("At least 5 types", len(bandwidths_types) >= 5),
    ("n_donors_per_type <= max A-donors per type",
     all(len(bank.A_eligible_donors(ct)) >= 2 for ct in bandwidths_types)),
    ("k_cells_per_donor <= min cells per donor",
     K_CELLS_PER_DONOR <= MIN_CELLS_PER_DONOR_TYPE * 2),
    ("Combined batch < 10,000",
     ELBO_BATCH_SIZE + real_mmd_cells < 10_000),
]

for name, ok in checks:
    marker = "✓" if ok else "✗"
    print(f"  {marker} {name}")

print("\n" + "=" * 70)