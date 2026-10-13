import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import scvi
import anndata as ad
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from scCuddle.data import load_bank
from scCuddle.mmd import compute_bandwidths_per_type, MMDPlan
from scCuddle.training_plan import MMDTrainingPlan
from scCuddle.callbacks import MMDABCallback

import logging
logging.getLogger("scvi").setLevel(logging.WARNING)
logging.getLogger("scvi.model").setLevel(logging.WARNING)
logging.getLogger("scvi.data").setLevel(logging.WARNING)




BASE = Path('/home/vlad/scGRIP')
DEFAULT_HEALTHY = BASE / 'data/bank/healthy_only.h5ad'
DEFAULT_WARMUP_DIR = BASE / 'runs/warmup'
DEFAULT_RUN_DIR = BASE / 'runs/mmd'


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--healthy', type=Path, default=DEFAULT_HEALTHY)
    p.add_argument('--warmup-dir', type=Path, default=DEFAULT_WARMUP_DIR)
    p.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument('--skip-warmup', action='store_true')
    p.add_argument('--skip-mmd', action='store_true')
    p.add_argument('--n-latent', type=int, default=30)
    p.add_argument('--warmup-epochs', type=int, default=15)
    p.add_argument('--mmd-epochs', type=int, default=20)
    p.add_argument('--elbo-batch-size', type=int, default=512)
    p.add_argument('--lambda-mmd', type=float, default=100.0)
    p.add_argument('--kl-weight', type=float, default=0.0375)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-donors-per-type', type=int, default=12)
    p.add_argument('--k-cells-per-donor', type=int, default=32)
    p.add_argument('--min-cells-per-donor-type', type=int, default=30)
    p.add_argument('--min-a-eligible', type=int, default=4)
    p.add_argument('--min-b-eligible', type=int, default=1)
    p.add_argument('--n-cells-per-donor-metric', type=int, default=30)
    p.add_argument('--compute-every', type=int, default=1)
    p.add_argument('--window-size', type=int, default=1)
    p.add_argument('--threshold', type=float, default=0.1)
    p.add_argument('--factor', type=float, default=0.5)
    p.add_argument('--min-lambda', type=float, default=1.0)
    p.add_argument('--cooldown', type=int, default=2)
    p.add_argument('--early-stop-patience', type=int, default=4)
    p.add_argument('--min-delta', type=float, default=0.001)
    p.add_argument('--no-early-stop', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    scvi.settings.seed = args.seed
    WARMUP_DIR = args.warmup_dir
    MMD_DIR = args.run_dir
    WARMUP_DIR.mkdir(parents=True, exist_ok=True)
    MMD_DIR.mkdir(parents=True, exist_ok=True)
    SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
    BANDWIDTH_POINTS = 512
    BANDWIDTH_FLOOR = 1e-6

    print("=" * 60); print("STEP 1: Load healthy data"); print("=" * 60)
    adata = ad.read_h5ad(args.healthy)
    print(f"{adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(adata.obs['group'].value_counts())

    print("\n" + "=" * 60); print("STEP 2: Setup scVI"); print("=" * 60)
    scvi.model.SCVI.setup_anndata(adata, batch_key='dataset_id', layer='counts')

    if args.skip_warmup:
        print("\n" + "=" * 60); print(f"STEP 3: SKIP warmup, loading from {WARMUP_DIR}"); print("=" * 60)
        model = scvi.model.SCVI.load(str(WARMUP_DIR), adata=adata)
        print("Warmup loaded.")
    else:
        print("\n" + "=" * 60); print(f"STEP 3: Warmup ({args.warmup_epochs} epochs, no MMD)"); print("=" * 60)
        model = scvi.model.SCVI(adata, n_latent=args.n_latent, n_layers=2, n_hidden=128)
        model.train(max_epochs=args.warmup_epochs, batch_size=args.elbo_batch_size, early_stopping=False, enable_progress_bar=True)
        model.save(str(WARMUP_DIR), overwrite=True)
        print(f"Warmup saved: {WARMUP_DIR}")

    if args.skip_mmd:
        print("\n" + "=" * 60); print("SKIP_MMD=True, stopping after warmup"); print("=" * 60)
        return

    print("\n" + "=" * 60); print("STEP 4: Build BankData (A/B split)"); print("=" * 60)
    ct_counts = adata.obs['cell_type'].value_counts()
    mmd_types = tuple(ct_counts[ct_counts >= 1000].index)
    print(f"MMD types ({len(mmd_types)})")

    bank = load_bank(
        path=args.healthy, donor_col='donor_id', cell_type_col='cell_type', batch_col='dataset_id',
        holdout_frac=0.2, split_seed=args.seed,
        min_cells_per_donor_type=args.min_cells_per_donor_type,
        min_A_eligible=args.min_a_eligible, min_B_eligible=args.min_b_eligible,
        mmd_types=mmd_types)
    batch_mapping = [str(c) for c in model.adata_manager.get_state_registry('batch').categorical_mapping]
    bank.adata.obs['dataset_id'] = pd.Categorical(bank.adata.obs['dataset_id'].astype(str), categories=batch_mapping)
    assert not bank.adata.obs['dataset_id'].isna().any(), "dataset_id не совпал с маппингом модели"
    loader_codes = bank.adata.obs['dataset_id'].cat.codes.to_numpy()
    assert (loader_codes == adata.obs['_scvi_batch'].to_numpy()).all(), "коды батчей загрузчика != scVI"
    print("batch codes loader == scVI: OK")
    print(f"\nA donors: {bank.A_donors.size}, B donors: {bank.B_donors.size}, eligible types: {len(bank.eligible_donors)}")

    print("\n" + "=" * 60); print("STEP 5: Compute bandwidths"); print("=" * 60)
    latents_all = torch.tensor(model.get_latent_representation(adata), dtype=torch.float32)
    latents_by_type = {}
    for ct in bank.eligible_donors:
        a_elig = bank.A_eligible_donors(ct)
        if len(a_elig) < args.min_a_eligible: continue
        idx = np.concatenate([bank.cells_for(ct, str(d)) for d in a_elig])
        latents_by_type[ct] = latents_all[idx]
    bandwidths = compute_bandwidths_per_type(latents_by_type, scales=SCALES, max_points=BANDWIDTH_POINTS, floor=BANDWIDTH_FLOOR)
    print(f"Bandwidths for {len(bandwidths)} cell types")

    A_donors_by_type = {ct: bank.A_eligible_donors(ct) for ct in bandwidths}
    mmd_plan = MMDPlan(
        A_donors_by_type=A_donors_by_type, bandwidths=bandwidths,
        n_donors_per_type=args.n_donors_per_type, k_cells_per_donor=args.k_cells_per_donor)
    print(f"MMDPlan: {len(mmd_plan.cell_types)} cell types")

    print("\n" + "=" * 60); print("STEP 6: Create MMDTrainingPlan"); print("=" * 60)
    plan = MMDTrainingPlan(
        module=model.module, bank=bank, mmd_plan=mmd_plan, adata=adata,
        elbo_batch_size=args.elbo_batch_size, lambda_mmd=args.lambda_mmd, seed=args.seed,
        kl_weight=args.kl_weight)
    print(f"Plan: {plan.n_batches} batches/epoch, lambda={plan.lambda_mmd}, kl_weight={plan.kl_weight}")

    print("\n" + "=" * 60); print("STEP 7: Callbacks"); print("=" * 60)
    mmd_ab_callback = MMDABCallback(
        model=model, bank=bank, mmd_plan=mmd_plan, bandwidths=bandwidths,
        rng=np.random.default_rng(args.seed + 1),
        n_cells_per_donor=args.n_cells_per_donor_metric,
        n_donors_per_type=None,
        compute_every=args.compute_every,
        log_per_type=True, verbose=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(MMD_DIR / 'checkpoints'), filename='best-{epoch:02d}-{mmd_ab:.4f}',
        monitor='mmd_ab', mode='min', save_top_k=1, save_last=True, verbose=True)

    callbacks = [mmd_ab_callback, checkpoint_callback]
    if not args.no_early_stop:
        early_stop = EarlyStopping(
            monitor='mmd_ab',
            patience=args.early_stop_patience,
            mode='min',
            min_delta=args.min_delta,
            verbose=True,
        )
        callbacks.insert(1, early_stop)
        print(f"EarlyStopping: ON (patience={args.early_stop_patience}, min_delta={args.min_delta})")
    else:
        print("EarlyStopping: OFF")

    print("\n" + "=" * 60); print(f"STEP 8: MMD phase ({args.mmd_epochs} epochs)"); print("=" * 60)
    trainer = pl.Trainer(
        max_epochs=args.mmd_epochs, accelerator='gpu', devices=1,
        callbacks=callbacks,
        enable_progress_bar=True, default_root_dir=str(MMD_DIR), log_every_n_steps=10)
    trainer.fit(plan)
    

if __name__ == "__main__":
    main()