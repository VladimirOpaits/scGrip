from __future__ import annotations

import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from .mmd import mmd2_biased


def compute_mmd_ab_per_type(model, bank, mmd_plan, bandwidths, rng,
    *, n_cells_per_donor: int = 30, n_donors_per_type: int | None = None,) -> dict[str, float]:
    import torch

    model.module.eval()
    results = {}

    for cell_type in mmd_plan.cell_types:
        a_donors = list(mmd_plan.A_donors_by_type[cell_type])
        eligible = bank.eligible_donors.get(cell_type, [])
        a_set = bank.A_donor_set
        b_donors = [d for d in eligible if str(d) not in a_set]

        if len(a_donors) < 2 or len(b_donors) < 1:
            continue

        if n_donors_per_type is not None:
            if len(a_donors) > n_donors_per_type:
                idx = rng.choice(len(a_donors), size=n_donors_per_type, replace=False)
                a_donors = [a_donors[i] for i in idx]
            if len(b_donors) > n_donors_per_type:
                idx = rng.choice(len(b_donors), size=n_donors_per_type, replace=False)
                b_donors = [b_donors[i] for i in idx]

        a_cells = _sample_cells(bank, cell_type, a_donors, n_cells_per_donor, rng)
        b_cells = _sample_cells(bank, cell_type, b_donors, n_cells_per_donor, rng)

        if a_cells.size < 2 or b_cells.size < 2:
            continue

        try:
            with torch.no_grad():
                a_lat = _get_latents(model, bank, a_cells)
                b_lat = _get_latents(model, bank, b_cells)
            mmd = mmd2_biased(a_lat, b_lat, bandwidths[cell_type])
            results[cell_type] = float(mmd.item())
        except Exception as e:
            print(f"[mmd_ab] failed for {cell_type}: {e}")
            continue

    return results


def _sample_cells(bank, cell_type, donors, n_cells_per_donor, rng):
    parts = []
    for d in donors:
        cells = bank.cells_for(cell_type, str(d))
        take = min(n_cells_per_donor, len(cells))
        if take > 0:
            parts.append(rng.choice(cells, size=take, replace=False))
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


def _get_latents(model, bank, cell_indices):
    import torch
    adata_subset = bank.adata[cell_indices]
    latents = model.get_latent_representation(adata_subset)
    return torch.tensor(latents, dtype=torch.float32)


class MMDABCallback(pl.Callback):
    def __init__(self, model, bank, mmd_plan, bandwidths, rng=None, *,
        n_cells_per_donor: int = 30, n_donors_per_type: int | None = None,
        compute_every: int = 1, log_per_type: bool = True, verbose: bool = True,):
        super().__init__()
        self.model = model
        self.bank = bank
        self.mmd_plan = mmd_plan
        self.bandwidths = bandwidths
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.n_cells_per_donor = n_cells_per_donor
        self.n_donors_per_type = n_donors_per_type
        self.compute_every = compute_every
        self.log_per_type = log_per_type
        self.verbose = verbose

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.compute_every != 0: return

        per_type = compute_mmd_ab_per_type(model=self.model, bank=self.bank,
            mmd_plan=self.mmd_plan, bandwidths=self.bandwidths, rng=self.rng,
            n_cells_per_donor=self.n_cells_per_donor, 
            n_donors_per_type=self.n_donors_per_type,)

        if not per_type:
            if self.verbose:
                print("[mmd_ab] No valid cell types")
            return

        avg = float(np.mean(list(per_type.values())))

        pl_module.log("mmd_ab", avg, on_epoch=True, prog_bar=True)
        pl_module.log("lambda_mmd", pl_module.lambda_mmd, on_epoch=True)

        if self.log_per_type:
            for ct, val in per_type.items():
                safe = ct.replace(" ", "_").replace(",", "").replace("/", "_")
                pl_module.log(f"mmd_ab/{safe}", val, on_epoch=True)

        if self.verbose:
            train_loss = trainer.callback_metrics.get('train_loss')
            mmd_loss = trainer.callback_metrics.get('mmd_loss')
            train_str = f"{float(train_loss):.2f}" if train_loss is not None else "n/a"
            mmd_str = f"{float(mmd_loss):.6f}" if mmd_loss is not None else "n/a"
            print(f"\n[mmd_ab] epoch={trainer.current_epoch} avg={avg:.4f} "
                  f"types={len(per_type)} lambda={pl_module.lambda_mmd:.2f} "
                  f"elbo={train_str} mmd={mmd_str}")