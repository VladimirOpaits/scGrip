from __future__ import annotations
import numpy as np
import torch
from anndata import AnnData
from scvi.train import TrainingPlan

from .data import BankData
from .dataloader import MMDDataLoader
from .mmd import MMDPlan, mmd_loss_from_batch

class MMDTrainingPlan(TrainingPlan):
    def __init__(self, module, bank: BankData, mmd_plan: MMDPlan, adata: AnnData, *, 
                 elbo_batch_size: int = 512, lambda_mmd: float = 100.0, seed: int = 0, 
                 use_cuda: bool | None = None, n_batches: int | None = None,
                 kl_weight: float = 0.0375, **kwargs):
        super().__init__(module, n_epochs_kl_warmup=None, n_steps_kl_warmup=None,
                         max_kl_weight=kl_weight, **kwargs)
        if len(mmd_plan.cell_types) == 0: raise ValueError("MMDPlan is empty. No eligible cell types. " "Check load_bank / compute_bandwidths_per_type.")
        
        self.bank = bank
        self.mmd_plan = mmd_plan
        self.adata = adata 
        self.elbo_batch_size = elbo_batch_size
        self.lambda_mmd = lambda_mmd
        self.rng = np.random.default_rng(seed)
        
        if use_cuda is None: use_cuda = torch.cuda.is_available()
        self.use_cuda = use_cuda

        if n_batches is None: n_batches = self._compute_n_batches()
        self.n_batches = n_batches
        
        self._loader = MMDDataLoader(
            plan=self.mmd_plan,
            bank=self.bank,
            adata=self.adata,
            elbo_batch_size=self.elbo_batch_size,
            rng=self.rng,
            n_batches=self.n_batches,
            use_cuda=self.use_cuda)

    def train_dataloader(self): return self._loader # Lightning хук

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.module.train()
        
    def _compute_n_batches(self) -> int:   # сколько батчей за эпоху
            n_a = 0
            for cell_type, donors in self.mmd_plan.A_donors_by_type.items():
                for donor in donors:
                    n_a += len(self.bank.cells_for(cell_type, str(donor)))  # получаем массив индексов для конкретной пары
            return max(1, n_a // self.elbo_batch_size)
        
        
    def training_step(self, batch, batch_idx):
            inference_outputs, generative_outputs, scvi_losses = self.module(
                batch, loss_kwargs={"kl_weight": self.kl_weight})  # inference = latent с z, qz: torch.distribution.Normal, qzm, qzv, library, ql
            elbo_loss = scvi_losses.loss
            
            n_mmd = batch.get("n_mmd_cells", 0)
            if n_mmd == 0 or "mmd_batch" not in batch: raise RuntimeError(f"Expected MMD batch, got n_mmd={n_mmd}, "f"has_mmd_batch={'mmd_batch' in batch}")
            
            z = inference_outputs["z"]
            z_mmd = z[-n_mmd:]   # из-за concat раннее
            
            mmd_loss = mmd_loss_from_batch(z_mmd, batch["mmd_batch"], self.bank, self.rng)
            total_loss = elbo_loss + self.lambda_mmd * mmd_loss
            
            self.log("train_loss", elbo_loss, prog_bar=True)
            self.log("mmd_loss", mmd_loss, prog_bar=True)
            self.log("mmd_weighted", self.lambda_mmd * mmd_loss)
            self.log("kl_weight", self.kl_weight)

            return total_loss