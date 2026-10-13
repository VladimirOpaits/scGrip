from __future__ import annotations

from dataclasses import dataclass
from .data import BankData
import numpy as np
import torch
from scipy.sparse import issparse
from .mmd import MMDPlan, sample_mmd_batch
from anndata import AnnData
from typing import Iterator


@dataclass(frozen = True)
class MMDDataLoader:
    plan: MMDPlan
    bank: BankData
    adata: AnnData
    elbo_batch_size: int
    rng: np.random.Generator
    n_batches: int
    use_cuda: bool = True
    
    def __iter__(self) -> Iterator[dict]:   # что бы можно было проходиться по содержимому через for
        for _ in range(self.n_batches):
            batch = self._make_batch()
            if batch is not None:
                yield batch      # генератор, автоматически создает итератор
    
    def __len__(self): return self.n_batches
    
    def _make_batch(self) -> dict | None:
        mmd_batch = sample_mmd_batch(self.plan, self.bank, self.rng)
        if mmd_batch is None: return None
        
        elbo_indices = self._sample_elbo_cells()   
        mmd_indices_flat, mmd_groups = self._flatten_mmd_batch(mmd_batch)
        all_indices = np.concatenate([elbo_indices, mmd_indices_flat])
        
        X = self._extract_X(all_indices)
        batch_index = self._extract_batch_index(all_indices)
        library_size = self._compute_library_size(X)    # scVI использует как scale фактор
        categorical_covariates = torch.zeros(X.shape[0], 0, dtype=torch.long, device=X.device)
        labels = torch.zeros(X.shape[0], 1, dtype=torch.long, device=X.device)
        
        return {
            "X": X,
            "batch": batch_index,
            "library_size": library_size,
            "categorical_covariates": categorical_covariates,
            "labels": labels,
            "mmd_batch": mmd_batch,
            "mmd_groups": mmd_groups,
            "n_mmd_cells": len(mmd_indices_flat),
            "n_elbo_cells": len(elbo_indices),
        }
        
        
        
        
        
        
    def _sample_elbo_cells(self) -> np.ndarray:
        all_a_cells = []
        for cell_type, donors in self.plan.A_donors_by_type.items():
            for donor in donors:
                cells = self.bank.cells_for(cell_type, str(donor))
                all_a_cells.append(cells)
        if not all_a_cells:
            return np.empty(0, dtype=np.int64)
        all_a_cells = np.concatenate(all_a_cells)

        n_take = min(self.elbo_batch_size, len(all_a_cells))
        return self.rng.choice(all_a_cells, size=n_take, replace=False)

    def _flatten_mmd_batch(
        self, mmd_batch
    ) -> tuple[np.ndarray, list[list[list[int]]]]:
        flat_indices = []
        mmd_groups: list[list[list[int]]] = []

        cursor = 0  # локальный счутчик в батче
        for cell_type in mmd_batch.donor_cells:
            type_groups: list[list[int]] = []
            for donor, cells in mmd_batch.donor_cells[cell_type].items():
                n = len(cells)
                type_groups.append(list(range(cursor, cursor + n)))
                flat_indices.append(cells)
                cursor += n
            mmd_groups.append(type_groups)

        if not flat_indices:
            return np.empty(0, dtype=np.int64), []

        return np.concatenate(flat_indices), mmd_groups

    def _extract_X(self, indices: np.ndarray) -> torch.Tensor:
        X = self.adata.X[indices]
        if issparse(X):
            X = X.toarray()
        X = torch.tensor(X, dtype=torch.float32)
        if self.use_cuda:
            X = X.cuda()
        return X

    def _extract_batch_index(self, indices: np.ndarray) -> torch.Tensor:
        batch_col = self.bank.adata.obs["dataset_id"].astype("category")
        codes = batch_col.cat.codes.to_numpy()[indices]
        batch_index = torch.tensor(codes, dtype=torch.long).unsqueeze(1)
        if self.use_cuda:
            batch_index = batch_index.cuda()
        return batch_index

    def _compute_library_size(self, X: torch.Tensor) -> torch.Tensor:
        return torch.log(X.sum(dim=1) + 1e-6).unsqueeze(1)