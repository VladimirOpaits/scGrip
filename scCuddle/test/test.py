import numpy as np
import torch
from dataclasses import dataclass

class MockBank:
    
    def __init__(self, cell_types, donors, cells_per_donor):
        self._cells = {}
        all_idx = 0
        for ct in cell_types:
            for d in donors:
                n = cells_per_donor
                self._cells[(str(ct), str(d))] = np.arange(all_idx, all_idx + n)
                all_idx += n
    
    def cells_for(self, cell_type, donor):
        return self._cells[(str(cell_type), str(donor))]


import sys
sys.path.insert(0, '/home/vlad/scGRIP')

from scCuddle.mmd import (
    multiscale_rbf,
    mmd2_biased,
    _median_heuristic,
    compute_bandwidths_per_type,
    MMDPlan,
    MMDBatch,
    _balanced_rest_sample,
    sample_mmd_batch,
    _each_vs_rest_mmd,
    mmd_loss_from_batch,
)


def check(name, condition, msg=""):
    status = "PASS" if condition else "FAIL"
    marker = "✓" if condition else "✗"
    print(f"  {marker} {name}" + (f" — {msg}" if msg else ""))
    if not condition:
        raise AssertionError(f"{name} failed: {msg}")


def test_multiscale_rbf():
    print("\n[TEST 1] multiscale_rbf")
    
    x = torch.randn(10, 5)
    y = torch.randn(15, 5)
    sigmas = (0.5, 1.0, 2.0)
    
    K = multiscale_rbf(sigmas, x, y)
    
    check("shape", K.shape == (10, 15), f"got {K.shape}")
    check("non-negative", (K >= 0).all().item())
    check("bounded by 1", (K <= 1.0 + 1e-6).all().item())
    
    K_xx = multiscale_rbf(sigmas, x, x)
    check("symmetry K(x,x)", torch.allclose(K_xx, K_xx.T, atol=1e-5))
    
    diag = torch.diag(K_xx)
    check("diag ≈ 1", torch.allclose(diag, torch.ones_like(diag), atol=1e-4))
    
    try:
        multiscale_rbf(sigmas, torch.randn(10), torch.randn(15))
        check("dim check", False, "should raise")
    except ValueError:
        check("dim check", True)
    
    try:
        multiscale_rbf((), x, y)
        check("empty sigmas", False, "should raise")
    except ValueError:
        check("empty sigmas", True)


def test_mmd2_biased():
    print("\n[TEST 2] mmd2_biased")
    
    sigmas = (0.5, 1.0, 2.0)
    
    torch.manual_seed(42)
    x = torch.randn(100, 5)
    y = x.clone()  
    mmd_same = mmd2_biased(x, y, sigmas)
    check("MMD(x,x) ≈ 0", mmd_same.item() < 0.01, f"got {mmd_same.item():.6f}")
    
    x = torch.randn(100, 5)
    y = torch.randn(100, 5) + 5.0  
    mmd_diff = mmd2_biased(x, y, sigmas)
    check("MMD(x, x+5) > 0", mmd_diff.item() > 0.1, f"got {mmd_diff.item():.4f}")
    
    for _ in range(10):
        x = torch.randn(50, 5)
        y = torch.randn(50, 5)
        mmd = mmd2_biased(x, y, sigmas)
        check(f"MMD ≥ 0 (iter)", mmd.item() >= 0, f"got {mmd.item():.6f}")


def test_median_heuristic():
    print("\n[TEST 3] _median_heuristic")
    
    x = torch.arange(100).float().unsqueeze(1)  
    median = _median_heuristic(x, max_points=1000, floor=1e-6)
    check("median reasonable", 20 < median < 50, f"got {median:.2f}")
    
    x = torch.tensor([[1.0]])
    median = _median_heuristic(x, max_points=1000, floor=1e-6)
    check("single point → floor", median == 1e-6)
    
    x = torch.ones(10, 3)
    median = _median_heuristic(x, max_points=1000, floor=1e-6)
    check("all same → floor", median == 1e-6)


def test_compute_bandwidths():
    print("\n[TEST 4] compute_bandwidths_per_type")
    
    latents = {
        "T cells": torch.randn(500, 10) * 2.0,
        "B cells": torch.randn(300, 10) * 1.0,
        "Monocytes": torch.randn(400, 10) * 0.5,
    }
    scales = (0.5, 1.0, 2.0)
    
    bandwidths = compute_bandwidths_per_type(
        latents,
        scales=scales,
        max_points=200,
        floor=1e-6,
    )
    
    check("3 types", len(bandwidths) == 3)
    check("T cells 3 sigmas", len(bandwidths["T cells"]) == 3)
    
    check(
        "T > B (scale)",
        bandwidths["T cells"][1] > bandwidths["B cells"][1],
        f"T={bandwidths['T cells'][1]:.3f}, B={bandwidths['B cells'][1]:.3f}",
    )


def test_balanced_rest_sample():
    print("\n[TEST 5] _balanced_rest_sample")
    
    rng = np.random.default_rng(42)
    
    donor_cells = {
        "d1": np.arange(0, 100),
        "d2": np.arange(100, 200),
        "d3": np.arange(200, 300),
        "d4": np.arange(300, 400),
    }
    
    k_cells = 60
    sample = _balanced_rest_sample(donor_cells, exclude_donor="d1", k_cells=k_cells, rng=rng)
    
    # DEBUG
    print(f"  sample size: {len(sample)}")
    print(f"  sample min: {sample.min()}, max: {sample.max()}")
    print(f"  sample first 10: {sample[:10]}")
    print(f"  sample last 10: {sample[-10:]}")
    print(f"  unique cells: {len(np.unique(sample))}")
    
    for d, cells in [("d2", (100, 200)), ("d3", (200, 300)), ("d4", (300, 400))]:
        n = ((sample >= cells[0]) & (sample < cells[1])).sum()
        print(f"  {d}: {n} cells")
    
    check("size = k_cells", len(sample) == k_cells, f"got {len(sample)}")
    check("no d1 cells", (sample < 100).sum() == 0)
    check("unique cells", len(np.unique(sample)) == len(sample))
    
    for d, cells in [("d2", (100, 200)), ("d3", (200, 300)), ("d4", (300, 400))]:
        n = ((sample >= cells[0]) & (sample < cells[1])).sum()
        check(f"{d} present", n > 0, f"got {n}")


def test_sample_mmd_batch():
    print("\n[TEST 6] sample_mmd_batch")
    
    rng = np.random.default_rng(42)
    
    bank = MockBank(
        cell_types=["T cells", "B cells", "Monocytes"],
        donors=[f"d{i}" for i in range(10)],
        cells_per_donor=50,
    )
    
    plan = MMDPlan(
        A_donors_by_type={
            "T cells": np.array([f"d{i}" for i in range(10)]),
            "B cells": np.array([f"d{i}" for i in range(10)]),
            "Monocytes": np.array([f"d{i}" for i in range(10)]),
        },
        bandwidths={
            "T cells": (0.5, 1.0, 2.0),
            "B cells": (0.5, 1.0, 2.0),
            "Monocytes": (0.5, 1.0, 2.0),
        },
        n_donors_per_type=5,
        k_cells_per_donor=20,
    )
    
    batch = sample_mmd_batch(plan, bank, rng)
    
    check("batch not None", batch is not None)
    check("3 types", len(batch.donor_cells) == 3)
    
    for ct in batch.donor_cells:
        donors = batch.donor_cells[ct]
        check(f"{ct}: ≤ 5 donors", len(donors) <= 5, f"got {len(donors)}")
        for d, cells in donors.items():
            check(f"{ct}/{d}: ≤ 20 cells", len(cells) <= 20, f"got {len(cells)}")
            check(f"{ct}/{d}: ≥ 2 cells", len(cells) >= 2)


def test_each_vs_rest_mmd():
    print("\n[TEST 7] _each_vs_rest_mmd")
    
    rng = np.random.default_rng(42)
    
    donor_cells = {
        "d1": np.arange(0, 20),
        "d2": np.arange(20, 40),
        "d3": np.arange(40, 60),
    }
    
    latents_same = torch.randn(60, 5)
    mmd_same = _each_vs_rest_mmd(
        donor_cells, latents_same, (0.5, 1.0, 2.0), k_cells=15, rng=rng,
    )
    check("same dist → small", mmd_same.item() < 0.1, f"got {mmd_same.item():.4f}")
    
    latents_shifted = torch.randn(60, 5)
    latents_shifted[:20] += 5.0 
    mmd_diff = _each_vs_rest_mmd(
        donor_cells, latents_shifted, (0.5, 1.0, 2.0), k_cells=15, rng=rng,
    )
    check("shifted → large", mmd_diff.item() > 0.1, f"got {mmd_diff.item():.4f}")


def test_mmd_loss_from_batch():
    print("\n[TEST 8] mmd_loss_from_batch (full pipeline)")
    
    rng = np.random.default_rng(42)
    
    bank = MockBank(
        cell_types=["T cells", "B cells"],
        donors=[f"d{i}" for i in range(5)],
        cells_per_donor=30,
    )
    
    plan = MMDPlan(
        A_donors_by_type={
            "T cells": np.array([f"d{i}" for i in range(5)]),
            "B cells": np.array([f"d{i}" for i in range(5)]),
        },
        bandwidths={
            "T cells": (0.5, 1.0, 2.0),
            "B cells": (0.5, 1.0, 2.0),
        },
        n_donors_per_type=4,
        k_cells_per_donor=15,
    )
    
    batch = sample_mmd_batch(plan, bank, rng)
    check("batch created", batch is not None)
    
    n_total = sum(
        len(idx)
        for ct in batch.donor_cells
        for idx in batch.donor_cells[ct].values()
    )
    
    latents = torch.randn(n_total, 5, requires_grad=True)
    
    loss = mmd_loss_from_batch(latents, batch, bank, rng)
    
    check("loss is scalar", loss.dim() == 0)
    check("loss ≥ 0", loss.item() >= 0)
    check("loss is tensor", isinstance(loss, torch.Tensor))
    
    loss.backward()
    check("grad exists", latents.grad is not None)
    check("grad non-zero", latents.grad.abs().sum().item() > 0)


def test_gradient_flow():
    print("\n[TEST 9] gradient flow")
    
    rng = np.random.default_rng(42)
    
    bank = MockBank(
        cell_types=["T cells"],
        donors=[f"d{i}" for i in range(4)],
        cells_per_donor=20,
    )
    
    plan = MMDPlan(
        A_donors_by_type={"T cells": np.array([f"d{i}" for i in range(4)])},
        bandwidths={"T cells": (0.5, 1.0, 2.0)},
        n_donors_per_type=4,
        k_cells_per_donor=10,
    )
    
    batch = sample_mmd_batch(plan, bank, rng)
    n_total = sum(
        len(idx) for ct in batch.donor_cells
        for idx in batch.donor_cells[ct].values()
    )
    
    latents = torch.randn(n_total, 5, requires_grad=True)
    loss = mmd_loss_from_batch(latents, batch, bank, rng)
    loss.backward()
    
    grad_norm = latents.grad.norm().item()
    check("grad norm > 0", grad_norm > 0, f"got {grad_norm:.4f}")
    
    optimizer = torch.optim.SGD([latents], lr=0.1)
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = mmd_loss_from_batch(latents, batch, bank, rng)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    check("loss decreases", losses[-1] < losses[0], f"{losses[0]:.4f} → {losses[-1]:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("MMD MODULE TESTS")
    print("=" * 60)
    
    tests = [
        test_multiscale_rbf,
        test_mmd2_biased,
        test_median_heuristic,
        test_compute_bandwidths,
        test_balanced_rest_sample,
        test_sample_mmd_batch,
        test_each_vs_rest_mmd,
        test_mmd_loss_from_batch,
        test_gradient_flow,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  ✗ FAILED: {test.__name__}")
            print(f"    {type(e).__name__}: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULT: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        exit(1)