"""
Test MMDDataLoader with synthetic data.

Run: python test/test_loader.py
"""
import numpy as np
import torch
import anndata as ad
from scipy.sparse import csr_matrix

import sys
sys.path.insert(0, '/home/vlad/scGRIP')

# Импорт из твоего модуля (адаптируй путь)
from scCuddle.data import BankData
from scCuddle.mmd import MMDPlan
from scCuddle.dataloader import MMDDataLoader


# ===== Утилита =====
def check(name, condition, msg=""):
    status = "✓" if condition else "✗"
    print(f"  {status} {name}" + (f" — {msg}" if msg else ""))
    if not condition:
        raise AssertionError(f"{name} failed: {msg}")


# ===== Синтетические данные =====
def make_synthetic_adata():
    """AnnData: 100 клеток × 20 генов, raw counts."""
    rng = np.random.default_rng(42)
    n_cells = 100
    n_genes = 20
    
    # Counts: пуассоновские
    X = rng.poisson(lam=2.0, size=(n_cells, n_genes)).astype(np.float32)
    
    obs = {
        "donor_id": [f"d{i % 5}" for i in range(n_cells)],  # 5 доноров
        "cell_type": [f"type{i % 3}" for i in range(n_cells)],  # 3 типа
        "dataset_id": [f"ds{i % 2}" for i in range(n_cells)],  # 2 датасета
    }
    
    adata = ad.AnnData(X=csr_matrix(X))
    adata.obs = ad.obs = obs
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    
    return adata


def make_mock_bank(adata):
    """BankData с полями, которые нужны MMDDataLoader."""
    # Доноры для A (все 5)
    A_donors = np.array(["d0", "d1", "d2", "d3", "d4"], dtype=object)
    B_donors = np.array([], dtype=object)
    
    # type_donor_cells: (cell_type, donor) → индексы клеток
    type_donor_cells = {}
    for ct in adata.obs["cell_type"].unique():
        for d in adata.obs["donor_id"].unique():
            mask = (adata.obs["cell_type"] == ct) & (adata.obs["donor_id"] == d)
            indices = np.where(mask)[0]
            if len(indices) > 0:
                type_donor_cells[(str(ct), str(d))] = indices
    
    # eligible_donors: cell_type → доноры
    eligible_donors = {}
    for ct in adata.obs["cell_type"].unique():
        donors = []
        for d in adata.obs["donor_id"].unique():
            if (str(ct), str(d)) in type_donor_cells:
                donors.append(str(d))
        eligible_donors[str(ct)] = np.array(sorted(donors), dtype=object)
    
    # Создаём BankData через dataclass (используем __init__ напрямую)
    bank = BankData(
        adata=adata,
        A_donors=A_donors,
        B_donors=B_donors,
        type_donor_cells=type_donor_cells,
        eligible_donors=eligible_donors,
    )
    return bank


def make_mock_plan(bank):
    """MMDPlan с bandwidths."""
    A_donors_by_type = {}
    bandwidths = {}
    for ct in bank.eligible_donors:
        a_elig = bank.A_eligible_donors(ct)
        if len(a_elig) >= 2:
            A_donors_by_type[ct] = a_elig
            bandwidths[ct] = (0.5, 1.0, 2.0)
    
    plan = MMDPlan(
        A_donors_by_type=A_donors_by_type,
        bandwidths=bandwidths,
        n_donors_per_type=3,
        k_cells_per_donor=5,
    )
    return plan


# ===== TEST 1: __len__ =====
def test_len():
    print("\n[TEST 1] __len__")
    adata = make_synthetic_adata()
    bank = make_mock_bank(adata)
    plan = make_mock_plan(bank)
    
    loader = MMDDataLoader(
        plan=plan, bank=bank, adata=adata,
        elbo_batch_size=10, rng=np.random.default_rng(0),
        n_batches=5, use_cuda=False,
    )
    
    check("len = 5", len(loader) == 5)
    check("iter works", iter(loader) is not None)


# ===== TEST 2: __iter__ =====
def test_iter():
    print("\n[TEST 2] __iter__")
    adata = make_synthetic_adata()
    bank = make_mock_bank(adata)
    plan = make_mock_plan(bank)
    
    loader = MMDDataLoader(
        plan=plan, bank=bank, adata=adata,
        elbo_batch_size=10, rng=np.random.default_rng(0),
        n_batches=3, use_cuda=False,
    )
    
    batches = list(loader)
    check("3 batches", len(batches) == 3)
    for i, b in enumerate(batches):
        check(f"batch {i} is dict", isinstance(b, dict))


# ===== TEST 3: структура батча =====
def test_batch_structure():
    print("\n[TEST 3] _make_batch structure")
    adata = make_synthetic_adata()
    bank = make_mock_bank(adata)
    plan = make_mock_plan(bank)
    
    loader = MMDDataLoader(
        plan=plan, bank=bank, adata=adata,
        elbo_batch_size=10, rng=np.random.default_rng(0),
        n_batches=1, use_cuda=False,
    )
    
    batch = next(iter(loader))
    
    required_keys = ["X", "batch_index", "library_size", "mmd_groups", "n_mmd_cells", "n_elbo_cells"]
    for key in required_keys:
        check(f"has {key}", key in batch)
    
    check("X is tensor", isinstance(batch["X"], torch.Tensor))
    check("batch_index is tensor", isinstance(batch["batch_index"], torch.Tensor))
    check("library_size is tensor", isinstance(batch["library_size"], torch.Tensor))
    check("mmd_groups is list", isinstance(batch["mmd_groups"], list))
    check("n_mmd_cells int", isinstance(batch["n_mmd_cells"], int))
    check("n_elbo_cells int", isinstance(batch["n_elbo_cells"], int))


# ===== TEST 4: размерности =====
def test_dimensions():
    print("\n[TEST 4] dimensions")
    adata = make_synthetic_adata()
    bank = make_mock_bank(adata)
    plan = make_mock_plan(bank)
    
    loader = MMDDataLoader(
        plan=plan, bank=bank, adata=adata,
        elbo_batch_size=10, rng=np.random.default_rng(0),
        n_batches=1, use_cuda=False,
    )
    
    batch = next(iter(loader))
    
    n_total = batch["n_mmd_cells"] + batch["n_elbo_cells"]
    
    check("X rows = n_total", batch["X"].shape[0] == n_total, f"got {batch['X'].shape[0]}, expected {n_total}")
    check("X cols = n_genes", batch["X"].shape[1] == adata.n_vars)
    check("batch_index len = n_total", batch["batch_index"].shape[0] == n_total)
    check("library_size len = n_total", batch["library_size"].shape[0] == n_total)


# ===== TEST 5: _sample_elbo_cells =====
def test_sample_elbo_cells():
    print("\n[TEST 5] _sample_elbo_cells")
    adata = make_synthetic_adata()
    bank = make_mock_bank(adata)
    plan = make_mock_plan(bank)
    
    loader = MMDDataLoader(
        plan=plan, bank=bank, adata=adata,
        elbo_batch_size=10, rng=np.random.default_rng(0),
        n_batches=1, use_cuda=False,
    )
    
    indices = loader._sample_elbo_cells()
    check("size = 10", len(indices) == 10, f"got {len(indices)}")
    check("unique", len(np.unique(indices)) == len(indices))
    check("all in adata range", (indices >= 0).all() and (indices < adata.n_obs).all())


# ===== TEST 6: _flatten_mmd_batch =====
def test_flatten_mmd_batch():
    print("\n[TEST 6] _flatten_mmd_batch")
    from scCuddle.mmd import sample_mmd_batch
    
    adata = make_synthetic_adata()
    bank = make_mock_bank(adata)
    plan = make_mock_plan(bank)
    
    loader = MMDDataLoader(
        plan=plan, bank=bank, adata=adata,
        elbo_batch_size=10, rng=np.random.default_rng(0),
        n_batches=1, use_cuda=False,
    )
    
    mmd_batch = sample_mmd_batch(plan, bank, loader.rng)
    check("mmd_batch not None", mmd_batch is not None)
    
    flat_indices, mmd_groups = loader._flatten_mmd_batch(mmd_batch)
    
    check("flat_indices 1D", flat_indices.ndim == 1)
    check("mmd_groups list", isinstance(mmd_groups, list))
    check("mmd_groups matches types", len(mmd_groups) == len(mmd_batch.donor_cells))
    
    # Проверка: локальные индексы корректны (все в диапазоне [0, len(flat_indices)))
    total_local = 0
    for type_groups in mmd_groups:
        for donor_groups in type_groups:
            for idx in donor_groups:
                check(f"idx {idx} in range", 0 <= idx < len(flat_indices))
                total_local += 1
    
    check("total local = n_flat", total_local == len(flat_indices))


# ===== TEST 7: _extract_X =====
def test_extract_X():
    print("\n[TEST 7] _extract_X")
    adata = make_synthetic_adata()
    bank = make_mock_bank(adata)
    plan = make_mock_plan(bank)
    
    loader = MMDDataLoader(
        plan=plan, bank=bank, adata=adata,
        elbo_batch_size=10, rng=np.random.default_rng(0),
        n_batches=1, use_cuda=False,
    )
    
    indices = np.array([0, 1, 2, 3, 4])
    X = loader._extract_X(indices)
    
    check("X dtype float32", X.dtype == torch.float32)
    check("X shape", X.shape == (5, adata.n_vars))
    check("X matches adata", torch.allclose(X, torch.tensor(adata.X[indices].toarray(), dtype=torch.float32)))


# ===== TEST 8: _extract_batch_index =====
def test_extract_batch_index():
    print("\n[TEST 8] _extract_batch_index")
    adata = make_synthetic_adata()
    bank = make_mock_bank(adata)
    plan = make_mock_plan(bank)
    
    loader = MMDDataLoader(
        plan=plan, bank=bank, adata=adata,
        elbo_batch_size=10, rng=np.random.default_rng(0),
        n_batches=1, use_cuda=False,
    )
    
    indices = np.array([0, 1, 2, 3, 4])
    batch_index = loader._extract_batch_index(indices)
    
    check("batch_index dtype long", batch_index.dtype == torch.long)
    check("batch_index shape", batch_index.shape == (5,))
    
    # Проверка значений
    batch_col = adata.obs["dataset_id"].astype("category")
    expected = batch_col.cat.codes.to_numpy()[indices]
    check("batch_index values", (batch_index.numpy() == expected).all())


# ===== TEST 9: _compute_library_size =====
def test_compute_library_size():
    print("\n[TEST 9] _compute_library_size")
    adata = make_synthetic_adata()
    bank = make_mock_bank(adata)
    plan = make_mock_plan(bank)
    
    loader = MMDDataLoader(
        plan=plan, bank=bank, adata=adata,
        elbo_batch_size=10, rng=np.random.default_rng(0),
        n_batches=1, use_cuda=False,
    )
    
    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    lib_size = loader._compute_library_size(X)
    
    expected = torch.log(torch.tensor([6.0, 15.0]) + 1e-6)
    check("library_size correct", torch.allclose(lib_size, expected, atol=1e-5))


# ===== TEST 10: batch order — ELBO first, MMD last =====
def test_batch_order():
    print("\n[TEST 10] batch order (ELBO first, MMD last)")
    adata = make_synthetic_adata()
    bank = make_mock_bank(adata)
    plan = make_mock_plan(bank)
    
    loader = MMDDataLoader(
        plan=plan, bank=bank, adata=adata,
        elbo_batch_size=10, rng=np.random.default_rng(0),
        n_batches=1, use_cuda=False,
    )
    
    batch = next(iter(loader))
    
    # n_elbo_cells первые строки X — ELBO
    # n_mmd_cells последние — MMD
    n_elbo = batch["n_elbo_cells"]
    n_mmd = batch["n_mmd_cells"]
    
    check("n_elbo = elbo_batch_size", n_elbo == 10, f"got {n_elbo}")
    check("n_mmd > 0", n_mmd > 0, f"got {n_mmd}")
    check("X rows = n_elbo + n_mmd", batch["X"].shape[0] == n_elbo + n_mmd)


# ===== TEST 11: интеграция с MMD loss =====
def test_mmd_integration():
    print("\n[TEST 11] MMD integration")
    from scCuddle.mmd import mmd_loss_from_batch
    
    adata = make_synthetic_adata()
    bank = make_mock_bank(adata)
    plan = make_mock_plan(bank)
    
    loader = MMDDataLoader(
        plan=plan, bank=bank, adata=adata,
        elbo_batch_size=10, rng=np.random.default_rng(0),
        n_batches=1, use_cuda=False,
    )
    
    batch = next(iter(loader))
    
    n_mmd = batch["n_mmd_cells"]
    
    # Симулируем латенты (n_total, n_latent)
    n_total = batch["X"].shape[0]
    z = torch.randn(n_total, 5, requires_grad=True)
    
    # MMD loss на последних n_mmd клетках
    z_mmd = z[-n_mmd:]  # MMD последние
    loss = mmd_loss_from_batch(z_mmd, batch["mmd_batch"], bank, loader.rng)
    check("loss scalar", loss.dim() == 0)
    check("loss ≥ 0", loss.item() >= 0)
    
    loss.backward()
    check("grad exists", z.grad is not None)


# ===== TEST 12: воспроизводимость по seed =====
def test_reproducibility():
    print("\n[TEST 12] reproducibility")
    adata = make_synthetic_adata()
    bank = make_mock_bank(adata)
    plan = make_mock_plan(bank)
    
    loader1 = MMDDataLoader(
        plan=plan, bank=bank, adata=adata,
        elbo_batch_size=10, rng=np.random.default_rng(42),
        n_batches=1, use_cuda=False,
    )
    loader2 = MMDDataLoader(
        plan=plan, bank=bank, adata=adata,
        elbo_batch_size=10, rng=np.random.default_rng(42),
        n_batches=1, use_cuda=False,
    )
    
    b1 = next(iter(loader1))
    b2 = next(iter(loader2))
    
    check("same X", torch.allclose(b1["X"], b2["X"]))
    check("same batch_index", (b1["batch_index"] == b2["batch_index"]).all())
    check("same mmd_groups", b1["mmd_groups"] == b2["mmd_groups"])


# ===== Run =====
if __name__ == "__main__":
    print("=" * 60)
    print("MMD DATALOADER TESTS")
    print("=" * 60)
    
    tests = [
        test_len,
        test_iter,
        test_batch_structure,
        test_dimensions,
        test_sample_elbo_cells,
        test_flatten_mmd_batch,
        test_extract_X,
        test_extract_batch_index,
        test_compute_library_size,
        test_batch_order,
        test_mmd_integration,
        test_reproducibility,
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