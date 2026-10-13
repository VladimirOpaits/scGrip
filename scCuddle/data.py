
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import anndata as ad  #type: ignore
import numpy as np

MMD_CELL_TYPES: tuple[str, ...] = (
    "T cells",
    "Monocytes",
    "B cells",
    "ILC",
    "DC",
    "Macrophages",
    "Myelocytes",
)

@dataclass(frozen = True)
class BankData:
    adata: ad.AnnData
    A_donors: np.ndarray
    B_donors: np.ndarray
    
    type_donor_cells: dict[tuple[str, str], np.ndarray]  # (cell_type, donor_id), массив индексов клеток в AnnData для MMD
    eligible_donors: dict[str, np.ndarray]               # донор-тип с достаточным количеством клеток на тип
    
    def __post_init__(self): 
        if np.intersect1d(self.A_donors, self.B_donors).size:
            raise ValueError("A and B donors overlap")
        adata_donors = set(self.adata.obs["donor_id"].astype(str).unique())
        for label, donors in (("A", self.A_donors), ("B", self.B_donors)):
            missing = set(map(str, donors)) - adata_donors
            if missing:
                raise ValueError(f"{label} contains unknown donors: {sorted(missing)[:3]}")
        available = set(self.type_donor_cells.keys())
        for cell_type, donors in self.eligible_donors.items():
            for donor in donors:
                if (str(cell_type), str(donor)) not in available:
                    raise ValueError(
                        f"eligible_donors has ({cell_type}, {donor}) "
                        "not in type_donor_cells")
            
    @property
    def A_donor_set(self) -> set[str]:
        return set(map(str, self.A_donors)) # str гарантирует конкретный тип, set дает O(1) вставку 
    
    def A_eligible_donors(self, cell_type: str) -> np.ndarray: # для MMD
        eligible = self.eligible_donors.get(cell_type, np.array([], dtype=object))
        a_set = self.A_donor_set
        return np.array(
            [d for d in eligible if str(d) in a_set],
            dtype=object,
        )
        
    def cells_for(self, cell_type: str, donor: str) -> np.ndarray:   # массив индексов для конкретной пары
        return self.type_donor_cells[(str(cell_type), str(donor))]
    
def _split_donors(obs: pd.DataFrame, *, donor_col: str, batch_col: str, holdout_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]: #type: ignore
    df = obs[[donor_col, batch_col]].drop_duplicates()  
    df[donor_col] = df[donor_col].astype(str)
    df[batch_col] = df[batch_col].astype(str)

    rng = np.random.default_rng(seed)
    A_parts: list[np.ndarray] = []
    B_parts: list[np.ndarray] = []

    for _, group in df.groupby(batch_col, observed=True):   # убеждаемся, что каждый датасет с двумя и больше донорами будет репрезентирован и в A и в B
        donors = np.sort(group[donor_col].to_numpy())
        n = len(donors)
        if n < 2:
            A_parts.append(donors)
            continue
        n_b = max(1, min(n - 1, int(round(n * holdout_fraction))))
        b_idx = rng.choice(n, size=n_b, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[b_idx] = True
        A_parts.append(donors[~mask])
        B_parts.append(donors[mask])

    A = np.sort(np.concatenate(A_parts)) if A_parts else np.empty(0, dtype=object)
    B = np.sort(np.concatenate(B_parts)) if B_parts else np.empty(0, dtype=object)
    return A, B

def _build_type_donor_index(obs: pd.DataFrame, *, donor_col: str, cell_type_col: str) -> dict[tuple[str, str], np.ndarray]: #type: ignore
    index: dict[tuple[str, str], np.ndarray] = {}
    for (ct, d), group in obs.groupby([cell_type_col, donor_col], observed = True):       # (ct, d) ключ группы, group это сам датафрейм с содержимым
        index[(str(ct), str(d))] = group.index.to_numpy(dtype = np.int64)
    return index

def _build_eligible_donors(type_donor_cells: dict[tuple[str, str], np.ndarray], *, min_cells: int, A_donors: set[str], B_donors: set[str], min_A_eligible: int = 4, # A B для дополнительной проверки
    min_B_eligible: int = 1) -> dict[str, np.ndarray]:  # type_donor_cells это результат _build_type_donor_index, содержит numpy массив с индексом клеток для всех типов всех доноров, прошедших отбор
    by_type: dict[str, list[str]] = {}
    for (ct, donor), cells in type_donor_cells.items():
        if cells.size >= min_cells:
            by_type.setdefault(ct, []).append(donor)
            
    out: dict[str, np.ndarray] = {}
    for ct, donors in by_type.items():
        n_A = sum(1 for d in donors if d in A_donors)
        n_B = sum(1 for d in donors if d in B_donors)
        if n_A >= min_A_eligible and n_B >= min_B_eligible:        # проверяем выполнение MMD требований
            out[ct] = np.array(sorted(donors), dtype = object)     # переводим в numpy API только для консистентности
        else:
            print(f"  skipping {ct!r}: A={n_A}, B={n_B} "
                f"(need A>={min_A_eligible}, B>={min_B_eligible})")
    
    return out



def load_bank(path: Path, *, donor_col: str = 'donor_id', cell_type_col: str = "celltypist", batch_col: str = 'dataset_id', holdout_frac: float = 0.2, split_seed: int = 0, 
    min_cells_per_donor_type: int = 5, min_A_eligible: int = 4, min_B_eligible: int = 1, mmd_types: tuple[str, ...] = MMD_CELL_TYPES) -> BankData:        # * делает передачу только по имени 
    if not path.exists():
        raise FileNotFoundError(path)
    if not mmd_types:
        raise ValueError("mmd_types cannot be empty")

    adata = ad.read_h5ad(path)
    adata.obs = adata.obs.reset_index(drop=True)

    for col in (donor_col, cell_type_col, batch_col):
        if col not in adata.obs.columns:
            raise ValueError(f"missing obs column: {col!r}")
        adata.obs[col] = adata.obs[col].astype(str)

    A_donors, B_donors = _split_donors(adata.obs, donor_col=donor_col, batch_col=batch_col, holdout_fraction=holdout_frac, seed=split_seed)

    type_donor_cells = _build_type_donor_index(adata.obs[adata.obs[cell_type_col].isin(mmd_types)], donor_col=donor_col, cell_type_col=cell_type_col)

    eligible_donors = _build_eligible_donors(type_donor_cells, min_cells=min_cells_per_donor_type, A_donors=set(map(str, A_donors)), B_donors=set(map(str, B_donors)), min_A_eligible=min_A_eligible, min_B_eligible=min_B_eligible)

    print(f"bank {path.name}: {adata.shape[0]} cells x {adata.shape[1]} genes, "
        f"A={A_donors.size} B={B_donors.size} donors, "
        f"{len(type_donor_cells)} (type, donor) pairs, "
        f"{len(eligible_donors)} eligible cell types")

    return BankData(adata=adata, A_donors=A_donors, B_donors=B_donors, type_donor_cells=type_donor_cells, eligible_donors=eligible_donors)

def load_query_signal(path: Path) -> ad.AnnData:
    if not path.exists():
        raise FileNotFoundError(path)
    adata = ad.read_h5ad(path)
    return adata