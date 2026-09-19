import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scvi
import torch
from scipy.stats import spearmanr

from sceris.lora import attach, default_targets, first_linear

COMPARTMENTS = [
    ("plasma",     ("plasmablast", "plasma cell")),
    ("pDC",        ("plasmacytoid dendritic",)),
    ("DC",         ("dendritic",)),
    ("monocyte",   ("monocyte", "macrophage")),
    ("NK",         ("natural killer",)),
    ("B",          ("b cell", "plasmablast")),
    ("CD4 T",      ("cd4",)),
    ("CD8 T",      ("cd8",)),
    ("other T",    ("t cell", "thymocyte", "mucosal invariant", "gamma-delta", "regulatory")),
    ("progenitor", ("progenitor", "hematopoietic stem")),
]

def compartment(labels):
    out = pd.Series("other", index=labels.index, dtype=object)
    low = labels.str.lower()
    assigned = pd.Series(False, index=labels.index)
    for name, keys in COMPARTMENTS:
        hit = pd.Series(False, index=labels.index)
        for k in keys:
            hit |= low.str.contains(k, regex=False)
        take = hit & ~assigned
        out[take] = name
        assigned |= take
    return out

p = argparse.ArgumentParser()
p.add_argument("--base-model", required=True)
p.add_argument("--run-dir", required=True)
p.add_argument("--old-adata", required=True)
p.add_argument("--new-adata", required=True)
p.add_argument("--rank", type=int, default=4)
p.add_argument("--donor", default="donor_id")
p.add_argument("--n-replay", type=int, default=70)
p.add_argument("--n-eval", type=int, default=25)
p.add_argument("--ns-energy", type=float, default=0.99)
p.add_argument("--batch", type=int, default=2048)
p.add_argument("--out-csv", default=None)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = scvi.model.SCVI.load(args.base_model)
model.module.eval()
census_base = ad.read_h5ad(args.old_adata)

rng = np.random.default_rng(0)  # тот же сплит, что в обучении
replay_idx, eval_idx = [], []
for donor, idx in census_base.obs.groupby(args.donor, observed=True).indices.items():
    take = rng.choice(idx, size=min(args.n_replay, len(idx)), replace=False)
    replay_idx.append(take)
    remaining = np.setdiff1d(idx, take)
    eval_idx.append(rng.choice(remaining, size=min(args.n_eval, len(remaining)), replace=False))
replay_idx = np.concatenate(replay_idx)
eval_idx = np.concatenate(eval_idx)
replay = census_base[replay_idx].copy()
eval_old = census_base[eval_idx].copy()

mu_ref = model.get_latent_representation(replay)
mu_ref_sigma = mu_ref.std(axis=0)
mu_eval_ref = model.get_latent_representation(eval_old)

new_adata = ad.read_h5ad(args.new_adata)
combined = ad.concat([replay, new_adata], join="outer", label="_replay_source",
                     keys=["replay", "new"], index_unique="_")
scvi.model.SCVI.setup_anndata(combined, batch_key=args.donor)
q_model = scvi.model.SCVI.load(args.run_dir, adata=combined)
q_model.module.eval()
q_model.module.to(device)
adapters = attach(q_model.module, rank=args.rank, n_cov=0, device=device)
adapters.load_state_dict(torch.load(f"{args.run_dir}/adapters.pt", map_location=device))


def collect(scvi_model, blocks, adata, indices=None, per_cell=None):  # моменты X^T X, либо поклеточная энергия вне подпространства
    C, rows = {}, {n: [] for n in blocks}
    handles = []
    for name, blk in blocks.items():
        def pre(_m, inp, name=name):
            x = inp[0].detach().double()
            if per_cell is None:
                C[name] = C[name] + x.T @ x if name in C else x.T @ x
            else:
                P = per_cell[name]
                rows[name].append(((x @ P) ** 2).sum(1).div(( x ** 2).sum(1).clamp_min(1e-12)).cpu().numpy())
        handles.append(first_linear(blk).register_forward_pre_hook(pre))
    with torch.no_grad():
        scvi_model.get_latent_representation(adata, indices=indices, batch_size=args.batch)
    for h in handles:
        h.remove()
    if per_cell is None:
        return {n: c.cpu() for n, c in C.items()}
    return {n: np.concatenate(r) for n, r in rows.items()}


ns_idx = np.setdiff1d(np.arange(census_base.n_obs), eval_idx)
base_blocks = {n: b for n, (_i, _o, b) in default_targets(model.module).items()}
q_blocks = {n: b for n, (_i, _o, b) in default_targets(q_model.module).items()}
C = collect(model, base_blocks, census_base, indices=ns_idx)
proj = {}
for name, Cn in C.items():
    evals, evecs = torch.linalg.eigh(Cn)
    evals, evecs = evals.flip(0).clamp_min(0), evecs.flip(1)
    k = int(((torch.cumsum(evals, 0) / evals.sum()) < args.ns_energy).sum()) + 1
    U = evecs[:, :k]
    proj[name] = (torch.eye(U.shape[0], dtype=torch.float64) - U @ U.T).to(device)
    print(f"{name}: k={k}", flush=True)

# поклеточная энергия вне подпространства на ФАКТИЧЕСКОМ входе блока (с адаптерами) + норма коррекции адаптера
delta_norm = {n: [] for n in adapters.adapters}
dh = []
for name, a in adapters.adapters.items():
    dh.append(a.register_forward_hook(lambda _m, _i, out, name=name: delta_norm[name].append(out.detach().norm(dim=1).cpu().numpy())))
outside = collect(q_model, q_blocks, eval_old, per_cell=proj)
for h in dh:
    h.remove()
delta_norm = {n: np.concatenate(v) for n, v in delta_norm.items()}

mu_eval_adapted = q_model.get_latent_representation(eval_old)
diff = mu_eval_adapted - mu_eval_ref
df = pd.DataFrame({
    "cell_type": eval_old.obs["cell_type"].astype(str).to_numpy(),
    "compartment": compartment(eval_old.obs["cell_type"].astype(str)).to_numpy(),
    "donor": eval_old.obs[args.donor].astype(str).to_numpy(),
    "drift_raw": np.linalg.norm(diff, axis=1),
    "drift_sigma": np.linalg.norm(diff / mu_ref_sigma, axis=1),
    "outside_enc0": outside["enc0"], "outside_enc1": outside["enc1"],
    "delta_enc0": delta_norm["enc0"], "delta_enc1": delta_norm["enc1"],
})

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 80)
q = [0.5, 0.9, 0.99, 0.999, 1.0]
print("\n=== квантили поклеточного дрейфа (held-out census) ===")
print(df[["drift_raw", "drift_sigma", "outside_enc0", "outside_enc1", "delta_enc0", "delta_enc1"]].quantile(q).round(4).to_string())

print("\n=== Spearman: дрейф vs ===")
for col in ["outside_enc0", "outside_enc1", "delta_enc0", "delta_enc1"]:
    print(f"  {col}: {spearmanr(df['drift_raw'], df[col]).statistic:+.3f}")

thr = df["drift_raw"].quantile(0.99)
df["top1pct"] = df["drift_raw"] >= thr
by_ct = df.groupby("cell_type").agg(n=("drift_raw", "size"), drift_mean=("drift_raw", "mean"),
                                    drift_p99=("drift_raw", lambda s: s.quantile(0.99)),
                                    outside_enc0=("outside_enc0", "mean"), outside_enc1=("outside_enc1", "mean"),
                                    n_top1pct=("top1pct", "sum"))
by_ct["enrich_top1pct"] = (by_ct["n_top1pct"] / df["top1pct"].sum()) / (by_ct["n"] / len(df))
print(f"\n=== по cell_type (top1% порог drift_raw >= {thr:.3f}) ===")
print(by_ct.sort_values("drift_mean", ascending=False).round(4).to_string())

by_d = df.groupby("donor")["top1pct"].sum().sort_values(ascending=False)
print(f"\n=== концентрация top1% по донорам: {int((by_d > 0).sum())} доноров из {df['donor'].nunique()} содержат хотя бы одну; "
      f"топ-10 доноров держат {by_d.head(10).sum()} из {int(df['top1pct'].sum())} ===")
print(by_d.head(10).to_string())

print("\n=== топ-25 клеток по дрейфу ===")
print(df.sort_values("drift_raw", ascending=False).head(25).round(4).to_string(index=False))

if args.out_csv:
    df.to_csv(args.out_csv, index=False)
