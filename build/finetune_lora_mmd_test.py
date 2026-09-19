import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scvi
import torch

from sceris.lora import attach, verify_identical

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

STRAT_MIN_CELLS = 20
STRAT_MIN_DONOR_FRAC = 0.5

def stratified_dist(mu, ct, donor, ref_centroid, sigma):  # CHANGED: + sigma, для нормировки в тех же единицах что anchor
    frame = pd.DataFrame(mu)
    frame["compartment"] = ct.to_numpy() if hasattr(ct, "to_numpy") else ct
    frame["donor"] = donor
    dc_grp = frame.groupby(["compartment", "donor"], observed=True)
    dc = dc_grp.mean()[dc_grp.size() >= STRAT_MIN_CELLS]
    shared = [c for c in ref_centroid.index if c in dc.index.get_level_values("compartment")]
    per_ct = {}
    for c in shared:
        sub = dc.xs(c, level="compartment")
        per_ct[c] = pd.Series(np.linalg.norm((sub.to_numpy() - ref_centroid.loc[c].to_numpy()) / sigma, axis=1), index=sub.index)  # CHANGED: делим на sigma до нормы
    df = pd.DataFrame(per_ct)
    keep = df.columns[df.notna().mean() >= STRAT_MIN_DONOR_FRAC] if len(df) else []
    per_donor = df[keep].mean(axis=1) if len(keep) else pd.Series(dtype=float)
    return per_donor, keep


# ---- NEW: stratified-MMD building blocks, портированы из scEris/build/finetune_mmd_stratified.py ----
MMD_MIN_PER_SIDE = 32       # NEW: компартмент нужен >= столько клеток на КАЖДОЙ стороне, чтобы участвовать в MMD
MMD_K_PER_COMPARTMENT = 64  # NEW: сколько клеток на компартмент на сторону сэмплируем на шаг

def rbf_kernel(x, y, sigmas):  # NEW
    x2 = (x ** 2).sum(1, keepdim=True)
    y2 = (y ** 2).sum(1, keepdim=True)
    dist2 = (x2 + y2.T - 2 * x @ y.T).clamp_min(0)
    K = 0
    for s in sigmas:
        K = K + torch.exp(-dist2 / (2 * s ** 2))
    return K

def mmd2(x, y, sigmas):  # NEW: Kyy отброшено — y это замороженный mu_ref, без градиента
    return rbf_kernel(x, x, sigmas).mean() - 2 * rbf_kernel(x, y, sigmas).mean()

def median_heuristic(x, y, max_samples=512):  # NEW
    z = torch.cat([x, y], dim=0).detach()
    if z.shape[0] > max_samples:
        z = z[torch.randperm(z.shape[0], device=z.device)[:max_samples]]
    return (torch.cdist(z, z, p=2) ** 2).median().sqrt().clamp_min(1e-3).item()

def multiscale_sigmas(x, y, scales=(0.25, 0.5, 1.0, 2.0, 4.0)):  # NEW
    base = median_heuristic(x, y)
    return [base * s for s in scales]
# ---- конец блока MMD ----


p = argparse.ArgumentParser()
p.add_argument("--base-model", required=True)      # путь к базовой модели
p.add_argument("--new-adata", required=True)        # h5ad новой когорты
p.add_argument("--out-dir", required=True)
p.add_argument("--rank", type=int, default=8)
p.add_argument("--donor", default="donor_id")
p.add_argument("--old-adata", required=True)
p.add_argument("--disease-col", default="disease")
p.add_argument("--healthy-label", default="normal")
p.add_argument("--lam", type=float, default=30.0)   # вес anchor-лосса на census replay
p.add_argument("--lam-new", type=float, default=0.0)
p.add_argument("--lam-mag", type=float, default=0.0)  # штраф на магнитуду корекции
p.add_argument("--lam-mmd", type=float, default=0.0)  # NEW: вес стратифицированного MMD (healthy_new vs census)
p.add_argument("--run-name", default=None)
p.add_argument("--n-replay", type=int, default=25)  # клеток на донора census для replay
p.add_argument("--n-eval", type=int, default=25)    # старые клетки для оценки якоря
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--device", default=None)
args = p.parse_args()

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}", flush=True)

model = scvi.model.SCVI.load(args.base_model)
module = model.module
module.eval()  # всегда замороженные веса

census_base = ad.read_h5ad(args.old_adata)

rng = np.random.default_rng(0)
replay_idx = []
eval_idx = []
for donor, idx in census_base.obs.groupby(args.donor, observed=True).indices.items():
    take = rng.choice(idx, size = min(args.n_replay, len(idx)), replace = False)
    replay_idx.append(take)   # проходимся именно по донорам
    remaining = np.setdiff1d(idx, take)  #
    held = rng.choice(remaining, size = min(args.n_eval, len(remaining)), replace = False)  # выбираем ограниченое
    eval_idx.append(held)
replay_idx = np.concatenate(replay_idx)
eval_idx = np.concatenate(eval_idx)
replay = census_base[replay_idx].copy()
eval_old = census_base[eval_idx].copy()

mu_ref = model.get_latent_representation(replay)  # без адаптеров
mu_ref_sigma = mu_ref.std(axis = 0)  # для скейлинга
mu_eval_ref = model.get_latent_representation(eval_old)
eval_ct = compartment(eval_old.obs["cell_type"].astype(str))  # типы held out доноров
eval_donor = eval_old.obs[args.donor].astype(str).to_numpy()

replay_ct = compartment(replay.obs["cell_type"].astype(str))
ref_frame = pd.DataFrame(mu_ref)
ref_frame["compartment"] = replay_ct.to_numpy()
ref_frame["donor"] = replay.obs[args.donor].astype(str).to_numpy()
ref_dc_grp = ref_frame.groupby(["compartment", "donor"], observed=True)
ref_dc = ref_dc_grp.mean()[ref_dc_grp.size() >= STRAT_MIN_CELLS]
ref_centroid = ref_dc.groupby("compartment", observed=True).mean()

new_adata = ad.read_h5ad(args.new_adata)

new_healthy_local_mask = (new_adata.obs[args.disease_col].values == args.healthy_label)
new_ct_healthy = compartment(new_adata.obs["cell_type"].astype(str))[new_healthy_local_mask]
new_donor_healthy = new_adata.obs[args.donor].astype(str).to_numpy()[new_healthy_local_mask]

combined = ad.concat([replay, new_adata], join="outer", label="_replay_source",
                          keys=["replay", "new"], index_unique="_")

scvi.model.SCVI.setup_anndata(combined, batch_key=args.donor)
q_model = scvi.model.SCVI.load_query_data(combined, model)   # freeze=True по умолчанию
q_model.is_trained_ = True  # костыль

q_module = q_model.module
q_module.eval()
q_module.to(device)

adapters = attach(q_module, rank=args.rank, n_cov=0, device=device)
verify_identical(q_model, adapters)



is_new = (combined.obs["_replay_source"] == "new").values
is_replay = ~is_new
healthy_new_mask = is_new & (combined.obs[args.disease_col].values == args.healthy_label)  # для penalty
new_adata_healthy = combined[healthy_new_mask].copy()
train_adata = combined

trainable_backbone = [p for p in q_module.parameters() if p.requires_grad]    # новые декодер веса для scArches
optimizer = torch.optim.Adam(
    list(adapters.parameters()) + trainable_backbone, lr=args.lr
)

from scvi.dataloaders import AnnDataLoader  # CHANGED: перенесено выше — нужен здесь же для одноразовой предзагрузки MMD-тензоров

# ---- NEW: подготовка стратифицированного MMD — какие компартменты участвуют и позиции клеток по сторонам ----
healthy_new_pos = np.where(healthy_new_mask)[0]  # позиции healthy_new клеток В combined
new_ct_healthy_arr = new_ct_healthy.to_numpy()   # NEW: тот же порядок, что и healthy_new_pos (обе идут от new_adata[new_healthy_local_mask] построчно)

mmd_comps = []  # NEW
idx_ref_c = {}  # NEW: позиции внутри replay/mu_ref (индексируются напрямую в numpy-массив mu_ref)
local_idx_new_c = {}  # CHANGED: теперь позиции ВНУТРИ healthy_all_tensors (см. ниже), а не внутри combined — избегаем AnnDataLoader на каждый шаг
for c in [name for name, _ in COMPARTMENTS]:   # NEW: "other" сюда не входит по построению COMPARTMENTS
    ref_pos_c = np.where(replay_ct.to_numpy() == c)[0]
    local_pos_c = np.where(new_ct_healthy_arr == c)[0]  # CHANGED: локальная позиция внутри healthy_new_pos/new_ct_healthy_arr
    if len(ref_pos_c) >= MMD_MIN_PER_SIDE and len(local_pos_c) >= MMD_MIN_PER_SIDE:
        mmd_comps.append(c)
        idx_ref_c[c] = ref_pos_c
        local_idx_new_c[c] = local_pos_c
print(f"stratified MMD compartments ({len(mmd_comps)}): {mmd_comps}", flush=True)  # NEW
mmd_rng = np.random.default_rng(2)  # NEW

healthy_all_tensors = None  # NEW (perf): все healthy_new тензоры грузим ОДИН раз здесь, а не AnnDataLoader на каждый компартмент на каждом шаге
if mmd_comps:  # NEW
    _healthy_all_loader = AnnDataLoader(q_model.adata_manager, indices=healthy_new_pos, batch_size=len(healthy_new_pos), shuffle=False)  # NEW
    healthy_all_tensors = next(iter(_healthy_all_loader))  # NEW
    healthy_all_tensors = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in healthy_all_tensors.items()}  # NEW
# ---- конец подготовки MMD ----


ep_rng = np.random.default_rng(1)
n_replay = len(replay)
new_pos = np.where(is_new)[0]
n_healthy_new = len(healthy_new_pos)

for epoch in range(args.epochs):
    replay_perm = ep_rng.permutation(n_replay)
    mu_ref_epoch = mu_ref[replay_perm]  # шаффлим сами, что бы была гарантия соответствия

    new_loader = AnnDataLoader(q_model.adata_manager, indices = new_pos, batch_size = 256, shuffle = True)
    replay_loader = AnnDataLoader(q_model.adata_manager, indices = replay_perm, batch_size = 256, shuffle = False)
    replay_iter = iter(replay_loader)
    replay_ptr = 0

    if args.lam_new > 0 and n_healthy_new > 0:
        healthy_perm = ep_rng.permutation(n_healthy_new)
        healthy_loader = AnnDataLoader(q_model.adata_manager, indices = healthy_new_pos[healthy_perm], batch_size = 256, shuffle = False)
        healthy_iter = iter(healthy_loader)

    elbo_sum = 0.0
    anchor_sum = 0.0
    delta_sum = 0.0
    mag_sum = 0.0
    mmd_sum = 0.0   # NEW
    n_steps = 0

    for new_tensors in new_loader:
        new_tensors = {k: v.to(device) for k, v in new_tensors.items()}

        try:
            replay_tensors = next(replay_iter)
        except StopIteration:
            replay_iter = iter(replay_loader)
            replay_ptr = 0
            replay_tensors = next(replay_iter)
        replay_tensors = {k: v.to(device) for k, v in replay_tensors.items()}

        bs = replay_tensors["X"].shape[0]
        mu_ref_batch = torch.as_tensor(mu_ref_epoch[replay_ptr:replay_ptr + bs], dtype=torch.float32, device=device)
        replay_ptr += bs

        optimizer.zero_grad()
        adapters.clear()
        _, _, loss_out = q_module(new_tensors, loss_kwargs={"kl_weight": 1.0})
        elbo = loss_out.loss

        adapters.clear()
        inf_replay = q_module.inference(**q_module._get_inference_input(replay_tensors))
        mu_new = inf_replay["qz"].loc
        sigma = torch.as_tensor(mu_ref_sigma, dtype=torch.float32, device=device)
        anchor = (((mu_new - mu_ref_batch) / sigma) ** 2).mean()
        mag_pen = adapters.penalty() if args.lam_mag > 0 else torch.tensor(0.0, device=device)  # last уже закеширован

        delta_pen = torch.tensor(0.0, device=device)
        if args.lam_new > 0 and n_healthy_new > 0:
            try:
                healthy_tensors = next(healthy_iter)
            except StopIteration:
                healthy_iter = iter(healthy_loader)
                healthy_tensors = next(healthy_iter)
            healthy_tensors = {k: v.to(device) for k, v in healthy_tensors.items()}
            adapters.clear()
            _ = q_module.inference(**q_module._get_inference_input(healthy_tensors))
            delta_pen = adapters.penalty()

        mmd = torch.tensor(0.0, device=device)  # NEW
        if args.lam_mmd > 0 and mmd_comps:  # NEW: стратифицированный MMD healthy_new vs census, по компартментам
            sizes = []  # NEW (perf): вместо форварда на каждый компартмент — один общий батч на ВСЕ компартменты
            local_picks = []  # NEW
            ref_picks = []  # NEW
            for c in mmd_comps:  # NEW
                k = min(MMD_K_PER_COMPARTMENT, len(local_idx_new_c[c]), len(idx_ref_c[c]))  # NEW
                sizes.append(k)  # NEW
                local_picks.append(mmd_rng.choice(local_idx_new_c[c], size=k, replace=False))  # NEW: позиции в healthy_all_tensors
                ref_picks.append(mmd_rng.choice(idx_ref_c[c], size=k, replace=False))  # NEW: позиции в replay/mu_ref
            local_all = torch.as_tensor(np.concatenate(local_picks), dtype=torch.long, device=device)  # NEW
            mmd_tensors = {kk: (vv[local_all] if torch.is_tensor(vv) else vv) for kk, vv in healthy_all_tensors.items()}  # NEW: срез готовых тензоров, без AnnDataLoader
            adapters.clear()  # NEW
            inf_mmd = q_module.inference(**q_module._get_inference_input(mmd_tensors))  # NEW: ОДИН forward на все компартменты сразу
            z_all = inf_mmd["qz"].loc  # NEW
            terms = []  # NEW
            off = 0  # NEW
            for c, k, sr in zip(mmd_comps, sizes, ref_picks):  # NEW
                z_c = z_all[off:off + k]  # CHANGED: срез уже посчитанного батча вместо нового forward'а
                off += k  # NEW
                y_c = torch.as_tensor(mu_ref[sr], dtype=torch.float32, device=device)  # замороженный census-таргет, без градиента
                terms.append(mmd2(z_c, y_c, multiscale_sigmas(z_c.detach(), y_c)))
            mmd = torch.stack(terms).mean()

        loss = elbo + args.lam * anchor + args.lam_new * delta_pen + args.lam_mag * mag_pen + args.lam_mmd * mmd  # CHANGED: + args.lam_mmd * mmd
        loss.backward()
        optimizer.step()

        elbo_sum += float(elbo.detach())
        anchor_sum += float(anchor.detach())
        delta_sum += float(delta_pen.detach())
        mag_sum += float(mag_pen.detach())
        mmd_sum += float(mmd.detach())  # NEW
        n_steps += 1

    mu_eval_adapted = q_model.get_latent_representation(eval_old)
    eval_drift = np.linalg.norm(mu_eval_adapted - mu_eval_ref, axis=1)
    anchor_eval = float((((mu_eval_adapted - mu_eval_ref) / mu_ref_sigma) ** 2).mean())  # NEW: та же формула что и anchor, для честного сравнения

    mu_new_healthy = q_model.get_latent_representation(new_adata_healthy)
    strat_per_donor, keep_ct = stratified_dist(mu_new_healthy, new_ct_healthy, new_donor_healthy, ref_centroid, mu_ref_sigma)  # CHANGED: + mu_ref_sigma
    strat_mean = float(strat_per_donor.mean()) if len(strat_per_donor) else float("nan")

    strat_eval_per_donor, keep_ct_eval = stratified_dist(mu_eval_adapted, eval_ct, eval_donor, ref_centroid, mu_ref_sigma)  # CHANGED: + mu_ref_sigma
    strat_eval_mean = float(strat_eval_per_donor.mean()) if len(strat_eval_per_donor) else float("nan")

    print(f"epoch {epoch}: elbo={elbo_sum / n_steps:.4f} | "
          f"anchor={anchor_sum / n_steps:.4f} anchor_held_out={anchor_eval:.4f} | "
          f"mag={mag_sum / n_steps:.4f} mmd={mmd_sum / n_steps:.4f} held_out_drift_max={eval_drift.max():.4f} | "
          f"strat_healthy_new={strat_mean:.4f} strat_held_out={strat_eval_mean:.4f}", flush=True)

import os
run_name = args.run_name or f"mmdtest_rank{args.rank}_lam{args.lam}_lammmd{args.lam_mmd}"
out_path = os.path.join(args.out_dir, run_name)
os.makedirs(out_path, exist_ok=True)
q_model.save(out_path, overwrite=True)
torch.save(adapters.state_dict(), os.path.join(out_path, "adapters.pt"))
