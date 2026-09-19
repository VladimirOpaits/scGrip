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
p.add_argument("--run-name", default=None)
p.add_argument("--n-replay", type=int, default=25)  # клеток на донора census, СВЕЖАЯ подвыборка из пула каждую эпоху (CHANGED: раньше фиксировалась один раз)
p.add_argument("--n-eval", type=int, default=25)    # старые клетки для оценки якоря, held-out на весь прогон
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
eval_idx = []
replay_pool_idx = []
for donor, idx in census_base.obs.groupby(args.donor, observed=True).indices.items():
    held = rng.choice(idx, size = min(args.n_eval, len(idx)), replace = False)
    eval_idx.append(held)
    remaining = np.setdiff1d(idx, held)  # CHANGED: весь остаток идёт в пул, не только n_replay штук — из него каждая эпоха берёт свежую подвыборку
    replay_pool_idx.append(remaining)  # NEW
eval_idx = np.concatenate(eval_idx)
replay_pool_idx = np.concatenate(replay_pool_idx)  # NEW
replay_pool = census_base[replay_pool_idx].copy()  # NEW: весь пул кандидатов на replay (до 100-n_eval клеток/донора), НЕ сама выборка
eval_old = census_base[eval_idx].copy()

mu_ref_pool = model.get_latent_representation(replay_pool)  # NEW: латент всего пула без адаптеров, считаем один раз
mu_ref_sigma = mu_ref_pool.std(axis = 0)  # CHANGED: масштаб теперь по всему пулу, не по узкой фиксированной выборке
mu_eval_ref = model.get_latent_representation(eval_old)
eval_ct = compartment(eval_old.obs["cell_type"].astype(str))  # типы held out доноров
eval_donor = eval_old.obs[args.donor].astype(str).to_numpy()

pool_ct = compartment(replay_pool.obs["cell_type"].astype(str))  # CHANGED: replay -> replay_pool
ref_frame = pd.DataFrame(mu_ref_pool)  # CHANGED
ref_frame["compartment"] = pool_ct.to_numpy()  # CHANGED
ref_frame["donor"] = replay_pool.obs[args.donor].astype(str).to_numpy()  # CHANGED
ref_dc_grp = ref_frame.groupby(["compartment", "donor"], observed=True)
ref_dc = ref_dc_grp.mean()[ref_dc_grp.size() >= STRAT_MIN_CELLS]
ref_centroid = ref_dc.groupby("compartment", observed=True).mean()

new_adata = ad.read_h5ad(args.new_adata)

new_healthy_local_mask = (new_adata.obs[args.disease_col].values == args.healthy_label)
new_ct_healthy = compartment(new_adata.obs["cell_type"].astype(str))[new_healthy_local_mask]
new_donor_healthy = new_adata.obs[args.donor].astype(str).to_numpy()[new_healthy_local_mask]

combined = ad.concat([replay_pool, new_adata], join="outer", label="_replay_source",  # CHANGED: replay -> replay_pool
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



from scvi.dataloaders import AnnDataLoader

ep_rng = np.random.default_rng(1)
new_pos = np.where(is_new)[0]
healthy_new_pos = np.where(healthy_new_mask)[0]
n_healthy_new = len(healthy_new_pos)

pool_pos_by_donor = replay_pool.obs.groupby(args.donor, observed=True).indices  # NEW: позиции внутри пула = позиции внутри combined (пул — первый блок конкатенации)

for epoch in range(args.epochs):
    replay_pos_epoch = np.concatenate([
        ep_rng.choice(pos, size=min(args.n_replay, len(pos)), replace=False)
        for pos in pool_pos_by_donor.values()
    ])  # NEW: СВЕЖАЯ подвыборка replay-клеток из пула на каждую эпоху — ключевое отличие от finetune_lora.py
    replay_perm = ep_rng.permutation(len(replay_pos_epoch))  # CHANGED: шаффлим позиции этой эпохи
    replay_idx_epoch = replay_pos_epoch[replay_perm]  # NEW
    mu_ref_epoch = mu_ref_pool[replay_idx_epoch]  # CHANGED: индексация в общий mu_ref_pool, не в узкий mu_ref

    new_loader = AnnDataLoader(q_model.adata_manager, indices = new_pos, batch_size = 256, shuffle = True)
    replay_loader = AnnDataLoader(q_model.adata_manager, indices = replay_idx_epoch, batch_size = 256, shuffle = False)  # CHANGED
    replay_iter = iter(replay_loader)
    replay_ptr = 0
    n_replay_epoch = len(replay_idx_epoch)  # NEW: пересчитывается каждую эпоху (для held_out_strat_dist ниже не нужно, но полезно для отчёта)

    if args.lam_new > 0 and n_healthy_new > 0:
        healthy_perm = ep_rng.permutation(n_healthy_new)
        healthy_loader = AnnDataLoader(q_model.adata_manager, indices = healthy_new_pos[healthy_perm], batch_size = 256, shuffle = False)
        healthy_iter = iter(healthy_loader)

    elbo_sum = 0.0
    anchor_sum = 0.0
    delta_sum = 0.0
    mag_sum = 0.0
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

        loss = elbo + args.lam * anchor + args.lam_new * delta_pen + args.lam_mag * mag_pen
        loss.backward()
        optimizer.step()

        elbo_sum += float(elbo.detach())
        anchor_sum += float(anchor.detach())
        delta_sum += float(delta_pen.detach())
        mag_sum += float(mag_pen.detach())
        n_steps += 1

    mu_eval_adapted = q_model.get_latent_representation(eval_old)
    eval_drift = np.linalg.norm(mu_eval_adapted - mu_eval_ref, axis=1)
    anchor_eval = float((((mu_eval_adapted - mu_eval_ref) / mu_ref_sigma) ** 2).mean())  # NEW: та же формула что и anchor, для честного сравнения replay vs held-out

    mu_new_healthy = q_model.get_latent_representation(new_adata_healthy)
    strat_per_donor, keep_ct = stratified_dist(mu_new_healthy, new_ct_healthy, new_donor_healthy, ref_centroid, mu_ref_sigma)  # CHANGED: + mu_ref_sigma
    strat_mean = float(strat_per_donor.mean()) if len(strat_per_donor) else float("nan")

    strat_eval_per_donor, keep_ct_eval = stratified_dist(mu_eval_adapted, eval_ct, eval_donor, ref_centroid, mu_ref_sigma)  # CHANGED: + mu_ref_sigma
    strat_eval_mean = float(strat_eval_per_donor.mean()) if len(strat_eval_per_donor) else float("nan")

    print(f"epoch {epoch}: elbo={elbo_sum / n_steps:.4f} | "
          f"anchor={anchor_sum / n_steps:.4f} anchor_held_out={anchor_eval:.4f} | "
          f"mag={mag_sum / n_steps:.4f} held_out_drift_max={eval_drift.max():.4f} | "
          f"strat_healthy_new={strat_mean:.4f} strat_held_out={strat_eval_mean:.4f}", flush=True)

import os
run_name = args.run_name or f"freshreplay_rank{args.rank}_lam{args.lam}_lamnew{args.lam_new}"
out_path = os.path.join(args.out_dir, run_name)
os.makedirs(out_path, exist_ok=True)
q_model.save(out_path, overwrite=True)
torch.save(adapters.state_dict(), os.path.join(out_path, "adapters.pt"))
