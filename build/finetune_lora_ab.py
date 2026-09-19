import argparse
import gc
import os
import time

import anndata as ad
import numpy as np
import pandas as pd
import scvi
import torch
import torch.nn as nn
from scvi.dataloaders import AnnDataLoader

from sceris.lora import attach, verify_identical, default_targets, first_linear

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

def stratified_dist(mu, ct, donor, ref_centroid, sigma):
    frame = pd.DataFrame(mu)
    frame["compartment"] = ct
    frame["donor"] = donor
    grp = frame.groupby(["compartment", "donor"], observed=True)
    dc = grp.mean()[grp.size() >= STRAT_MIN_CELLS]
    per_ct = {}
    for c in ref_centroid.index:
        if c not in dc.index.get_level_values("compartment"):
            continue
        sub = dc.xs(c, level="compartment")
        per_ct[c] = pd.Series(np.linalg.norm((sub.to_numpy() - ref_centroid.loc[c].to_numpy()) / sigma, axis=1), index=sub.index)
    df = pd.DataFrame(per_ct)
    if df.empty:
        return pd.Series(dtype=float)
    keep = df.columns[df.notna().mean() >= STRAT_MIN_DONOR_FRAC]
    return df[keep].mean(axis=1)


MMD_MIN_PER_SIDE = 32
MMD_K_PER_COMPARTMENT = 64

def rbf_kernel(x, y, sigmas):
    x2 = (x ** 2).sum(1, keepdim=True)
    y2 = (y ** 2).sum(1, keepdim=True)
    dist2 = (x2 + y2.T - 2 * x @ y.T).clamp_min(0)
    return sum(torch.exp(-dist2 / (2 * s ** 2)) for s in sigmas)

def mmd2(x, y, sigmas):  # Kyy отброшен: y — замороженный census-таргет без градиента
    return rbf_kernel(x, x, sigmas).mean() - 2 * rbf_kernel(x, y, sigmas).mean()

def multiscale_sigmas(x, y, scales=(0.25, 0.5, 1.0, 2.0, 4.0), max_samples=512):
    z = torch.cat([x, y], dim=0).detach()
    if z.shape[0] > max_samples:
        z = z[torch.randperm(z.shape[0], device=z.device)[:max_samples]]
    base = (torch.cdist(z, z) ** 2).median().sqrt().clamp_min(1e-3).item()
    return [base * s for s in scales]


def input_moments(scvi_model, blocks, adata, indices, batch_size):  # X^T X входа первого Linear каждого блока
    C, handles = {}, []
    for name, blk in blocks.items():
        def pre(_m, inp, name=name):
            x = inp[0].detach().double()
            C[name] = C[name] + x.T @ x if name in C else x.T @ x
        handles.append(first_linear(blk).register_forward_pre_hook(pre))
    with torch.no_grad():
        scvi_model.get_latent_representation(adata, indices=indices, batch_size=batch_size)
    for h in handles:
        h.remove()
    return {name: c.cpu() for name, c in C.items()}


parser = argparse.ArgumentParser()
parser.add_argument("--base-model", required=True)
parser.add_argument("--old-adata", required=True)
parser.add_argument("--new-adata", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--run-name", required=True)
parser.add_argument("--donor", default="donor_id")
parser.add_argument("--disease-col", default="disease")
parser.add_argument("--healthy-label", default="normal")
parser.add_argument("--rank", type=int, default=4)
parser.add_argument("--gelu", action="store_true")  # по умолчанию линейный адаптер
parser.add_argument("--lam", type=float, default=30.0)
parser.add_argument("--lam-mag", type=float, default=5.0)
parser.add_argument("--lam-mmd", type=float, default=10.0)
parser.add_argument("--ns-energy", type=float, default=0.99)  # 0 = без проекции
parser.add_argument("--ns-batch", type=int, default=2048)
parser.add_argument("--n-replay", type=int, default=70)
parser.add_argument("--n-eval", type=int, default=25)
parser.add_argument("--metric-ref-n", type=int, default=50)
parser.add_argument("--holdout-frac", type=float, default=0.35)  # доля здоровых доноров в B: они не участвуют в MMD
parser.add_argument("--epochs", type=int, default=20)  # итоговое число эпох, в том числе при дообучении
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=0)  # инициализация адаптера и порядок батчей
parser.add_argument("--split-seed", type=int, default=0)  # replay/eval и A/B; держим одинаковым между вариантами, чтобы линейка была одна
parser.add_argument("--resume-from", default=None)
parser.add_argument("--start-epoch", type=int, default=0)  # только для чекпойнтов без train_state.pt
parser.add_argument("--device", default=None)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

model = scvi.model.SCVI.load(args.base_model)
module = model.module
module.eval()
model_vars = pd.Index(torch.load(os.path.join(args.base_model, "model.pt"), map_location="cpu", weights_only=False)["var_names"])

census = ad.read_h5ad(args.old_adata)
census.X = None  # X дублирует layers['counts'], а реестр модели читает counts
assert np.array_equal(census.var_names, model_vars)

split_rng = np.random.default_rng(args.split_seed)
replay_idx, eval_idx = [], []
for idx in census.obs.groupby(args.donor, observed=True).indices.values():
    take = split_rng.choice(idx, size=min(args.n_replay, len(idx)), replace=False)
    rest = np.setdiff1d(idx, take)
    replay_idx.append(take)
    eval_idx.append(split_rng.choice(rest, size=min(args.n_eval, len(rest)), replace=False))
replay_idx, eval_idx = np.concatenate(replay_idx), np.concatenate(eval_idx)
replay = census[replay_idx].copy()
eval_old = census[eval_idx].copy()

mu_ref = model.get_latent_representation(replay)
sigma = mu_ref.std(axis=0)
mu_eval_ref = model.get_latent_representation(eval_old)
eval_ct = compartment(eval_old.obs["cell_type"].astype(str)).to_numpy()
eval_donor = eval_old.obs[args.donor].astype(str).to_numpy()
replay_ct = compartment(replay.obs["cell_type"].astype(str)).to_numpy()

# референс strat-метрик: фиксированная выборка census, не зависит от n_replay и --seed
metric_rng = np.random.default_rng(12345)
metric_idx = np.concatenate([
    metric_rng.choice(idx, size=min(args.metric_ref_n, len(idx)), replace=False)
    for idx in census.obs.groupby(args.donor, observed=True).indices.values()
])
metric_obs = census.obs.iloc[metric_idx]
frame = pd.DataFrame(model.get_latent_representation(census, indices=metric_idx))
frame["compartment"] = compartment(metric_obs["cell_type"].astype(str)).to_numpy()
frame["donor"] = metric_obs[args.donor].astype(str).to_numpy()
grp = frame.groupby(["compartment", "donor"], observed=True)
ref_centroid = grp.mean()[grp.size() >= STRAT_MIN_CELLS].groupby("compartment", observed=True).mean()
del frame, grp, metric_obs

base_blocks = {name: blk for name, (_i, _o, blk) in default_targets(module).items()}
C_census = C_eval = None
if args.ns_energy > 0:  # до загрузки когорты: полный census и combined вместе не влезают в память
    ns_idx = np.setdiff1d(np.arange(census.n_obs), eval_idx)  # без held-out, иначе метрики защиты нечестные
    C_census = input_moments(model, base_blocks, census, ns_idx, args.ns_batch)
    C_eval = input_moments(model, base_blocks, eval_old, None, args.ns_batch)
del census
gc.collect()

new = ad.read_h5ad(args.new_adata)
new.X = None
new_donor = new.obs[args.donor].astype(str).to_numpy()
new_ct = compartment(new.obs["cell_type"].astype(str)).to_numpy()
healthy = new.obs[args.disease_col].astype(str).to_numpy() == args.healthy_label

# B — здоровые доноры, которых MMD не видит: честная проверка интеграции и честная группа сравнения для сигнала болезни
healthy_donors = np.unique(new_donor[healthy])
ab_rng = np.random.default_rng(args.split_seed + 1)
donors_b = ab_rng.choice(healthy_donors, size=max(1, round(len(healthy_donors) * args.holdout_frac)), replace=False)
group = np.where(~healthy, "disease", np.where(np.isin(new_donor, donors_b), "B", "A"))
print(f"healthy donors A={len(healthy_donors) - len(donors_b)} B={len(donors_b)} | disease cells={int((~healthy).sum())}", flush=True)

combined = ad.concat([replay, new], join="outer", label="_source", keys=["replay", "new"], index_unique="_")
del new
combined = combined[:, model_vars].copy()  # outer concat сортирует гены по алфавиту, а энкодер ждёт порядок модели
assert np.array_equal(combined.var_names, model_vars)
gc.collect()

new_pos = np.where(combined.obs["_source"].to_numpy() == "new")[0]  # concat сохраняет порядок строк когорты
pos_a, pos_b = new_pos[group == "A"], new_pos[group == "B"]

scvi.model.SCVI.setup_anndata(combined, batch_key=args.donor, layer="counts")
q_model = scvi.model.SCVI.load_query_data(combined, model)
q_model.is_trained_ = True  # свой цикл вместо Trainer
q_module = q_model.module
q_module.eval()
q_module.to(device)

adapters = attach(q_module, rank=args.rank, device=device, act=nn.GELU if args.gelu else nn.Identity)
verify_identical(q_model, adapters)
mu_check = q_model.get_latent_representation(combined, indices=np.arange(512))
assert np.abs(mu_check - mu_ref[:512]).max() < 1e-4, "replay в combined кодируется не так, как отдельно: порядок генов или данные не совпадают"
print(f"combined/replay identity check passed: max diff {np.abs(mu_check - mu_ref[:512]).max():.2e}", flush=True)

healthy_a = combined[pos_a].copy()
healthy_b = combined[pos_b].copy()

proj = {}
if args.ns_energy > 0:
    q_blocks = {name: blk for name, (_i, _o, blk) in default_targets(q_module).items()}
    C_new = input_moments(q_model, q_blocks, combined, new_pos, args.ns_batch)
    for name, a in adapters.adapters.items():
        evals, evecs = torch.linalg.eigh(C_census[name])
        evals, evecs = evals.flip(0).clamp_min(0), evecs.flip(1)
        k = int((torch.cumsum(evals, 0) / evals.sum() < args.ns_energy).sum()) + 1
        U = evecs[:, :k]
        P = torch.eye(U.shape[0], dtype=torch.float64) - U @ U.T
        outside = {key: round(float((P * Cx[name]).sum() / torch.trace(Cx[name])), 4)
                   for key, Cx in [("census", C_census), ("held_out", C_eval), ("new", C_new)]}
        print(f"[nullspace] {name}: k={k}/{U.shape[0]} energy outside: {outside}", flush=True)
        proj[name] = P.float().to(device)
        with torch.no_grad():
            a.down.bias.zero_()
            a.up.bias.zero_()
            a.down.weight.copy_(a.down.weight @ proj[name])
        a.down.bias.requires_grad_(False)  # bias сдвигал бы все клетки в обход проекции
        a.up.bias.requires_grad_(False)
    del C_census, C_eval, C_new

trainable_backbone = [p for p in q_module.parameters() if p.requires_grad]  # слот новых доноров в декодере
optimizer = torch.optim.Adam(list(adapters.parameters()) + trainable_backbone, lr=args.lr)

a_ct = new_ct[group == "A"]
mmd_comps, ref_pos_c, a_pos_c = [], {}, {}
for c, _ in COMPARTMENTS:
    r = np.where(replay_ct == c)[0]
    q = np.where(a_ct == c)[0]
    if len(r) >= MMD_MIN_PER_SIDE and len(q) >= MMD_MIN_PER_SIDE:
        mmd_comps.append(c)
        ref_pos_c[c], a_pos_c[c] = r, q
print(f"MMD compartments: {mmd_comps}", flush=True)

a_tensors = None
if args.lam_mmd > 0 and mmd_comps:  # клетки A грузим один раз, дальше только индексируем
    loader = AnnDataLoader(q_model.adata_manager, indices=pos_a, batch_size=len(pos_a), shuffle=False)
    a_tensors = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in next(iter(loader)).items()}

ep_rng = np.random.default_rng(args.seed + 1000)
mmd_rng = np.random.default_rng(args.seed + 2000)
mu_ref_t = torch.as_tensor(mu_ref, dtype=torch.float32, device=device)
sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=device)


def evaluate():
    mu_eval = q_model.get_latent_representation(eval_old)
    diff = mu_eval - mu_eval_ref
    out = {
        "anchor_held_out": float(((diff / sigma) ** 2).mean()),
        "drift_max": float(np.linalg.norm(diff, axis=1).max()),
        "strat_held_out": float(stratified_dist(mu_eval, eval_ct, eval_donor, ref_centroid, sigma).mean()),
    }
    for g, adata_g in [("A", healthy_a), ("B", healthy_b)]:
        mu_g = q_model.get_latent_representation(adata_g)
        out[f"strat_{g}"] = float(stratified_dist(mu_g, new_ct[group == g], new_donor[group == g], ref_centroid, sigma).mean())
    return out


start_epoch = 0
if args.resume_from:
    sd = torch.load(os.path.join(args.resume_from, "model.pt"), map_location=device, weights_only=False)["model_state_dict"]
    sd.pop("pyro_param_store", None)  # служебный ключ scvi, не параметр модуля
    q_module.load_state_dict(sd)
    adapters.load_state_dict(torch.load(os.path.join(args.resume_from, "adapters.pt"), map_location=device))
    state_path = os.path.join(args.resume_from, "train_state.pt")
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)  # set_rng_state принимает только CPU-тензор
        optimizer.load_state_dict(st["optimizer"])
        ep_rng.bit_generator.state = st["ep_rng"]
        mmd_rng.bit_generator.state = st["mmd_rng"]
        torch.set_rng_state(st["torch_rng"])
        start_epoch = st["epoch"]
    else:
        start_epoch = args.start_epoch
        for _ in range(start_epoch):
            ep_rng.permutation(len(replay_idx))
        print("WARNING: no train_state.pt, Adam starts fresh", flush=True)
    print(f"resumed at epoch {start_epoch}", flush=True)
else:
    m = evaluate()
    print("epoch -1 (pre-training): " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)


n_rep = len(replay_idx)
t0 = time.time()
for epoch in range(start_epoch, args.epochs):  # эпоха = один проход по всем клеткам новой когорты
    perm = ep_rng.permutation(n_rep)  # replay — первые n_rep строк combined
    perm_t = torch.as_tensor(perm, device=device)
    new_loader = AnnDataLoader(q_model.adata_manager, indices=new_pos, batch_size=args.batch_size, shuffle=True)
    replay_loader = AnnDataLoader(q_model.adata_manager, indices=perm, batch_size=args.batch_size, shuffle=False)
    replay_iter, ptr = iter(replay_loader), 0
    sums = {"elbo": 0.0, "anchor": 0.0, "mag": 0.0, "mmd": 0.0}
    n_steps = 0

    for new_tensors in new_loader:
        try:
            replay_tensors = next(replay_iter)
        except StopIteration:
            replay_iter, ptr = iter(replay_loader), 0
            replay_tensors = next(replay_iter)
        new_tensors = {k: v.to(device) for k, v in new_tensors.items()}
        replay_tensors = {k: v.to(device) for k, v in replay_tensors.items()}
        bs = replay_tensors["X"].shape[0]
        target = mu_ref_t[perm_t[ptr:ptr + bs]]
        ptr += bs

        optimizer.zero_grad()
        adapters.clear()
        elbo = q_module(new_tensors, loss_kwargs={"kl_weight": 1.0})[2].loss

        adapters.clear()
        mu_new = q_module.inference(**q_module._get_inference_input(replay_tensors))["qz"].loc
        anchor = (((mu_new - target) / sigma_t) ** 2).mean()
        mag = adapters.penalty() if args.lam_mag > 0 else torch.zeros((), device=device)  # кэш того же replay-forward

        mmd = torch.zeros((), device=device)
        if a_tensors is not None:
            sizes, a_pick, r_pick = [], [], []
            for c in mmd_comps:
                k = min(MMD_K_PER_COMPARTMENT, len(a_pos_c[c]), len(ref_pos_c[c]))
                sizes.append(k)
                a_pick.append(mmd_rng.choice(a_pos_c[c], size=k, replace=False))
                r_pick.append(mmd_rng.choice(ref_pos_c[c], size=k, replace=False))
            idx_t = torch.as_tensor(np.concatenate(a_pick), dtype=torch.long, device=device)
            batch = {k: (v[idx_t] if torch.is_tensor(v) else v) for k, v in a_tensors.items()}
            adapters.clear()
            z_all = q_module.inference(**q_module._get_inference_input(batch))["qz"].loc  # один forward на все компартменты
            terms, off = [], 0
            for k, rp in zip(sizes, r_pick):
                z_c = z_all[off:off + k]
                y_c = mu_ref_t[torch.as_tensor(rp, device=device)]
                off += k
                terms.append(mmd2(z_c, y_c, multiscale_sigmas(z_c, y_c)))
            mmd = torch.stack(terms).mean()

        loss = elbo + args.lam * anchor + args.lam_mag * mag + args.lam_mmd * mmd
        loss.backward()
        optimizer.step()
        if proj:  # Adam не знает про ограничение: возвращаем строки down в ортогональное дополнение
            with torch.no_grad():
                for name, a in adapters.adapters.items():
                    a.down.weight.copy_(a.down.weight @ proj[name])

        for key, val in [("elbo", elbo), ("anchor", anchor), ("mag", mag), ("mmd", mmd)]:
            sums[key] += float(val.detach())
        n_steps += 1

    m = evaluate()
    print(f"epoch {epoch}: " + " ".join(f"{k}={v / n_steps:.4f}" for k, v in sums.items())
          + " | " + " ".join(f"{k}={v:.4f}" for k, v in m.items())
          + f" | t={time.time() - t0:.0f}s", flush=True)


out_path = os.path.join(args.out_dir, args.run_name)
os.makedirs(out_path, exist_ok=True)
q_model.save(out_path, overwrite=True)
torch.save(adapters.state_dict(), os.path.join(out_path, "adapters.pt"))
torch.save({
    "optimizer": optimizer.state_dict(),
    "ep_rng": ep_rng.bit_generator.state,
    "mmd_rng": mmd_rng.bit_generator.state,
    "torch_rng": torch.get_rng_state(),
    "epoch": args.epochs,
    "donors_b": list(donors_b),  # пригодится скрипту оценки сигнала болезни
}, os.path.join(out_path, "train_state.pt"))
