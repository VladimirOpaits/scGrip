import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scvi
import torch

import time  # NEW
import torch.nn as nn  # NEW: для nn.Identity в --linear
from sceris.lora import attach, verify_identical, default_targets, first_linear, AdapterSet  # CHANGED: + AdapterSet для фильтрации блоков

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
p.add_argument("--ns-energy", type=float, default=0.99)  # NEW: доля энергии census-активаций, чьё подпространство запрещено адаптеру; 0 = проекция выключена
p.add_argument("--ns-batch", type=int, default=2048)     # NEW: батч прохода по census при сборе активаций
p.add_argument("--layers", default=None)  # NEW: список блоков через запятую (например "enc0"), None = все блоки
p.add_argument("--lip-target", type=float, default=0.0)  # NEW: целевая верхняя граница spectral-gain адаптера (down·up); 0 = выключено
p.add_argument("--lip-min-elbo-weight", type=float, default=0.1)  # NEW: не душить ELBO ниже этой доли
p.add_argument("--seed", type=int, default=0)  # NEW: единая точка, управляющая всей случайностью прогона
p.add_argument("--linear", action="store_true")  # NEW: убрать GELU из адаптера — чистый линейный LoRA (up@down@x)
p.add_argument("--metric-ref-n", type=int, default=50)  # NEW: клеток на донора для ФИКСИРОВАННОГО референса strat-метрик (не зависит от n_replay)
p.add_argument("--resume-from", default=None)  # NEW: папка чекпойнта для дообучения (model.pt, adapters.pt, опционально train_state.pt)
p.add_argument("--start-epoch", type=int, default=0)  # NEW: с какой эпохи продолжать, если в чекпойнте нет train_state.pt; --epochs = ИТОГОВОЕ число эпох
args = p.parse_args()
torch.manual_seed(args.seed)  # NEW: BUGFIX — down.weight инициализировался через nn.init.normal_ без зафиксированного torch-seed'а, отсюда разный старт коррекции между "одинаковыми" прогонами

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}", flush=True)

model = scvi.model.SCVI.load(args.base_model)
module = model.module
module.eval()  # всегда замороженные веса

census_base = ad.read_h5ad(args.old_adata)
census_base.X = None  # NEW (OOM): X — точная копия layers['counts'] (проверено), а реестр базовой модели читает именно counts; без X concat не тащит вторую копию в combined

rng = np.random.default_rng(args.seed)  # CHANGED: было default_rng(0)
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

# CHANGED: референс для strat-метрик больше НЕ строится из replay — иначе при маленьком n_replay ни одна пара
# (донор, компартмент) не набирает STRAT_MIN_CELLS и метрика разваливается, а при среднем n_replay меняется её смысл.
# Берём фиксированную выборку census с собственным rng, одинаковую для всех прогонов: линейка не должна зависеть от бюджета replay.
metric_rng = np.random.default_rng(12345)  # NEW: намеренно НЕ зависит от --seed
metric_ref_idx = np.concatenate([
    metric_rng.choice(idx, size=min(args.metric_ref_n, len(idx)), replace=False)
    for idx in census_base.obs.groupby(args.donor, observed=True).indices.values()
])  # NEW
metric_ref_obs = census_base.obs.iloc[metric_ref_idx]  # CHANGED (OOM): только obs, без копии матрицы экспрессии
mu_metric_ref = model.get_latent_representation(census_base, indices=metric_ref_idx)  # CHANGED (OOM): по индексам, без .copy() на 80k клеток
ref_frame = pd.DataFrame(mu_metric_ref)  # CHANGED: было mu_ref (replay)
ref_frame["compartment"] = compartment(metric_ref_obs["cell_type"].astype(str)).to_numpy()  # CHANGED
ref_frame["donor"] = metric_ref_obs[args.donor].astype(str).to_numpy()  # CHANGED
ref_dc_grp = ref_frame.groupby(["compartment", "donor"], observed=True)
ref_dc = ref_dc_grp.mean()[ref_dc_grp.size() >= STRAT_MIN_CELLS]
ref_centroid = ref_dc.groupby("compartment", observed=True).mean()
print(f"[metric-ref] {len(metric_ref_idx)} cells, compartments: {list(ref_centroid.index)}", flush=True)  # NEW
del metric_ref_obs, mu_metric_ref, ref_frame, ref_dc_grp, ref_dc  # CHANGED (OOM): освобождаем всё промежуточное

# CHANGED (OOM): моменты census считаем ДО загрузки новой когорты и сразу отпускаем census_base —
# иначе полный census, new_adata и combined живут одновременно и это пик, на котором процесс убивали.
def input_moments(scvi_model, target_blocks, adata, indices=None):  # несмещённый второй момент X^T X входа первого Linear каждого блока
    C = {}
    handles = []
    for name, blk in target_blocks.items():
        def pre(_m, inp, name=name):
            x = inp[0].detach().double()
            C[name] = C[name] + x.T @ x if name in C else x.T @ x
        handles.append(first_linear(blk).register_forward_pre_hook(pre))
    with torch.no_grad():
        scvi_model.get_latent_representation(adata, indices=indices, batch_size=args.ns_batch)
    for h in handles:
        h.remove()
    return {name: c.cpu() for name, c in C.items()}

import gc  # CHANGED (OOM): перенесено сюда
C_census = C_eval = None
n_ns_cells = 0
if args.ns_energy > 0:
    ns_idx = np.setdiff1d(np.arange(census_base.n_obs), eval_idx)  # весь census КРОМЕ held-out eval — иначе held-out метрики нечестные
    base_blocks = {name: blk for name, (_i, _o, blk) in default_targets(module).items()}
    t_ns = time.time()
    C_census = input_moments(model, base_blocks, census_base, indices=ns_idx)
    C_eval = input_moments(model, base_blocks, eval_old)
    n_ns_cells = len(ns_idx)
    print(f"[nullspace] moments over {n_ns_cells} census cells collected in {time.time() - t_ns:.0f}s", flush=True)

del census_base  # CHANGED (OOM): дальше нужен только replay/eval_old (отдельные копии)
gc.collect()

new_adata = ad.read_h5ad(args.new_adata)
new_adata.X = None  # NEW (OOM): то же самое — X дублирует counts
gc.collect()

new_healthy_local_mask = (new_adata.obs[args.disease_col].values == args.healthy_label)
new_ct_healthy = compartment(new_adata.obs["cell_type"].astype(str))[new_healthy_local_mask]
new_donor_healthy = new_adata.obs[args.donor].astype(str).to_numpy()[new_healthy_local_mask]

combined = ad.concat([replay, new_adata], join="outer", label="_replay_source",
                          keys=["replay", "new"], index_unique="_")
combined = combined[:, replay.var_names].copy()  # BUGFIX: outer concat сортирует гены по алфавиту, а энкодер ждёт порядок census-модели (2997/3000 генов стояли не на месте)
assert np.array_equal(combined.var_names, replay.var_names)

scvi.model.SCVI.setup_anndata(combined, batch_key=args.donor, layer="counts")  # CHANGED (OOM): X теперь None, данные лежат в counts (как и в реестре базовой модели)
q_model = scvi.model.SCVI.load_query_data(combined, model)   # freeze=True по умолчанию
q_model.is_trained_ = True  # костыль

q_module = q_model.module
q_module.eval()
q_module.to(device)

targets = default_targets(q_module)  # CHANGED: фильтруем блоки перед attach, если задан --layers
if args.layers:
    keep = set(args.layers.split(","))
    targets = {name: t for name, t in targets.items() if name in keep}
    print(f"adapters attached only to: {list(targets.keys())}", flush=True)
adapters = AdapterSet(targets, rank=args.rank, n_cov=0, act=nn.Identity if args.linear else nn.GELU).to(device)  # CHANGED: было attach(q_module, ...); + act
verify_identical(q_model, adapters)



is_new = (combined.obs["_replay_source"] == "new").values
is_replay = ~is_new
healthy_new_mask = is_new & (combined.obs[args.disease_col].values == args.healthy_label)  # для penalty
new_adata_healthy = combined[healthy_new_mask].copy()
train_adata = combined

del new_adata  # NEW (OOM): всё нужное от неё уже в combined и в new_ct_healthy/new_donor_healthy
gc.collect()

# ---- NEW: null-space проекция — подпространство входов каждого адаптируемого блока, занятое census ----
proj = {}  # NEW: имя адаптера -> проектор на ортогональное дополнение census-подпространства
if args.ns_energy > 0:
    q_blocks = {name: blk for name, (_i, _o, blk) in default_targets(q_module).items()}
    C_new = input_moments(q_model, q_blocks, combined, indices=np.where(is_new)[0])  # адаптеры ещё нулевые, выход идентичен базе

    def outside(P, Cx):  # доля энергии Cx вне census-подпространства: trace(P Cx) / trace(Cx)
        return float((P * Cx).sum() / torch.trace(Cx))

    for name in adapters.adapters.keys():
        evals, evecs = torch.linalg.eigh(C_census[name])
        evals, evecs = evals.flip(0).clamp_min(0), evecs.flip(1)
        cum = torch.cumsum(evals, 0) / evals.sum()
        k = int((cum < args.ns_energy).sum().item()) + 1
        U = evecs[:, :k]
        P = torch.eye(U.shape[0], dtype=torch.float64) - U @ U.T
        print(f"[nullspace] {name}: in_dim={U.shape[0]} k={k} | energy outside: "
              f"census_train={outside(P, C_census[name]):.4f} census_held_out={outside(P, C_eval[name]):.4f} "
              f"new_cohort={outside(P, C_new[name]):.4f}", flush=True)
        proj[name] = P.float().to(device)
    del C_census, C_eval, C_new

    with torch.no_grad():
        for name, a in adapters.adapters.items():
            a.down.bias.zero_()
            a.down.bias.requires_grad_(False)  # bias даёт константный сдвиг ВСЕМ клеткам, в обход проекции
            a.up.bias.zero_()
            a.up.bias.requires_grad_(False)
            a.down.weight.copy_(a.down.weight @ proj[name])
# ---- конец null-space ----

def adapter_lipschitz():  # NEW: верхняя граница усиления адаптера (down·up spectral norm) — считается ТОЛЬКО из своих же весов, без held-out
    return max(
        float(torch.linalg.matrix_norm(a.down.weight, ord=2)) * float(torch.linalg.matrix_norm(a.up.weight, ord=2))
        for a in adapters.adapters.values()
    )

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
mmd_rng = np.random.default_rng(args.seed + 2000)  # CHANGED: было default_rng(2)

healthy_all_tensors = None  # NEW (perf): все healthy_new тензоры грузим ОДИН раз здесь, а не AnnDataLoader на каждый компартмент на каждом шаге
if mmd_comps:  # NEW
    _healthy_all_loader = AnnDataLoader(q_model.adata_manager, indices=healthy_new_pos, batch_size=len(healthy_new_pos), shuffle=False)  # NEW
    healthy_all_tensors = next(iter(_healthy_all_loader))  # NEW
    healthy_all_tensors = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in healthy_all_tensors.items()}  # NEW
# ---- конец подготовки MMD ----


ep_rng = np.random.default_rng(args.seed + 1000)  # CHANGED: было default_rng(1)
n_replay = len(replay)
new_pos = np.where(is_new)[0]
n_healthy_new = len(healthy_new_pos)

def eval_pre_training():  # NEW: те же метрики ДО обучения — нулевой уровень (естественный разброс доноров вокруг референса)
    mu_e = q_model.get_latent_representation(eval_old)
    a_e = float((((mu_e - mu_eval_ref) / mu_ref_sigma) ** 2).mean())
    s_n, _ = stratified_dist(q_model.get_latent_representation(new_adata_healthy), new_ct_healthy, new_donor_healthy, ref_centroid, mu_ref_sigma)
    s_e, _ = stratified_dist(mu_e, eval_ct, eval_donor, ref_centroid, mu_ref_sigma)
    print(f"epoch -1 (pre-training): anchor_held_out={a_e:.4f} | "
          f"strat_healthy_new={float(s_n.mean()):.4f} strat_held_out={float(s_e.mean()):.4f}", flush=True)

import os  # NEW: нужен уже здесь для путей чекпойнта
start_epoch = 0  # NEW: дообучение с чекпойнта вместо перезапуска с нуля
if args.resume_from:
    # сплит replay/eval, подпространство и combined пересобраны детерминированно из тех же --seed/--n-replay, поэтому формы совпадают
    ckpt_sd = torch.load(os.path.join(args.resume_from, "model.pt"), map_location=device, weights_only=False)["model_state_dict"]
    ckpt_sd.pop("pyro_param_store", None)  # служебный ключ сериализации scvi, не параметр модуля; strict-загрузка остаётся, чтобы ловить настоящие несовпадения
    q_module.load_state_dict(ckpt_sd)
    adapters.load_state_dict(torch.load(os.path.join(args.resume_from, "adapters.pt"), map_location=device))
    state_path = os.path.join(args.resume_from, "train_state.pt")
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)  # BUGFIX: torch.set_rng_state принимает только CPU-тензор; состояние Adam optimizer сам перенесёт на устройство параметров
        optimizer.load_state_dict(st["optimizer"])
        ep_rng.bit_generator.state = st["ep_rng"]
        mmd_rng.bit_generator.state = st["mmd_rng"]
        torch.set_rng_state(st["torch_rng"])
        start_epoch = st["epoch"]
    else:
        start_epoch = args.start_epoch
        for _ in range(start_epoch):  # прокручиваем поток ep_rng, чтобы не повторить порядок replay первых эпох
            ep_rng.permutation(n_replay)
            if args.lam_new > 0 and n_healthy_new > 0:
                ep_rng.permutation(n_healthy_new)
        print("WARNING: no train_state.pt — Adam moments start fresh, mmd/loader RNG streams differ from an uninterrupted run", flush=True)
    print(f"resumed from {args.resume_from} at epoch {start_epoch}", flush=True)
else:
    eval_pre_training()  # NEW

t0 = time.time()  # NEW
for epoch in range(start_epoch, args.epochs):  # CHANGED: было range(args.epochs)
    elbo_weight = 1.0  # NEW: самозатухающий вес ELBO — считается ТОЛЬКО из весов адаптера, доступно в любом реальном сценарии
    current_lip = adapter_lipschitz()  # NEW: печатаем всегда, даже без троттлинга — для калибровки --lip-target
    if args.lip_target > 0:
        elbo_weight = max(args.lip_min_elbo_weight, min(1.0, args.lip_target / max(current_lip, 1e-8)))
    print(f"  [lipschitz] gain={current_lip:.4f} target={args.lip_target:.4f} elbo_weight={elbo_weight:.4f}", flush=True)

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

        loss = elbo_weight * elbo + args.lam * anchor + args.lam_new * delta_pen + args.lam_mag * mag_pen + args.lam_mmd * mmd  # CHANGED: elbo -> elbo_weight * elbo
        loss.backward()
        optimizer.step()
        if proj:  # NEW: после каждого шага возвращаем строки down.weight в ортогональное дополнение census-подпространства
            with torch.no_grad():
                for name, a in adapters.adapters.items():
                    a.down.weight.copy_(a.down.weight @ proj[name])

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
          f"strat_healthy_new={strat_mean:.4f} strat_held_out={strat_eval_mean:.4f} | t={time.time() - t0:.0f}s", flush=True)  # CHANGED: + время

import os
run_name = args.run_name or f"nullspace_e{args.ns_energy}_rank{args.rank}_lam{args.lam}_lammag{args.lam_mag}_lammmd{args.lam_mmd}{'_linear' if args.linear else ''}_seed{args.seed}"  # CHANGED: + seed, + linear
out_path = os.path.join(args.out_dir, run_name)
os.makedirs(out_path, exist_ok=True)
q_model.save(out_path, overwrite=True)
torch.save(adapters.state_dict(), os.path.join(out_path, "adapters.pt"))
torch.save({  # NEW: состояние оптимизатора и генераторов — чтобы следующее дообучение через --resume-from продолжалось без сброса Adam
    "optimizer": optimizer.state_dict(),
    "ep_rng": ep_rng.bit_generator.state,
    "mmd_rng": mmd_rng.bit_generator.state,
    "torch_rng": torch.get_rng_state(),
    "epoch": args.epochs,
}, os.path.join(out_path, "train_state.pt"))
