import gc
import json
import os
import time

import anndata as ad
import numpy as np
import pandas as pd
import scvi
import torch
import torch.nn as nn
from scvi.dataloaders import AnnDataLoader

from sceris.cli.build_parser import build_argparse
from sceris.functions import (
    compartment,
    mmd2,
    mmd_compartment_positions,
    multiscale_sigmas,
    reference_geometry,
    sample_per_donor,
    stratified_dist,
    summarize_stratified,
)
from sceris.lora import attach, verify_identical


args = build_argparse().parse_args()
torch.manual_seed(args.seed)
device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}", flush=True)

run_name = args.run_name or (
    f"rank{args.rank}_lam{args.lam}_mmd{args.lam_mmd}_mag{args.lam_mag}"
)
out_path = os.path.join(args.out_dir, run_name)
checkpoint_dir = os.path.join(args.out_dir, f"{run_name}_checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
with open(os.path.join(checkpoint_dir, "run_config.json"), "w") as handle:
    json.dump(vars(args), handle, indent=2, sort_keys=True)


def save_evaluator_checkpoint(completed_epoch):
    """Save only state needed to re-encode one historical training epoch."""
    if completed_epoch not in args.save_epochs:
        return
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{completed_epoch:03d}.pt")
    torch.save(
        {
            "format": "sceris_lora_adapter_epoch_v1",
            "epoch": completed_epoch,
            "adapter_state_dict": adapters.state_dict(),
            "rank": args.rank,
            "gelu": args.gelu,
        },
        checkpoint_path,
    )
    print(f"saved evaluator checkpoint: {checkpoint_path}", flush=True)


def to_device(tensors):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in tensors.items()
    }


model = scvi.model.SCVI.load(args.base_model)
module = model.module
module.eval()
model_vars = pd.Index(
    torch.load(
        os.path.join(args.base_model, "model.pt"),
        map_location="cpu",
        weights_only=False,
    )["var_names"]
)

census = ad.read_h5ad(args.old_adata)
census.X = None
assert np.array_equal(census.var_names, model_vars)

split_rng = np.random.default_rng(args.split_seed)
replay_idx, eval_idx = [], []
for donor_idx in census.obs.groupby(args.donor, observed=True).indices.values():
    replay_take = split_rng.choice(
        donor_idx,
        size=min(args.n_replay, len(donor_idx)),
        replace=False,
    )
    replay_idx.append(replay_take)
    remaining = np.setdiff1d(donor_idx, replay_take)
    eval_idx.append(
        split_rng.choice(
            remaining,
            size=min(args.n_eval, len(remaining)),
            replace=False,
        )
    )
replay_idx = np.concatenate(replay_idx)
eval_idx = np.concatenate(eval_idx)
replay = census[replay_idx].copy()
eval_old = census[eval_idx].copy()
n_replay_cells = len(replay)

mu_ref = model.get_latent_representation(replay)
anchor_sigma = np.maximum(mu_ref.std(axis=0), 1e-6)
mu_eval_ref = model.get_latent_representation(eval_old)
replay_ct = compartment(replay.obs["cell_type"].astype(str)).to_numpy()

metric_rng = np.random.default_rng(12345)
metric_idx = sample_per_donor(
    census.obs,
    args.donor,
    args.metric_ref_n,
    metric_rng,
)
metric_obs = census.obs.iloc[metric_idx]
mu_metric_ref = model.get_latent_representation(census, indices=metric_idx)
metric_ct = compartment(metric_obs["cell_type"].astype(str)).to_numpy()
metric_donor = metric_obs[args.donor].astype(str).to_numpy()
ref_centroid, metric_sigma, ref_support = reference_geometry(  # фикс
    mu_metric_ref,
    metric_ct,
    metric_donor,
)
metric_sigma = np.maximum(metric_sigma, 1e-6)
print(
    "strat reference support (Census donors): "
    + ", ".join(f"{name}={count}" for name, count in ref_support.items()),
    flush=True,
)
del metric_obs, metric_idx, mu_metric_ref, replay_idx, eval_idx, census
gc.collect()

new = ad.read_h5ad(args.new_adata)
new.X = None
new_donor = new.obs[args.donor].astype(str).to_numpy()
new_ct = compartment(new.obs["cell_type"].astype(str)).to_numpy()
healthy = new.obs[args.disease_col].astype(str).to_numpy() == args.healthy_label

healthy_donors = np.unique(new_donor[healthy])
if not len(healthy_donors):
    raise ValueError("The new cohort has no donors with the requested healthy label.")
ab_rng = np.random.default_rng(args.split_seed + 1)
if len(healthy_donors) == 1:
    donors_b = np.array([], dtype=healthy_donors.dtype)
else:
    n_b = min(
        len(healthy_donors) - 1,
        max(1, round(len(healthy_donors) * args.holdout_frac)),
    )
    donors_b = ab_rng.choice(healthy_donors, size=n_b, replace=False)
group = np.where(
    ~healthy,
    "disease",
    np.where(np.isin(new_donor, donors_b), "B", "A"),
)
print(
    f"healthy donors A={len(healthy_donors) - len(donors_b)} "
    f"B={len(donors_b)} | disease cells={int((~healthy).sum())}",
    flush=True,
)

combined = ad.concat(
    [replay, new],
    join="outer",
    label="_source",
    keys=["replay", "new"],
    index_unique="_",
)
del new
combined = combined[:, model_vars].copy()
assert np.array_equal(combined.var_names, model_vars)
del model_vars
gc.collect()

new_pos = np.flatnonzero(combined.obs["_source"].to_numpy() == "new")
assert len(new_pos) == len(group)
pos_a = new_pos[group == "A"]
pos_b = new_pos[group == "B"]

scvi.model.SCVI.setup_anndata(combined, batch_key=args.donor, layer="counts")
q_model = scvi.model.SCVI.load_query_data(combined, model)
q_model.is_trained_ = True  
q_module = q_model.module
q_module.eval()
q_module.to(device)

for parameter in q_module.parameters():
    parameter.requires_grad_(False)  # обязательно замораживаем веса
adapters = attach(
    q_module,
    rank=args.rank,
    device=device,
    act=nn.GELU if args.gelu else nn.Identity,
)
verify_identical(q_model, adapters)
n_check = min(512, n_replay_cells)
mu_check = q_model.get_latent_representation(combined, indices=np.arange(n_check))
assert np.abs(mu_check - mu_ref[:n_check]).max() < 1e-4, (
    "replay in combined is encoded differently: check gene order and counts"
)
print(
    f"combined/replay identity check passed: "
    f"max diff {np.abs(mu_check - mu_ref[:n_check]).max():.2e}",
    flush=True,
)

healthy_a = combined[pos_a].copy()
healthy_b = combined[pos_b].copy()
optimizer = torch.optim.Adam(adapters.parameters(), lr=args.lr)

a_ct = new_ct[group == "A"]                      
mmd_a = mmd_compartment_positions(replay_ct, a_ct)
print(f"MMD compartments (replay x A): {list(mmd_a)}", flush=True)

a_tensors = None
if args.lam_mmd > 0 and mmd_a:
    a_loader = AnnDataLoader(
        q_model.adata_manager,
        indices=pos_a,
        batch_size=len(pos_a),
        shuffle=False,
    )
    a_tensors = to_device(next(iter(a_loader)))


def format_stratified(summaries, common_compartments):
    """Compact per-compartment report with the number of eligible donors."""  # фикс
    chunks = [f"strat common={list(common_compartments)}"]
    for name, (means, support) in summaries.items():
        cells = [
            f"{compartment_name}={means[compartment_name]:.3f}[n={int(support[compartment_name])}]"
            for compartment_name in means.index
            if support[compartment_name] > 0
        ]
        chunks.append(f"strat_{name}_by_comp: " + (", ".join(cells) or "none"))
    return " | ".join(chunks)


def evaluate():
    mu_eval = q_model.get_latent_representation(eval_old)
    diff = mu_eval - mu_eval_ref
    out = {
        "anchor_held_out": float(((diff / anchor_sigma) ** 2).mean()),
        "drift_max": float(np.linalg.norm(diff, axis=1).max()),
    }
    summaries = {}
    for name, adata_group in [("A", healthy_a), ("B", healthy_b)]:
        if adata_group.n_obs:
            mu_group = q_model.get_latent_representation(adata_group)
            distances = stratified_dist(
                mu_group,
                new_ct[group == name],
                new_donor[group == name],
                ref_centroid,
                metric_sigma,
            )
        else:
            distances = pd.DataFrame(columns=ref_centroid.index, dtype=float)
        summaries[name] = summarize_stratified(distances)  # храним mean и coverage отдельно по компартментам

    means_a, support_a = summaries["A"]
    means_b, support_b = summaries["B"]
    common_compartments = means_a.index[(support_a > 0) & (support_b > 0)]  # фикс: A/B scalar считается на одном и том же support
    out["strat_A"] = (
        float(means_a.loc[common_compartments].mean())
        if len(common_compartments)
        else float("nan")
    )
    out["strat_B"] = (
        float(means_b.loc[common_compartments].mean())
        if len(common_compartments)
        else float("nan")
    )
    return out, summaries, common_compartments


ep_rng = np.random.default_rng(args.seed + 1000)
mmd_rng = np.random.default_rng(args.seed + 2000)
mu_ref_t = torch.as_tensor(mu_ref, dtype=torch.float32, device=device)
anchor_sigma_t = torch.as_tensor(anchor_sigma, dtype=torch.float32, device=device)

start_epoch = 0
if args.resume_from:
    model_state = torch.load(
        os.path.join(args.resume_from, "model.pt"),
        map_location=device,
        weights_only=False,
    )["model_state_dict"]
    model_state.pop("pyro_param_store", None)
    q_module.load_state_dict(model_state)
    adapters.load_state_dict(
        torch.load(
            os.path.join(args.resume_from, "adapters.pt"),
            map_location=device,
            weights_only=False,
        )
    )
    state_path = os.path.join(args.resume_from, "train_state.pt")
    if os.path.exists(state_path):
        train_state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(train_state["optimizer"])
        ep_rng.bit_generator.state = train_state["ep_rng"]
        mmd_rng.bit_generator.state = train_state["mmd_rng"]
        torch.set_rng_state(train_state["torch_rng"])
        start_epoch = train_state["epoch"]
    else:
        start_epoch = args.start_epoch
        for _ in range(start_epoch):
            ep_rng.permutation(n_replay_cells)
        print("WARNING: no train_state.pt; Adam state starts fresh", flush=True)
    print(f"resumed at epoch {start_epoch}", flush=True)
else:
    metrics, strat_summaries, common_compartments = evaluate()
    print(
        "epoch -1 (pre-training): "
        + " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        + "\n  "
        + format_stratified(strat_summaries, common_compartments),
        flush=True,
    )


t0 = time.time()
for epoch in range(start_epoch, args.epochs):
    replay_perm = ep_rng.permutation(n_replay_cells)
    replay_perm_t = torch.as_tensor(replay_perm, device=device)
    new_loader = AnnDataLoader(
        q_model.adata_manager,
        indices=new_pos,
        batch_size=args.batch_size,
        shuffle=True,
    )
    replay_loader = AnnDataLoader(
        q_model.adata_manager,
        indices=replay_perm,
        batch_size=args.batch_size,
        shuffle=False,
    )
    replay_iter = iter(replay_loader)
    replay_ptr = 0
    sums = {"elbo": 0.0, "anchor": 0.0, "mag": 0.0, "mmd": 0.0}
    n_steps = 0

    for new_tensors in new_loader:
        try:
            replay_tensors = next(replay_iter)
        except StopIteration:
            replay_iter = iter(replay_loader)
            replay_ptr = 0
            replay_tensors = next(replay_iter)

        new_tensors = to_device(new_tensors)
        replay_tensors = to_device(replay_tensors)
        batch_size = replay_tensors["X"].shape[0]
        target = mu_ref_t[replay_perm_t[replay_ptr:replay_ptr + batch_size]]
        replay_ptr += batch_size

        optimizer.zero_grad(set_to_none=True)
        adapters.clear()
        elbo = q_module(new_tensors, loss_kwargs={"kl_weight": 1.0})[2].loss

        adapters.clear()
        mu_replay = q_module.inference(
            **q_module._get_inference_input(replay_tensors)
        )["qz"].loc
        anchor = (((mu_replay - target) / anchor_sigma_t) ** 2).mean()
        mag = (
            adapters.penalty()
            if args.lam_mag > 0
            else torch.zeros((), device=device)
        )

        mmd = torch.zeros((), device=device)
        if a_tensors is not None:
            sampled_pairs = []
            for compartment_name, (ref_positions, a_positions) in mmd_a.items():
                n_pick = min(
                    args.mmd_k_per_compartment,
                    len(ref_positions),
                    len(a_positions),
                )
                sampled_pairs.append(
                    (
                        compartment_name,
                        mmd_rng.choice(ref_positions, size=n_pick, replace=False),
                        mmd_rng.choice(a_positions, size=n_pick, replace=False),
                    )
                )

            a_pick = np.concatenate([a_positions for _, _, a_positions in sampled_pairs])
            a_pick_t = torch.as_tensor(a_pick, dtype=torch.long, device=device)
            mmd_batch = {
                key: value[a_pick_t] if torch.is_tensor(value) else value
                for key, value in a_tensors.items()
            }
            adapters.clear()
            z_all = q_module.inference(
                **q_module._get_inference_input(mmd_batch)
            )["qz"].loc

            terms = []
            offset = 0
            for _, ref_positions, a_positions in sampled_pairs:
                n_pick = len(a_positions)
                z_compartment = z_all[offset:offset + n_pick]
                ref_compartment = mu_ref_t[
                    torch.as_tensor(ref_positions, dtype=torch.long, device=device)
                ]
                offset += n_pick
                terms.append(
                    mmd2(
                        z_compartment,
                        ref_compartment,
                        multiscale_sigmas(z_compartment, ref_compartment),
                    )
                )
            mmd = torch.stack(terms).mean()

        loss = (
            elbo
            + args.lam * anchor
            + args.lam_mag * mag
            + args.lam_mmd * mmd
        )
        loss.backward()
        optimizer.step()

        for key, value in [("elbo", elbo), ("anchor", anchor), ("mag", mag), ("mmd", mmd)]:
            sums[key] += float(value.detach())
        n_steps += 1

    metrics, strat_summaries, common_compartments = evaluate()
    print(
        f"epoch {epoch}: "
        + " ".join(f"{key}={value / n_steps:.4f}" for key, value in sums.items())
        + " | "
        + " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        + f" | t={time.time() - t0:.0f}s"
        + "\n  "
        + format_stratified(strat_summaries, common_compartments),
        flush=True,
    )
    save_evaluator_checkpoint(epoch + 1)


q_model.save(out_path, overwrite=True)
torch.save(adapters.state_dict(), os.path.join(out_path, "adapters.pt"))
torch.save(
    {
        "optimizer": optimizer.state_dict(),
        "ep_rng": ep_rng.bit_generator.state,
        "mmd_rng": mmd_rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "epoch": args.epochs,
        "donors_b": list(donors_b),
    },
    os.path.join(out_path, "train_state.pt"),
)
with open(os.path.join(out_path, "run_config.json"), "w") as handle:
    json.dump(vars(args), handle, indent=2, sort_keys=True)
