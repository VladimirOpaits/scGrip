---
name: mmd-alignment-inconclusive
description: "MMD(z(cohort2_normal), z(census)) alignment term tried and deprioritized -- overshoots the batch-removal target, but the whole comparison is confounded by n=1 runs and mismatched stopping rules, not settled science. Not resumed unless the paper specifically needs it."
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
  modified: 2026-08-25T20:14:43.529Z
---

2026-08-24/25. The "missing member" flagged in [[anchored-finetune-result]]
(λ=30 vs λ=100 nearly identical → anchor and batch-removal are orthogonal,
residual batch isn't fixed by λ) was implemented as a two-phase scheme in
`MMD-anchor-scVI.ipynb` + `build/finetune_mmd_phase2*.py`: phase 1 = ELBO +
anchor only (warm start, `models/scvi_mmd_phase1_only`), phase 2 = phase 1 +
`lam_align * MMD(z_new, mu_ref)` on top (multi-kernel RBF, `mmd2_train` drops
the constant Kyy term, real magnitude via `mmd2_report`).

**Result, swept lam_align = 30 / 10 / 1 (5000 steps each, same phase-1
start):** all three landed at nearly the same real `c2normal_sigma` (-0.94,
-0.94, drifting toward -1.29 by proxy) — MMD fully converges its own
objective (mmd(full) → ~0.01 regardless of weight) and overshoots PAST the
target of ~0, not just short of it. `AUC_c2n_vs_train` flipped from 0.822
(anchor-only, real, `anchored_lam30`) to ~0.18 (MMD, real) -- same
separability magnitude, opposite sign: batch offset flipped, not removed.

**Bigger finding, undercutting the whole comparison:** the real (full-encode,
non-proxy) check on `phase1_only` itself -- ELBO+anchor, ZERO MMD steps --
already gave `c2normal_sigma = -0.206`, `AUC_c2n_vs_train = 0.453` (near the
0.5 target) and BETTER `gap_sigma`/`AUC_sle_vs_c2n`/IFN-r than the original
`anchored_lam30` branch (+1.049 / 0.822). So on paper phase1_only alone beats
anchored_lam30, and MMD only makes it worse from there.

**Why this is NOT a settled result:** `phase1_only` (1000 steps, round
number, no stopping criterion) and `anchored_lam30` (up to 100 epochs,
early-stopped on `val_elbo`, unrelated to `c2normal_sigma`) are two different
training protocols compared as single runs (n=1 each). No seed replication
exists for either. No control was run for "ELBO+anchor alone, trained as
long as the MMD runs (5000+ steps) -- does c2normal_sigma drift negative on
its own, without MMD ever touching it?" That control is the single biggest
missing piece: it would separate "MMD actively overshoots" from "extended
unfrozen fine-tuning drifts this way regardless of MMD, which merely
accelerates a pre-existing dynamic."

Proxy `c2normal_sigma` (computed on the 25-cells/donor replay subsample, used
for cheap in-training monitoring) disagreed with the real full check by ~2x
in magnitude on the same checkpoint (-0.65 proxy vs -0.206 real) though same
sign/direction as the two real MMD endpoints (-0.94, -0.94) -- trust the
proxy's direction within one run, not its magnitude or absolute
zero-crossing point.

**Decision (2026-08-25, Vlad's call): paused, not resumed for now.** Chasing
this further (replication runs, the missing anchor-only-longer control,
early-stopping-on-real-metric) is method-validation work that doesn't move
the paper's actual Step 1 goals -- AB-MIL implementation (not started) and
cohort 3/4 biomarker replication ([[lupus-cohort34-replication]], the
paper's stated main result, also not started). `anchored_lam30` remains the
standing baseline used throughout this project's cohort-2 analysis (MELD/
DA-seq, composition work); `phase1_only`'s apparently-better numbers are
noted here but NOT adopted as the new baseline until (if ever) the missing
control/replication work above is actually done.

**UPDATE (2026-08-25, cohort 4 replication with a real held-out split).** Resumed
specifically to diagnose cohort 4's batch-removal guard failure (see
[[lupus-cohort34-anchored-result]]). This run finally had the missing control from the
list above: cohort 4's healthy donors split into group A (in the MMD loss) and group B
(held out from MMD, still gets ELBO like phase 1 did). Two findings:

1. **The overshoot replicates exactly** on this second, independent cohort, and now on
   a genuinely held-out group (not a same-data proxy): running lam_align=10 to 5000
   steps flips `AUC_c2n_vs_train` 0.982 -> 0.168 (opposite-direction separability, same
   magnitude) -- same pathology as the original cohort-2 result, now confirmed not to be
   an artifact of the missing validation split.
2. **A live, signed, held-out stopping signal fixes it.** `c2n_sigma(B)` -- computed by
   RE-ENCODING the census replay through the CURRENT module every check (not the frozen
   pre-training mu_ref snapshot -- an early version of this used the frozen snapshot and
   was off by a full sign flip, +2.96 true vs -0.75 measured, a real bug, fixed by
   recomputing the reference live) -- crosses zero in just ~20 steps (lam_align=10,
   checked every 5 steps). Stopping there: `c2normal_sigma` 2.958 -> 0.283,
   `census_pairwise_CV` UNCHANGED (0.281 -> 0.281, geometry literally untouched at this
   step count), and both guards cohort 4 was failing now pass (`AUC_sle_vs_c2n` 0.849 ->
   0.883, `r_ifn_cohort2` 0.745 -> 0.822).

**Revised verdict:** MMD-as-implemented converges to the SAME overshot endpoint
regardless of lam_align (1/10/30 all land near the same place, now on two cohorts) --
lam_align controls *speed*, not the destination. That means lam_align is not the
right tuning knob; STEP COUNT with an honest held-out stop is. The original "paused,
not resumed" verdict was about lam_align sweeps; this held-out-early-stop design is the
missing piece the pause was waiting for, and it worked cleanly on cohort 4. Worth
retrying on cohort 2's original overshoot (mmd_phase2_mmd, mmd_phase2_lamalign10) with
this same held-out+early-stop recipe before generalizing further.

Scripts: `build/finetune_mmd_phase2_cohort4.py` (plain, unsigned offset/spread
tracking -- kept for the offset/spread decomposition view, but its unsigned metric did
NOT catch the overshoot, only the signed one did), `build/finetune_mmd_phase2_cohort4_earlystop.py`
(the fixed, signed, live-reference version that actually worked).

**If resumed later:** start with the control run (continue `phase1_only`
ELBO+anchor-only, no MMD, real checks every ~1000 steps out to 5000+) before
touching `lam_align` again -- scripts already exist:
`build/finetune_mmd_phase2.py`, `build/finetune_mmd_phase2_earlystop.py`,
`scenarios/encode_mmd_phase2_cohort2.py` (parametrized `--model-dir`/
`--prefix`), checkpoints `models/scvi_mmd_phase1_only`,
`models/scvi_mmd_phase2_{mmd,lamalign10,lamalign1_earlystop}`.
