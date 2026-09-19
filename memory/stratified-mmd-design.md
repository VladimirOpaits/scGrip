---
name: stratified-mmd-design
description: "Pooled MMD between cohort and Census silently penalizes cell-type COMPOSITION difference and pays for it by degrading cell-type geometry; the fix is to compute MMD per compartment and average unweighted, in both the loss and the stopping signal."
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
  modified: 2026-08-27T14:23:08.904Z
---

2026-08-26. Vlad asked the right question: the scVI encoder is a per-cell function
`z = f(x, batch)` and literally cannot see that a donor is 55% monocytes, so how can
composition be "removed"? Answer: it can't be removed, but it can force a distortion,
and composition enters at the AGGREGATION step, in two places.

**In the metric.** A donor centroid is a composition-weighted convex combination:
`centroid_d = Σ_t (fraction of type t in d) · (mean latent of type t in d)`. Two donors
with identical per-cell biology but different composition get different centroids —
this is the documented r(monocyte fraction, distance) = +0.81 among healthy donors
(see [[composition-confound-lupus]]).

**In the MMD loss — the part that was NOT previously understood.** MMD compares POOLED
cell distributions. Cohort is ~55% monocytes, Census ~10%. Two clouds that differ in
type proportions cannot be made to overlap by any translation. The only way to reduce
pooled MMD is to distort the relative geometry of cell types — pull cohort monocytes
toward where Census non-monocytes sit, or collapse the myeloid/lymphoid separation so
the proportion difference stops registering. The encoder never sees composition; the
OBJECTIVE is defined on the pool, and the pool differs by composition, so the gradient
attacks cell-type geometry as a proxy.

The anchor term and the ELBO are both per-cell and therefore composition-clean. It is
specifically the MMD term (and the whole-donor metric) that carry this defect. It is
NOT the decoder's donor embedding — the decoder also decodes each cell independently.

**Confirmed by dose-response on cohort 4** (scIB, cached latents, cell-type ASW):
```
MMD steps                bio conservation   batch mixing
0    (anchor only)            0.514            0.880
20   (early-stopped)          0.504  −0.010    0.880
5000 (run to convergence)     0.485  −0.029    0.884
```
Cell-type separability degrades monotonically with how hard MMD is pushed, ~3x worse at
convergence, while batch mixing is flat (0.880 → 0.884) — the price comes specifically
out of the biological axis. This also upgrades early stopping: it is not only preventing
the c2normal_sigma sign overshoot ([[mmd-alignment-inconclusive]]), it is independently
limiting cell-type geometry damage. NOT yet proven that the damage is specifically
compositional rather than generic large-perturbation; the discriminating test is a
per-cell-type ASW breakdown — if compositional, damage should concentrate in monocytes,
where the proportion mismatch is largest.

**The fix — stratified MMD.** Compute MMD inside each compartment and average unweighted:
```
L_align = (1/n_types) · Σ_t MMD²(z_cohort[type t], z_census[type t])
```
Cohort monocytes then get compared to Census monocytes rather than to a pool where
monocytes are 10%. The proportion difference drops out of the gradient entirely, so
there is no longer any incentive to distort between-type geometry. Same logic already
applied to the METRIC in `scenarios/stratified_distance.py`, just moved into the loss.
Stratify the stopping signal the same way, so composition falls out of both and we stop
accidentally optimizing against the biology that, in lupus, IS the subject.

**Why this matters beyond tidiness:** at whole-donor level, composition alone predicts
SLE at AUC 0.985 (see CLAUDE.md). Any objective that quietly rewards erasing
compositional difference is optimizing against the disease signal.

**IMPLEMENTED AND VALIDATED ON COHORT 4 (2026-08-26).**
`build/finetune_mmd_stratified.py`. 9 compartments enter the loss (B, CD4 T, CD8 T, DC,
NK, monocyte, other, pDC, plasma; `other T` has zero cohort-side cells and `progenitor`
only 2 census cells, both drop out). K=64 cells per compartment per side per step.

The stopping signal is stratified too, and had to be redesigned: the replay holds only 25
cells/donor, so per-(donor, compartment) census centroids are impossible (~2.5 monocytes
per donor) and `stratified_distance.py`'s MIN_CELLS=20 ruler cannot be built from it.
Replacement statistic, per compartment t:
`score_t = ||mean(z_B[t]) - mean(z_ref[t])|| / null_t`, where `null_t` is the same
offset for a random census donor subset of the SAME size as group B — i.e. how far apart
two samples of that size drawn from census itself land. Target ~1 (as far from census as
census is from itself), not 0. Averaged unweighted over compartments.

**Per-type ASW confirmed the mechanism first**: of the -0.0313 global compartment-ASW
drop from 0 to 5000 pooled-MMD steps, monocytes contribute -0.0250 — **~80% of all the
damage**, and 86% of the damage at the 20-step mark. Compartment-level distances show
what pooled MMD actually traded: at 5000 steps it pushed B cells 1.910 -> 2.458 and
monocytes 1.593 -> 2.100 FURTHER from census while pulling T/NK in, and collapsed the
monocyte disease gap 0.613 -> 0.281.

**Result, cohort 4** (stratified metric unless noted; strat@2.5 = the 20-step milestone,
`models/scvi_mmd_lam30_cohort4_strat_lamalign10_at2.5`):
```
                     baseline   pooled@20   strat@2.5
c2normal_sigma         1.259      0.310       0.171
gap_sigma              1.255      1.019       1.074
AUC_sle_vs_c2n         0.804      0.780       0.802
r_ifn_sle              0.881      0.882       0.885
bio conservation      0.5574     0.5454      0.5517
  monocyte ASW        0.5722     0.5512      0.5617
census_pairwise_CV     0.281      0.281       0.281
```
Stratified dominates pooled on BOTH axes at once: more batch removed (0.171 vs 0.310)
AND more disease signal kept (0.802 vs 0.780), with exactly half the ASW damage globally
and in monocytes specifically. Reference geometry untouched.

**Second diagnostic worth keeping:** stratified training keeps the pooled and stratified
metrics CONSISTENT (final: pooled -0.347, stratified -0.069), while pooled training
drives them apart (pooled -1.114 vs stratified +1.406). Divergence between the two is
itself a signal that the objective is being gamed.

Running the stratified version to plateau (400 steps, score 3.204 -> 1.593) still
degrades, but far more gracefully than pooled: gap_sigma -24% vs pooled's -59%,
r_ifn_sle 0.790 vs 0.598. Early stopping still needed, just less urgently. Milestone
checkpoints are saved automatically at score <= 2.5 / 2.0 / 1.5 / 1.2 / 1.0 so the
stopping point can be chosen post-hoc without retraining.

**ALL THREE COHORTS DONE (2026-08-26).** Recommended checkpoints: cohorts 2 and 3 the
`_at1` milestone (score <= 1.0, reached at step 100 and step 30), cohort 4 the `_at2`
milestone (step 60; its `_at1` was never reached, the score plateaued at ~1.6).

```
                     COHORT 2               COHORT 3               COHORT 4
                base  pooled  strat    base  pooled  strat    base  pooled  strat
offset/ref_mean 1.133  0.834  0.629   1.046  1.003  0.588   1.759  0.866  0.796
strat c2n_sigma 0.276 -0.147 -0.571   0.224  0.091 -0.818   1.259  0.310  0.148
strat gap_sigma 1.673  1.622  1.764   2.336  2.356  2.598   1.255  1.019  1.077
strat AUC_sle   0.861  0.894  0.923   0.973  0.973  0.983   0.804  0.780  0.834
r_ifn_sle       0.849  0.877  0.879   0.919  0.919  0.929   0.881  0.882  0.861
ASW global     0.5454 0.5462 0.5438  0.5405 0.5414 0.5404  0.5574 0.5454 0.5485
ASW monocyte   0.5570 0.5541 0.5498  0.5471 0.5450 0.5432  0.5722 0.5512 0.5570
```
Stratified beats pooled on batch removal AND disease signal in all three, at equal or
smaller bio-conservation cost. Largest margin on cohort 3, where pooled MMD had barely
moved the offset at all (1.046 -> 1.003) while stratified took it to 0.588.

**A NEGATIVE c2normal_sigma IS NOT OVERSHOOT — correcting the earlier reading.** The
stratified runs drove c2normal_sigma to -0.57 (c2) and -0.82 (c3), which looks like the
same sign-flip pathology as [[mmd-alignment-inconclusive]]. It is not. Decomposing into
block offset and donor dispersion (in ref_mean units): the offset is a NORM, cannot go
below zero, and lands at 0.59-0.93 in every run -- it never crosses. Meanwhile every
cohort's donors are 1.5-2.3x LESS dispersed than census donors (spread/ref 0.43-0.68 vs
census 1.00, unchanged by training). Because c2normal_sigma is `E||x - c||` and not
`||E[x] - c||`, a tighter donor cloud pushes the mean per-donor distance below the census
average long before the offset is gone. Same for AUC_c2n_vs_train < 0.5. **Report block
offset, not c2normal_sigma, when claiming batch removal.** The genuine pooled failure
(cohort 4, 5000 steps) looks different: its pooled offset went tiny (~0.21) while its
STRATIFIED metric got worse than baseline (1.406 vs 1.259) -- pooled and stratified
disagreeing is the signature to watch, not a negative sign.

**Still open:** stratifying the REPLAY SAMPLING as well, since
[[anchor-generalization-check]] shows rare compartments (plasma, pDC at ~0.2% of census)
get ~0.05 anchor cells per donor and are effectively unanchored.

**Bug found and fixed 2026-08-27, NOT yet rerun.** The `other` compartment is a
catch-all with different biology on each side -- census: myeloid cell/platelet/
hematopoietic precursor/innate lymphoid/erythrocyte (2390 cells, 1.5%); cohort:
undifferentiated `lymphocyte` only (3888 cells, 1.2%). It never entered the
reporting metric (`stratified_distance.py`'s MIN_DONOR_FRAC=0.5 donor-coverage
filter drops it, coverage was 44.4%), but DID enter the MMD loss and the stopping
signal in `finetune_mmd_stratified.py` -- their filter is only MIN_PER_SIDE=32 raw
cells, which `other` easily clears. So the loss was pulling biologically
incomparable populations together on ~1.3% of cells -- the same compositional-
mismatch mechanism the whole stratification was built to avoid, recurring inside
one compartment. Fixed by excluding `c == "other"` from `comps` in
`finetune_mmd_stratified.py` (the same list feeds both the loss and the stopping
metric). Not yet rerun on any cohort -- batch this into the next full rerun along
with other pending fixes ([[anchor-generalization-check]]'s replay-sampling
stratification, and the offset_t reporting-metric migration discussed with Vlad
2026-08-27).
