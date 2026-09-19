---
name: composition-confound-lupus
description: Whole-patient centroid distance partly measures cell-type COMPOSITION, not cell state; unweighted per-compartment stratification removes it and halves the apparent batch.
metadata:
  type: project
---

2026-08-21, lupus/SLE MIL pipeline. The naive patient metric (donor centroid over
ALL cells -> distance to healthy census centroid) is partly a disguised measure of
cell-type composition.

Evidence, 22 healthy cohort2 donors: r(monocyte fraction, IFN score) = -0.814,
r(monocyte fraction, distance) = +0.812, so r(IFN, distance) = -0.546 (their
product — a common cause, not a relation). Controlling for monocyte fraction flips
it to +0.337. Absent in SLE (r(mono, IFN) = -0.06) because the real IFN response
swamps the compositional trace. Root cause: census is ~10% monocytes, cohort2 ~55%
(different protocol).

Fix = per-(donor, compartment) centroids vs the reference centroid OF THE SAME
compartment, then UNWEIGHTED mean over compartments — composition cancels by
construction. `scenarios/stratified_distance.py`.

Blocker on the way: census uses fine CL labels, cohort2 coarse ones; only 7 labels
match literally, 2 survive coverage filters. Both sides must be collapsed onto a
coarse compartment vocabulary (CD4 T / CD8 T / NK / B / monocyte / DC / pDC /
plasma) — then 5 compartments and all 142 donors pass.

Result (anchored λ=30), metric change ONLY, no retraining:
c2normal_sigma 1.049 -> 0.276; AUC_c2n_vs_train 0.822 -> 0.656 (CLAUDE.md target
0.5); r_ifn_normal -0.503 -> +0.464; r_ifn_sle 0.825 -> 0.849.
Cost: AUC_sle_vs_c2n 0.900 -> 0.861 — composition carries REAL disease signal too
(SLE genuinely expands monocytes), and unweighted averaging discards it with the
artifact.

**Why:** much of what looked like residual batch was compositional difference
between corpora, and a metric fix beat a training fix for it.

**How to apply:** report both pooled and stratified acceptance tables; never read
a whole-patient distance as "cell state" without the composition control
(r(monocyte fraction, distance) among healthy donors). Separating real
compositional biology from compositional artifact is exactly the job for AB-MIL
attention, not for either averaging scheme. See [[celltype-compartment-axis]],
[[anchored-finetune-result]].
