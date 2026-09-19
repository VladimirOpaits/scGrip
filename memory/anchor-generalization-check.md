---
name: anchor-generalization-check
description: "The census latent anchor generalizes to never-replayed cells (no memorization), but it restrains SHAPE not position; warp concentrates in rare types and, among common types, in monocytes -- and the MMD step adds warp to monocytes 2-4x more than to any other common type."
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
  modified: 2026-08-26T19:36:24.954Z
---

2026-08-26. Vlad's concern: the anchor in `build/finetune_anchored.py` is a soft L2
penalty on a FIXED replay subsample (25 cells/donor, 40125 of 160500), minibatched from
that same fixed set for 75-99 epochs. Could the encoder satisfy it by memorizing those
cells while the map drifts elsewhere? `census_pairwise_CV` does not settle it -- donor
centroids average ~25 anchored + ~75 free cells, so free cells could move
compensatingly and leave it flat.

Test (`scenarios/anchor_generalization.py`): displacement of replay vs held-out census
cells, in three nested views -- raw, residual after the best similarity transform fitted
ON REPLAY cells, residual after the best affine map. Removing similarity/affine matters
because the latent demonstrably rescaled; without it you re-measure that expansion and
call it damage (the same mistake made earlier in the session, see
[[unsupervised-arc-closed]]). Reference latent had to be encoded fresh
(`scenarios/encode_base_census.py` -> `data/lupus/base_census_latent.npy`):
`scarches_train_latent.npy` is NOT exactly the base output (mean coord discrepancy 0.017
~= 0.077 in norm units vs a ~1.01 signal); the fresh encode reproduces the stored
`mu_ref` to 5.9e-06.

**Answer: no memorization.** held-out/replay displacement ratio = 1.04 raw, 1.10 after
similarity, 1.12 after affine. Cells the anchor never touched moved essentially as much
as the ones it pinned.

**More important, and it corrects a wrong mental model: the anchor does not hold cells
in place at all.** The replay cells themselves moved 1.013 on average, against a typical
latent norm of 3.76 (~27%). About half of that is a global similarity transform
(1.013 -> 0.526); ~0.48 of nonlinear warp remains AT THE ANCHORED CELLS. This was
visible in the training logs all along (val_anchor plateaus ~1.05 = rms drift ~1.02
sigma/dim). So "anchored" means the SHAPE of the reference cloud is preserved
(census_pairwise_CV 0.273 vs base 0.253), not that coordinates are frozen. Conclusions
built on it are unaffected (everything is computed within one model), but do not describe
it as pinning cells.

**Where the warp lives -- rarity dominates, monocytes are the anomaly.** Affine residual
for held-out cells, by compartment (share of census in brackets): plasma 0.938 (0.24%),
DC 0.694 (0.67%), pDC 0.685 (0.20%), other 0.676 (1.5%), **monocyte 0.615 (10.3%)**,
other T 0.555, B 0.526, NK 0.520, CD4 T 0.517, CD8 T 0.517. Main axis is rarity -- 25
cells/donor sampled uniformly gives a 0.2%-abundant type ~0.05 anchor cells per donor.
The declared monocyte prediction was therefore only partly right: monocytes are beaten by
the rare types, but among COMMON types they are a clear outlier (0.615 vs 0.517-0.555)
despite being more abundant than B or other T.

**Per donor** (all donors have exactly 100 cells, so cell count cannot confound):
spearman(monocyte fraction, held-out warp) = **+0.760**; but also +0.648 against warp at
the donor's own REPLAY cells, and +0.435 against the held/replay ratio. Two stacked
effects: monocyte-rich donors are warped more everywhere, and the anchor additionally
generalizes slightly worse for them. Donor-to-donor spread of held-out warp is moderate,
p95/p5 = 1.32x.

**Phase 1 vs phase 2 -- direct confirmation of the compositional mechanism in
[[stratified-mmd-design]].** Added warp from the 10 MMD steps, by compartment:
monocyte 0.615 -> 0.653 (**+6.2%**), B +2.7%, NK +2.6%, other T +2.4%, CD4 T +2.2%,
CD8 T +1.4%. MMD adds 2.3-4.4x more warp to monocytes than to any other common type --
exactly what the pooled-MMD compositional argument predicts (cohort pool ~55% monocyte
vs census ~10%, unmatchable by translation, so the gradient bends monocyte geometry).
Note also that overall displacement from base DECREASED over those steps (1.013 ->
0.870) while nonlinear warp rose: MMD is not moving everything, it is locally bending,
selectively where the composition mismatch is largest.

**Consequence:** strengthens the case for stratified MMD ([[stratified-mmd-design]]) and
suggests a second fix worth considering -- stratify the REPLAY SAMPLING too, so rare
compartments get anchor cells proportional to need rather than to abundance. Currently
plasma/pDC are effectively unanchored.
