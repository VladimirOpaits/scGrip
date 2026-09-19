---
name: anchor-lambda-tradeoff
description: "RETRACTS the documented 'anchor and batch removal are orthogonal' claim. The original lam sweep tested only lam=30 vs lam=100 in RAW mode -- both deep in the over-anchored plateau. A sigma-mode sweep reveals a smooth monotone trade-off curve; batch removal saturates around lam=10 and the optimum sits far weaker than anything previously tested."
metadata: 
  node_type: memory
  type: project
  originSessionId: 4c10a894-795a-443a-9bce-1a5686bd1785
  modified: 2026-08-27T18:18:42.896Z
---

2026-08-27. CLAUDE.md states: "lam=100 vs lam=30 почти не отличаются -> якорь и снятие
батча ортогональны, остаточный батч упирается не в lam." **That conclusion is wrong.**
It was measured at two points that are both on the flat part of the curve.

Per [[anchor-scale-inconsistency]], the original sweep ran in RAW anchor mode, where
lam=30 and lam=100 correspond to SIGMA lam~432 and lam~1440 -- i.e. both massively
over-anchored. Re-running the sweep in sigma mode on cohort 2 (all `--ref-floor 0`,
phase 1 only) gives a clean monotone trade-off:

```
                     lam=3   lam=10  lam=30  lam=100 |  RAW lam=30
census_pairwise_CV   0.298   0.285   0.279   0.275   |   0.273     (target ~0.27)
c2normal_sigma       0.343   0.330   0.496   0.745   |   1.049     (target ~0)
AUC_c2n_vs_train     0.643   0.635   0.686   0.756   |   0.822     (target 0.5)
gap_sigma            2.205   2.392   2.523   2.503   |   2.387
AUC_sle_vs_c2n       0.903   0.905   0.909   0.908   |   0.900
r_ifn_cohort2        0.829   0.838   0.842   0.839   |   0.834
```

Geometry degrades monotonically as the anchor weakens (CV 0.275 -> 0.298); batch leaves
monotonically as it weakens (c2normal_sigma 0.745 -> 0.330). They are in direct tension,
not orthogonal. The old sweep simply could not see it.

**Where the optimum is.** lam=10 is best on batch (c2normal_sigma 0.330, AUC_c2n_vs_train
0.635, both sweep minima) at a small geometry cost. lam=30 is best on disease (gap_sigma
2.523, AUC_sle_vs_c2n 0.909, both sweep maxima). lam=3 is OVER-relaxed: batch stops
improving (0.343, worse than lam=10) while geometry visibly slips (0.298) and disease
signal drops (gap 2.205). So batch removal saturates near lam=10.

All four points pass the guards (AUC_sle_vs_c2n >= 0.87, r_ifn >= 0.80), so lam=10 vs
lam=30 is a deliberate batch-vs-signal trade, not an admissibility question. Standing
choice: **lam=30 in sigma mode** -- phase 2's MMD closes much of the remaining batch gap
anyway, while gap_sigma and AUC are what the headline table reports.

**Also note sigma lam=30 beats RAW lam=30 outright** (the only difference between
`anchored_lam30` and `anchored_lam30_f0sigma` is the anchor formula): batch halved
(c2normal_sigma 1.049 -> 0.496), gap 2.387 -> 2.523, AUC 0.900 -> 0.909, r_ifn 0.834 ->
0.842, for census_pairwise_CV 0.273 -> 0.279. The historical branch was over-anchored.

**SETTLED 2026-08-27 by running phase 2 from BOTH anchors.** The lam=10-vs-lam=30
question was decided with data, not argument: stratified MMD (lam_align=10) was run from
`anchored_lam10sigma` and from `anchored_lam30_f0sigma`, everything else identical.

```
                     pooled                stratified
                lam=10  lam=30   d      lam=10  lam=30   d
gap_sigma       2.609   2.757  +0.148   1.672   1.791  +0.119
AUC_sle_vs_c2n  0.939   0.947  +0.008   0.933   0.940  +0.007
c2normal_sigma -0.491  -0.577          -0.633  -0.662
AUC_c2n_vs_train 0.361  0.330           0.267   0.258
census_pairwise_CV 0.282 0.269
```
lam=10's phase-1 batch advantage (0.330 vs 0.496) is ERASED by MMD -- both end up
overshot negative. lam=30's phase-1 disease advantage (+0.131) SURVIVES and grows
(+0.148 pooled, +0.119 stratified), above the 0.05 noise threshold declared in advance.
The mechanism is exactly as argued: batch has a second instrument downstream, signal does
not, so conceding batch is cheap and conceding signal is not. **lam=30 is now justified by
measurement, not inheritance.**

**IMPORTANT SIDE FINDING -- the stopping signal is blind to what matters.** Both runs
early-stopped at the SAME step (70) with essentially the same score (0.993 vs 0.994), and
their whole score trajectories agree to the third decimal. Yet the resulting models differ
by 0.148 in gap_sigma. The stopping statistic measures only residual batch on group B; it
says when the batch is gone and nothing about how much disease signal survived. Never read
an identical stopping score as identical models, and when choosing among milestone
checkpoints, check the disease axis separately.

**For the paper:** the old lam=30/lam=100 table is internally consistent and may be shown,
but its conclusion must be restated as "insensitive within the strong-anchor regime", NOT
"lam does not matter", and the sentence "we use lam=30 for the rest of the analysis" is
false as written -- the rest of the analysis uses a ~14x weaker anchor with the same
nominal lam.

Both trainers now print their full config, so lam/anchor_scale are recoverable from logs.
Related: [[anchored-finetune-result]] (carries the retracted claim),
[[stratified-replay-negative-result]].
