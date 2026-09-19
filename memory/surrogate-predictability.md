---
name: surrogate-predictability
description: "The scVI de-confounding gain is PARTLY PREDICTABLE from a cheap no-scVI linear feature (gap to the biological ceiling) — oos R2=0.33, Spearman 0.56 on CRC; answers Vlad's \"must we always sweep?\" with a partial analytic law + irreducible deep-model noise."
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
  modified: 2026-07-26T14:45:24.406Z
---

2026-07-26. Vlad pushed back on the whole method being brute-force experimental ("нет каких-то
способов посчитать это без перебора?"). Built a SURROGATE test (scripts/surrogate.py): can cheap,
no-scVI quantities PREDICT the expensive scVI de-confounding gain?

**Setup.** Target = GAIN = transfer(quantile) − transfer(noref) per (held × core × rho), scVI-measured
(from crc_statrun parquet). Aggregate over seeds → 66 independent CRC units. Cheap features (linear
logreg on RFF signatures, seconds, NO scVI): lin_ceiling (readability with a diverse un-confounded
corpus), lin_baseline (linear transfer from the confounded core only), **lin_gap = ceiling − baseline**,
emp_rho, confound_sev. Validation = leave-ONE-HELD-out (never split rows randomly → no circularity).

**Result — the law is PARTLY there.**
- lin_gap ↔ scVI GAIN: pearson **+0.66** (p=1.4e-9), spearman +0.61. Single interpretable feature.
- Out-of-group (leave-one-held-out): **R2=+0.33, Spearman=+0.56** predicting an UNSEEN study's gain.
- lin_baseline alone −0.61 (low confounded-baseline separability = more room = bigger gain).
- confound_sev useless (~0); emp_rho weak and ADDING features HURTS oos R2 (0.33→0.29) → no nonlinear
  signal for an RF to grab. Simple-first vindicated; RF would overfit 66 grouped units.

**Interpretation (the honest answer to "must we always sweep?").** THREE-tier: (1) provable/no-sweep =
identifiability/overlap (off-diagonal cell empty → no method works, ex ante); (2) predictable = this
surrogate — cheap geometry pre-screens & RANKS cohorts by benefit (Spearman 0.56) so we skip near-zero-
gap ones and spend GPU on high-gap ones; (3) irreducible = the remaining ~2/3 variance is the deep
scVI + finite-sample noise, no closed form (generalization bounds too loose). So sweeping shrinks to
"predict + spot-check", it doesn't vanish. Also quantifies the project thesis "effect tracks READABILITY
/ gap to ceiling" ([[enrichment-heldout-result]]) as a validated PREDICTOR, not just a description.
The sweeping is largely a symptom of the black-box deep downstream — a linear de-confounder would be
analytically predictable (see [[explainability-geometry]] w-rotation).

Figures/code: scripts/surrogate.py, figures/nearopp/surrogate_scatter.png. Next if pursued: replicate
on lung (needs lung signatures pulled), add MMD-coverage feature, or build the overlap ex-ante screen.
Also open: AUC is a coarse/saturating target — swap to margin / batch-vs-bio η² / within-diagnosis MMD
for a more mechanistic readout.
