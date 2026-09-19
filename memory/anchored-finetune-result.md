---
name: anchored-finetune-result
description: Latent-distillation anchor on census replay preserves reference geometry during scVI fine-tuning; batch removal and geometry turn out to be orthogonal, so MMD is the missing third force.
metadata:
  type: project
---

2026-08-21, lupus/SLE. Unanchored `unfrozen=True` fine-tuning on cohort2_normal
removed the batch but wrecked the reference: census pairwise-distance CV 0.253 ->
0.462 and the cloud inflated 2.2x (ref_mean 1.15 -> 2.69). ELBO stayed healthy
throughout — the damage was geometric, not reconstructive, which is why
CLAUDE.md's `λ·L_ELBO` replay would NOT have caught it (the decoder just
compensates for a drifting latent).

Fix = `build/finetune_anchored.py`: two sources, loss chosen by origin.
cohort2_normal -> plain ELBO (kl_weight fixed at 1.0, no warmup — warmup would
free the latent exactly while we pin it). Census replay (25 cells x 1605 donors)
-> ONLY `lam * ||mu_new - mu_ref||^2`; the decoder is not even executed for them
(`module.inference()` directly, take `qz.loc`). SLE cells appear in no loss term.
Hand-rolled torch loop, not a TrainingPlan — scvi's dataloader does not expose row
indices, which the distillation target needs.

Result: geometry fully restored (CV 0.274, ref_mean 1.09) AND the best disease
signal of any space (gap 2.387σ, AUC_sle_vs_c2n 0.900, r_IFN 0.834 at λ=30).
But batch only ~40% removed (c2normal_sigma 1.05 vs target 0).

Key negative result: λ=30 vs λ=100 changed almost nothing (CV 0.273 vs 0.274,
c2normal 1.05 vs 1.24). No smooth geometry-vs-batch trade curve — so the residual
batch is NOT the anchor's fault.
**RETRACTED 2026-08-27 — see [[anchor-lambda-tradeoff]].** Both arms were RAW-mode and
deep in the over-anchored plateau (= sigma λ~432 and ~1440). A sigma-mode sweep shows a
clean monotone trade-off curve; anchor strength and batch removal are in direct tension,
not orthogonal. The MMD term is still worth having, but the stated reason was wrong. The loss simply contains no term demanding that
healthy donors from different sources coincide; unanchored training faked that job
by inflating the reference until the query fell inside it.

**Why:** it proves the anchor and the alignment force are orthogonal, so MMD is a
missing third term, not an optional extra.

**How to apply:** next step is `lam_align * MMD(z(c2_normal), z(census))` on top of
the anchor. Acceptance table via `scenarios/anchor_report.py <prefix>...`; artifacts
are `models/scvi_anchored_lam{30,100}` and `data/lupus/anchored_lam*_*`. Note much
of the "residual batch" turned out to be composition — see
[[composition-confound-lupus]] before tuning further.
