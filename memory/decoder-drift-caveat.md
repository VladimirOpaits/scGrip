---
name: decoder-drift-caveat
description: "The scVI decoder (and its shared per-gene dispersion) only ever gets gradient from cohort cells, never from census -- so its fidelity on reference cells is untested and possibly degraded. Irrelevant to the current latent-only pipeline; matters only if decoder-based methods (built-in DE, imputation, counterfactual generation) get used later."
metadata: 
  node_type: memory
  type: project
  originSessionId: 4c10a894-795a-443a-9bce-1a5686bd1785
  modified: 2026-08-27T14:49:38.773Z
---

2026-08-27. Vlad's question: if we fine-tune the shared decoder on new cohort
cells while only anchoring census cells in LATENT space (via
`||mu_new - mu_ref||^2`, decoder never executed for census rows -- see
[[anchored-finetune-result]]), doesn't the decoder get worse at reconstructing
old (census) cells?

**Mechanism, confirmed structurally.** scVI's decoder is ONE shared MLP across
all batches -- batch enters as a one-hot concatenated into the input, not as a
separate per-batch subnetwork. Dispersion is `dispersion='gene'` (one value per
gene for the whole model, not per batch). Every training script in this project
(`finetune_anchored.py`, `finetune_mmd_stratified.py`) runs the decoder ONLY on
new/cohort rows -- anchor and MMD terms both stop at `qz.loc`, no decoder call.
So the only gradient the decoder (and the shared per-gene dispersion) ever gets,
post-base-training, comes from cohort ELBO. Nothing corrects it back toward
census. Reconstruction fidelity on census cells after fine-tuning is currently
**not measured anywhere** in the pipeline.

**Why this is not a live bug:** the entire current arc (composition/batch
diagnostics, stratified distance, the planned AB-MIL) reads only encoder output
(`z` / `qz.loc`). The decoder is never consulted downstream. Anchoring only the
latent mean is exactly matched to that scope.

**Where it would bite, if ever:** scvi-tools' built-in differential-expression
(Bayes factor on decoder-reconstructed normalized expression, using the shared
dispersion as the noise model), imputation/counterfactual generation, or any
comparison that treats census-decoded values as a reference norm. DE done
WITHIN one cohort (SLE vs cohort_normal) is unaffected -- both groups are
"new" cells sharing the same decoder self-consistently.

**Cheap check if/when it becomes relevant:** forward-pass-only (no training) NB
reconstruction log-likelihood of a held-out census sample through the phase-1
vs phase-2 decoder -- quantifies the actual fidelity loss instead of guessing.
Deliberately deferred, not queued as a pre-rerun fix -- current objective is
latent-space disease signal only.
