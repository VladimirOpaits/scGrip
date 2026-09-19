---
name: ewc-fisher-decoder-dilution
description: "2026-09-07 EWC baseline (build/finetune_ewc.py) consistently underperforms anchor across 4 orders of magnitude of lam-ewc, likely because the Fisher diagonal is dominated by decoder params, not the encoder params that determine the measured latent geometry."
metadata: 
  node_type: memory
  type: project
  originSessionId: 9e54ad71-2c38-4bf4-b67f-ff624399e3a4
  modified: 2026-09-07T18:06:58.393Z
---

Ran `build/finetune_ewc.py --lam-ewc {1e7,1e8,1e9,1e10} --lr 1e-2` (EWC-only, --lam-anchor 0)
on cohort 2, encoded + measured with `paper/geometry.py`'s `geometry()`. All four give
IDENTICAL numbers to 3 decimals:

```
                                         census_CV  block_offset  gap_sigma  AUC_sle_vs_norm
anchor (distillation only, 100% donors)      0.279         0.971      2.523            0.909
final (anchor + stratified MMD)              0.269         0.544      2.757            0.947
ewc_lam7 / lam8 / lam9 / lam10                0.406         1.098      1.80-1.81         0.857-0.858
```

EWC is worse than plain anchor on every axis (reference geometry MORE distorted than
anchor: 0.406 vs 0.279 census_CV; batch offset bigger, not smaller: 1.098 vs 0.971;
disease signal weaker: gap_sigma 1.80 vs 2.52, AUC 0.857 vs 0.909) and totally
insensitive to lam-ewc across 4 orders of magnitude.

**Likely mechanism, visible directly in the Fisher printout** (same top-5 at every
lam value, since Fisher is estimated once before training and doesn't depend on
lam): the four highest-Fisher parameters are ALL in `decoder.px_decoder.fc_layers`
(Fisher ~22-28), with the first ENCODER parameter (`z_encoder.mean_encoder.bias`)
only 5th, at Fisher ~16, already lower than any decoder layer. EWC's penalty
`sum_i F_i*(theta_i-theta*_i)^2` sums over ALL parameters, so with decoder Fisher
systematically larger, the penalty budget is spent mostly protecting DECODER
reconstruction fidelity, not the ENCODER output (`get_latent_representation`,
i.e. `qz.loc`) that `census_CV`/`block_offset`/etc. actually measure. Anchor, by
contrast, penalizes `qz.loc` drift directly and only. This is a structural reason
EWC-as-implemented (global, uniform-target Fisher penalty) is a weaker baseline for
THIS task than a target-specific latent anchor — not just a mistuned lambda; raising
lam by 1000x changed nothing because the encoder/decoder penalty SPLIT didn't change,
only the overall magnitude did (and Adam's per-parameter normalization plausibly also
saturates that magnitude increase, independently).

**Not yet tried, would directly test the mechanism**: restrict the EWC penalty sum to
`z_encoder.*` parameters only (mirror what anchor targets), and/or check whether Adam
itself is saturating the effective step size once lam*grad is large (try SGD or a much
smaller lr to rule this in/out independently of the Fisher-dilution story).

**How to apply**: don't re-run the same lam sweep expecting a different outcome above
1e7 — the flatness is confirmed, not noise. If EWC is worth pursuing further as a
baseline for the paper, the next real experiment is encoder-only Fisher penalty, not
more lambda values. If it's still not competitive there, EWC-as-a-serious-baseline is
probably a negative result worth reporting as-is (anchor's task-specific target beats
EWC's global parameter regularization) rather than something to keep tuning.
See [[paper-pivot-anchor-mmd-ewc]].
