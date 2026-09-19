---
name: clcvae-ewc-precedent
description: "2026-09-08 deep read of the actual CLCVAE paper (Hediyeh-zadeh/Lotfollahi/Theis, ICML CompBio 2022 workshop) — explains why their EWC \"succeeded\" while ours (in build/finetune_ewc.py) underperforms plain anchor; the two setups are not comparable."
metadata: 
  node_type: memory
  type: project
  originSessionId: 9e54ad71-2c38-4bf4-b67f-ff624399e3a4
  modified: 2026-09-08T10:27:45.143Z
---

Full-text read of https://icml-compbio.github.io/2022/papers/WCBICML2022_paper_68.pdf
("Continual single-cell architecture surgery for reference mapping"), triggered by
Vlad asking how to reconcile CLCVAE's reported EWC success with our own negative
result (see [[ewc-fisher-decoder-dilution]], [[ewc-anchor-combo-and-donorfrac5]]).

**Architecture is fundamentally different, not just hyperparameters.** CLCVAE is
built on a Conditional VAE (Sohn et al. 2015) in the scArches "architecture surgery"
style of Lotfollahi et al. 2022 (*Nature Biotechnology* 2022): the encoder has
explicit BATCH-CONDITIONAL weights. Standard (non-CL) architecture surgery for a new
batch works by: freeze the shared non-linear "projection function" (backbone)
entirely, and ADD NEW per-batch conditional weights that absorb whatever's different
about the new data. Quote: "the non-linear projection function and conditional
weights learned on the reference are fixed, and new conditional weights are added to
the model to capture new batches added to the data." CLCVAE's ONLY change: instead
of freezing the backbone outright, let it move a little under an EWC penalty (their
loss is literally `L = L_CVAE(theta) + (lambda/2) * sum_i F_i*(theta_i-theta*_i)^2`
-- same form we implemented). Crucially, the new per-batch conditional weights are
NOT Fisher-penalized (they didn't exist at reference time, Fisher=0 for them, exactly
like our own scArches surgery params) -- they're a DEDICATED, unconstrained capacity
that absorbs the batch effect, separate from what EWC protects.

**Our pipeline has no such dedicated capacity.** `encode_covariates=False` is a
deliberate choice in this project (see CLAUDE.md: frozen surgery under
`encode_covariates=False` provably cannot move `get_latent_representation()` at all
-- no conditional batch slot exists to add new weights to). So in
`build/finetune_ewc.py`, ALL adaptation must flow through the SAME shared
encoder/decoder weights that EWC is simultaneously trying to hold near theta*.
CLCVAE's EWC only has to solve an easy sub-problem (nudge the backbone slightly;
the new conditional weights do the heavy lifting of batch removal). Ours has to
solve a genuinely harder, more adversarial one (the exact same weights must both
remove the batch AND stay pinned near the reference) -- directly consistent with
why our EWC penalty structurally dilutes onto decoder parameters
([[ewc-fisher-decoder-dilution]]) instead of protecting the encoder geometry we
measure: there's no separate low-Fisher parameter set for the useful adaptation to
hide in.

**Even CLCVAE's own reported win is modest, and against different baselines than
ours.** Fig 1d (scIB on the pancreas 9-technology benchmark): CLCVAE is
"marginally better" than plain frozen architecture-surgery on sil_batch, sil_labels,
celltype_F1, graph_connectivity -- and clearly beats naive UNCONSTRAINED
fine-tuning (which catastrophically forgets the reference, their explicit negative
control, Fig 1b showing tech-specific latent clumping). CLCVAE never compares EWC
against anything like a target-specific latent-distillation anchor (our method) --
that comparison doesn't exist in their experiment space. So "EWC succeeded at
CLCVAE" really means: EWC beats no-regularization and is on par with (slightly
better than) full-freeze, in an architecture where most of the adaptation burden
is already handled by a separate parameter set EWC doesn't even touch. That's a
different, easier claim than "EWC beats a purpose-built latent anchor" -- which is
what our sweep tested and where EWC lost.

Their other 2 experiments (melanoma cross-cell-type mapping to an immune reference,
lung mapping to HCLA) are about a DIFFERENT capability entirely -- flagging query
cells/states NOT present in the reference as `unknown` more reliably (uncertainty
score fatter-tailed for genuinely novel cell types) -- not about latent geometry
preservation at all. Not directly relevant to our anchor/MMD geometry claims.

**How to apply**: this is not evidence our EWC implementation is buggy, or that a
different lambda would fix it -- the architectural precondition CLCVAE's success
depends on (a dedicated new-batch parameter slot outside the Fisher-penalized set)
doesn't exist in an `encode_covariates=False` pipeline. If EWC is revisited, the only
faithful reproduction would require adding a genuine conditional/batch-specific
parameter block to the encoder first (i.e. abandoning `encode_covariates=False`) --
a much bigger architecture change than tuning `--lam-ewc`. Otherwise, report the
comparison as-is: a fair baseline under our architecture, and it lost. See
[[paper-pivot-anchor-mmd-ewc]].
