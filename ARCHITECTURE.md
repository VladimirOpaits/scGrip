# scGRIP: reference-mapping architecture

This document defines the replacement for the monolithic `build/finetune_*`
scripts.  It reflects the current design decision, which supersedes the older
post-block-adapter / disease-ELBO note in `TODO.md`.

## Mainline: phase I

The reference scVI model is frozen.  The only trainable component is an update
to the first expression-facing encoder linear layer (`W0`, and optionally its
bias).  The default parameterisation is a full `W0` update.  A gene-wise
affine calibration is kept only as an explicit ablation, not as a hidden
variant or an extra latent channel.

| cells | ELBO | functional anchor | MMD | gradients |
| --- | --- | --- | --- | --- |
| reference replay | no | yes, to frozen base latent | no | adapter only |
| independent healthy reference metric bank | no | no | MMD target only | none |
| query healthy A | yes | no | MMD source | adapter only |
| query healthy B | yes | no | no | adapter only |
| query disease | no | no | no | none |

`replay`, `metric reference`, and `held-out reference` are disjoint.  Disease
cells are encoded only after fitting.  They must not decide a checkpoint.

## Modules

```
scgrip/
  config.py       explicit experiment/data/adapter specifications
  adapters.py     zero-initialised first-layer updates and identity checks
  reference.py    model-gene validation, donor splits, healthy A/B roles
  strata.py       tissue-agnostic stratum geometry and donor distances
  mmd.py          fixed-bandwidth, full MMD² objective
  phase1.py       training/checkpoint/provenance orchestration
  evaluation.py   reconstruction of states and independent metrics
  signatures.py   pre-specified, stratum-aware positive controls
build/
  train_phase1.py
  evaluate_phase1.py
  prepare_reference_bank.py
  prepare_external_cohort.py
```

## Reference bank

The trainer accepts a compact, model-compatible reference bank rather than a
hard-coded full Census file.  It contains raw/count-layer inputs, stable donor
and stratum labels, and enough cells to form disjoint replay, metric, and
held-out roles.  This is what makes a large public reference model practical:
functional anchoring needs paired counts and base latents, not merely 50-D
coordinates.

For a broad reference, the bank must be filtered to the relevant healthy
tissue before training.  A blood metric reference is not a valid MMD target for
skin, colon, or lung cells.

`build/prepare_reference_bank.py` and
`build/prepare_external_cohort.py` make these artifacts.  They use the exact
ordered model panel, retain raw counts at the layer expected by the saved scVI
registry, and retain no implicit gene padding.  A stratum crosswalk is an
explicit `source,target` CSV; missing labels cause an error rather than being
collapsed into `other`.

The default `strict` input policy permits observed integer-like raw counts
only.  A legacy input with documented panel surgery can be passed only via
`audited_imputation`; the run manifest records the exact filled gene(s), fill
value, number of filled entries, and verification that every declared
zero-padded model gene is actually zero.  This is a sensitivity-analysis flag,
not a claim that imputation restores biological variation.

## Strata and metrics

The old PBMC keyword compartment map is replaced by a required `stratum`
column (preferably a Cell Ontology label/ID).  Metrics compare only shared
strata.  The reference centre is a **global all-cell centre per stratum**;
query donor centres are compared to it.  Support is reported per stratum and
never silently turned into a scalar `NaN`.

MMD uses balanced samples within shared strata.  Kernel bandwidths are computed
once from frozen base latents and then held fixed.  Training logs full MMD²
(`Kxx + Kyy - 2Kxy`), so its sign remains interpretable.

## Checkpoints and selection

Every state stores only the adapter update, optimizer/RNG state, model-panel
hash, reference-role provenance, and configuration.  A fresh frozen base is
always reconstructed for evaluation.

Selection is constrained: minimise healthy-B stratified distance only among
states satisfying a predeclared held-out reference-drift budget.  Disease,
IFN, fibrosis, tumour, or treatment signatures are never used for tuning.

## Future phase II (not part of phase I)

Disease-aware fitting is deliberately a separate experimental protocol.  It
will require a disease-aware leave-study-out reference, functional anchors for
both the original reference and phase-I healthy mapping, and a retention loss
that prevents disease-to-healthy collapse.  It is not enabled by a flag in the
phase-I trainer.
