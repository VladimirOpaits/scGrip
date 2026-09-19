# Experimental roadmap after phase I

This is a decision log, not a menu of switches to turn on together.  Each item
should be tested with repeated healthy-donor A/B splits and a fixed external
evaluation protocol.

## Priority now

1. **Establish the primary W0 method.**  Use full `W0` adaptation, functional
   replay anchoring, and fixed-bandwidth healthy-A MMD.  Sweep a small,
   predeclared MMD grid (for example 5, 20, 50) across several A/B seeds.  A
   model that improves A while consistently worsening B is rejected as
   overfitting even if a disease readout happens to improve.
2. **Make the reference valid before tuning.**  Use a healthy, tissue-matched
   bank and explicit stratum vocabulary.  The biggest apparent win from MMD can
   otherwise be a blood-vs-tissue mismatch disguised as batch correction.
3. **Quantify sequencing-depth sensitivity.**  Report donor-level latent
   distance after regressing or stratifying on log library size and technical
   run/lane.  Run a depth-matched subset as a sensitivity analysis.  This
   diagnoses whether `W0` is correcting chemistry/depth or erasing biological
   structure; it is not a training target in phase I.
4. **Use external held-out studies.**  The central evidence is a cohort unseen
   by the frozen reference model.  A CellxGene/Census-trained model cannot be
   evaluated honestly on cells already used to train it.

## Clean ablations

| Ablation | What it answers | What it must not become |
| --- | --- | --- |
| `affine_w0` | Can per-gene calibration explain most transfer? | A hidden disease-conditioned normalizer |
| row-normalised/DoRA-style `W0` | Does preserving first-layer direction while learning magnitude reduce reference drift? | A default before the full-W0 baseline is understood |
| stronger `W0` L2 / spectral constraint | Does it improve out-of-reference behaviour beyond functional replay anchoring? | A substitute for held-out reference monitoring |
| donor-balanced MMD draws | Does equalising donors within a stratum avoid domination by abundant reference donors? | A change made after looking at disease signatures |
| linear-time/RFF MMD | Can larger samples be used without quadratic cost? | A different target distribution or a moving kernel scale |

The current implementation already keeps MMD samples small and fixed-bandwidth;
linear/RFF MMD is an engineering follow-up only if profiling shows it dominates
runtime.

## Explicitly deferred phase II

After a phase-I healthy mapping is fixed, a separate disease-aware experiment
could train a tightly constrained input-layer update on disease cells with
anchors to both the original reference and the phase-I healthy map.  It would
need a disease-aware leave-study-out reference and an explicit anti-collapse
objective.  It must not reuse IFN/fibrosis/tumour signatures for model
selection.  This is deliberately absent from the phase-I CLI.

## Multi-cohort lupus test

Combining two lupus cohorts is valuable only after patient identity is
deduplicated across studies.  The correct split is by patient/study before any
healthy A/B split; duplicated disease donors must not appear on both sides.
The first implementation should build a merged query artifact with a stable
cross-study patient ID and retain original study/run metadata for evaluation.
