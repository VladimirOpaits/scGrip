---
name: mil-power-benchmark-ground-truth
description: "2026-09-07 spike-in capability test (real SLE cells mixed into real control bags, known ground truth) on AB-MIL over the lupus scVI latent — parked with the AB-MIL pivot, but the numbers are real and reusable if that direction reopens."
metadata: 
  node_type: memory
  type: project
  originSessionId: 9e54ad71-2c38-4bf4-b67f-ff624399e3a4
  modified: 2026-09-07T15:34:54.315Z
---

Before the pivot away from AB-MIL (see [[paper-pivot-anchor-mmd-ewc]]), ran a capability benchmark with exact ground truth (script now deleted with the rest of `scenarios/hard_negative_matching/`, construction after scMILD, doi 10.1101/2025.01.09.632256): real control-donor cells split into paired case/control bags, case bags get a controlled fraction (witness rate w) of real SLE cells swapped in, spike matched to the displaced cells' compartment so composition can't be the shortcut.

**Findings, all on the p2_f0sigma (anchor+MMD) latent, ABMIL(hidden=16,attn=8)**:
- Null (w=0) clean at 0.49–0.51 in every config tried.
- Sham control (spike from OTHER HEALTHY donors, not SLE) stays at ~0.5 — detection is disease-specific, not "foreign donor" leakage.
- Detection threshold ≈15% witness rate on 21 clean donors; below 10%, nothing.
- More cells per donor (500→2000) bought ZERO improvement at any witness rate — the constraint is donor count, not per-donor depth.
- Donor count helps monotonically holding background composition fixed (21→111 background donors: AUC 0.50→0.77 at w=0.20).
- **Control-group purity dominates cohort size, by a lot**: same 21 donors, same w=0.20 — AUC 0.972 with clean controls vs 0.496 when 80% of "controls" secretly carry signal. A 5x larger but 80%-contaminated cohort (111 donors) still only reaches 0.768, below the small clean cohort.
- **Attention does NOT localize the signal even when classification is perfect**: at w=0.35, donor-level AUC = 1.000 but cell-level attention AUROC (spiked vs non-spiked cells) is only 0.71; at w=0.5 attention AUROC actually DROPS to 0.65 while donor AUC stays 1.000. This is an architecture problem (softmax attention pooling), not a data-volume problem — flagged as needing entmax/sparsemax or attention regularization or instance supervision before attention output can be trusted as a biomarker-localization tool.

**How to apply:** if AB-MIL work resumes, don't re-derive these — the threshold and the control-purity>size result directly answer "is it worth collecting more donors" (mostly no, worth vetting existing controls first) and the attention-localization gap is a known, unresolved prerequisite, not something a bigger cohort fixes.
