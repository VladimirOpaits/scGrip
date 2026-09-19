---
name: stratified-replay-negative-result
description: "Stratifying the census replay by (donor, compartment) so rare types get anchor cells WORKS mechanically but buys nothing measurable -- neutral on every acceptance metric and on ASW, slightly worse on composition-controlled held-out warp, and 2x slower to converge in phase 2. Recommendation: keep floor=0."
metadata: 
  node_type: memory
  type: project
  originSessionId: 4c10a894-795a-443a-9bce-1a5686bd1785
  modified: 2026-08-27T17:37:02.268Z
---

2026-08-27. Tested the fix proposed in [[anchor-generalization-check]]: replay was
stratified by DONOR only (25 cells/donor uniform), leaving plasma/pDC/DC anchored in
~5-15% of the donors that have them. New scheme (`--ref-floor 2` in
`build/finetune_anchored.py`): reserve up to 2 cells per compartment present in the
donor, then fill the rest of the 25-cell budget uniformly. Total replay size unchanged
(40125), `--ref-floor 0` reproduces the old sampling bit-for-bit.

**Mechanically it works.** Replayed cells: plasma 94 -> 365, DC 250 -> 934, pDC 83 ->
313. Donor coverage hits the ceiling set by the data (every donor that HAS a rare cell
now gets one replayed).

**But it buys nothing.** Old-vs-new replay, both in SIGMA anchor mode (see
[[anchor-scale-inconsistency]]):
```
                    COHORT 2        COHORT 3        COHORT 4
                  f0      f2      old     f2      old     f2
c2normal_sigma  -0.056  -0.060   0.224   0.244   1.259   1.147
gap_sigma        1.624   1.634   2.336   2.421   1.255   1.266
AUC_sle_vs_c2n   0.887   0.886   0.973   0.971   0.804   0.801
r_ifn_sle        0.850   0.858   0.919   0.922   0.881   0.883
```
All within noise, signs inconsistent across cohorts. Per-compartment ASW likewise: global
-0.0008, no compartment moves more than +-0.003, INCLUDING the rare ones the fix targets
(DC -0.0020, pDC -0.0021).

**A trap worth remembering.** `anchor_generalization.py` appeared to show a big win:
held-out displacement 1.0860 -> 1.0612, replay/held ratio 1.0304 -> 1.0019 (near-perfect
generalization). It is a COMPOSITION ARTIFACT. Rare cells have high displacement
(1.5-2.0) and the new scheme moves them out of held-out into replay: held-out rare-cell
count drops 1360 -> 166. Reweighting both runs to identical census composition reverses
the sign: nonlinear warp 0.5987 -> 0.6110, i.e. new replay is slightly WORSE. Mechanism:
the 25-cell budget is fixed, so giving cells to rare types takes them from common ones
(CD4 T replayed 15741 -> 11464, -27%), and common types are 90%+ of the data.

**Phase 2 is worse too.** Stratified MMD on cohort 2 from each phase-1 checkpoint:
old replay early-stopped at step 70 (score 0.994), new replay needed step 150 (0.928) --
2x the steps, visibly noisier trajectory. Final acceptance after phase 2 is again a wash
(gap_sigma 1.791 vs 1.809, AUC 0.940 vs 0.944, r_ifn_sle 0.880 vs 0.872) and well inside
the stopping-signal noise (SE 0.11-0.27 sigma, n=1 per arm, no seed replication).

**Recommendation: keep `--ref-floor 0`.** The flag stays in the script (default 2 was set
during the experiment -- CHANGE THE DEFAULT BACK TO 0 or pass it explicitly). The sigma
anchor mode, adopted in the same session, DOES matter and should be kept.

**Separate finding, still open.** DC/pDC/plasma DO enter the stratified MMD loss
(MIN_PER_SIDE=32 on the pooled cohort side lets them in) but never enter the stopping
signal: group B has only 8 donors and fewer than 32 cells of those types, so
`stratified_score()` silently skips them. We optimize compartments we cannot measure and
stop on the five common ones. Not resolved.
