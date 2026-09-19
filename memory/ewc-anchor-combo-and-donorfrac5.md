---
name: ewc-anchor-combo-and-donorfrac5
description: 2026-09-07 EWC+anchor combined loss does not beat plain anchor on any geometry/scIB axis; anchor donor-frac=5% (~80 donors) shows the first small plateau erosion.
metadata: 
  node_type: memory
  type: project
  originSessionId: 9e54ad71-2c38-4bf4-b67f-ff624399e3a4
  modified: 2026-09-07T19:36:51.000Z
---

Follow-up to [[ewc-fisher-decoder-dilution]] and [[anchor-donor-frac-plateau]]. Full numbers
in `logs/sweep_report_table2.log`, script `scenarios/sweep_report.py`.

**EWC + anchor (`build/finetune_ewc.py --lam-ewc 1e8 --lam-anchor 30 --lr 1e-2`)**:
```
                census_CV  block_offset  gap_sigma  AUC_sle_vs_norm  block_asw(mean)
anchor              0.279        0.971       2.523            0.909            0.886
ewc-only(1e9)       0.406        1.098       1.805            0.857            0.872
ewc+anchor          0.282        1.326       2.265            0.878            0.886
```
The anchor term dominates the combined loss (census_CV/block_asw pulled back to
roughly plain-anchor level), but the combo is not BETTER than anchor alone on any
axis, and `block_offset` (raw census-cohort centroid distance) is WORSE than either
arm alone (1.326, vs anchor 0.971 and even vs ewc-only 1.098) -- silhouette-based
metrics (block_asw) don't catch this because they measure separability, not raw
displacement; the two disagree here. Reading: EWC's Fisher-diagonal parameter
penalty doesn't compose cleanly with the anchor's ELBO+distillation objective --
plausibly the same tension flagged in TODO.md for EWC+MMD (fighting the parameter
movement the other loss needs), here between EWC and the anchor/ELBO term instead.
No configuration of EWC (alone or combined) beats plain anchor; `final` (anchor +
stratified MMD) remains the best arm on almost every metric.

**scIB donor-ASW (`conservation.py::scib_bio`) is flat across all 11 arms tested**
(bio ~0.49-0.50, batch/donor ASW ~0.875-0.880) -- NOT a null result, this metric is
already documented in `paper/conservation.py`'s own docstring as blind to this
method's batch effect by construction (it asks whether donors are separable WITHIN
a cell type; our batch effect is a rigid whole-cohort-block displacement). The
informative scIB variant here is `block_asw` (block=census-vs-cohort, silhouette
within compartment) -- used above.

**Anchor donor-frac=5% (~80 donors)**: first real (if small) erosion below the
25-100% plateau -- `gap_sigma` 2.523→2.363, `AUC_sle_vs_norm` 0.909→0.905,
`block_asw` 0.886→0.883. Still far better than EWC or unanchored, and the drop is
modest, but it's a genuine, not-noise-level move (unlike 25/50/75/100%, which were
all indistinguishable). The anchor-donor-budget threshold is therefore somewhere
between 5% (~80) and 25% (~401) donors, not below 5% and not at ~200 as the old
TODO guessed.

**How to apply**: don't pursue EWC+anchor combined-loss experiments further without
a new idea (this data point closes that avenue as currently designed -- see
[[paper-pivot-anchor-mmd-ewc]] for the sequencing agreement that said stop if EWC
alone isn't competitive, which it wasn't, and the combo didn't rescue it either).
If refining the donor-budget number matters for the paper, next step is bisecting
between 5% and 25% (e.g. 10%, 15%) rather than going below 5%.
