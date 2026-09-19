---
name: batch-carrier-map
description: "Research plan surfaced building scEris per-compartment bases — distinguish cell types carrying REAL technical batch from those that merely CORRELATE with batch, via per-compartment cross-batch MMD at fixed biology + a per-compartment separability ceiling; buildable in sceris/stats.py."
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
  modified: 2026-07-29T22:35:45.377Z
---

2026-07-30. Built the first universal per-compartment PCA library (11 CL compartments, 60-dim, 3000 HVG,
~316k cells across lung/CRC/brain/blood, in data/universal_bases/*.npz via cloud). That surfaced the
measurement plan for the compartment axis — two per-compartment maps, project a labeled cohort into each
compartment basis (apply_basis) -> per-donor RFF signatures, then:

1. **Separability ceiling** = within-study diagnosis AUC per compartment -> WHERE the biological signal
   lives (within-study CV, not leave-study-out, to avoid batch leakage).
2. **Batch-carrier** = cross-batch/study MMD among SAME-diagnosis patients (biology fixed) per
   compartment -> distinguishes REAL technical batch (separation persists at fixed biology) from
   SPURIOUS (biology correlated-with-batch; vanishes once biology controlled). Negative control = a
   biologically-stable reference cell type as a pure-batch ruler.

Refines de-confounding: dilute compartments carrying REAL batch, spare compartments carrying biology
(avoids over-correction — the classic integration failure). Bounded by identifiability: where batch≡
biology in a compartment (empty off-diagonal), real-vs-spurious is unidentifiable.

**Literature (batch IS cell-type-specific):** RNA content; dissociation sensitivity (van den Brink 2017,
immediate-early/HSP artifacts); ambient RNA from high-RNA neighbors (SoupX, DecontX); platform interaction
(nuclei-vs-cell, Smart-seq2-vs-10x, v2-vs-v3); stress. scIB benchmark (Luecken 2022) shows type-specific
integration difficulty. Our per-type RFF can re-check this on our data and validate the original
"technology dominates biology ~13×" ([[first-validation-result]]) as concentrated in high-RNA/platform-
sensitive types, not uniform.

Code: scenarios/cell_type_distance (project CRC tumor/normal cohort into compartment bases) + sceris/stats.
Also logged repo-local OPEN_QUESTIONS #7 (this) and #8 (HVG inner-join gene-loss + corpus-versioning of
the HVG panel: join="inner" drops non-shared genes before HVG; panel depends on corpus so bases are
versioned; fix = fixed reference gene panel + batch-aware HVG). Anchors [[celltype-compartment-axis]],
[[unified-deconfound-method]] (real-vs-spurious ties to its overlap/identifiability caveat).
