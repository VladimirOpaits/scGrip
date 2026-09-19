---
name: usefulness-validation-plan
description: The plan to validate the retrieval-augmentation as a USEFUL drop-in tool (not just a proven mechanism) — the "benefit vs difficulty" surface + does-no-harm, over a 2D (confound-degree × cohort-size) grid on realistic (non-adversarial) user cohorts.
metadata:
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
---

2026-07-18. Strategic reframe (Vlad's question: "what best validates this as a useful tool that
plugs into a standard pipeline?"). Everything so far ([[scvi-lever-result]], [[retrieval-augmented
-harmony]]) proved the MECHANISM on an ADVERSARIAL construction (fully-confounded core: disease from
one study + normal from another, i.e. diagnosis-batch correlation rho=1). That proves "it works" but
is not how a user's data looks. Validating it as a TOOL needs: (1) realistic difficulty regime, not
worst-case; (2) DOES-NO-HARM guarantee (a plug-in that can degrade results won't be adopted); (3)
field-standard metrics (scIB), not just our internal transfer-AUC; (4) fair baselines (scVI-alone,
scArches-alone, scVI+random-extra-data = our samebatch control).

**THE pivotal experiment (chosen): the "benefit vs difficulty" SURFACE + does-no-harm.**
Difficulty axis = 2D (Vlad's pick): (a) CONFOUND DEGREE rho = correlation between diagnosis and
batch in the user cohort (0 = balanced/mixed across batches, 1 = fully confounded = our current
setup); (b) COHORT SIZE n = number of patient samples (underpowered -> ample). Our existing
experiments are just the (rho=1, n=50) corner; this fills the grid.
Per (rho, n) cell: build a realistic user cohort with that confound+size from REAL paired studies
(studies that HAVE both classes, used as the cohort's batches, with diagnosis assigned to batches at
target correlation rho -- connects to crc_diffuse.py machinery). Then: baseline = standard scVI on
the user cohort alone -> held-out task; augmented = scVI + retrieved references -> held-out task.
Benefit = augmented - baseline. Plot the benefit SURFACE over (rho, n). Expected shape = the product
pitch: benefit RISES with rho and FALLS with n; at rho~0 (balanced data) benefit ~0 but NOT negative
(does-no-harm); and paired-vs-samebatch still isolates smart-retrieval from data-volume.
This answers the only question a user/reviewer asks: "when do I turn this on, and can it hurt me?"

**Supporting layers (in ROI order, on top of the surface):**
- scIB-standard metrics on the integrated latent (batch mixing + bio conservation) -> speak the
  field's language; upgrade our scIB-lite.
- comparison to scArches/scVI-alone -> prove retrieval adds value over what scVI already does.
- DE-recovery readout: "after integration are the known disease-DE genes / disease cell-state still
  there?" -- legible to practitioners (biology not lost), more concrete than AUC.
- multi-study held-pool (the strict biology-generalization stressor, from [[scvi-lever-result]] TODO).

**Tool shape:** thin layer over the LanceDB corpus + scVI reference-mapping:
augment(user_adata, diagnosis_col) -> retrieved reference patients / augmented model. pip-installable
drop-in. (Product direction from [[lancedb-platform]] / [[project-direction]].)

**How to apply / NEXT:** the one NEW piece to build = a core-builder parameterized by (rho, n) that
draws from real paired studies with a tunable diagnosis-batch correlation (generalize setup_core;
rho=1 reproduces current). Everything else reuses crc_scvi.py (scVI ref-mapping, arms, RFF+logreg
readout, samebatch control). Start on CRC. Deliverable = the benefit surface plot + the does-no-harm
claim at rho~0. Confirm the rho-knob construction with Vlad before coding (subtle part: assigning
diagnosis to batches at a target correlation using paired studies as the batches).
