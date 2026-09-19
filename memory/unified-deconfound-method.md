---
name: unified-deconfound-method
description: The unified method formulation Vlad and Claude converged on — group-balanced retrieval de-confounding via ONE RFF-MMD metric used in TWO directions (far same-diagnosis = invariance; near opposite-diagnosis = hard-negative boundary), anchored in causal matching + contrastive learning.
metadata:
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
  modified: 2026-07-26T10:40:30.265Z
---

2026-07-24. After a long strategic conversation (SOTA positioning for the paper Guillem asked to
frame well), the method crystallized into ONE clean statement. This supersedes the old "smart
selection beats random" framing (weak/retracted, see [[retrieval-augmented-harmony]]) and the
"two separate levers" framing.

**THE UNIFIED METHOD — group-balanced retrieval de-confounding.**
Frame the confound as a spurious-correlation / group-robustness problem (cow-on-beach / Waterbirds):
Y = diagnosis, A = batch(study/assay/…); the (Y × A) group matrix is confounded when only the
diagonal is populated (disease≡its study). De-confound = retrieve references from a corpus that
POPULATE THE OFF-DIAGONAL cells, breaking diagnosis ⊥ batch. This is GroupDRO / group-balancing /
DFR-style, not ad-hoc "diversity."

**ONE metric (RFF-MMD patient signature), TWO opposite directions — Vlad's synthesis:**
- **far, SAME diagnosis** (RFF-MMD max) = "cow on beach": disease patients from other batches ->
  INVARIANCE (disease looks the same across batches). = diverse positives / augmentation.
  THIS is the direction we validated (4 diseases, samebatch control, cluster-bootstrap CIs).
- **near, OPPOSITE diagnosis** (RFF-MMD min) = "pig on grass": a control that SHARES the cases'
  batch -> DISCRIMINATION, a clean boundary at fixed batch. = **HARD NEGATIVE MINING** (metric/
  contrastive learning: near-in-embedding + opposite-label = most informative for the boundary) =
  **matched case-control** (causal inference).

**2026-07-26 UPDATE — pig-on-grass is NOW VALIDATED (no longer just future work).** Added `nearopp`
arm to crc_statrun.py (retrieve K normals nearest the core DISEASE centroid + K disease nearest the
core NORMAL centroid; MIN dist, label flipped; leak-free, balanced 2K). Ran on cloud, CRC (11 held ×3
cores) + lung/IPF (5 held ×2 cores), rho=1. Result: pig **de-confounds as strongly as cow** — DECONF
(pig−samebatch) transfer +0.138 [+0.018,+0.382] CRC, +0.156 [+0.062,+0.233] lung, positive in 100% of
held — AND **dilutes batch HARDER**: lowest batch_eta2 of every arm in BOTH tissues (CRC 0.103, lung
0.114 vs ~0.16 cow/random), batch_mix up vs noref in 100% held. No over-correction (cell-purity flat
0.92-0.95). Character: pig kills batch harder but transfers slightly BELOW cow (0.79 vs 0.84 CRC) — no
diverse positives, just boundary sharpening. Confirms Vlad's synthesis empirically. Figures (patient +
cell UMAP) + results page in repo: figures/nearopp/, docs/pig_on_grass.html, scripts/figures/. So the
"B direction" can now be a validated part of the paper, not just general form.

**Key correction (Claude was wrong earlier):** near-matching "pulls the batch" is a FEATURE, not a
bug. In matched design you WANT case+control to share the confounder so it cancels and the residual
difference = diagnosis. Vlad's point: since batch and biology aren't fully separable, small RFF-MMD
distance ≈ batch similarity, so a super-close opposite-diagnosis pair localizes the disease boundary
precisely (batch held ~fixed). The earlier "circular/pulls batch" caveat was confused.

**Dual theoretical anchor (great for positioning across audiences):**
causal inference (matching / covariate balancing / positivity-overlap) FOR the stats/bio reviewer,
AND contrastive/metric learning (hard-negative mining, invariance) FOR the ML reviewer.

**Honest caveats to state in the paper:**
1. Near-matching controls BATCH but not other patient covariates (age/sex/PMI). "batch⊄bio not
   separable" -> residual = diagnosis PLUS unmatched covariates; pure-disease needs matching the
   full covariate vector -> the covariate-balancing generalization. (Patient itself is a confound
   source; neurodegeneration = age≡diagnosis, often UNBREAKABLE -> may re-explain the brain null:
   we de-confounded study, not the dominant age/PMI axis.)
2. Near opposite-diagnosis pairs are RARE for strongly compositional disease (disease shifts the
   embedding, so case≠control aren't close). When they exist they're boundary/mild cases — most
   informative but availability tracks disease strength.
3. **Overlap / identifiability (causal inference):** you can only fill an off-diagonal cell the
   corpus actually has. Single-arm studies (a COVID study is all-COVID) mean some cells are empty ->
   the method balances the matrix only as far as corpus OVERLAP allows. An overlap-diagnostic ("which
   confounds are breakable in your corpus") is a valuable standalone feature.

**Paper scope (converged, for Guillem):** ONE focused paper = group-balanced retrieval de-confounding,
FAR-same-diagnosis direction validated on 4 diseases (CRC/COVID/IPF/dementia) on Harmony+scVI with the
samebatch control + CIs, effect tracks disease READABILITY/headroom not tissue (CRC clean tight-CI >>
blood/lung positive-heterogeneous >> brain null). The near-opposite-diagnosis (hard-negative/matching)
direction + covariate-balancing + overlap-diagnostic + frozen-foundation-encoder+DFR + skewed-corpus
test = the general form / future work that shows depth WITHOUT diluting the validated core. Do NOT
claim the unvalidated B-direction as a result. Guillem (2026-07) asked: viable on OTHER cancers (we
have 1 cancer CRC + non-cancers; a 2nd cancer = lung adenocarcinoma is cheap, HLCA already on disk),
proper SOTA analysis, a meeting before end of summer. See [[lung-ipf-feasibility]], [[scvi-lever-result]].
