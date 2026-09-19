---
name: celltype-compartment-axis
description: "Research direction surfaced by BUILDING the scEris tool — disease separability is cell-type-specific, so signatures/overlap should be per-compartment; plus Vlad's universal tissue-agnostic per-type PCA idea and the overlap-diagnostic p>>n trap."
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
  modified: 2026-07-28T19:59:40.007Z
---

2026-07-28. Building the scEris retrieval tool surfaced research directions invisible in pure
research (Vlad's point: implementation forces confronting the real data flow). Recorded in repo
OPEN_QUESTIONS.md #4/#5.

**Compartment axis.** Whole-patient RFF signature POOLS all cells -> compartment-localized disease
signal (disease in ~5% of cells) is DILUTED by uninformative types -> FALSE NEGATIVE. So
overlap/separability must be a per-compartment MAP, not a global yes/no; the tool's job = find the
compartment where signal is present AND de-confoundable. Architecture: per-cell-type PCA -> per-type
RFF block -> CONCAT into patient signature (distance = sum of per-compartment MMDs); gotchas =
missing blocks (mask / compare on shared compartments), min-cells-per-block (noisy kernel mean),
block weights, multiple-testing. No annotations -> unsupervised clusters (named types optional).

**Vlad's universal tissue-agnostic per-type PCA.** Library of cell-type bases reusable ACROSS tissues
("cell type as universal anchor" paradigm). Solves rare cells cheaply (pool the type across tissues),
keeps batch (PCA on raw counts, unintegrated), shippable reference artifact. TENSION: universality vs
resolution — universal basis's top axes get eaten by tissue/batch, disease drifts to minor components;
SOFTENED because RFF uses ALL kept components so signal isn't lost, just not dominant -> universal
bases just need larger n_pca. Test: build a universal T-cell basis, project CRC+blood T cells, see if/
where disease separability survives.

**Foundation/reference embeddings — filter for us.** Batch-INTEGRATED refs (scVI/scANVI/Census-scVI,
HLCA-scANVI) REMOVED the batch axis we de-confound -> useless for us (nothing to fix; this is why
research chose own-PCA-on-raw-counts). Only unintegrated/batch-retaining spaces qualify; test via
batch-predictability (batch predictable in the embedding -> axis preserved). Geneformer 1024-dim +
GPU-costly and we put RFF on top anyway -> own PCA(50) is faster and enough. Keep own PCA as baseline;
pretrained embeddings = pluggable alt "space" gated by the batch-preservation check.

**Overlap diagnostic p>>n trap.** In-sample propensity/logreg in p>>n separates perfectly ->
propensity collapses to 0/1 -> false "no overlap" (overfitting artifact). Fixes: cross-fitting (honest
OOS propensity), reduce to intrinsic dim (PCA-50, or cell-level where n>>p), regularization; prefer
distance/density/MMD-witness over a learned classifier. Pseudo-replication: cells within a patient are
correlated -> effective n ~ #patients not #cells -> conclude at patient level. Permutation-null:
shuffle Y, a trustworthy overlap MUST report full overlap. Overlap = tier-1 of the three-tier
(provable/overlap -> predictable/[[surrogate-predictability]] -> irreducible/deep).

Anchors [[unified-deconfound-method]] (overlap = its caveat 3), [[scvi-lever-result]]. Tool status:
core cow/pig retrieval + CLI + stats validated on CRC/blood; this is the next research iteration,
NOT yet built. Next concrete build step (in progress): per-cell-type signature pipeline (Basis PCA +
RFF/RBFSampler + per-patient aggregate) so the compartment experiment is even runnable.
