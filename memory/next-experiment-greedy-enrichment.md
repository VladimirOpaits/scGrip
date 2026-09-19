---
name: next-experiment-greedy-enrichment
description: "Planned v1 of the central experiment: greedy farthest-point cohort enrichment on pbmcsca, with the circularity guard and chosen metrics"
metadata: 
  node_type: memory
  type: project
  originSessionId: 506da36b-625b-49b9-b619-f9185b7b5811
---

Decided 2026-06-16. GSE295527 (modern 3' PBMC) is DONE as a generalization check: marginal cosine heatmap shows clean technology blocks (Flex far from 10x at ~0.04; GEM-X vs NextGEM close ~0.01–0.02) with donor F1/F5 collapsed to ~0.00 within each technology — confirms the pbmcsca finding on modern platforms. No stratification needed there (both donors healthy, no disease contrast, structure obvious). Absolute distances are small only because all three platforms are 3'-droplet-family (narrow tech axis); the *structure* is the confirmation, not the magnitude.

**Next step = the central experiment.** Anatomy: one biological category (fixed metadata) → small "original cohort" deliberately from ONE technology/lab (the batch confound) → pool of same-category samples across many technologies → greedy **farthest-point** selection to enrich → baselines (random, same-technology, nearest-point) → measure how batch effect dilutes vs N added samples.

Locked choices (updated 2026-06-21):
- **Selection ruler = RFF-MMD** (store.rff_signatures → selection.l2_distance_matrix), not centroid. Centroid still works as a coarse first-moment proxy but overstates tech-dominance and collapses sub-platform structure — see [[mmd-distribution-ruler]]. Farthest-point over patient-level MMD signatures.
- **Three datasets, three roles** (NEW — onek1k added this session):
  - **onek1k** (982 donors, all `10x 3' v2` = ONE batch; subsampled 40×1000 in geneformer.ipynb; ingested dataset="onek1k") = the realistic **homogeneous cohort showcase** + scIB benchmark substrate. onek1k.ipynb runs the enrichment demo (greedy from onek1k cohort over pbmcsca+gse pool).
  - **pbmcsca** (2 donors × 7 techs, has CellType) = label-free **donor-recovery** testbed AND scIB.
  - **gse295527** (2 donors × 3 platforms, NO CellType) = secondary label-free testbed; EXCLUDED from scIB (no labels).
- **Dual-track outcome (core decision):** the method (selection+ruler) must work **label-free**; cell types are only for the benchmark.
  - **PRIMARY = label-free donor-recovery** (`holdout_donor_generalization`): predict donor across a HELD-OUT technology; arms diverse/random/narrow at matched N; success = diverse>random>narrow. Needs donors shared across technologies → runs on **pbmcsca/gse, NOT onek1k** (onek1k is single-batch, its donors don't span techs).
  - **SECONDARY = scIB** (iLISI/kBET/batchASW + ARI/NMI/cLISI/cell-typeASW) on **onek1k+pbmcsca** = field-standard proof-benchmark. Requires harmonizing onek1k's 29 fine CELLxGENE ontology types → ~10 coarse pbmcsca types. Bio-conservation is the load-bearing side (Harmony removes batch regardless).

**Circularity guard (critical):** selection uses the Geneformer embedding, so evaluation must NOT be on the same embedding or it's self-fulfilling (we already proved distance≈technology). Plan: keep a **batch-predictability classifier** (predict technology/lab → AUC should fall toward chance) as the independent load-bearing metric, and/or run scIB on a *different* representation (PCA/scVI). scIB caveat: it's designed to evaluate integration (post-correction), whereas we do data selection pre-correction — so use it as "does diverse enrichment make subsequent Harmony/scVI integration more honest (bio easier to recover at same batch-removal)." The real non-circular payoff is downstream generalization (train on enriched cohort, test on held-out technology), deferred as heavier.

Relates to [[first-validation-result]], [[sequence-length-depth-axis]], [[project-direction]]. Greedy farthest-point sampling note + stratified-distance formalism live in obsidian/Retrieval/.
