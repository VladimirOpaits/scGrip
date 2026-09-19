---
name: quantile-sweet-spot-result
description: "On HLCA adeno-vs-normal confounded-start Harmony experiment, moderate-quantile retrieval (q=0.75) Pareto-dominates greedy-farthest: it resolves the 'maximally different = hardest to integrate' tension."
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
---

2026-07-01. First diagnosis-experiment on the [[hlca-adeno-dataset-build]] data. Code: hlca_experiment.py + selection.py; results in hlca_results.csv. This is the calibrated answer to [[greedy-picks-hardest-to-integrate]].

**Design — "confounded start":** cohort = 16 samples where diagnosis is correlated with technology (adeno all from Smart-seq2=T2, normal all from 10x 3'v2=T1). Fixed cohort size (replace, not add). Per-class matched replacement of k samples from a pool of same-diagnosis samples in OTHER technologies. Arms: PURE / RETRIEVAL (greedy farthest, q=1) / QUANTILE q=0.5,0.75 / RANDOM (5 seeds). k=2 and k=5.

**Pipeline per arm:** raw counts → normalize/log1p/HVG2000/scale/PCA(50) → Harmony(batch=technology) → metrics. Selection is done on Geneformer 1152-dim space; evaluation in independent PCA space (non-circular guard).

**Metrics:** headline predictability (kNN macro-F1 on class-balanced subsamples): **BioPred** = predict diagnosis (project's biology metric), **BatchMix** = 1 − predict technology. Plus standard scIB: iLISI/kBET by technology; NMI/ARI/silhouette_label/cLISI/graph_connectivity by **cell_type** (NOT diagnosis — diagnosis is not a cell-level cluster structure; using it gave NMI≈0, a bug caught and fixed). Census gives cell_type in a unified CL ontology (106 types) so no vocab harmonization needed.

**Result (k=5, the informative regime):**
| arm | BioPred | BatchMix | NMI(cell-type) | ½+½ |
|---|---|---|---|---|
| PURE | 0.491 | 0.522 | 0.638 | 0.507 |
| RETRIEVAL farthest | 0.628 | 0.718 | 0.654 | 0.673 |
| QUANT q=0.5 | 0.733 | 0.679 | 0.653 | 0.706 |
| **QUANT q=0.75** | 0.695 | **0.748** | 0.655 | **0.722** |
| RANDOM | 0.766 | 0.591 | 0.694 | 0.678 |

**Findings:** (1) Enrichment recovers diagnosis: PURE BioPred 0.49 (≈chance, erasure — Harmony removes technology and the confounded diagnosis with it); any enrichment raises it. (2) **Moderate quantile q=0.75 Pareto-dominates greedy-farthest** on BOTH axes → the sweet spot; order q0.75 > q0.5 > RANDOM ≈ RETRIEVAL > PURE. Don't go to max diversity. (3) Cell-type scIB (NMI/graph/sil) is high and ~equal across all arms → general biology is preserved everywhere; the action is specifically on diagnosis×technology.

**Caveats:** only 43 samples (unit = patient), RANDOM noisy (Bio 0.55–0.83), quantiles single-run (no seeds). k=2 under-powered. Confounded start is deliberately strong. Next: cloud re-run on 150–300 samples (HLCA has adeno 138 / normal 435 donors available) to sweep quantiles 0.3/0.5/0.7/0.9 with seeds for a real Pareto front + CIs; Geneformer-space spot-check; vary confound strength. Immediate next task (user, 2026-07-01): held-out CLASSIFICATION — train a diagnosis classifier on the enriched cohort, test on held-out samples, to show enrichment improves generalization (the practical payoff beyond probing). Related: [[pivot-to-diagnosis-data]], [[mmd-distribution-ruler]].
