---
name: external-pca-mmd-plan
description: "2026-07-04 plan pivot: go beyond Census with user-sourced external datasets, one shared PCA over all cells (comparable basis) -> RFF-MMD patient signatures -> gap-to-ceiling enrichment; scSet optional if enough data."
metadata:
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
---

2026-07-04. Census is EXHAUSTED for paired lung studies (only 3, none with minority-class >=10 — see [[enrichment-heldout-result]]). To validate the de-confounding finding on more/better studies, we go OUTSIDE Census. **User owns data acquisition** (browsing the internet for datasets); Claude owns the pipeline. Do NOT pull more Census raw counts — user redirected away from that.

**Agreed pipeline:** (1) user drops external h5ads; (2) harmonize genes across datasets by **Ensembl ID intersection** (critical — symbols differ across datasets/versions); (3) log-normalize consistently; (4) **ONE shared PCA fit on the pooled cells** (or fit-on-pool, project held-out) so all cells' vectors are comparable across datasets — this is the whole point (PCA is universal AND batch-PRESERVING, so it doubles as the free "compressed(scVI) vs uncompressed batch" contrast we wanted from Geneformer); (5) **RFF-MMD** patient signatures (distribution, not centroid — [[mmd-distribution-ruler]]; centroid/pseudobulk overstates batch); (6) same gap-to-within-study-ceiling enrichment experiment (`scripts/heldout_enrichment.py` logic). scSet can be trained on the representation too IF enough data.

**Acquisition spec given to user (what to look for):** the bottleneck is PAIRED studies with real normal n — studies containing BOTH tumor AND normal in the same study, normal >=10-20 samples; plus several studies/techs of the same diagnosis for the enrichment pool. Must be RAW counts (.X/.raw, not normalized/scaled/embeddings), genes as Ensembl IDs, per-sample metadata (sample/donor id, disease, technology, study id), >=200 cells/sample. h5ad preferred; lung to stay comparable, other tissue OK as a separate track. Negative result on PCA is itself a paper point (representation-dependence). Build the ingestion pipeline only AFTER seeing 1-2 real files (match true format, avoid rework).
