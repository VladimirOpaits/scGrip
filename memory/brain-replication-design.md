---
name: brain-replication-design
description: "Second-tissue replication = brain, single-assay (10x 3' v3), neurodegeneration vs control — batch is lab/protocol only, which kills the \"just regress out the assay\" objection."
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
---

2026-07-14. Frozen design for the cross-tissue replication (found via
[[paired-study-scarcity-is-a-cancer-problem]]):

  tissue  = brain, **assay = 10x 3' v3 ONLY**
  y       = neurodegeneration (PD/AD/ALS/FTD/LBD/dementia) vs normal control  [broad = primary]
  y2      = dementia family only (AD/FTD/LBD/Pick/PSP)                        [narrow = sensitivity]
  11 paired studies, 1474 donors inside them; 2733 donors total (1107 / 1626).
  Biggest: 37a17b78 (PD/LBD, 699 vs 256) — capped at 200 donors/study or it outvotes the other ten.

**Why single-assay matters:** on CRC a reviewer can say "your batch axis is just technology, put
assay in as a covariate". Here technology is constant, so between-study variance IS lab / dissociation
protocol / year. If near->biology, far->batch and the enrichment gap-closing replicate here, the
claim stops being about technology and becomes about batch as such — strictly stronger than v1.

**How to apply:** representation is raw counts -> our OWN PCA (`scripts/brain_build.py`), NOT the
Census scVI latent — scVI is trained with dataset_id as a batch covariate and subtracts the very
axis we measure (this also makes the old lung d_batch a lower bound, see [[scset-pivot]]).
Build order mirrors CRC: counts -> PCA50 -> stratified geosketch landmarks -> Nystrom signatures ->
d_bio/d_batch -> quantile enrichment -> batch_frac + leave-one-cell-type-out ablation
([[explainability-geometry]], [[enrichment-heldout-result]]).

**Machine safety:** Census obs/X pulls must run under a cgroup cap
(`systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0`) — an unguarded pandas pull of
full-Census obs swapped Vlad's machine to death twice. Never `to_pylist()` an Arrow batch; aggregate
inside Arrow. RLIMIT_AS does NOT work (TileDB reserves virtual address space; fails at thread pool).
