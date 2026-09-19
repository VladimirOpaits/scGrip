---
name: pivot-to-diagnosis-data
description: "2026-06-21 decision to leave PBMC cell-type trade-off and move to a large multi-tech/multi-lab dataset of a DIFFERENT tissue with diagnosis labels (healthy-vs-disease min, more diagnoses better)."
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
---

Decided 2026-06-21: the onek1k/pbmcsca cell-type experiment hit a structural ceiling (healthy PBMC, monolithic single-tech onek1k, tiny 14-sample pool, cross-dataset cell-type harmonization is noisy). Pivot to a new dataset.

Requirements for the new dataset (user's spec): different tissue entirely; large; multiple technologies AND labs (the batch axis); at minimum healthy-vs-disease, but more diagnoses better if data is sufficient. Use **diagnosis presence as the biology metric** instead of cell type — simpler, faster, and the actual product scenario ("enrich a patient cohort within same diagnosis"). Critical confound to avoid: the same diagnosis must appear across multiple technologies/labs, else diagnosis and batch are inseparable.

**Why:** Diagnosis-as-bio matches CLAUDE.md hypothesis ("same diagnosis, age, sex; different labs/tech"); cell-type was always a proxy and required harmonizing incompatible vocabularies.

**How to apply:** New metric design carries over from harmony_eval.py — predictability axis: BioPred = F1 predicting diagnosis (binary or multi-class), BatchMix = 1 − F1 predicting technology/dataset, both on class-balanced subsamples. Then compare greedy selection at multiple distance quantiles (not just farthest — see [[greedy-picks-hardest-to-integrate]]). Related: [[project-direction]], [[next-experiment-greedy-enrichment]].
