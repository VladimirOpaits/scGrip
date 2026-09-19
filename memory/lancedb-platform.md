---
name: lancedb-platform
description: "Vector-DB platform: LanceDB store of cell-level Geneformer embeddings, schema, and the patient-as-distribution design"
metadata: 
  node_type: memory
  type: project
  originSessionId: 29a4987b-59e9-4515-84c7-f37d0ab3e315
---

Decided 2026-06-20: the project's data layer is **LanceDB** (embedded, files-only; chosen over Qdrant/pgvector for minimal infra, file-backup, numpy-native — heavy set-distance math runs in numpy anyway). DuckDB deferred (only needed later for SQL JOINs on small tables; not installed yet). `bioml` env already has `lancedb`.

`build_lancedb.py` ingests `embeddings/pbmcsca.csv` + `embeddings/gse295527.csv` into `vectors/cells.lance`. **`cells` = source of truth** (one row per cell), patient-level objects (pseudobulk, distribution signatures) are derived on the fly. Unified schema across datasets: `cell_id, dataset, technology, donor, cell_type, replicate, sample_id, vector(fixed_size_list<float,768>)`. **43021 cells total**: pbmcsca (31021, has `cell_type`, donors pbmc1/pbmc2, 14 samples, technology from `Method`) + gse295527 (12000, NO cell_type, donors F1/F5, 12 samples, 1000 cells/sample). `sample_id` = the patient unit (donor × technology), e.g. "pbmc1 | Smart-seq2", "Flex | F1A". 26 samples, 12 technologies (plate → legacy droplet → modern Flex/GEMX3P/NextGEM3P).

Gotcha: LanceDB default search metric is **L2 (euclidean)** — the validated ~13× result used **cosine**; use `.metric("cosine")` or store L2-normalized vectors for the ANN shortlist. `n_genes`/depth not in CSVs — add from tokenized data later if needed.

Design locked this session (see obsidian/Retrieval/): patient = distribution not a point; **unit of redundancy = patient**; two-tier retrieval = RFF-MMD single-vector shortlist (ANN) → partial-Hausdorff rerank on cells; Hausdorff is L2-only (can't linearize), MMD bridges L2↔L3; set-distance beats stratification for selection (stratification = analysis-only, sees 1/3 axes). Relates to [[next-experiment-greedy-enrichment]], [[first-validation-result]], [[geneformer-embedding-setup]], [[project-direction]].
