---
name: hlca-adeno-dataset-build
description: "The concrete HLCA lung dataset built for the diagnosis experiment: lung adenocarcinoma vs normal, balanced across 3 technologies, Geneformer-embedded into a new LanceDB."
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
---

Built 2026-07-01 (local, RTX 4050 6GB, bioml env) the pivot dataset from [[pivot-to-diagnosis-data]].

**Task chosen:** binary `lung adenocarcinoma` vs `normal`, lung, from CELLxGENE Census (version pinned `2025-11-08`). Picked because it is the least technology-confounded strong candidate: disease spans 10 assays / 5 labs, and both classes co-occur in the same technologies (10x 3'v2, 10x 3'v3, Smart-seq2) — see recon in data/disease_coverage.csv, data/hlca_samples.csv.

**Final sample set (data/selected_samples.csv, 43 of 46 downloaded):** balanced per (disease x assay):
10x 3'v2 = 7 adeno / 8 normal; 10x 3'v3 = 8 / 6; Smart-seq2 = 8 / 6. 3 samples lost to a chronically slow Census shard (dataset 1b350d0a: donors D503, D496 normal 10x3'v3; plus adeno Goveia_Carmeliet_2020_patient_46 as collateral in the same chunk). ~1000 cells/sample, 39969 cells total, disease-balanced (adeno 20372 / normal 19597).

**Pipeline scripts (repo root):** select_samples.py -> download_cells.py -> tokenize_hlca.py -> embed_build.py. census_recon.py did the metadata recon. Model = Geneformer **V2-104M** (hidden_size 1152, so embeddings are **1152-dim**, NOT 768 as an older note said), cached at ~/.cache/huggingface/hub/models--ctheodoris--Geneformer. Cell embedding = CLS token (emb_mode="cls", model_version="V2").

**Key build gotchas (why the code looks the way it does):**
- Census value_filter must use DOUBLE quotes because assay names contain apostrophes (`10x 3' v2`).
- Per-sample queries each re-scan all ~2.3M lung obs -> slow + hang-prone. Fixed: ONE bulk get_obs, then pull expression in **cell-count chunks (~2500 cells)** by soma_joinid with a per-chunk SIGALRM timeout (240s). Chunking by whole dataset failed (18k-cell adeno pull timed out).
- Must match samples by the full tuple (dataset_id, donor_id, assay) — donor_id collides across datasets.
- Sequence lengths reach 4096 (V2 context; Smart-seq2 dense). On 6GB this OOMs via O(seq^2) attention -> truncate input_ids to **2048** before embedding (keeps CLS + top-expressed genes). At batch=8/2048 it fits in ~4.2/6.1 GB.
- fsspec 2026.6.0 vs datasets 5.0.0 pin warning was harmless; tokenization worked without downgrade.

**Output:** vectors/cells.lance rebuilt by embed_build.py (schema: cell_id, dataset="hlca", technology=assay, donor, cell_type, disease, sample_id, vector[1152]). Embedding run ~4h overnight on 4050.

**Next:** adapt harmony_eval.py to disease-mode (bio=disease, batch=technology); with balanced technologies the full scIB battery (iLISI/kBET/sil_batch/graph_conn + NMI/ARI/sil_label/cLISI) is finally valid — see [[greedy-picks-hardest-to-integrate]] for why it wasn't on onek1k. Then greedy selection at multiple distance quantiles. Related: [[lancedb-platform]], [[geneformer-embedding-setup]].
