---
name: sequence-length-depth-axis
description: "Geneformer token length = sequencing depth; a technology-correlated batch axis worth investigating, and why GSE295527 embeddings are slow"
metadata: 
  node_type: memory
  type: project
  originSessionId: 506da36b-625b-49b9-b619-f9185b7b5811
---

Observed 2026-06-15: GSE295527 embedding extraction was ~8× slower than pbmcsca (~2.4 vs ~18 cells/s, ETA 4h for 36k cells) because modern platforms (Flex, GEM-X v4, Parse, Scale) detect far more genes per cell → tokenized length median 2377 with 67% of cells hitting the 2048 model cap, vs pbmcsca's short ~700–1000. Transformer attention is ~quadratic in length, so (2377/900)²≈7× matches. GPU was genuinely compute-bound (100%, 85°C), not a bug.

Practical fix: the analysis mean-pools each sample into one centroid, so 3000 cells/sample is overkill. Subsampled the existing tokenized dataset to 1000 cells/sample → `tokenized/gse295527_1k.dataset` (12000 cells), no re-tokenization needed, ETA ~1.3h. Point the extract cell's `input_data_file` there.

Vladimir flagged this as potentially **fundamental** (wants to revisit after his exam): genes-detected-per-cell is itself strongly technology-dependent, so sequence length is another technology-correlated batch axis, and Geneformer's truncation at 2048 is a technology-dependent information bottleneck (deep platforms lose everything beyond top-2048 genes) that could create/mask batch signal. Open question: does pairwise embedding distance correlate with median-length difference (i.e. depth leaking into the "molecular" metric)? Full writeup + control ideas in obsidian/Retrieval/"Длина последовательности = глубина секвенирования — скрытая технологическая ось.md". Relates to [[first-validation-result]], [[geneformer-embedding-setup]].
