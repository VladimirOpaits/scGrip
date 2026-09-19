---
name: first-validation-result
description: First empirical confirmation that technology dominates biology in Geneformer embedding distance (pbmcsca)
metadata: 
  node_type: memory
  type: project
  originSessionId: 506da36b-625b-49b9-b619-f9185b7b5811
---

First validation of the core hypothesis (2026-06-13), on pbmcsca, Geneformer-V2-104M cell embeddings (mean-pooled to pseudobulk per donor×Method sample, cosine distance over 14 samples). See [[geneformer-embedding-setup]].

**Result: technology dominates biology by ~13× in embedding distance.**
- Mean cosine dist same-method pairs ≈ 0.015; same-donor pairs ≈ 0.201; all pairs ≈ 0.191.
- Control 1 (replicates): the two 10x-v2 technical replicates of pbmc1 (A vs B) are each other's nearest neighbour, d=0.003. Metric is sane.
- Control 2 (tech vs bio axis): same-tech-different-donor closer than same-donor-different-tech in 88% of triples; **10/10 samples** have technology winning on average. Smart-seq2/CEL-Seq2 (plate) sit ~0.46–0.51 from same-donor droplet samples but ~0.008 from the same plate method.
- Secondary structure: droplet methods (10x, Drop-seq, Seq-Well, inDrops) cluster together, plate methods (Smart-seq2, CEL-Seq2) cluster separately; the few "bio wins" are inside the droplet family where tech signal is weak enough for biology to show — i.e. the metric is not blind to biology, batch just overrides it.

**Why it matters:** confirms the premise that selecting "far" samples = selecting different technologies = diluting batch. This was the v1 simple baseline (marginal pseudobulk + cosine). Advanced ideas (stratification by CellType, inverse-variance reliability weight n_eff/(n_eff+n0), James-Stein shrinkage of small-n centroids) are parked in the user's Obsidian for later. Distances live in geneformer.ipynb; raw embeddings at embeddings/pbmcsca.csv.
