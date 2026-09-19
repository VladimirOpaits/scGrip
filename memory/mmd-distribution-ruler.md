---
name: mmd-distribution-ruler
description: "MMD(RFF) distribution ruler compresses tech-dominance 39x->6.4x, validated; and the greedy-MMD cross-dataset selection ordering"
metadata: 
  node_type: memory
  type: project
  originSessionId: 29a4987b-59e9-4515-84c7-f37d0ab3e315
---

Validated 2026-06-20 (cv.ipynb, code in store.rff_signatures + selection.l2_distance_matrix). Patient signature = mean of RFF features of its cells (kernel mean embedding) → L2 between signatures = MMD with RBF kernel, σ via median-heuristic. This is the principled generalization of the centroid pseudobulk (centroid = MMD with linear kernel = first moment only).

**Key result:** on 26 samples (pbmcsca + gse295527), centroid-cosine gives tech-vs-bio ratio 39× (same-tech 0.0036, same-donor 0.1406) — but that **overstates batch dominance** because centroid sees only the location shift (which is mostly technology). MMD (n_features=2048) compresses it to **6.4×** (same-tech 0.0936, same-donor 0.6024): biology is far more recoverable than the centroid implied, it just lived in distribution shape (coverage/composition), not the center. Validated: nearest pairs = technical replicates (within-platform A/B, d≈0.02–0.04, calibration OK); σ-robust (ratio 5.0/6.4/7.3 at σ×0.5/×1/×2); gse-block no longer collapses to 0 — MMD separates sub-platforms (NextGEM3P ~0.60 closer to 10x lineage than Flex/GEMX ~0.77). Centroid is fine as a coarse tech-shift detector but blind to within-block structure; MMD is the honest single-vector ruler for selection + outcome. Recorded in obsidian/Retrieval/Результат — технология доминирует над биологией.md + [[RFF-MMD note]].

**Greedy-MMD selection (diverse arm)** from a 10x-v2 cohort (start: pbmc1/pbmc2 10x v2 A/B), order of farthest-point picks (d = min-to-cohort at pick time): pbmc2|Smart-seq2 (1.165), Flex|F5B (0.785), pbmc1|CEL-Seq2 (0.556), pbmc1|inDrops (0.525), NextGEM3P|F1B (0.512), pbmc1|10xv3 (0.410), pbmc2|Seq-Well (0.329), GEMX3P|F5A (0.303), pbmc2|inDrops (0.302), pbmc1|Smart-seq2 (0.284), pbmc2|Drop-seq, pbmc1|Drop-seq, pbmc1|Seq-Well, pbmc2|CEL-Seq2, then the gse-block dedups to the bottom (0.02–0.07). Smart-seq2 is #1 in both centroid and MMD.

Relates to [[lancedb-platform]], [[first-validation-result]], [[next-experiment-greedy-enrichment]]. Next: label-free outcome metric = donor-recovery (holdout_donor_generalization) — see [[next-experiment-greedy-enrichment]]; constraint: only 2 donors in pbmcsca (binary, chance 50%), donor-recovery is within-dataset (gse donors F1/F5 disjoint from pbmcsca pbmc1/pbmc2).
