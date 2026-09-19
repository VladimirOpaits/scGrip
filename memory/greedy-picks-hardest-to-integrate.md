---
name: greedy-picks-hardest-to-integrate
description: "On pbmcsca/onek1k, greedy-farthest retrieval selects the LEAST integrable technologies — batch becomes more predictable, not less, after Harmony."
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
---

Harmony trade-off experiment (onek1k as monolithic target cohort + pbmcsca pool, k=3 cell-level replacement, 40 samples ×1000 cells fixed). On the honest metric (kNN macro-F1 predictability, computed on class-balanced subsamples), **RETRIEVAL mixed batches WORSE than RANDOM**: batch_pred ≈0.79 (greedy) vs ≈0.62–0.66 (random); BioPred ≈equal (~0.7). Reason: greedy_farthest picks the most distinct technologies (Smart-seq2/inDrops/CEL-Seq2 plate/early-droplet) — exactly the ones Harmony cannot merge. "Maximally different" = "hardest to integrate." This is the central tension for the project, possibly tautological — needs testing whether it persists with more data / mid-quantile selection.

**Why:** Contradicts the earlier optimistic read (which was a kBET artifact of 96/4 batch imbalance — see below). The real signal only appeared after fixing metrics.

**How to apply:** Don't claim "diverse retrieval improves integration" without controlling for selection extremity. Test greedy at multiple distance quantiles (not just farthest) to find the sweet spot: diverse enough to dilute batch, close enough to integrate.

Method notes that made metrics trustworthy: (1) onek1k is one technology → batch metrics dominated by 96% imbalance; iLISI structurally ≈0 and was dropped; kBET was inflated by imbalance (0.976 → 0.004 once batches balanced). (2) All batch metrics now computed on class-balanced subsamples. (3) Switched headline axis to predictability (BioPred = F1 on cell_type, BatchMix = 1 − F1 on technology) — robust to imbalance, matches the project's batch-predictability plan. Code in harmony_eval.py; experiment in onek1k.ipynb. Related: [[mmd-distribution-ruler]], [[next-experiment-greedy-enrichment]], [[first-validation-result]].
