---
name: composition-as-offset
description: Feed patient cell-type composition into the MIL classifier as a fitted nuisance offset, not concatenated input, so attention is forced onto what proportions cannot explain.
metadata:
  type: project
---

2026-08-22, lupus MIL design. AB-MIL's softmax attention produces a weighted
AVERAGE, so the pooled patient vector is a state descriptor that is structurally
blind to abundance. Composition is therefore genuinely complementary information
that the pooling operator discards — measured on cohort 2 (n=142, 5-fold x 20):
composition alone logloss 0.119 / AUC 0.985, state alone 0.113 / 0.994, both
0.092 / 0.998.

Do NOT simply concatenate composition into the classifier input. Composition
alone already reaches AUC 0.985, so BCE would be nearly satisfied by it and the
attention branch would receive almost no gradient — giving a great AUC and
useless attention weights, i.e. losing the "which cells" answer that is the whole
point.

Correct form — composition as a nuisance covariate fitted FIRST, attention on the
residual:

    logit(y) = offset(composition_CLR) + w . attention_pool(cells)

**Why:** it defines a biomarker as what proportions cannot explain, which is the
stated goal (cell-state biomarkers, not a cell counter), and it makes the
compositional contribution explicit, readable and ablatable instead of leaking in
implicitly. Also composition is ~half technical (same-suspension reruns differ by
9% of cells vs 16% between donors, ratio 1.7), so folding it into an offset keeps
that batch contamination out of the attention branch.

Note the direction differs from the CVAE/QC-passthrough analogy it came from:
there covariates go to the DECODER to relieve the encoder; here composition is a
predictor, so relief requires the offset/residual construction, not plain
conditioning.

**How to apply:** use CLR, not raw proportions (simplex, not additive) — plain
log was used for the numbers above. Evaluate the composition term's contribution
specifically on held-out cohorts, where proportions should degrade more than
state. See [[composition-confound-lupus]], [[lupus-cohort34-replication]].
