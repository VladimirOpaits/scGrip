---
name: lupus-meld-daseq-monocyte-result
description: "MELD/DA-seq rerun on batch-corrected embeddings across all 3 cohorts: the cell-type ORDERING replicates strongly (DA-seq Spearman 0.88-0.94), monocytes top in every cohort -- but the original '6x more likely than any other type' headline was a binarization artifact; the honest continuous lead is only 1.12-1.30x."
metadata: 
  node_type: memory
  type: project
  originSessionId: 4c10a894-795a-443a-9bce-1a5686bd1785
  modified: 2026-08-27T18:48:56.925Z
---

2026-08-27. Reran `scenarios/meld_daseq_analysis.py` on the batch-corrected phase-2
embeddings for all three cohorts (added `--prefix` / `--out-tag` so it can consume an
already-encoded latent instead of rebuilding the unadapted projection, which is what the
original run used and which had batch and disease entangled).

**The monocyte finding survives batch removal.** On cohort 2, classical monocyte
DA-significant fraction 0.282 (projection) -> 0.298 (corrected), MELD likelihood
0.551 -> 0.552. This was a real risk: monocytes are the compartment with the largest
compositional mismatch against census (55% vs 10%), so they could have been pure batch.
They were not.

**But the "~6x more likely than any other cell type" headline was a BINARIZATION
ARTIFACT and must be retracted.** `frac_da_significant` thresholds each cell at the 95th
percentile of its own permutation null. When a type's score mass sits near that
threshold, a small shift in the continuous score flips disproportionately many cells.
Measured: on cohort 2 a **30%** difference in continuous DA-seq score became a **350%**
difference in the binary fraction.

Continuous DA-seq score, monocytes vs the next non-monocyte type:
```
cohort 2   0.714 vs 0.550   +0.164   1.30x      (binary said 3.5x)
cohort 3   0.778 vs 0.694   +0.084   1.12x      (binary said 1.14x)
cohort 4   0.680 vs 0.543   +0.137   1.25x      (binary said 2.3x)
```

**What DOES replicate, and it is the strongest result here: the ORDERING.**
Spearman across the 11 cell types, between cohorts with disjoint donors and batches:
```
              DA-seq    MELD
c2 vs c3      +0.936   +0.773
c2 vs c4      +0.927   +0.645
c3 vs c4      +0.882   +0.600
```
Monocytes (classical and non-classical) rank 1-2 in all three cohorts; plasmablast, pDC
and progenitor sit at the bottom in all three. DA-seq replicates markedly better than
MELD.

**Absolute fractions are NOT comparable across cohorts** (classical monocyte
frac_da_significant: 0.298 / 0.699 / 0.534 for c2/c3/c4). Cohort 3 scores high on
everything because its global disease signal is far stronger (AUC_sle_vs_c2n 0.973-0.995,
gap_sigma up to 3.014 vs cohort 2's 0.887/1.624) -- in that cohort the disease is legible
in ALL cell types, not selectively in monocytes. Donor counts also differ a lot
(142 / 41 / 92), and the permutation null shuffles at donor level, so its resolution
differs too.

**Reporting rule going forward: report the continuous DA-seq score and MELD likelihood.**
`frac_da_significant` is threshold-dependent and exaggerates; include it only as a
secondary column with the caveat stated.

**CLOSED 2026-08-27: MELD now has its own donor-permutation null.** `run_meld` gained
`n_perm` (graph fitted once, reused across permutations; the dead duplicate
`meld_permutation_test` was removed), run at n_perm=200 on all three cohorts.

The two tests agree strongly WITHIN each cohort -- Spearman of significance fractions
across the 11 types: c2 +0.954, c3 +0.815, c4 +1.000. Graph-diffusion density and
multiscale kNN neighbour-fraction are methodologically independent, so this convergence
is real support, not one method echoing the other.

Classical-monocyte MELD-significant fraction: 0.652 / 0.708 / 0.664 (c2/c3/c4) -- top or
near-top everywhere, now backed by a permutation null. MELD is systematically MORE
liberal than DA-seq (c2 classical monocyte 0.652 vs 0.298), as expected from graph
smoothing concentrating signal.

Cross-cohort replication of MELD significance: +0.726 / +0.963 / +0.745 -- good but
weaker than DA-seq's continuous scores (0.88-0.94), consistent with DA-seq being the more
transferable readout.

**One caveat:** in cohort 3, `lymphocyte` (0.944) outranks monocytes under MELD. That
label is the unassigned catch-all with only 358 cells there, so it is fragile, but it
means "monocytes rank first" is exactly true only for c2 and c4 under MELD.

Supersedes the 2026-08-23 within-cohort-2 projection result. Related:
[[stratified-mmd-design]], [[composition-confound-lupus]], [[unsupervised-arc-closed]].
