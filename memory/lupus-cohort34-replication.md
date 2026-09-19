---
name: lupus-cohort34-replication
description: "Cohorts 3 and 4 of the lupus atlas can each rerun the whole cohort-2 experiment self-contained, which makes the 57 cross-cohort repeated donors irrelevant."
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
  modified: 2026-08-22T15:17:42.469Z
---

2026-08-21. The lupus atlas (261 donors, 1.26M cells, all 10x 3' v2) splits into 4
Processing_Cohorts. Only cohort 2 has been used so far (its 22 normals in the ELBO
term of anchored fine-tuning; its 120 SLE encoded only). Cohorts 1/3/4 are
completely untouched — the base scVI is trained on CENSUS healthy PBMC, not on the
lupus atlas at all.

    cohort   SLE   normal
      1.0      0       47
      2.0    120       22   <- used
      3.0     23       18
      4.0     52       44

Plan: rerun the ENTIRE cohort-2 pipeline separately for cohort 3 and cohort 4 —
fine-tune the census base on that cohort's own normals with the census-replay
latent anchor, then measure its own SLE against its own normals.

**Why:** as a self-contained replication per cohort, the 57 donors that recur
across cohorts stop mattering. The comparison never crosses a cohort boundary, so
a donor appearing in both 2 and 4 is not leakage — it is just the same person
measured twice in two independent experiments. Donor recurrence only breaks the
design if cohorts 3/4 are used as a held-out set for a cohort-2-trained model
(there 53 of cohort 2's 142 donors recur — 32 SLE, 21 of the 22 normals — and the
clean non-overlapping pool shrinks to 42 SLE + 77 normal).

**How to apply:** two independent replications of [[anchored-finetune-result]],
which is a far stronger claim than one cohort. Reuse build/finetune_anchored.py
and scenarios/anchor_report.py with the cohort swapped; cohort 1 has no SLE so it
serves only as extra healthy reference or as a held-out healthy batch. Report the
stratified metric alongside pooled — see [[composition-confound-lupus]].

**Paper framing (2026-08-22, Vlad's call):** the main result the paper is built
around is biomarker gene-set replication across cohorts 2/3/4 (same genes,
independently, on each cohort's own SLE-vs-normal contrast) — not yet started,
this cohort-3/4 rerun is the prerequisite. Diagnostics like the monocyte
reference-noise check (paired within-donor distance ~1.6, comparable to the
2.1 signal itself even after 3x more cells — see census_monocyte_supplement)
are secondary metrics: fine to report as-is with an honest note on how they were
derived (cell count, paired-noise caveat), not something to over-engineer before
moving on to the main replication result.
