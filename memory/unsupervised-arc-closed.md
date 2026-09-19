---
name: unsupervised-arc-closed
description: "Final consolidated result closing the unsupervised (scVI batch-removal) arc: MMD+held-out-early-stop applied to all three cohorts (2/3/4), acceptance table + scIB gathered; MELD/DA-seq flagged as not yet rerun on the fixed embeddings."
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
  modified: 2026-08-25T23:17:12.513Z
---

2026-08-26. Applied the held-out-early-stop MMD recipe validated on cohort 4
(see [[mmd-alignment-inconclusive]]) to cohorts 2 and 3 as well
(`build/finetune_mmd_earlystop.py`, generalized from the cohort-4 script via
`--cohort-tag`). All three cohorts now have a clean, guard-passing anchored+MMD
checkpoint:

| | cohort2 base -> MMD | cohort3 base -> MMD | cohort4 base -> MMD |
|---|---|---|---|
| c2normal_sigma (target ~0) | 1.049 -> 0.222 | 0.444 -> 0.327 | 2.958 -> 0.283 |
| census_pairwise_CV (target ~0.27) | 0.273 -> 0.270 | 0.291 -> 0.282 | 0.281 -> 0.281 |
| AUC_c2n_vs_train (target ~0.5) | 0.822 -> 0.609 | 0.707 -> 0.675 | 0.982 -> 0.624 |
| AUC_sle_vs_c2n (guard >=0.87) | 0.900 -> 0.927 | 0.995 -> 0.993 | 0.849 -> 0.883 |
| r_ifn_cohort2 (guard >=0.80) | 0.834 -> 0.859 | 0.939 -> 0.942 | 0.745 -> 0.822 |

Early-stop crossed the target almost immediately in all three: cohort2 in ~10
steps, cohort3 at step 0 (was already close), cohort4 in ~20 steps -- confirms
[[mmd-alignment-inconclusive]]'s revised verdict that lam_align controls speed
of convergence to an overshot endpoint, not the destination; step-count with an
honest held-out stop is the real knob. scIB (bio conservation ~0.50, batch
mixing ~0.88-0.89) UNCHANGED before/after on all three -- the correction is
targeted at the offset axis, doesn't touch the aggregate clustering metrics.

Checkpoints: `models/scvi_mmd_{lam30,lam30_cohort3}_lamalign10_earlystop`,
`models/scvi_mmd_phase2_cohort4_lamalign10_earlystop`. Encoded outputs:
`data/lupus/mmd_{lam30,lam30_cohort3}_lamalign10_earlystop_*`,
`data/lupus/mmd_phase2_cohort4_lamalign10_earlystop_*`.

**Real OOM twice on cohort2's encode** (320k cells, the largest cohort) --
confirmed via `journalctl -k`, not the watchdog's fault (kernel killed both
attempts at 6.8GB/8.3GB anon-rss, faster than the 5s-interval free-check could
catch). Two fixes applied: (1) `scenarios/encode_anchored_cohort2.py`'s
`score_ifn()` was doing a full `adata.copy()` just to score ~15 IFN genes --
score_genes needs the FULL gene panel for its control-gene sampling so the gene
axis can't be subset, but obs/uns/other layers can be dropped -- rewritten to
build a minimal `AnnData(X=counts.copy(), var=var[[]])` instead of copying
everything. (2) Docker Desktop's qemu VM was holding ~4GB RSS in the background
on this 14GB machine; stopped it (`systemctl --user stop docker-desktop.service`)
before the retry succeeded. Worth checking `systemctl --user status
docker-desktop.service` before any cohort-2-scale (largest cohort) job on this
machine again.

**MELD/DA-seq (see [[lupus-meld-daseq-monocyte-result]]) gathered as-is, NOT
rerun** on the new MMD-corrected embeddings -- still the projection-only,
cohort-2-only result (monocytes ~6x more likely to sit in an SLE-exclusive
neighborhood, both MELD-likelihood and DA-seq-style permutation agree).
Re-running it on the corrected embeddings, and replicating on cohort 3/4
independently, is the natural next step if the paper needs it -- flagged, not
done.

**Where this leaves the project:** all three cohorts have their own
independently-anchored-and-MMD-corrected scVI embedding, batch-removal guards
now pass everywhere. This closes the unsupervised half of CLAUDE.md's Step 1
architecture. Remaining Step-1 work is AB-MIL (not started) and, per this
result, potentially rerunning MELD/DA-seq + biomarker gene-set replication on
the now-corrected cohort 3/4 embeddings instead of projection-only ones.
