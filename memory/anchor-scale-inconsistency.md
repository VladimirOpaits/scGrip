---
name: anchor-scale-inconsistency
description: "Cohort 2's headline anchored checkpoint was trained with the RAW anchor loss (sum over dims) while cohorts 3 and 4 used the SIGMA-normalized one (mean over dims / sigma^2) -- a 14.4x difference in effective anchor strength at the same lam=30, so the published 3-cohort comparison is confounded."
metadata: 
  node_type: memory
  type: project
  originSessionId: 4c10a894-795a-443a-9bce-1a5686bd1785
  modified: 2026-08-27T17:20:12.730Z
---

2026-08-27. Found while relaunching phase 1 with the new stratified replay: the
per-epoch log line labels its units, and the old logs disagree.

```
anchored_lam30.log         (cohort 2)  -> "units/dim"  first train_anchor 1.5120   RAW
anchored_lam100.log        (cohort 2)  -> (no label)   first train_anchor 1.4890   RAW
anchored_lam30_cohort3.log             -> "sigma/dim"  first train_anchor 0.1200   SIGMA
anchored_lam30_cohort4.log             -> "sigma/dim"  first train_anchor 0.1296   SIGMA
```

`build/finetune_anchored.py` has `--anchor-scale {sigma,raw}`, default now `sigma`:
```python
if args.anchor_scale == "sigma":
    return ((d / sigma_t) ** 2).mean()   # mean over cells AND dims
return (d ** 2).sum(-1).mean()           # sum over dims
```
raw/sigma effective scale ratio = `20 / mean(1/sigma^2)` = **14.4x** with the observed
per-dim sigmas. At the same `lam=30`, cohort 2's anchor gradient was ~14x stronger than
cohorts 3 and 4's.

**CLAUDE.md documents the RAW form** (`lam*||mu_new - mu_ref||^2`), which matches cohort
2. Cohorts 3/4 silently used sigma after the default changed. Earlier in this same
session I described the anchor loss to Vlad as the sigma-normalized version -- correct
for cohorts 3/4, WRONG for cohort 2, the source of most headline numbers.

**Mitigating measurement:** achieved drift is similar despite the 14x nominal gap,
because both settings sit near the anchor's plateau. Converting to common units
(units/dim = sigma/dim x mean_sigma): cohort2 RAW epoch1 0.2778, cohort2 SIGMA epoch1
0.2702, epoch50 0.2401. So the geometry outcome is probably not 14x different -- but
the cross-cohort comparison is still not apples-to-apples and cannot be published as if
it were.

**Decision:** go forward in SIGMA mode everywhere (scale-free, comparable across latent
sizes, already the script default and what cohorts 3/4 used). Rerun cohort 2's
old-replay baseline in sigma mode (`--ref-floor 0 --anchor-scale sigma --lam 30`) so
that (a) the old-vs-new replay comparison isolates the sampling change and (b) all three
cohorts finally share one anchor formula. The existing `models/scvi_anchored_lam30` is
RAW and must not be used as the baseline for that comparison.

Related: [[anchored-finetune-result]], [[stratified-mmd-design]],
[[anchor-generalization-check]].
