---
name: lupus-cohort34-anchored-result
description: "Anchored fine-tuning (lam=30) replicated independently on cohorts 3 and 4, per the lupus-cohort34-replication plan -- cohort 3 succeeds cleanly, cohort 4 fails the batch-removal guard rail."
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
  modified: 2026-08-25T20:14:28.364Z
---

2026-08-25. Ran the full [[anchored-finetune-result]] pipeline (build/finetune_anchored.py
--lam 30, then scenarios/encode_anchored_cohort2.py) independently on cohort 3
(Processing_Cohort "3.0") and cohort 4 ("4.0") of the lupus atlas, per the plan in
[[lupus-cohort34-replication]]. New generic scripts: `build/lupus_cohortX_build.py`
(extracts a full held-out cohort by Processing_Cohort id, same chunked-h5py approach
as the cohort-2 extractor), and `--query-path`/`--tag` args added to
`build/finetune_anchored.py` + `scenarios/encode_anchored_cohort2.py` so the existing
cohort-2 scripts work unmodified for any cohort.

Extracted: cohort3_full.h5ad = 117484 cells / 41 donors (23 SLE, 18 normal); cohort4_full.h5ad
= 111482 cells / 92 donors (52 SLE, 40 normal after the ≥300-cells/donor QC filter — atlas
table said 44 normal, 4 dropped on QC). Checkpoints: `models/scvi_anchored_lam30_cohort3`,
`models/scvi_anchored_lam30_cohort4`. Encoded outputs: `data/lupus/anchored_lam30_cohort{3,4}_*`.

**Acceptance table (anchor_report.py, all three side by side):**
```
                       cohort2(orig)  cohort3   cohort4
ref_mean                     1.100     1.168     1.053
ref_sigma                    0.326     0.347     0.313
census_pairwise_CV           0.273     0.291     0.281
c2normal_sigma                1.049     0.444     2.958
sle_sigma                     3.436     3.458     4.669
gap_sigma                     2.387     3.014     1.711
AUC_sle_vs_c2n                0.900     0.995     0.849
AUC_c2n_vs_train              0.822     0.707     0.982
r_ifn_cohort2                 0.834     0.939     0.745
```

**Cohort 3: clean replication, actually better than the original cohort-2 result** --
c2normal_sigma closer to the 0 target (0.444 vs 1.049), geometry intact
(census_pairwise_CV 0.291 close to the ~0.27 projection-only reference), stronger
SLE/normal separation (AUC_sle_vs_c2n 0.995, gap_sigma 3.014).

**Cohort 4: FAILS the batch-removal guard rail.** c2normal_sigma=2.958 -- almost 3x
worse than cohort 2's own already-imperfect 1.049, batch offset barely touched.
AUC_c2n_vs_train=0.982 -- cohort 4's healthy donors are almost perfectly separable
from train_normal by distance alone, i.e. essentially unintegrated. r_ifn_cohort2=0.745
drops below the anchor_report.py guard threshold (>=0.80) -- this is a real, named
regression, not a rounding artifact. Geometry itself is not broken
(census_pairwise_CV=0.281, ref_CV in line with the other two), so this isn't a repeat
of the old "unanchored destroys geometry" failure mode -- it's specifically that
lam=30, transplanted unchanged from cohort 2's calibration, is not removing cohort 4's
batch offset.

**DECOMPOSITION (2026-08-25, later same day) -- reinterprets the table above.**
`c2normal_sigma` is the mean of PER-DONOR distances to the reference centre, so it
conflates two different quantities: the cohort's rigid block offset and the spread of
its donors around their own mean. They combine as
`mean_dist ~ sqrt(offset^2 + spread^2)`, verified exact to <1% on all three cohorts.
Decomposed in RAW latent units (the same units as ref_mean / the document's distance
tables -- do NOT divide these by ref_sigma: that quantity is the sd of the SCALAR
distance distribution, the correct denominator for c2normal_sigma but a category error
for a vector displacement, and dividing by it inflates offset ~3x into meaningless
"sigma" values):

```
                        cohort2  cohort3  cohort4   census
ref_mean                  1.100    1.168    1.053
healthy mean dist         1.442    1.321    1.979    <- matches document table (1.44)
c2normal_sigma            1.049    0.444    2.958
OFFSET (block shift)      1.246    1.221    1.851    0 (by construction)
spread (donor dispersion) 0.740    0.507    0.722    1.100
```

Geometric picture: each cohort is a COMPACT clump (spread 0.51-0.74, tighter than
census's own 1.10 -- expected, one cohort is more homogeneous than 1605 pooled donors)
sitting DISPLACED by roughly one typical-donor-distance from the census centre.

* **Cohort 4 differs from cohort 2 almost purely in OFFSET** (1.851 vs 1.246, +49%); its
  donor spread is identical to cohort 2's (0.722 vs 0.740). Its healthy donors are not
  anomalous or heterogeneous -- the whole cohort is rigidly displaced further. That is
  exactly the failure mode a mean/distribution-alignment loss term can fix.
* **Cohort 3's apparent "clean replication" is mostly SPREAD, not batch removal.** Its
  offset (1.221) is within 2% of cohort 2's (1.246); its whole advantage comes from
  healthy donors being ~31% less dispersed (0.507 vs 0.740). So the ranking 3 > 2 > 4 is
  substantially a ranking of each cohort's internal donor homogeneity, not of how well
  the anchor removed batch.

**Consequence for tuning:** `c2normal_sigma` is a bad single objective -- it can be
lowered by collapsing donor spread rather than removing offset, and a held-out healthy
group does NOT catch that (held-out donors collapse along with the trained ones).
Always monitor the pair (offset -> target 0, spread -> must stay ~2.3, not collapse)
plus SLE donor spread as a third guard. For cohort 4 collapse is a weak temptation
(offset dominates: zeroing spread would only move c2normal_sigma 2.958 -> 2.545), for
cohort 3 it is a real one (0.444 -> 0.156).

**RESOLVED (2026-08-25, later still) -- see [[mmd-alignment-inconclusive]] for the full
MMD-side story.** Cohort 4's offset was fixed with an MMD alignment term (group-A/group-B
held-out split, live-recomputed signed stopping metric), early-stopped at just 20 steps:
`c2normal_sigma` 2.958 -> 0.283, `census_pairwise_CV` unchanged (0.281 -> 0.281, geometry
untouched), both previously-failing guards now pass (`AUC_sle_vs_c2n` 0.849 -> 0.883,
`r_ifn_cohort2` 0.745 -> 0.822). Checkpoint: `models/scvi_mmd_phase2_cohort4_lamalign10_earlystop`,
encoded outputs `data/lupus/mmd_phase2_cohort4_lamalign10_earlystop_*`.

**Open question, not yet investigated:** whether cohort 4 needs its own lam sweep
(same logic as the cohort-2 lam 30/10/1 sweep in [[mmd-alignment-inconclusive]], but
here for the anchor term itself, not MMD), or whether something about cohort 4's
composition/technical profile (92 donors, largest of the three, mixed SLE:normal
52:40) makes this held-out cohort structurally harder to anchor. Do not treat cohort 4
as "done" -- report it explicitly as the negative/harder case alongside cohort 3's
clean pass, per the same non-cherry-picking discipline established for
[[mmd-alignment-inconclusive]].

**Status:** cohorts 3 and 4 checkpoints + encoded latents are saved and available for
downstream work (AB-MIL, etc.) -- but cohort 4's own-cohort SLE-vs-normal contrast rides
on an anchored fit that has NOT met the batch-removal bar cohort 2/3 met. Any biomarker
gene-set replication claim using cohort 4 should flag this caveat.
