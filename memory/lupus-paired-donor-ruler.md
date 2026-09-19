---
name: lupus-paired-donor-ruler
description: 57 lupus-atlas donors recur across processing cohorts with non-overlapping libraries, giving a direct same-person batch ruler; 40 healthy pairs even share the cell suspension.
metadata:
  type: project
---

2026-08-21. In the lupus atlas, `library_uuid` (the actual 10x run) NEVER crosses a
Processing_Cohort boundary — 0 of 88. So a donor appearing in two cohorts really
was measured in two independent runs. `sample_uuid` is donor-level and is shared,
so it does NOT indicate a shared measurement; check `library_uuid` /
`suspension_uuid` instead.

73 paired observations (donor x cohort-pair), splitting by whether the cell
suspension is shared:

  * **weak pairs, 53** (same `suspension_uuid`, different library + cohort):
    **40 normal** + 13 SLE. Biology is IDENTICAL by construction — literally the
    same cell suspension — so the whole distance is technical. For healthy donors
    this is the near-perfect batch ruler.
  * **strong pairs, 20** (different suspension), **all SLE**, 17 of them 2<->4.
    Full stack including a separate draw/prep, but SLE activity can genuinely
    differ between draws, so biology is not guaranteed fixed.

Median ~3500 cells per side, so equal-n subsampling is easy.

**Why:** it converts our numbers from unitless to calibrated. Today "c2normal_sigma
= 0.276" has no scale; against a same-person distance it becomes "inside / outside
measurement noise". It also gives the only real generalization test available for
the fine-tune: pairs involving cohort 2 should shrink, and pairs NOT involving it
(3<->4, 1<->3) reveal whether the correction is generic or cohort-2-specific.

**How to apply:** always use a calibration ladder, never a bare paired distance —
split-half within one library (sampling-noise floor), then weak pairs (run batch),
then strong pairs, then different donors same cohort (inter-individual biology),
then healthy vs SLE (the signal). All at matched cell counts, repeated subsamples,
reported in census reference-sigma units. Ties into
[[composition-confound-lupus]] (run pooled and stratified) and
[[lupus-cohort34-replication]].
