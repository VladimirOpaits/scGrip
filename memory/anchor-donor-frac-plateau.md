---
name: anchor-donor-frac-plateau
description: "2026-09-07 anchor donor-count ablation on cohort 2 — 25%/50%/75%/100% of the 1605 census reference donors give statistically indistinguishable geometry; the anchor budget plateau starts at or below 25% (~400 donors), lower than the TODO's prior guess of ~200."
metadata: 
  node_type: memory
  type: project
  originSessionId: 9e54ad71-2c38-4bf4-b67f-ff624399e3a4
  modified: 2026-09-07T19:25:31.077Z
---

Added `--donor-frac` to `build/finetune_anchored.py` (subsamples the census reference
donor pool once, before the per-donor replay loop -- donors not picked contribute
nothing to the anchor, not even at reduced weight). Ran `--lam 30 --donor-frac
{0.25,0.5,0.75}` on cohort 2, encoded + measured with `paper/geometry.py`:

```
                          census_CV  block_offset  gap_sigma  AUC_sle_vs_norm
anchor 100% (1605 donors)     0.279         0.971      2.523            0.909
anchor_donorfrac75 (~1204)    0.277         0.971      2.569            0.908
anchor_donorfrac50 (~803)     0.278         0.965      2.508            0.907
anchor_donorfrac25 (~401)     0.277         0.976      2.523            0.907
```

All four are within noise of each other on every metric. This directly extends the
still-open "Свип бюджета якоря" question in TODO.md: the plateau does NOT start
around ~200 donors as speculated there -- it's already flat at 25% (~401 donors),
the smallest fraction tried so far. Practical consequence, same as the TODO already
flagged: if 400 donors is already enough, anchoring on a much smaller EXTERNAL
healthy-donor pool (when no matched internal train set exists) looks realistic --
the open question "is 20 external donors enough" is now bounded below by evidence,
not pure guesswork, but the bound itself still needs to be found (this ablation
didn't go below 25%).

**How to apply / next step**: push the ablation lower --10%, 5%, a handful of
donors (e.g. 20, 50, 100 in absolute count rather than fraction) -- to find where it
actually breaks, not just confirm it hasn't broken yet at 25%. Reuses the same
`--donor-frac` flag; for small absolute counts, add a `--donor-n` int alternative if
fractions get awkward at low donor counts. See [[paper-pivot-anchor-mmd-ewc]].

**Sanity check that this isn't a pooling artifact (Vlad pushed back, rightly).**
`geometry()`'s `census_CV`/`ref_mean` are computed on the RE-ENCODED FULL 1605-donor
census base regardless of `--donor-frac` -- donors not in the replay set still get
measured, just never entered the anchor loss. So a real degradation in the
non-replayed 1204 donors could in principle hide inside a pooled 1605-donor number.
Directly checked on the donorfrac25 run by splitting donor centroids into "was in
`ref_idx`" (401) vs "wasn't" (1204): mean distance to centre 1.097 vs 1.103,
census_CV 0.281 vs 0.294 -- a small (~5% relative), not a masked collapse. The
plateau is real, not a pooling artifact. Mechanistically this matches
[[anchor-generalization-check]]: the encoder is donor-blind
(`encode_covariates=False`), so a constraint applied to any representative
expression-space sample generalizes to unseen cells/donors with similar expression,
regardless of whether that specific donor's id ever appeared in the anchor loss --
donor identity was never an input the encoder could have memorized against.
