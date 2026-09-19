---
name: final-rerun-numbers
description: "2026-08-29 canonical rerun of all three cohorts with `other` excluded from the stratified MMD loss -- the numbers that go in the paper, plus the paper errors this rerun exposed (block offset was group-A-only, cohort 4 A/B split is 26/14 not 27/13, unanchored offset is recoverable)."
metadata:
  node_type: memory
  type: project
---

2026-08-29. Reran phase 2 for cohorts 3 and 4 with `other` excluded from the MMD loss
(cohort 2's `p2_f0sigma` already had it excluded). New prefixes `data/lupus/final_c3`,
`data/lupus/final_c4`. Cohort 3 early-stopped at step 30 (score 0.986); cohort 4 never
reached target and used the `at2` milestone (step 50), same rule as before.

**All paper numbers now come from one script: `scenarios/paper_numbers.py`** (geometry,
A/B control, scIB, compartment ASW, within-compartment fine ASW, block ASW, DA-seq).

```
                  c2_base c2_anch c2_fin c2_unanch | c3_base c3_anch c3_fin | c4_base c4_anch c4_fin
census_CV          0.2541  0.2793  0.2693  0.4625  | 0.2541  0.2905  0.2695 | 0.2541  0.2808  0.2742
block_offset       1.4459  0.9708  0.5436  0.8786  | 1.6800  1.0456  0.5978 | 2.2148  1.7588  0.8238
donor_spread       0.7849  0.6284  0.6440  0.4083  | 0.5711  0.4341  0.4782 | 0.8417  0.6857  0.6630
gap_sigma          2.1771  2.5234  2.7571  1.1489  | 2.4350  3.0144  3.5426 | 1.6736  1.7112  1.9312
AUC_sle_vs_norm    0.8644  0.9095  0.9473  0.8701  | 0.9541  0.9952  0.9928 | 0.8385  0.8490  0.8764
r_ifn_all          0.7431  0.8420  0.8724  0.8154  | 0.8731  0.9389  0.9539 | 0.7242  0.7453  0.7813
scib_bio           0.5043  0.4953  0.4880  0.4887  | 0.5105  0.4997  0.4876 | 0.5254  0.5144  0.5019
scib_batch_donor   0.8810  0.8763  0.8778  0.8897  | 0.8979  0.8911  0.8879 | 0.8862  0.8800  0.8804
```
Offset reduction factor 2.66 / 2.81 / 2.69 -- tighter than the pre-rerun 2.7-2.9.

**Four errors in the previous draft that this rerun exposed.**
1. `block offset` in the headline tables was computed on **group A only**, not all healthy
   donors. All-donor cohort 2: base 1.446 (draft said 1.466 = A), anchor 0.971 (1.001 = A),
   final 0.544 (0.533 = A). Same for the other cohorts.
2. Cohort 4's A/B split is **26/14**, not 27/13 -- `held_n = round(40*0.35) = 14`.
3. The unanchored arm's block offset **is recoverable** (`data/lupus/finetune`): 0.879.
   The draft's "--" hid a number that helps the argument -- naive fine-tuning removes
   batch BETTER than the anchor (0.879 vs 0.971) while wrecking the reference (CV 0.462).
4. "The disease signal collapses" is false: AUC rises 0.864 -> 0.870 under unanchored
   fine-tuning; only gap sigma falls, and only because census dispersion doubles.

**Lambda sweep now has a batch column** (was missing, so the appendix caption promised a
trade-off the table could not show):
```
                 lam3    lam10   lam30   lam100
census_CV        0.2978  0.2851  0.2793  0.2752
block_offset     0.9420  0.9204  0.9708  1.0449
gap_sigma        2.2047  2.3923  2.5234  2.5027
```
Geometry monotone in lambda, batch best at lam=10, disease best at lam=30 -- consistent
with [[anchor-lambda-tradeoff]].

**Two DA-seq claims in the draft are wrong.** "plasmablast, pDC and progenitors rank last
in all three" fails for cohort 4, where the bottom three are progenitor (0.250),
plasmablast (0.302) and **B cell** (0.355), with pDC fourth at 0.368. And cohort 3's
progenitor entry rests on **10 cells**, below the pipeline's own 20-cell floor.

**Operational trap hit twice today:** `pgrep -f`/`pkill -f` on a pattern that also occurs
in the invoking shell's own command line matches ITSELF. A queued job that waited on
`pgrep -f "paper_numbers.py"` deadlocked forever, and a later `pkill -f "tmp/daseq.sh"`
killed the shell issuing it. Never pattern-match a string that appears in your own
command line.

Related: [[scib-bio-drop-mechanism]], [[stratified-mmd-design]], [[anchor-lambda-tradeoff]].
