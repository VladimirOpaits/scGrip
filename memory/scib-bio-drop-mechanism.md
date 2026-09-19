---
name: scib-bio-drop-mechanism
description: "Located WHERE the scIB bio-conservation drop from stratified MMD comes from: coarse compartment structure is essentially preserved (T/NK even improve), the loss is within-compartment fine-subtype resolution, uniform across compartments and worst in B cells. Same failure mode as pooled MMD, moved one level down."
metadata: 
  node_type: memory
  type: project
  originSessionId: 4c10a894-795a-443a-9bce-1a5686bd1785
  modified: 2026-08-28T16:25:49.656Z
---

2026-08-28. Broke the headline "bio ASW falls 0.016-0.025" down two ways on all three
cohorts (`scenarios/asw_by_celltype.py` for coarse, a fine-label variant for within-
compartment), comparing base projection -> anchor -> anchor+stratified MMD.

**Coarse compartment ASW barely moves, and the sign is not uniform.** Global
-0.005 / -0.003 / -0.007 (c2/c3/c4). Per compartment, base -> final:
```
             c2       c3       c4
CD4 T     +0.007   +0.015   +0.011     <- lymphoid GAINS
NK        +0.002   +0.011   +0.004
CD8 T     +0.003   -0.002   +0.003
B         -0.011   -0.015   -0.009
monocyte  -0.014   -0.020   -0.016     <- myeloid loses
DC        -0.025   -0.031   -0.018     <- worst
pDC       -0.030   -0.017   -0.005
other     -0.010   -0.015   -0.018
```
Myeloid + catch-all lose, T/NK improve. Consistent with myeloid carrying the largest
cohort-vs-census displacement (census ~10% monocyte vs cohort 35-69%) -- see
[[composition-confound-lupus]], [[anchor-generalization-check]].

**The real loss is one level down: fine subtype structure inside compartments.**
Silhouette on the fine `cell_type` vocabulary, computed separately within each
compartment so cross-compartment distance cannot help:
```
             c2 base -> final    c4 base -> final
GLOBAL_fine   0.5214  0.5077      0.5380  0.5228   (-0.014 / -0.015)
B             0.5285  0.4996      0.5534  0.5230   (-0.029 / -0.030)  <- worst
CD8 T         0.5003  0.4847      0.5210  0.5026
monocyte      0.5424  0.5292      0.5597  0.5420
CD4 T         0.5132  0.5018      0.5280  0.5138
NK            0.5334  0.5226      0.5361  0.5234
```
Every compartment loses here, including CD4 T and NK which GAINED at coarse level.
That reconciles the two numbers: the headline scIB figure uses fine CL labels, so it
reports mostly this.

**Mechanism.** The MMD is stratified at compartment granularity, so inside a
compartment it may match the two distributions by any transport, including one that
blurs subtypes -- compartment-level distribution matching constrains nothing below
compartment level. This is exactly the pooled-MMD failure of [[stratified-mmd-design]]
displaced one level down: pooled MMD damaged BETWEEN-compartment structure
(monocytes ~80% of the damage), stratifying fixed that, and the residual damage moved
inside the compartments.

**Why it is not fixable by stratifying finer.** Census is annotated at fine CL
granularity, the cohorts coarsely; only 7 labels match literally and 2 survive the
filters. The stratification resolution is bounded by annotation compatibility, not by
the method.

**Two things that ARE actionable.**
1. DC and pDC take the largest coarse-level hit and are NOT in the early-stopping
   score (verified: the score covers only B, CD4 T, CD8 T, NK, monocyte -- group B has
   too few DC/pDC cells). They receive gradient with no control.
2. All three REPORTED models (`strat_c2_at1`, `strat_c3_at1`, `strat_c4_at2`) were
   trained with `other` still in the MMD loss; only the later `p2_f0sigma`/`p2_lam10`
   runs exclude it. `other` loses 0.010-0.018 coarse ASW in every cohort. The rerun
   that was already planned should remove this.

**Framing for the paper.** Donor ASW (batch mixing) is flat at 0.876-0.890 across every
arm, because the batch being removed is a block offset of one cohort relative to the
atlas -- a geometry donor ASW is insensitive to. So scIB structurally can show the COST
of the alignment and not its BENEFIT. State that explicitly, or the flat batch column
reads as "the method did nothing" while the bio column reads as pure loss.

Related: [[stratified-mmd-design]], [[anchor-lambda-tradeoff]], [[unsupervised-arc-closed]].

**2026-08-28 follow-up: scIB CAN see our batch if `batch_key` is the BLOCK, not the
donor.** Donor ASW is flat by construction -- our batch is a rigid translation of the
whole cohort block relative to a fixed reference, which moves all cohort donors
together and is therefore invisible to a metric computed on donor labels. Swapping
`batch_key` to census-vs-cohort-control makes `silhouette_batch` measure block
separability within cell type. Labels MUST be coarse compartments (a fine label_key
lets the metric separate the blocks by annotation vocabulary alone) and SLE cells must
be dropped (otherwise disease scores as batch).

```
block-ASW = 1 - |silhouette(census vs cohort)|, within compartment; 1.0 = indistinguishable
              cohort 2: base  anchor  final   d        cohort 4: base  anchor  final   d
mean                    0.8803 0.8865 0.8917 +0.011              0.8323 0.8454 0.8785 +0.046
B                       0.9099 0.9367 0.9549 +0.045              0.8622 0.8931 0.9341 +0.072
CD4 T                   0.9513 0.9651 0.9724 +0.021              0.9072 0.9200 0.9657 +0.059
NK                      0.9252 0.9375 0.9505 +0.025              0.8979 0.9177 0.9476 +0.050
DC                      0.8343 0.8338 0.8431 +0.009              0.7585 0.8036 0.8204 +0.062
monocyte                0.8686 0.8601 0.8657 -0.003              0.8213 0.8203 0.8356 +0.014
pDC                     0.7920 0.7728 0.7631 -0.029              0.7068 0.7009 0.7514 +0.045
```
Monotone across arms, and cohort 4 -- which starts ~2x further out -- gains ~4x more.

**Caveat that must be stated if this is published:** block-ASW with two blocks and
compartment labels is essentially our own stratified offset, normalised and dressed in
standard vocabulary. It answers the "scIB shows nothing" objection but is NOT
independent evidence.

Two diagnoses it produced: **monocytes barely move** (-0.003 / +0.014) despite carrying
the largest offset -- the most stubborn compartment; and **pDC gets WORSE in cohort 2**
(-0.029) while improving in cohort 4 (+0.045) -- pDC is in the loss but not in the
stopping signal, so in cohort 2 it is being damaged without being aligned. Concrete
support for putting DC/pDC into the stopping score or removing them from the loss.

Script: `$CLAUDE_JOB_DIR/tmp/asw_block.py` (scratch; worth moving into `scenarios/` if kept).
