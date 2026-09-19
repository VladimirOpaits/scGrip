---
name: brain-boundary-result
description: "Brain replication FAILED on the biology side, cleanly — dementia in cortex is barely readable from the donor nuclear transcriptome (ceiling 0.5–0.74 vs CRC 0.88). It's a boundary of the method, not a bug."
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
---

2026-07-15. Ran the brain replication ([[brain-replication-design]]) to the ceiling check and
stopped there. Verdict: the LEVER cannot be shown in cortex because there is no biology to recover.

Setup was clean and confound-controlled: cortex only (region fixed), 10x 3' v3 only (tech fixed),
single coherent disease axis (dementia/Alzheimer vs normal — Parkinson/ALS dropped because a
PD-trained model tested on AD measures disease-transfer, not batch), leak-free pool-only PCA.

Within-study ceiling (batch fixed -> pure biology), RFF-MMD signature, C tuned by nested CV:
  6f7fd0f1 0.74 | 37a17b78 0.62 | 5e57cd50 ~0.50 (permutation: real 0.482 sits in null band
  [0.33,0.66] -> genuinely no signal, not a fixable confound). CRC ceiling was 0.88.

Ruled OUT as causes (so the low ceiling is real, not method error): under-regularization (C swept +
nested CV, envelope still 0.53/0.62/0.74); representation (whole-donor AND per-compartment RFF-MMD,
incl. microglia — all weak); mixed label (single disease); region (fixed to prefrontal family, incl.
Brodmann-9/46 aliases); transductive leakage (pool-only build).

**Why:** neurodegeneration is a subtle shift in cell STATE at nearly constant composition; a mean
kernel embedding over a donor's nuclei averages it toward zero. Tumor (CRC) is a compositional
OVERTURN (malignant cells appear, immune influx) — trivially visible. So the lever de-confounds
where disease is compositionally expressed and has nothing to recover where it is a whisper.

**How to apply:** brain is a characterized NEGATIVE CONTROL / boundary case for the paper, not a
failed experiment — pair it with CRC to show *when* the method helps. Next tissue for a POSITIVE
mechanism replication should have a composition-shifting disease: blood/COVID (lymphopenia, monocyte
influx; 18 paired studies in [[paired-study-scarcity-is-a-cancer-problem]]). Data is built and
reusable: data/brain/ (counts_hv.npz, pca50.npy, signatures, cell_types). Do NOT re-pull.

AGGRESSIVENESS FINDING (confounded-start logreg-transfer enrichment, leak-free PFC signatures,
averaged over held x confounder x seed, ~24-36 configs): on brain, **farthest (q=1.0) > random >
quantile**, and farthest even beats q=0.9. This REVERSES the CRC/lung result
([[enrichment-heldout-result]]: quantile>=random>farthest) — and NOT because of Harmony (Vlad
corrected me: CRC's final conclusion was the SAME logreg-transfer, integration was only a side
probe). Leading hypothesis: **assay homogeneity sets the optimal aggressiveness.** CRC/lung are
multi-assay -> farthest overshoots into toxic extreme-technology outliers (Smart-seq2/inDrop) that
transfer poorly -> moderate quantile wins. Brain is single-assay (10x 3' v3 only) -> no tech
outliers -> maximal diversity is safe and just reaches the plateau fastest. farthest confirms the
EFFICIENCY/speed story (same endpoint, ~3x fewer donors); the peak-then-flatten is reservoir
exhaustion, not "excess harms". Effect is small & noisy on brain (biology weak). Candidate robust
default q~0.9 (near-farthest for homogeneous, pulled-back for heterogeneous) — UNTESTED, needs the
same q-sweep re-run on CRC (multi-assay) to confirm 0.9 sits near the top in BOTH regimes.
