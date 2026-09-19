---
name: blood-covid-positive
description: "Blood/COVID is the POSITIVE second tissue — strong compositional biology (ceiling 0.85) AND a real transfer gap (0.12), the regime brain lacked. The pivot worked."
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
---

2026-07-16. Pivoted from brain ([[brain-boundary-result]]) to blood/COVID because COVID is a
COMPOSITIONAL disease (lymphopenia, monocyte influx) that a mean-kernel patient signature can see.
Corpus built: data/blood/ — 5' family (10x 5' v1/v2/5'-transcription; 3' excluded because COVID is
5'-native and healthy atlases are 3'-heavy -> mixing would confound disease with technology), 814
donors (427 COVID / 387 normal), 21 studies, own PCA on raw counts. Immune compartments clean
(Monocyte 24%, CD4T 23%, CD8T 19%, NK 12%, B 9%; Other 1.1%). Held = 5 paired COVID/normal studies
on 5' v1; pool = 16 studies / 532 donors. Leak-free (pool-only PCA/sigma/landmarks).

Within-study ceiling (COVID vs normal, C tuned in-fold) vs baseline transfer from the confounded
pool, leak-free signatures:
  21d3e683 ceil .911 base .685 gap .226 | 30cd5311 .998/.858/.140 | ebc2e1ff .987/.904/.083
  242c6e7f .778/.689/.089 | 2a498ace .564/.491/.073
  mean ceiling 0.847, baseline 0.725, gap 0.122; d_batch(eta^2)=0.341.

**Why this matters:** BOTH preconditions the lever needs are met, unlike brain. (1) biology is
strong (0.85, near CRC's 0.88, vs brain 0.6) -> composition-shift disease is readable; (2) a real
baseline<ceiling gap exists (0.12) -> the pool is confounded, so there is something to de-confound.
2a498ace is the null case (ceiling 0.56 ~ chance) — keep as a NEGATIVE CONTROL (a study where
biology is absent so the lever should do nothing), the blood analogue of CRC's b617ee1b.

**How to apply:** preconditions met but NOT YET the lever result — next is the enrichment
experiment (does retrieval close the gap toward ceiling?) + the aggressiveness q-sweep. Blood is
single-5'-family (relatively homogeneous like brain), so the [[brain-boundary-result]] hypothesis
predicts farthest/high-q should win here too — but strong biology may change the picture; test it.
Scripts: blood_design/build/celltypes/leakfree.py. d_batch 0.34 < brain 0.56 < CRC 0.68 (fewer
assays = less batch), but the gap exists, so confound is real. Do NOT re-pull; data/blood/ is built.