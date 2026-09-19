---
name: enrichment-heldout-result
description: "2026-07-03 central result: diversity-enrichment of a fixed-size training cohort monotonically improves cross-study diagnosis transfer (AUC 0.52->0.75) while negctrl stays flat at 0.5; on scSet/scVI embeddings."
metadata:
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
---

2026-07-03. First clean positive result for the platform thesis, run cheaply in scVI-scSet space (path C, staying in Census). Script: `scripts/heldout_enrichment.py`. Corpus = 710-patient scVI corpus, scSet re-pretrained on all 710 (val 0.085), re-embedded to 256-d `data/scset/patient_emb.parquet`.

**Design:** held-out = the TWO paired lung datasets (both adeno & normal in one study, so batch is shared across classes and can't be the discriminator): `576f193c` (13 adeno / 4 normal) and `9f222629` (3 adeno / 71 normal). Never trained on, donor+assay dedup vs pool. AUC computed **per study then macro-averaged** (must NOT pool them — diagnosis correlates with study, pooling reintroduces the confound). Training cohort FIXED size (25 adeno + 25 normal), seeded homogeneously from one technology (10x 3' v2), then homogeneous members are progressively **replaced** (k=0..20, size held fixed) by patients selected from OTHER technologies. 30 seeds.

**Result (macro-AUC, k=0 -> k=20):** quantile q=0.75 **0.516 -> 0.747** (best); random 0.516 -> 0.727; greedy_farthest 0.516 -> 0.674 (plateaus low — chases outliers, confirms [[quantile-sweet-spot-result]] and [[greedy-picks-hardest-to-integrate]] on new data); **negctrl (shuffled labels) flat ~0.50 at all k** (proves the rise is real biology, not a data-volume artifact). Homogeneous single-tech cohort barely transfers (0.52 ~ chance).

**Strong claims:** (1) diversity dose-response is real; (2) farthest < random (HOW you diversify matters). **Weak/not-yet-significant:** quantile beats random only by ~0.02 at sd~0.08 — need more seeds + bootstrap CI. Held-out still small (two imbalanced studies).

**Per-study breakdown (macro HIDES asymmetry — always report per-study, not macro).**

**REVISION (2026-07-03, later same day): the clean "diversity helps" story does NOT survive a 3rd study.** Broadened POS to tumor-vs-normal (any lung carcinoma; MALIG set) to unlock more paired held-out studies — but Census lung is CAPPED: only 3 paired studies exist (576f193c, 9f222629, b617ee1b), NONE with minority-class >=10. Pulled 4 extra tumor patients into 9f222629 (scripts/pull_heldout.py, tumor 4->8). Corpus now 714, re-embedded. 100-seed run:
- **576f193c** (21 tumor / 4 normal): homogeneous start FAILS (0.37, anti-correlated) -> enrichment HELPS -> 0.68.
- **9f222629** (8 tumor / 71 normal): 0.70 -> 0.81, helps.
- **b617ee1b** (6 tumor / 3 normal, narrow single-lab 10x): homogeneous start WORKS (0.64) -> enrichment HURTS, monotone decline in ALL arms (quantile 0.64->0.38, farthest 0.64->0.25 worst, random 0.64->0.36), crosses <0.5 by k~4-5. negctrl flat ~0.50 => the decline is REAL, not noise.

**Key honest conclusion (v1): enrichment is a TRADE-OFF, not a universal good** — effect flips with whether the homogeneous base cohort already matched the target. Superseded by the ceiling reframe below.

**GAP-TO-CEILING REFRAME (2026-07-04) — the correct metric and the real story.** Absolute cross-study AUC is uninterpretable because the homogeneous baseline is batch-confounded (diagnosis≈dataset: base cohort draws tumor & normal from DIFFERENT studies, so it learns a batch axis, not biology — this is why baselines can sit <0.5, a systematic flip, NOT a bug: within-study LOO-AUC is 0.95/0.98/0.61 so biology IS separable & embeddings are fine). Fix = measure each study's WITHIN-STUDY LOO-AUC ceiling (batch held constant -> isolates that study's own biology axis) and ask if enrichment closes the train->test gap to it. `scripts/heldout_enrichment.py` now computes ceilings + a gap table. Result (quantile, 50 seeds):
- 576f193c ceiling 0.95: gap 0.59 (k=0, baseline anti-transfers 0.37) -> 0.28 (k=20). CLOSES ~half.
- 9f222629 ceiling 0.98: gap 0.28 -> 0.16. closes ~40%.
- b617ee1b ceiling 0.61: gap **0.01** (k=0 baseline 0.60 already AT ceiling — b617 is 10x like base tech, transfers with no batch penalty) -> 0.22. enrichment OPENS gap 0.21 (nothing to close, only drift; ceiling noisy on 9 patients).

**Classifier-robustness check (2026-07-04, `scripts/clf_check.py`): the whole picture is NOT a linear-probe artifact.** Tested linear LogReg vs poly2 (PCA20+deg2) vs RBF-SVM vs MLP(64) vs RandomForest, on the k=0 baseline and k=20 quantile-enriched cohorts. The <0.5 baseline on 576f193c gets WORSE with more capacity (0.36 linear -> 0.25 RBF): higher-capacity models grab the batch axis harder, so linear was the most CONSERVATIVE choice, not the weak link — polynomial/kernel do NOT fix the confound, they deepen it. Enrichment's gap-closing on 576f193c holds in every classifier and is LARGER for nonlinear (RBF 0.25->0.74, closes 0.49 of 0.59). b617ee1b gives negative "closed" in all 5 (no gap to close). Secondary hint: on de-confounded (enriched) data, higher-capacity models transfer best (RBF 9f222629 @k=0 already 0.95≈ceiling) — capacity helps only AFTER de-confounding; caution on 50-patient cohorts.

**Corrected thesis: enrichment is a targeted DE-CONFOUNDING tool — it closes the transfer gap exactly where one exists (batch-confounded baseline: 576f193c, 9f222629) and is useless/slightly harmful where the baseline already reaches the biological ceiling (b617ee1b).** Not a random trade-off. quantile >= random > farthest on both studies with a real gap (farthest chases outliers, worst). This is the defensible framing. Note the subtype-heterogeneity caveat (POS lumps adeno+squamous+small-cell etc.; subtype correlates with study so it adds entanglement) — next rigor level would model per-subtype instead of lumping into "tumor".

Next: more seeds + bootstrap CI to test quantile-vs-random; sweep q; final "hold out one whole technology" test (GEXSCOPE or BD Rhapsody — droplet, NOT Smart-seq2 which is fundamentally different); later the [[scset-pivot]] compressed-batch(scVI) vs uncompressed-batch(Geneformer) contrast now justified as a positive follow-up. Related: [[pivot-to-diagnosis-data]], [[project-direction]].
