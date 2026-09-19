---
name: lung-ipf-feasibility
description: Lung/IPF fibrosis as the next strong-signal validation tissue — data landscape, why Census alone is insufficient, and the HLCA-native-h5ad route being pulled (gated on >=3 paired studies).
metadata:
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
  modified: 2026-07-23T17:08:02.969Z
---

2026-07-23. Vlad picked LUNG + IPF/pulmonary fibrosis as the next validation: a strong-signal
compositional disease in a NEW organ system (gut/CRC, blood/COVID, brain/dementia already done), to
show the lever works "where the nature of the diagnosis permits" -- the clean positive counterweight
to blood's equivocal result. Chose fibrosis over lung cancer to widen the disease CLASS beyond cancer.
He rejected Census-Crohn's because gut+CRC is too anatomically close.

**DATA LANDSCAPE (from data/census_paired_scan.csv + live Census obs, cached data/lung_obs.parquet).**
The scan machinery: scripts/census_paired_scan.py (Arrow-only, cgroup-capped). Census 2025-11-08.
- Lung has 45 datasets / 1053 donor-arms. Disease labels present: pulmonary fibrosis (55 donors),
  interstitial lung disease (61), NSIP (7), hypersensitivity pneumonitis (6), sarcoidosis (6) +
  cancers + COVID + COPD + cystic fibrosis.
- **BUT fibrosis/ILD is concentrated in only 2 Census dataset_ids with a normal arm:**
  (1) 9f222629 = **the integrated Human Lung Cell Atlas (HLCA) "full"**, 2.28M cells, 8 assays,
      73 fibrosis + 164 normal donors; (2) f14bc322 = an ILD-subtypes study, 472k cells, 1 assay,
      64 fibrosis + 42 normal. Two studies is too few for held-LOSO (need >=3: 2 for the core + held).
- Assay-as-batch fails: within HLCA, fibrosis+normal co-occur only on 10x 3' v2 (50 fib/60 nor);
  other assays are single-class -> assay is itself confounded with disease, can't be the held axis.
- **The standardized Census obs schema DROPS the study-of-origin field** (only dataset_id/donor_id/
  assay/disease/cell_type...). HLCA is an INTEGRATION of ~14 contributing studies; that internal
  `study`/`dataset` label -- the batch axis we need -- lives ONLY in the native submitted h5ad.

**DECISION (autonomous, Vlad away): pull the native HLCA full h5ad (20.4 GB) to the cloud, read its
obs backed, and GATE on >=3 fibrosis+normal-paired contributing studies before building anything.**
This is ONE self-contained, gene-harmonized, cell-type-annotated atlas file (the lung analogue of the
CRC atlas) -- NOT the fragile multi-GEO assembly ([[external-pca-mmd-plan]]). scripts/hlca_fetch_
inspect.py does download (cc.download_source_h5ad, dataset 9f222629) + the study x arm donor report ->
data/hlca_study_report.txt. Cloud needed cellxgene-census installed (was absent). Runs detached under
MemoryMax=20G. If gate PROCEEDs: build lung corpus like blood_build.py (subsample fib+normal donors,
<=400 cells/donor, raw counts -> HVG -> IncrementalPCA -> signatures -> cell_types from the atlas
annotation), register a "lung" tissue in harmony_sweep CONFIG + crc_scvi TISSUE_CFG, then run
crc_statrun.py --tissue lung (same scVI stat design as CRC/blood).

**RISK FLAGGED:** even with the native study field, HLCA's fibrosis likely comes from only a few
contributing ILD studies -> may still be thin (blood-like, ~3-4 held) and could come out
underpowered/heterogeneous. If the gate shows <3 paired studies, STOP and report -- do not force it.
The honest fallback finding may be: strong-signal + multi-paired-study is genuinely rare outside the
curated CRC atlas ([[paired-study-scarcity-is-a-cancer-problem]]).

**BUILT + RUNNING (2026-07-23).** Gate PASSED: HLCA native h5ad has `study` (5 fibrosis+normal studies:
Kaminski_2020 32/28, Banovich_Kropski_2020 20/38, Misharin_Budinger_2018 9/8, Sheppard_2020 5/3,
Schiller_2020 3/11) + a big normal reservoir. Raw counts in .raw.X (integer), Ensembl var, ann_level_*
cell types. Corpus built (scripts/lung_build.py, resume-checkpointed; skmisc needed for HVG seurat_v3):
data/lung/ = 292 donors (73 fib/219 normal), 114,905 cells, 2000 HVG, own PCA, leak-free signatures,
cell_type_coarse_lung from ann_level_2. Registered tissue "lung" in harmony_sweep CONFIG + crc_scvi
TISSUE_CFG + crc_statrun TISSUES.
- **ASYMMETRIC design (new machinery in crc_statrun, CORE_MODE="asym"):** fibrosis is study-scarce,
  normal abundant, and the both-arm retrieval reservoir crc/blood use barely exists -> core = fibrosis
  from study A x normal from study B (different study sets), rho=1.0 ONLY (rho<1 needs both arms per
  study). refs = K fibrosis medoids from diverse fibrosis-studies + K normal medoids from diverse
  normal-studies (build_refs_asym). 5 held x 2 cores x 4 arms x 2 seeds = 160 jobs. K=6 yields ~10 refs
  (only ~4 fibrosis source studies available -> effectively asymmetric K).
- **YELLOW FLAG from the smoke (1 config, Kaminski held, seed0):** noref 0.601, **samebatch 0.903**,
  quantile 0.682, random 0.710 -> paired < samebatch, i.e. DECONF would be NEGATIVE here. Either n=1
  scVI seed noise OR a real asym effect: more same-batch data -> better scVI integration -> better
  transfer (a data-volume win that samebatch is meant to catch as the null). If it holds over the full
  run, the honest cross-tissue conclusion is "on lung/IPF under the asym design, data-volume can beat
  diverse-retrieval de-confounding" -- a NEGATIVE for the lever on this tissue. Full 160-job run
  launched overnight (run_lung.sh) to average it out; analyze paired-vs-samebatch per held with the
  cluster bootstrap (sweep_stats-style). DO NOT spin a celebratory artifact until the sign is known.
- caveat for comparability: lung uses asym refs (not the paired refs of crc/blood) + rho=1 only + K
  effectively ~5 -> results are comparable WITH A CAVEAT, not apples-to-apples.

**RESULT (2026-07-23, 80 jobs, data/lung_statrun_results.parquet). The smoke YELLOW FLAG was n=1
noise -- averaged, the pattern is CORRECT (paired > samebatch), POSITIVE but HETEROGENEOUS like blood.**
Arm means: noref 0.621 | samebatch 0.589 | quantile 0.772 | random 0.779 (paired ~0.77 >> samebatch).
Per-held DECONF (paired-samebatch), minority-weighted, cluster bootstrap:
  Misharin +0.330, Schiller +0.383, Sheppard +0.242 (STRONG, paired reaches 0.85-0.98 near ceiling
  -> fibrosis IS strongly readable, the premise holds); Banovich +0.019 (flat); Kaminski -0.043 (neg,
  the study whose smoke samebatch=0.903 fluked). 4/5 positive, median +0.24.
- **DECONF weighted = +0.082, CI [-0.011, +0.332] CROSSES 0** -- because the TWO highest-weight held
  (Banovich, Kaminski, minority=18) are flat/negative and drag the precision-weighted mean; the big
  effects (Misharin/Schiller/Sheppard) carry low weight (minority 3-8). Same heterogeneity failure mode
  as blood: works big on most held, a couple don't cooperate, n=5 held too few -> CI crosses 0.
- **TOTAL (paired - noref) = +0.071, CI [+0.015, +0.235] EXCLUDES 0, 4/5** -- so retrieval DOES beat
  the no-reference baseline overall; it's the vs-samebatch contrast that's underpowered here.
- DATAQ (samebatch-noref) -0.011, crosses 0 -- control clean. scIB clean: batch_mix 0.11->0.25 (doubles),
  celltype_purity flat 0.94 -> batch mixed, biology preserved, NO over-correction (as CRC).
- **HONEST HEADLINE:** lung/IPF confirms the STRONG-SIGNAL premise (paired reaches near-ceiling where
  it works, big +0.24..+0.38 gains) but NOT a clean tight-CI win -- the vs-samebatch CI crosses 0,
  heterogeneous exactly like blood. Cross-tissue picture now: CRC clean (+0.15, tight CI) >> lung/blood
  positive-but-heterogeneous (big effects, wide CI, a couple of non-cooperating held) >> brain null.
  The wall is n_held (5-7), not the biology -- fibrosis signal is strong (ceiling high), the method
  works where headroom exists (low samebatch/noref -> big recovery), but 5 held can't give a tight CI.
  Do NOT claim lung as a clean replication; claim it as "large effects where the disease is readable,
  underpowered at the held level like blood." Artifacts (80 npz) on cloud for UMAP if wanted.
