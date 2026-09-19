---
name: scvi-lever-result
description: The retrieval de-confounding lever on scVI — now CONFIRMED with its own cluster-bootstrap CI (11 held x 3 cores x rho 0.5/1.0): +0.152 at rho=1, +0.047 at rho=0.5, both exclude 0; scIB shows no over-correction; artifacts saved for UMAP/geometry.
metadata:
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
  modified: 2026-07-23T15:13:48.870Z
---

2026-07-18. Phase pivot after maxing out unsupervised Harmony ([[retrieval-augmented-harmony]]):
move the downstream to **scVI** (scvi-tools 1.4.2, the de-facto standard scRNA integrator) because
(1) if the lever works there it's a contribution to the pipeline everyone uses, not a Harmony trick;
(2) scVI has INDUCTIVE reference-mapping (load_query_data / scArches surgery, native — no separate
scarches pkg) so held patients are PROJECTED into a reference latent without retraining -> closes the
transductive caveat we discussed (Harmony has no out-of-sample transform); (3) fits the product
vision: a mini-lib that augments your scVI reference with diagnosis-matched cross-batch patients.
Vlad chose reference-mapping straight away (not the throwaway joint-train replication).

Protocol (scripts/crc_scvi.py, reuses harmony_sweep data/core/refs/selection machinery): train scVI
on the reference cohort = confounded core (25 disease from study A + 25 normal from study B) + K
retrieved refs, batch_key=study, n_latent=10; then prepare_query_anndata + load_query_data to map the
held study's cells; RFF-MMD patient signatures on the concatenated scVI latent -> logreg(C=0.01) on
core+refs -> transfer AUC on held. Raw counts confirmed integer (counts_hv.npz, median lib 2518).
Same arms + samebatch control as Harmony.

**SMOKE RESULT (1 config, held=Chen_2024, DEM=Joanito, NOR=Lee, K=6, seed0) — LEVER REPLICATES,
STRONGER THAN HARMONY:**
  raw(PCA) 0.725 | scVI-noref 0.432 | scVI-samebatch 0.442 | scVI-coverage(paired) 0.755
- scVI OVER-CORRECTS the confounded 2-batch core HARDER than Harmony: noref 0.432 is BELOW chance
  (Harmony-noref was ~0.70). The confound flips the signal under scVI's stronger batch model.
- samebatch (more data, same 2 batches) stays stuck at 0.442 -> control behaves (data-quantity ~0).
- paired refs (coverage) RESCUE it to 0.755, above raw. **De-confounding effect (coverage-samebatch)
  = +0.313**, vs Harmony's ~+0.13. More over-correction -> more headroom -> bigger recovery.

Engineering: ~155s per scVI arm (200 ref + 100 query epochs) on RTX 4050; GPU barely used (0.22GB /
6GB -> epochs are the cost, not memory; can raise batch size / cut epochs, ref had early_stopping,
query fixed 100). raw arm 3.6s (no GPU). Mini-sweep (~3 held x 2 core x ~6 arm-seeds ~= 36 configs)
projected ~1.5h.

**How to apply / NEXT:** this is ONE config, ONE seed -- promising, NOT yet statistics. Wire the
mini-sweep in crc_scvi.py (multiple held x core x arm x seed, resumable parquet data/crc_scvi_results
.parquet) and run the SAME precision-weighted stats (scripts/sweep_stats.py adapted) to confirm the
paired>samebatch lever holds on scVI across held. If it does, scVI is the paper's headline downstream
(standard method + inductive protocol closes the transductive caveat + product framing). scANVI
(semi-supervised, disease-label-aware) is the later "even better" arm. Tune epochs down first (200
ref likely overkill given tiny GPU use and early_stopping) to make the sweep cheaper.
[DONE 2026-07-18: mini-sweep wired + running, tuned batch=512/ref150/query80 -> ~70s/arm, ~2x
faster; noref 0.52 / coverage 0.67 with tuned settings on first config = lever still holds, milder
gap because fewer epochs -> less over-correction.]

**FULL MINI-SWEEP (2026-07-18, crc_scvi.py, 3 held x 14 cores x 2 scVI-seeds, K=6, 140 rows,
data/crc_scvi_results.parquet). scVI-seed now = job seed so deterministic-selection arms get error
bars too.** THE LEVER REPLICATES ON scVI, SOLIDLY; the ordering sub-claim from the thin n=3 was an
artifact, RETRACTED:
  raw 0.675 | noref 0.715 | samebatch 0.713 | coverage 0.848 | random 0.849 | quantile 0.866
- **De-confounding (paired - samebatch) = +0.144** robust (on low-noref/high-confound configs reaches
  +0.24..+0.3, e.g. smoke Chen +0.31). **data-quantity (samebatch - noref) = -0.002 ~ 0** -- control
  clean, replicates Harmony. So "pairs de-confound, not data volume" holds on scVI, and STRONGER than
  Harmony (+0.144 vs +0.13; scVI over-corrects the confounded 2-batch core harder -> more headroom).
- **ORDERING IS SECONDARY (matches Harmony n=11, RETRACTS the n=3 "quantile best / coverage worse"):**
  coverage 0.848 ~= random 0.849 (coverage<random exactly 7/14 = 50%, mean diff -0.001 -- the earlier
  "coverage worse than random" was pure n=3 noise, Vlad flagged it as suspicious and was right).
  quantile +0.017 over random (10/14) but that is WITHIN run-to-run seed noise (random seed-std 0.028
  > 0.017), so not a robust edge. Selector choice doesn't matter once refs are paired, same conclusion
  as [[retrieval-augmented-harmony]]. Run-to-run scVI-seed noise ~0.014-0.028 std/config (small).
- note: aggregate noref 0.715 ~= raw 0.675 (scVI-noref does NOT systematically crash over 14 cores,
  unlike the single smoke config where it hit 0.43 -- over-correction severity is config-dependent,
  worst on the most-confounded cores). paired arms ~0.85-0.87 clearly above both baselines.
The reboot mid-sweep cost nothing (resumable parquet, flush every 6 jobs -> 84/140 survived, resumed).

**CONFIRMATORY STAT RUN — scVI now has its OWN CI (2026-07-22, scripts/crc_statrun.py,
data/crc_statrun_results.parquet, 528 rows).** Vlad's call: the real bottleneck was statistical
validity, so expand the two axes generalization rests on. Design = 11 held (all CRC paired studies
minority>=5, SAME set Harmony used -> directly comparable) x 3 DISJOINT cores per held x rho in
{0.5, 1.0} x arms {noref, samebatch, quantile, random} x K=6 x 2 seeds. Unit = held; minority-weighted
mean; cluster bootstrap over held (20k).
KEY INSIGHT that made it cheap (~6h not ~90h): SEPARATE the descriptive from the confirmatory.
crc_surface.py's cost is the 15-cell rho x n grid, which forced it to 3 held AND a DETERMINISTIC core
(one per held) -- too thin to carry a claim. The stat run DROPS the grid and spends the budget on
held + core instead. Keep crc_surface as the descriptive benefit MAP; crc_statrun as the test.
- **DECONF (paired[quantile,random] - samebatch): rho=1.0 -> +0.152, CI [+0.106, +0.211], 10/11 held
  positive. rho=0.5 -> +0.047, CI [+0.021, +0.086], 10/11.** BOTH exclude 0. So the lever is NOT an
  artifact of the degenerate fully-confounded regime -- it weakens smoothly as the confound weakens,
  which is what a genuine de-confounding mechanism should do. Magnitude matches Harmony (+0.152 vs
  +0.134 at rho=1, intervals overlap) -> same mechanism on a linear and on a deep integrator.
- **HONEST CAVEAT — DATAQ is NOT null at rho=0.5:** samebatch-noref = +0.024, CI [+0.010, +0.042],
  EXCLUDES 0 (at rho=1.0 it's clean: +0.008, crosses 0). Reading: under partial confound, adding more
  same-batch donors itself partly dilutes the imbalance. De-confounding still dominates ~2x
  (0.047 vs 0.024). Same pattern blood showed under Harmony. MUST be stated in the paper.
- Wu_2022_Cancer_Discov is the ONLY negative held (-0.019, negative at both rho) and its TOTAL
  (paired-noref) is just +0.014 -> looks like a no-headroom held (baseline already fine), the CRC
  analogue of the b617ee1b/2a498ace null cases. Not yet diagnosed.
- **scIB (first over-correction check ever run for scVI; harmony_sweep._eta2/_knn_scib reused so
  numbers are directly comparable to Harmony): celltype_purity FLAT across arms (0.924-0.930) while
  batch_mix DOUBLES with refs (0.12 -> 0.25). Batch mixes, cell biology preserved -> no
  over-correction on CRC.** Caveat to pre-empt: batch_eta2 is slightly HIGHER for ref arms
  (0.116-0.127 vs noref 0.111) -- not a contradiction, refs ADD new studies to the pool so
  between-study variance mechanically grows; kNN batch_mix is the honest metric here. Say this
  explicitly or a reviewer will read it as the batch getting worse.

**ARTIFACT HARVESTING (Vlad's point: training is the expensive step, so take everything off the
fitted model).** crc_statrun saves per job to data/statrun_art/*.npz (528 files, 108MB):
patient RFF signatures + classifier w + scaler + held probabilities (ALL jobs -> patient UMAP,
w-rotation geometry, ROC/calibration without retraining), plus the FULL cell latent +
per-cell study/cell_type for a subset (core_idx=0, seed=0 -> 88 jobs) for cell-level UMAPs.
Implementation note: hs._rows() is deterministic (rng seeded 0), so per-cell study/ct are recovered
by re-calling it rather than changing scvi_latent's signature -- crc_scvi/crc_surface untouched.
Deliberately on scVI, NOT scANVI: scVI never sees the diagnosis, so class separation in its latent
is a CONSEQUENCE of de-confounding; in scANVI the latent is fitted to a label classifier, so a pretty
UMAP would be partly an artifact of the setup.

**OPS (cost real time twice).** Long scVI sweeps OOM'd the box: ProcessPoolExecutor never recycles
workers, each grew to ~14.85GB anon-rss (confirmed in the serial console), 2 workers > 29GB.
py3.10 on the GCE box has no max_tasks_per_child -> use multiprocessing.Pool(maxtasksperchild=20)
instead, plus systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 so a leak kills the
scope, not the machine. After the fix the 524-job run finished with exactly 1 scope start, 0 restarts.
Also: SSH to the box drops (rc 255) while 2 workers run even at 5/29GB RAM (8 vCPU oversubscribed) --
launch detached with setsid + file markers, poll rarely with retries; never drive a run interactively.

**BLOOD stat run (2026-07-23, crc_statrun.py --tissue blood, data/blood_statrun_results.parquet,
224 rows). scVI lever is EQUIVOCAL on blood — does NOT reach significance, unlike CRC.** Design: 7 held
(all blood paired minority>=5) x 2 disjoint cores/held x rho{0.5,1.0} x {noref,samebatch,quantile,
random} x 2 seeds, n=20. Ran clean overnight (1 scope start, 0 OOM). Same engine, tissue-parametrized
(crc_scvi.init(tissue), crc_statrun --tissue; crc path unchanged).
- **DECONF: rho=1.0 -> +0.083, CI [-0.023, +0.251] CROSSES 0 (n.s.); rho=0.5 -> -0.002, crosses 0.**
  5/7 held positive at rho=1 but the distribution is BIMODAL and heavy-tailed: strong positives
  30cd5311 +0.443, 01ad3cd7 +0.263, ebc2e1ff +0.186; but TWO negatives 9dbab10c -0.098 (weight 25,
  the biggest) and 21d3e683 -0.042. Unweighted mean rho=1 is +0.125 and median +0.092 (both positive),
  but the high-weight negative 9dbab10c drags the precision-weighted mean to +0.083 and the n=7 cluster
  bootstrap CI crosses 0. So: lever works on MOST blood held (some hugely) but reverses on a couple,
  and 7 held isn't enough to call it.
- Headroom pattern MOSTLY holds: the big-positive held have low noref (~0.46-0.48, strong
  over-correction -> big recovery); 9dbab10c has higher noref (0.58, less headroom) and goes negative.
  EXCEPTION: 21d3e683 has low noref (0.45) yet DECONF negative -- not explained by headroom alone.
- **Contrast with Harmony-blood, which WAS significant (+0.035, CI [0.013,0.056], p=0.001,
  [[retrieval-augmented-harmony]]).** On scVI blood does NOT reach significance. Likely causes: scVI's
  higher per-config variance (deep model) widens CIs; only 2 cores/held + n=20 (blood has fewer paired
  studies than CRC); and genuine per-held heterogeneity. NOT a pipeline bug -- smoke + scIB behave.
- DATAQ (data-quantity control) NULL everywhere on blood scVI (rho=1 -0.014, rho=0.5 +0.014, both cross
  0) -- CLEANER than Harmony-blood where DATAQ was significant +0.006. So on scVI the samebatch control
  is clean; the issue is effect heterogeneity, not a data-volume confound.
- scIB fine: batch_mix rises with refs 0.18->0.29, celltype_purity flat 0.82 -> batch mixes, biology
  preserved (mechanism operates even where transfer AUC doesn't gain). 2a498ace confirmed ~null
  (noref~0.47, the known blood negative-control study).
- **CORES/SEEDS WON'T FIX BLOOD (checked per-core, no new compute).** The two negative held are
  negative on BOTH their cores (9dbab10c: -0.11/-0.14/-0.09/-0.06 across its 4 core-rho cells;
  21d3e683: 3/4 negative) -> consistent held-level heterogeneity, NOT a bad-core artifact. A cluster
  bootstrap CI is set by BETWEEN-held spread; cores/seeds only shrink within-held noise, so they can't
  narrow it. The only real lever is MORE HELD (more clusters) and blood is capped at 7 (8 paired
  studies). Blood is structurally underpowered at the held level -- an overnight core top-up is wasted.
- **Headroom pattern is the signal (more useful than a tight CI):** negative held are exactly the ones
  where scVI-noref is already >=0.5 (9dbab10c noref 0.57-0.66 -> little headroom -> refs mildly
  over-correct and HURT); the big positives are where noref CRASHES at rho=1 (30cd5311 0.49, 01ad3cd7
  0.45 -> strong over-correction -> refs rescue +0.3..+0.55). 17/28 (held x core x rho) cells positive,
  median +0.044. So the lever pays exactly where scVI over-corrects; blood just has several held with
  no headroom to recover.
- HONEST HEADLINE: CRC replicates on scVI with a tight CI; blood does NOT (equivocal, heterogeneous).
  This tracks the cross-tissue gradient (CRC>blood) already seen under Harmony -- effect size follows
  disease readability. Next options to make blood callable: (a) diagnose the 2 reversing held
  (9dbab10c, 21d3e683 -- over-correction? cell-type composition?); (b) add cores/seeds to tighten;
  (c) accept blood as "positive-but-weak/heterogeneous" and report the gradient honestly. NOT built:
  a blood artifact -- result too equivocal to visualize as a win.

**TODO PLAN — scVI evaluation modes (2026-07-18, from Vlad's design point).** How held is projected
is not one question but TWO different evaluations; keep both, they answer different things:
1. **Per-study LOSO (current mini-sweep)** = reviewer-facing STATISTICS. held = one study = one batch;
   many such folds, precision-weighted (minority-size) mean of DECONF (paired - samebatch). This is
   where the "lever works on scVI" claim lives. held=one-batch is fine here because we average the
   effect over many folds. Co-mapping that held cohort together is a VALID deployment mode (a user
   with a trained scVI embeds an incoming cohort in a batch) -- NOT a flaw to defend against. The
   per-patient-vs-cohort distinction barely bites when held is a single batch (all held share one
   study batch-offset anyway; only its precision changes).
2. **Multi-study held-pool (NEXT, not built yet)** = the STRONG biology-vs-batch test AND the most
   realistic product scenario. Build held as a POOL of patients drawn from MANY different held studies
   (each patient/study = its OWN batch), project each PER-PATIENT / per-its-own-batch (this is where
   separate projection finally becomes meaningful, because the batches are genuinely different), then
   predict disease ACROSS the heterogeneous pool. Tests whether the scVI latent encodes biology that
   survives integration across many unseen batches -- one held study is too thin to show this.
   Realistic because incoming patients come from varied sources, not one clean study.
Order: let the per-study mini-sweep finish (gives the base "works on scVI" stat), THEN build the
multi-study held-pool probe as a second, stricter layer. The per-patient strict projection is a
SUBSET of building the multi-study pool -- do them together. Deferred robustness item folded in here:
"project each held patient one-at-a-time" only matters for the multi-study-pool case.
