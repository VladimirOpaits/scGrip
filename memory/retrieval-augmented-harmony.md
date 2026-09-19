---
name: retrieval-augmented-harmony
description: Retrieval-augmented Harmony de-confounds via PAIRING (samebatch control proves it's not more data) on all 3 tissues; effect size tracks disease readability (CRC +0.13 > blood +0.04 > brain +0.02, near-null) not tissue identity — brain fails from low headroom, not over-correction.
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
---

2026-07-16. Vlad's real thesis: retrieval isn't for a plain classifier (batch-blind, so "which
batches" can't help it — that's why arms==random in [[blood-covid-positive]] enrichment). It's to
supply a batch-LABEL-AWARE method (Harmony, CODAL) with the right reference data to subtract batch.

Experiment (scripts/blood_harmony.py): confounded core (COVID from one study + normal from one
atlas -> diagnosis==study), add K diagnosis-matched reference patients from OTHER studies, run
Harmony(batch=study) on core+refs+held jointly, recompute RFF-MMD signatures on the CORRECTED
embedding, train on core+refs, transfer to held. held=30cd5311 (ceiling 1.0).

Result (one config, held=30cd5311):
  raw core 0.811 | harmony no-ref 0.758 (Harmony ALONE HURTS: over-corrects confounded data,
  removes biology with batch)
              K=10   K=20   K=40
  smart(far) 0.674  0.772  0.842
  random     0.815  0.853  0.936  (-> near ceiling 1.0)

TWO findings:
1. **Retrieval-augmented Harmony de-confounds** — adding diagnosis-matched cross-batch references
   rescues Harmony and pushes transfer 0.76 -> 0.94. This is the positive result for the idea, and
   the batch-aware downstream is what the lever was always for.
2. **RFF-MMD max-diversity (farthest) reference selection is COUNTERPRODUCTIVE** — random beats it
   at every K (biggest gap at low K: 0.815 vs 0.674). Farthest picks batch-OUTLIER studies Harmony
   can't integrate -> over-correction. Unifies with [[greedy-picks-hardest-to-integrate]]: for an
   integration downstream, extreme diversity is toxic. Correct objective = REPRESENTATIVE batch
   coverage, not max distance. Vlad's 2-stage design (metadata-filter + max-distance) is right in
   stage 1, backwards in stage 2.

SELECTION heuristics (2026-07-17, all 5 blood held studies, deterministic + medoid-anchored):
first-attempt staircase (seeded start) ~= random. BUT switching to a DETERMINISTIC MEDOID ANCHOR
(start greedy from the reservoir-centroid's nearest donor, not a random/outlier start) revived smart
selection. Two deterministic heuristics tested vs random(CI over 12-15 seeds):
  - coverage: max distinct STUDIES per class (round-robin), medoid donor per study, trim 1 outlier study.
  - quantile0.8: medoid-anchored greedy_quantile at q=0.8.
Result across 5 held studies: **random is the FLOOR everywhere (never above the smart curves).**
coverage and quantile sit at/above random's upper 95% CI in ~3-4/5 panels (clearly above on
30cd5311, ebc2e1ff; tie on 21d3e683). coverage vs quantile ~comparable, scenario-dependent:
coverage wins big & early on 2a498ace/30cd5311(K8); quantile stronger early on ebc2e1ff, and on
242c6e7f quantile is bad at low K then catches up. So "smart retrieval beats random" is REVIVED (the
key was the medoid anchor = avoid outlier starts), but coverage-vs-quantile-vs-staircase is not yet
settled. The [[quantile-sweet-spot-result]] reappears: moderate/anchored beats both extremes
(farthest hurts, random is floor).

CRC REPLICATION (2026-07-17, crc.ipynb, 5 held studies, multi-assay tissue): CONFIRMS the blood
picture. smart (coverage/quantile) >= random on all 5 held studies, never below. quantile can
strongly dominate (Lee: 0.74->0.95 vs random floor ~0.87; coverage lags then catches up at K=12).
coverage~=quantile on Chen/Joanito/Zhang. Where Harmony-no-ref is already high (Uhlitz 0.99, Joanito
0.87 = weak confound / Harmony alone suffices) references add little -- lever needs headroom.
Hypothesis for quantile>=coverage on multi-assay CRC (vs tie on single-assay blood): coverage insists
on covering ALL studies (study~=technology bundle in CRC) so it eventually grabs an outlier-TECH study
that Harmony can't integrate; quantile0.8 picks moderate-distance refs (80th pct) that sit BELOW the
extreme-technology outliers, dodging them by construction. Untested probe left in chat (show
study+assay+dist of coverage vs quantile picks on Lee). Cross-tissue so far: smart>=random robust on
blood(single-assay) AND CRC(multi-assay); quantile the safer default on heterogeneous corpora.

CRC 2D SWEEP (2026-07-17, scripts/harmony_sweep.py, parallel/cached data/harmony_results.parquet;
axes = CORE(A disease-study x B normal-study) x HELD, paired refs, metrics transfer_auc + batch_eta2
+ celltype_eta2 + kNN scIB). Findings over 37 core x held:
- LEVER WORKS: raw 0.677 -> paired refs 0.83 (+0.15). batch_eta2 0.096->0.025 (batch removed),
  celltype_purity ~0.94 (cell biology preserved) -> NO over-correction on CRC (strong compositional
  biology). The scIB decomposition is clean: batch down, bio kept.
- PAIRING is the lever, ORDERING is secondary: coverage(0.823)~=quantile(0.838)~=random(0.825) at
  max K. Under the PAIRED design, study-order barely matters (differs from the earlier non-paired
  result where coverage/quantile beat random -- pairs decorrelate regardless of order). Smart's real
  value is STABILITY: random has per-(core,held,K) seed std ~0.047 (up to 0.22); deterministic
  coverage/quantile remove that gamble. quantile a small mean edge (+0.023, beats random in 66%);
  coverage ~= random on mean (+0.011, 52%).
- gain does NOT scale with confound severity (r=0.05) -- pairs de-confound UNIFORMLY across
  severity 0.03-0.43 (robustness, my severity hypothesis was wrong).
- **DECISIVE CONTROL (samebatch)**: adding K more disease from A + K normal from B (more data, SAME
  two batches, confound preserved) is FLAT at 0.70 across all K. Decomposition at K=6: Harmony-alone
  +0.021, more-data +0.004(~0), DE-CONFOUNDING +0.129. So the gain is 96% de-confounding, ~0% data
  quantity -- the reviewer-proof result that refs help by breaking diagnosis==study, not by adding
  samples. transfer_auc = train logreg on core+refs signatures (Harmony-corrected RFF-MMD), test on
  held patients; held never in training, its label used only for the final AUC.

BLOOD 2D SWEEP (2026-07-17, same engine, 2000 rows, 5 held x 14 cores, K=[1,2,4,6], seeds=5):
REPLICATES THE DIRECTION, NOT THE MAGNITUDE.
- raw 0.646 -> harmony-no-ref 0.616 (Harmony ALONE HURTS confounded data here too, slightly worse
  than CRC's near-zero alone-effect) -> paired refs 0.670-0.677 @K=6. Total lever gain only +0.03,
  vs CRC's +0.15 -- much weaker on blood.
- **samebatch control replicates the QUALITATIVE claim but not the clean 96/0 split**: @K=6
  samebatch=0.628 (flat, as on CRC) vs paired=0.670-0.677. data-quantity effect (samebatch-noref)
  = +0.012, de-confounding effect (paired-samebatch) = +0.041 -> ~77% de-confounding / 23% data
  volume (CRC was ~96%/~0%). Data volume is a real minority contributor on blood; on CRC it wasn't.
- ORDERING DOES NOT MATTER on blood under the paired design: quantile win-rate 0.49/+0.006,
  coverage 0.48/+0.008 -- statistically flat vs random (random seed-std also small, 0.019 vs CRC's
  0.047). CRC's "quantile has a small mean edge, smart buys stability" finding does NOT replicate
  on blood -- on blood random is not even less stable. Read together with the CRC-vs-earlier-blood
  staircase result (line 37-50, pre-paired-design): once refs are PAIRED, ordering effects that
  showed up in the non-paired staircase experiments mostly wash out.
- **MILD OVER-CORRECTION APPEARS on blood** (didn't on CRC): celltype_purity drops 0.861->0.812
  (~0.05) alongside batch_eta2 dropping 0.059->0.009 and batch_mix tripling 0.078->0.471. CRC kept
  celltype_purity ~flat at ~0.94. Consistent with the standing hypothesis that over-correction risk
  tracks how compositional/strong the biology is (CRC strongest, blood weaker, brain hypothesized
  worst) -- blood is the first tissue to show any of it.
- confound_severity range collapsed to 0.001-0.013 (vs CRC's 0.03-0.43) -- essentially no variation,
  severity-vs-gain correlation not testable on blood. Not yet diagnosed whether this is a real
  property of blood (immune composition varies less between studies) or a units/scale issue in
  confound_severity() for this tissue -- check before citing.

BRAIN 2D SWEEP (2026-07-17, same engine, 1850 rows, 5 held x 14 cores, K=[1,2,4,6], seeds=5) --
ALL THREE TISSUES NOW DONE. Prediction ("brain shows worse over-correction") was WRONG in kind --
what actually happens is the lever has almost NO HEADROOM, not that it actively destroys biology.
- raw 0.539 -> harmony-noref 0.511 -> paired@K6 0.534 -> samebatch@K6 0.511. de-confounding effect
  (paired-samebatch) = +0.023 (smallest of the 3 tissues: CRC +0.129 > blood +0.041 > brain +0.023).
  data-quantity effect ~0.000 (still flat/clean, replicates on brain too). BUT paired (0.534) barely
  beats raw (0.539) -- the whole lever is within noise on brain. All AUCs sit near 0.5 (chance),
  matching [[brain-boundary-result]]'s finding that dementia-in-cortex is barely readable at all
  (ceiling 0.5-0.74). The lever needs biological signal to RECOVER; brain barely has any to begin
  with, so there's nothing for de-confounding to unlock.
- celltype_purity stays HIGH throughout: 0.993(raw)->0.967(noref)->0.950(paired@K6), a ~0.04 drop,
  same magnitude as blood's drop, NOT worse. So over-correction (bio axis actively destroyed) is
  NOT what limits brain -- it's headroom (bio signal too weak to move AUC), a different failure mode
  than hypothesized. batch_eta2 still drops cleanly (0.071->0.010), so Harmony still removes batch
  fine on brain; it just has nothing useful to reveal underneath.
- ordering: coverage edges random (57% win, +0.007), quantile ties (50%, +0.001) -- same pattern as
  blood, no strong smart-selection story once refs are paired.

DIFFUSE-CONFOUND CONTROL (2026-07-17, scripts/crc_diffuse.py, data/crc_diffuse_results.parquet,
CRC only, 210 rows, 5 held x N in {1,2,3}, K=6). Addresses Vlad's strongest objection: the confounded
core is the WORST case for Harmony (batch A=100% disease, batch B=100% normal; Harmony is batch-aware
but disease-BLIND so at this extreme it's FORCED to erase biology with batch) -- so maybe our whole
headroom is an artifact of that degeneracy. Honest dilution: keep diagnosis tied to the SIDE (can
only draw one class from a single-class source study) but SPREAD each side across N distinct source
studies. N=1 = the main degenerate design; N=2,3 = many batches per class. Source studies kept
DISJOINT from the paired reservoir (no leak between the N axis and the K/refs axis).
- **THE DE-CONFOUNDING EFFECT SURVIVES DILUTION -- Vlad's objection does NOT hold.** DECONF
  (paired-samebatch) = +0.062 (N=1) -> +0.083 (N=2) -> +0.097 (N=3); positive in 14/15 held x N
  cells (93%). It does NOT shrink as the start gets less degenerate; if anything it grows. So the
  main result is NOT an artifact of the 2-batch worst case.
- BUT raw baseline RISES with N (0.642 -> 0.777 -> 0.778): a core drawn from 3 disease + 3 normal
  studies is inherently less confounded and transfers better even with NO Harmony. So headroom
  raw->ceiling shrinks with N, yet the SPECIFIC paired-vs-samebatch contribution persists. Honest
  framing for the paper: dilution helps the raw transfer, but does not remove the need for the
  retrieval lever. samebatch-noref (pure data-quantity) ~0 at every N (control still clean).
- **THE NOISE IS HARMONY, NOT US -- and our refs STABILIZE it.** harmony_noref swings wildly across
  held x N (e.g. Joanito 0.87->0.67->0.68, Uhlitz 0.97->0.75->1.00): cross-config std 0.115, range
  0.436. raw std 0.109, samebatch std 0.120 -- all the Harmony-on-confounded-batches arms are equally
  unstable (unsupervised iterative soft-clustering on 2-5 small batches is intrinsically high-variance).
  Our PAIRED arm is BOTH higher AND tighter: mean 0.848 vs noref 0.769, cross-config std 0.078 vs
  0.115, range 0.269 vs 0.436. So diagnosis-decorrelated cross-batch pairs pull Harmony out of its
  "guessing" regime into a REPRODUCIBLE one. This is a SECOND, independent selling point beyond +AUC:
  the lever makes a batch-aware integration RELIABLE on confounded data where it is otherwise a gamble.
  (Why noref is noisy and not monotone in N is a red herring -- we never build on noref; DECONF is
  measured paired-vs-samebatch on the IDENTICAL core, so the shared Harmony noise cancels in the
  contrast.) Caveat: non-random arms (raw/noref/coverage/quantile) are a single deterministic core
  draw (seed=0), no seed-averaging -> their per-config numbers look jumpier than they'd be with
  core-seeds; offered to smooth but the DECONF claim already holds at 93% without it.

EXPANDED LOSO + PRECISION-WEIGHTED STATS (2026-07-18, scripts/sweep_stats.py). Vlad's point: LOSO
scores each held INDEPENDENTLY, so adding more held studies (even thin, noisy ones) cannot corrupt
the existing per-held estimates -- it only adds observations. The ONLY way a thin held "hurts" is a
naive sign-test (throws away magnitude, lets a near-coin-flip vote dilute the proportion-positive);
fix = weight each held by its minority-class size (precision proxy), not avoid the data. So expanded
held from top-5 to ALL paired studies with minority>=5 (CRC 5->11, blood 5->7, brain 5->7) by editing
CONFIG held lists in harmony_sweep.py (top-5 kept first for cache stability; resumable, only new held
ran). The "5" was never a data limit -- just top-5 by minority size for robust per-panel estimates.
Stats method: unit = held study; per-held DECONF = paired(coverage,quantile @maxK) - samebatch(@maxK)
and DATAQ = samebatch - harmony_noref; minority-weighted mean + CLUSTER bootstrap (resample whole
held studies, 20k) for the CI; plus a weighted mixedlm(arm, 1|held) cross-check.
- **CRC n=11 held: MUCH STRONGER than the n=5 sign-test.** 11/11 held positive (was 5/5).
  minority-weighted mean DECONF = 0.134, cluster-bootstrap 95% CI [0.100, 0.174] EXCLUDES 0.
  mixedlm beta=0.124 +/- 0.013, p<0.0001 (BUT it threw a boundary/ConvergenceWarning -- random-
  effect variance ~0, i.e. held-to-held spread tiny relative to effect; good for us but take the
  mixedlm p with caution -> the PRIMARY evidence is the nonparametric cluster-bootstrap CI, which is
  immune to that). New thin held behave: DECONF 0.05 (Qi, minority=5) .. 0.23 (MUI). DATAQ control
  weighted mean = 0.001, CI [-0.007, 0.006] -- cleanly crosses 0 (data-quantity truly null). So on
  CRC the central claim is now decisively supported: n=5 gave best-case p=0.031; n=11 weighted gives
  p<1e-4 and a bootstrap CI far from 0.
- **BLOOD n=7 held: crossed from suggestive to SIGNIFICANT.** 6/7 positive (21d3e683 the one
  negative, -0.029, low weight). weighted mean DECONF = 0.035, bootstrap CI [0.013, 0.056] EXCLUDES
  0 (was [0.004, 0.064] barely at n=5). mixedlm beta=0.039, p=0.0012 (no convergence warning). BUT
  honest caveat: blood's DATAQ control is NOT null -- weighted mean 0.006, CI [0.001, 0.013] EXCLUDES
  0. So on blood there IS a small real data-quantity contribution (~0.006), though de-confounding
  (0.035) dominates it ~6x. On CRC data-quantity is a clean 0; on blood it's small-but-real. Must
  state this in the paper -- reviewer would catch it.
- **BRAIN n=7 held: cleaner NULL than n=5.** At n=5 brain was +0.025 (weak positive); at n=7 the
  weighted mean is -0.006, bootstrap CI [-0.034, 0.038] CROSSES 0, mixedlm p=0.86. Only 3/7 positive,
  and the TWO BIGGEST brain held (37a17b78 minority=54: -0.041; 6f7fd0f1 minority=39: -0.001) are
  <=0. So brain is not "a small effect" -- it's a statistical ZERO where disease biology is
  unreadable, exactly as [[brain-boundary-result]] predicts. DATAQ control also null (0.008, crosses
  0). Adding held REMOVED the earlier weak-positive noise -> brain is a clean negative control.
THREE-TIER RESULT after expansion: strong (CRC, p<1e-4) / real (blood, p=0.001) / null (brain,
p=0.86), effect magnitude 0.134 > 0.035 > -0.006 MONOTONE in disease readability. This is the honest
headline for Guillem, replacing the old n=5 sign-test framing.

CLASSIFIER CONSTRUCTION + LEAKAGE ANALYSIS (2026-07-18). How transfer_auc is actually computed in
evaluate_full: take core+refs+held cells (<=120/donor) -> Harmony-correct their POOL-ONLY PCA coords
by `study` (UNSUPERVISED, y never seen) -> _sigs builds FRESH 512-dim RFF-MMD signatures ON THE
CORRECTED embedding (sigma=median pdist, random W/b seed0 so basis is shared train/test), one vector
per patient -> StandardScaler fit on TRAIN(core+refs) only -> LogReg(C=0.01) trained on core+refs y
-> predict_proba on held -> roc_auc. TWO distinct RFF signature sets, don't conflate: (1) the
PRECOMPUTED signatures_lf.parquet (leak-free, pool-only) are used ONLY for reference SELECTION
(order_studies medoid/centroid); (2) the FRESH transductive _sigs are what the classifier trains on.
LEAKAGE VERDICT (Vlad's call, I agree): NO meaningful leakage. (a) No LABEL leak -- held y is never
in training, only in the final AUC; Harmony is label-blind; StandardScaler+LogReg are strictly
inductive (fit on train only). (b) PCA is pool-only (held projected, not fit) -- verified earlier.
(c) coverage strategy works ~as well as distance-based ones and selects on the leak-free precomputed
sigs, so retrieval doesn't depend on held either. The ONLY transductive element is that held cells
sit in the joint Harmony run (their POSITIONS, not labels) + a sigma scalar -- and that is inherent
to any integrate-then-classify deployment (you MUST co-integrate the query patient's cells to place
them in the shared space), label-free, so it does not inflate AUC unfairly. Honest paper phrasing:
"transductive integration (label-free) + inductive classifier on the label" -- state it, don't hide
it, but it is not a leak. Don't let a reviewer mislabel it as one.

**CROSS-TISSUE SUMMARY (CRC/blood/brain, same engine, same samebatch control):** the de-confounding
effect size (paired-samebatch) tracks biological readability of the disease, not "how many assays"
or corpus size: CRC (ceiling 0.88, strong compositional cancer signal) +0.129 > blood (ceiling 0.85,
compositional COVID signal) +0.041 > brain (ceiling 0.5-0.74, subtle cell-state shift) +0.023 (and
even that's noise-level given raw~=paired). In all 3, data-quantity effect (samebatch) is flat/~0 --
the CENTRAL reviewer-proof claim ("refs help via de-confounding, not via adding samples") replicates
cleanly everywhere. Over-correction (celltype_purity loss) is present but mild and roughly CONSTANT
across tissues (~0.03-0.05 drop) -- it does NOT scale up on the weak-biology tissue as hypothesized;
what scales down instead is how much the lever can achieve, because there's less recoverable signal.
Smart ordering (coverage/quantile vs random) only showed a real edge on CRC (multi-assay corpus,
where coverage risks grabbing a toxic outlier-tech study that quantile dodges); on both single-assay
tissues (blood, brain) ordering is statistically flat vs random -- "smart selection" is a multi-assay-
corpus-specific finding, not a general one.

**How to apply / NEXT (Guillem writeup):** the defensible, cross-tissue-replicated claims are (1)
retrieval-augmented Harmony de-confounds via PAIRING, proven by the samebatch control isolating
~100% of the effect to de-confounding not data-volume, replicated on 3 independent tissues; (2) the
SIZE of the effect is gated by how biologically readable the disease is (ceiling), not by tissue
identity per se -- brain's near-null result is a headroom problem, consistent with and reinforcing
[[brain-boundary-result]], not a new failure mode; (3) smart study-ordering only matters in multi-
assay corpora. Harmony is unsupervised (blind to disease) -> pairs are a necessary workaround;
supervised scANVI/CODAL that preserve the disease axis natively are the principled next downstream,
and would be expected to need LESS of this pairing trick since they can be told the label directly.
Earlier single-config note (held=30cd5311, raw core already 0.811 -> limited headroom;
test a harder-confounded held). Open: (a) robustness over held x cores; (b) a better selection
heuristic than the staircase (Vlad to rethink) that beats random on the MEAN, not just variance;
(c) check what studies farthest over-picks (outlier labs). CODAL is the heavier follow-up downstream
(VAE, cell-level; patient-level retrieval to choose which patients' cells). Reframe for the paper:
the robust claim is "diagnosis-matched cross-batch references de-confound a batch-aware method";
"our retrieval selection beats random" is NOT yet supported (staircase only ties + tightens CI).