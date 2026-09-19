---
name: paper-pivot-anchor-mmd-ewc
description: 2026-09-07 hard scope pivot — first-paper plan is anchor+stratified-MMD (+planned EWC baseline) reference-preservation method on lupus c2/c3/c4; AB-MIL/biomarker-discovery/hard-negative-donor-matching dropped from paper scope.
metadata: 
  node_type: memory
  type: project
  originSessionId: 9e54ad71-2c38-4bf4-b67f-ff624399e3a4
  modified: 2026-09-07T15:34:35.195Z
---

Vlad decided (after [[external-matching-diagnostic]] and the [[mil-power-benchmark-ground-truth]] capability test both came back "not ready, real gaps") to stop chasing the biomarker-discovery arc for a first paper. New scope, explicit: **anchor (replay latent distillation) + stratified MMD, with an EWC baseline planned** (see literature check — [[clcvae-ewc-precedent]] if written, otherwise: Hediyeh-zadeh/Lotfollahi/Theis, ICML CompBio 2022 workshop, never journal-published, closest direct competitor) as the whole paper. Validated on lupus cohorts 2/3/4. RA (depth-covariate decoder surgery) stays in as a secondary applied-example vignette, not a biomarker claim.

**Dropped from paper scope, and removed from the repo 2026-09-07:**
- `sceris/abmil.py`, `sceris/bags.py` — AB-MIL model code, zero dependents in build/paper, deleted outright.
- `scenarios/hard_negative_matching/` (external_pool_diag.py, gene_space_signature.py, match_provenance.py, match_within_block.py, mil_power_benchmark.py) — existed only to support AB-MIL's contrastive external-donor step; deleted. The underlying finding (retrieval picks batch block not person; gene-space RFF-MMD is MORE batch-dominated than the trained latent, not less; witness-rate/contamination capability bounds) stays valid and stays in memory, just not in the codebase.
- `scenarios/abmil_re.py` — AB-MIL driver, deleted.
- Most of `sceris/` (retrieval.py, signatures.py, store.py, source.py, reference.py, dataset.py, project.py, cli.py, plus an unused duplicate anchor.py/latent.py, and compartments.py which only those used) — archived to `sceris/_legacy_platform/`, tests moved to `tests/_legacy_platform/`. This was leftover from an EVEN OLDER "patient retrieval across diseases" platform pivot, predating the lupus MIL work entirely; nothing in build/ or paper/ ever imported it. `sceris/__init__.py` emptied (was unconditionally importing store/retrieval, which broke `from sceris import config` once those moved — fixed).

**Kept, this is now the whole paper's engine:** `build/{census_healthy_base,census_base_nofloor,census_monocyte_supplement,train_census_scvi,lupus_base,lupus_cohortX_build,finetune_anchored,finetune_mmd_stratified,scArches,strata,ra_base,finetune_anchored_depthcov}.py`, `paper/{encode,common,geometry,conservation,neighbourhoods}.py`. `build/finetune_mmd_stratified.py` is confirmed (by output-path pattern, matches every live `_strat_` checkpoint name) to be the ONE script that produced every live MMD checkpoint across c2/c3/c4 — the earlier `finetune_mmd_phase2*.py` / `finetune_mmd_earlystop.py` (pooled, not stratified) are superseded precursors, archived in `build/_legacy/`. TODO.md's dependency-chain note naming the phase2 scripts as "live" is STALE — it describes the deferred 8000-gene track, not the current pipeline; verify against code, not that note.

**Next step implied by this pivot:** implement EWC as a real arm (Fisher-diagonal parameter penalty vs the base model's loaded weights) alongside anchor, run on c2/c3/c4 with the same geometry/held-out metrics already in `paper/geometry.py`, to get the external-baseline comparison the paper needs. Not yet built.

**How to apply:** any future biomarker/AB-MIL/donor-matching question is out of scope for THIS paper — answer honestly that it's parked (see [[external-matching-diagnostic]], [[mil-power-benchmark-ground-truth]] if that memory exists) but don't resume building it without Vlad explicitly reopening that direction.
