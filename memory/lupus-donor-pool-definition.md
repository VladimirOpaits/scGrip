---
name: lupus-donor-pool-definition
description: "Precise definition of \"pool\" (multiplexing stratum) and \"mixed pool\" for the AB-MIL permutation-null test on the lupus cohorts"
metadata: 
  node_type: memory
  type: project
  originSessionId: eb2fb708-8d4a-4796-b609-724e6e330ac3
  modified: 2026-09-02T22:06:20.826Z
---

**Pool**: the set of donors, within one cohort, whose `library_uuid` sets (over all their cells) are IDENTICAL as sets — not overlapping, equal. Built as `frozenset` of `library_uuid` per donor, then `pd.factorize` over those frozensets assigns the pool id (per cohort, prefixed e.g. `c2_0`). A donor is not nested inside one library — donor and library are crossed (cohort 2: 124 donors × 4 libraries each, 18 donors × 8 libraries each; cohort 4: 92 donors × 4 libraries each) — so "shuffle the label within the donor's library" is undefined; the pool (identical library set) is the smallest unit within which shuffling a donor label is well-defined.

**Mixed pool**: a pool containing at least one SLE donor AND at least one control donor. Test: `0 < bags.y[mask].sum() < mask.sum()`. Only mixed pools are permutable — a pure pool (all-SLE or all-control) has nothing to shuffle, so including it in the null either does nothing or falsely narrows it.

Measured composition (2026-09, cohort2_full/cohort3_full/cohort4_full.h5ad via h5py, `ind_cov`/`library_uuid`/`disease`):
- c2: 142 donors, 40 libraries, 18 pools, only 4 mixed (56 donors, 8 of 22 controls survive filtering)
- c3: 41 donors, 12 libraries, 6 pools, 4 mixed (31 donors, 18 of 18 controls survive)
- c4: 92 donors, 24 libraries, 6 pools, ALL 6 mixed (92 donors, all 40 controls survive) — c4 is the only cohort where the stratified permutation null is not degraded by filtering; c2's null should be read with caution (only 8 controls left).

Consequence: the CLAUDE.md instruction "shuffle within strata, ready-made stratum = 10x library" does not literally apply here — library is not a per-donor unit. Pool is the correct stratum. See [[lupus-cohort34-replication]] for the broader per-cohort rerun context.

The script that builds this table (`build/strata.py`, reads h5ad via h5py, writes `data/lupus/donor_strata.parquet` with donor/pool/cohort/n_lib) was given to Vlad in chat but has NOT been saved to disk as of 2026-09-03 — do not assume it exists; check before referencing it as a real file.
