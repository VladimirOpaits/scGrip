---
name: census-panel-frequency-floor
description: "The 3000-gene census panel's missing RA genes were cut by the RAM-driven 5% frequency floor, not by variance — ~half the panel is a memory workaround artifact."
metadata: 
  node_type: memory
  type: project
  originSessionId: fd271409-a943-42ae-8818-14ec597b0bf3
  modified: 2026-09-06T12:58:33.675Z
---

2026-09-06. `scenarios/ra_hvg_divergence.py` (written this session; streams full
`seurat_v3` incl. loess clipping over all 11288 genes × 481191 cells of
`census_healthy_base_filtered.h5ad`; validated against scanpy bit-for-bit,
max|diff| 1.6e-8, spearman 1.0).

The recorded diagnosis — "the census HVG panel misses canonical RA genes because it
was selected for variance in healthy PBMC, not RA" — is **wrong for 7 of 8**.
Ranks among healthy census donors, out of 11288: G0S2 **25**, IFIT2 111, CTLA4 457,
BLK 472, PADI4 829, CCR6 880, CD40 950 — all inside an honest top-3000. They never
reached HVG selection: all sit at 2.1–4.7% of cells, and `REDUCE_GENE_FRAC=0.05` in
`build/census_healthy_base.py` drops everything under 5% *before* calling
`highly_variable_genes`. That floor exists only to keep the in-memory matrix under
the 14GB ceiling. Only STAT4 (20.4% of cells, rank 5337) genuinely lost on variance;
PADI2 is out even of the 11288 pool (1% floor at `FILTERED_RAW`).

Not local: **45% of a floor-free top-3000 (1344 genes) sits below 5%**, and the
published panel overlaps a floor-free top-3000 by only **1654/3000**.

**Why:** it reverses the fix. The binding constraint is the frequency floor, not the
panel width — dropping the floor at an unchanged 3000 genes already recovers 7 of 8,
while 3000→8000 only adds STAT4 plus headroom. It also kills the deferred plan's
stated main risk (that these genes might be near-zero-variance in health, making any
widening futile).

**How to apply:** a floor-free base and model now exist alongside the published ones
(`census_healthy_base_nofloor_hvg.h5ad` / `models/scvi_census_healthy_base_nofloor`,
built by `build/census_base_nofloor.py`, trained via `train_census_scvi.py --tag
nofloor`). Same 160500 cells and same 3000-gene width as the published base — the
gene set is the *only* difference, by construction, so it is a clean A/B. It shares
1658/3000 genes with the published panel; of the 1342 new ones, 1338 were below the
5% floor. Nothing downstream has been run on it yet — all published numbers are
still the old base. The accumulator lives in `sceris/hvg.py`; reuse it rather than
rewriting. If the 8000-gene widening is later revisited (deferred, waiting on GPUs —
see [[anchored-finetune-result]]), note it buys only STAT4 on top of this. The base
path is hardcoded in six `build/` scripts while `sceris/`+`scenarios/` read
`sceris.config.CENSUS_HVG` from `.env`.
