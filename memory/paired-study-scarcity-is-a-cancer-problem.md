---
name: paired-study-scarcity-is-a-cancer-problem
description: "The paired-study bottleneck that blocked us for weeks was self-inflicted — it comes from insisting the disease axis be CANCER, not from a shortage of data in Census."
metadata: 
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
---

2026-07-14. A Census-wide scan (`scripts/census_paired_scan.py`, streams all 96.6M primary human
cells in Arrow, ranks every tissue by studies holding BOTH disease and control **donors**) killed
the premise we had been operating under.

Paired studies (donor-level disease + control, minority arm >= 5):
  blood 18, **brain 16**, lung 6, colon 6, small intestine 5, heart 4, kidney 4 ...
And most survive restriction to a SINGLE assay (brain 14, blood 17).

**Why:** oncology papers don't need controls to make their claim, so they don't collect normal
samples — normals live in separate reference atlases (different lab, year, protocol). Every
NON-cancer disease study needs controls by design (you cannot say anything about Parkinson's or
COVID without healthy donors), so pairing is built into the study design for free.

So the confound diagnosis<->study is real and structural (see [[explainability-geometry]] — that
framing stands and belongs in the paper), but the *data scarcity* was ours: we kept requiring
tumor-vs-normal because CRC started that way. The disease axis is a free parameter.

**Why:** we burned weeks on Census lung (only 3 weak paired studies) and nearly downloaded LuCA
and a breast atlas to solve a problem that did not exist.

**How to apply:** when a tissue "has no paired studies", check whether the filter is the tissue or
the disease label. Scan donor-level, never cell-level (normal CELLS inside a tumor are not normal
SAMPLES). Related: [[crc-atlas-base]], [[external-pca-mmd-plan]] (its acquisition spec is now
obsolete — no external h5ads needed).
