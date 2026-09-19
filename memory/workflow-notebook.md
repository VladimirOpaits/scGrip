---
name: workflow-notebook
description: "Code layout: pipeline/util scripts live in scripts/, run from repo root as python scripts/<name>.py (supersedes the old notebook-only guidance)."
metadata:
  node_type: memory
  type: feedback
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
---

2026-07-03: All pipeline/util `.py` scripts live in **`scripts/`** (user moved them there and asked to keep putting util scripts there).

**This SUPERSEDES the earlier "put analysis in geneformer.ipynb notebook" rule** — that was the old PBMC/Geneformer phase. The current [[scset-pivot]] work is standalone scripts in `scripts/`, not the notebook.

**How to apply:** put new util/pipeline scripts in `scripts/`. **Always run from the repo root** as `python scripts/<name>.py` (do NOT `cd scripts`). Two reasons it must be root: (1) data paths inside scripts are relative to CWD (e.g. `data/scvi_corpus/...`), and (2) cross-imports (`import selection`, `import scset_model`, …) resolve because Python puts the script's own dir (`scripts/`) on `sys.path[0]`. Verified 2026-07-03: all 10 modules import cleanly and `python scripts/scset_downstream.py` runs from root. Result CSVs currently still land at repo root (heldout_results.csv etc.). Relates to [[project-direction]].

**2026-07-05: `notebook.ipynb` (repo root) is the consolidated ORCHESTRATOR** over `scripts/` — markdown docs per stage + live `!{sys.executable} scripts/X.py` calls + result-loading/plot cells. Code stays in scripts/ (single source of truth); heavy builds (crc_extract/crc_pca/crc_signatures/scset_crc/crc_leakfree_build) are documented as commands but NOT recomputed (artifacts in data/crc/); experiment cells (crc_enrichment, crc_clf_check, crc_msi, crc_add, scSet compare) run live so "Run All" reproduces results in ~6-8 min. **Regenerate via `python scripts/make_notebook.py`** (nbformat generator — edit structure THERE, not the .ipynb directly). Executed clean end-to-end (nbconvert exit 0 = pipeline works cross-cut). Env: bioml kernel. Relates to [[crc-atlas-base]].
