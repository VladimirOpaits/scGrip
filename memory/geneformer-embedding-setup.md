---
name: geneformer-embedding-setup
description: "How to run Geneformer EmbExtractor in this project (env, model, GPU constraints)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 506da36b-625b-49b9-b619-f9185b7b5811
---

Embedding extraction runs in the `bioml` conda env (`source ~/miniconda3/etc/profile.d/conda.sh && conda activate bioml`). The project uses **Geneformer-V2-104M** (12 layers, hidden 768, vocab 20275), not the 316M default — it matches the `gc104M` tokenizer dictionaries the tokenization already used, and 316M is too heavy for the laptop's 6GB GPU. Weights live at `Geneformer/Geneformer-V2-104M/` (downloaded from HF subfolder `ctheodoris/Geneformer/Geneformer-V2-104M`; the git clone only had config.json).

GPU is an RTX 4050 Laptop, **6GB VRAM only**. Two hard constraints when using `EmbExtractor`:
1. It **hardcodes `device="cuda"`** in `get_embs` — no CPU fallback; the GPU driver must be up (`nvidia-smi` working).
2. `model_type="Pretrained"` loads `BertForMaskedLM`, whose forward computes the full **MLM head over vocab 20275 across the whole sequence** — a huge transient allocation. On 6GB this OOMs unless `forward_batch_size=8` (or lower) and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Throughput ~18 cells/s; full pbmcsca (31021 cells) ≈ 28 min.

`EmbExtractor` does NOT create the `output_directory` — `os.makedirs("embeddings", exist_ok=True)` first or it fails at the final `to_csv`. Output is `embeddings/pbmcsca.csv`: 768 emb dims + `Method`/`Experiment`/`CellType` columns. The runnable cell is in [[workflow-notebook|geneformer.ipynb]]. Relates to [[project-direction]].

**Tokenization (CPU) gotchas:** Always run `TranscriptomeTokenizer(..., nproc=1)`. With `nproc>1` the final truncation `datasets.map()` step (a) flakily dies with `multiprocess.context.TimeoutError`, and (b) each worker holds a full copy of the dataset, blowing up the laptop's 14GB RAM. With `nproc=1` peak RAM stays ~2GB — the tokenizer reads the h5ad in chunks of 512 cells in backed mode, so the sparse matrix is never densified. Background tasks do NOT survive a session restart/compaction, so re-verify `tokenized/<name>.dataset/` exists on disk after resuming. Note: `conda run -n bioml` errors ("not a conda environment"); must `source .../conda.sh && conda activate bioml`.
