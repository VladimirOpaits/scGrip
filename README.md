# scGRIP

`scgrip` is the current implementation of constrained reference mapping for a
frozen scVI model.  The older `build/finetune_*` and `build/evaluate_*` scripts
remain as historical experiments; they are not the entry points for new runs.
The design and non-negotiable loss roles are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Current phase-I workflow

1. Make a healthy, tissue-matched reference bank.  It needs raw counts, donor
   labels, a shared stratum vocabulary, and enough cells for disjoint metric,
   replay, and held-out reference roles.

   ```bash
   PYTHONPATH=$PWD python build/prepare_reference_bank.py \
     --input-adata /path/reference.h5ad \
     --base-model /path/frozen_scvi \
     --out-adata data/prepared/reference_bank.h5ad \
     --source-counts-layer counts \
     --donor-col donor_id --stratum-col cell_type \
     --filter-col tissue_general --filter-value blood \
     --max-cells-per-donor 300 --reserve-per-stratum 10
   ```

   If reference and query labels differ, give both preparation commands an
   explicit `--stratum-map labels.csv`, with `source,target` columns.  The same
   target vocabulary must be used on both sides.

2. Make a query cohort artifact.  It retains disease cells for post-fit
   encoding, but not for phase-I gradients.

   ```bash
   PYTHONPATH=$PWD python build/prepare_external_cohort.py \
     --input-adata /path/query.h5ad \
     --base-model /path/frozen_scvi \
     --out-adata data/prepared/query.h5ad \
     --source-counts-layer counts \
     --donor-col donor_id --stratum-col cell_type --disease-col disease
   ```

   `strict` is the default input policy.  Only use
   `--input-panel-policy audited_imputation` for documented legacy panel
   surgery.  For the current RA artifact this records the constant MALAT1 fill
   and all declared zero-padded genes; it does not turn that imputation into
   observed expression.

3. Run phase I.  Its invariant is fixed in code: healthy A+B get ELBO, replay
   gets a functional anchor, healthy A gets MMD against an independent metric
   reference, healthy B is held out, and disease cells get no gradients.

   ```bash
   PYTHONPATH=$PWD python build/train_phase1.py \
     --base-model /path/frozen_scvi \
     --reference-adata data/prepared/reference_bank.h5ad \
     --query-adata data/prepared/query.h5ad \
     --out-dir runs --run-name cohort_w0_mmd20 \
     --reference-donor-col donor_id --reference-stratum-col cell_type \
     --query-donor-col donor_id --query-disease-col disease --query-stratum-col cell_type \
     --healthy-label normal --counts-layer counts \
     --adapter full_w0 --lambda-anchor 20 --lambda-mmd 20 --device cuda
   ```

   The primary adapter is `full_w0`.  `affine_w0` is a labelled ablation only.
   No automatic “best” disease checkpoint is chosen.  An
   automatic healthy-only choice is allowed only after predeclaring
   `--selection-anchor-budget`.

4. Evaluate base and saved states from fresh frozen models.

   ```bash
   PYTHONPATH=$PWD python build/evaluate_phase1.py \
     --run-dir runs/cohort_w0_mmd20 \
     --out-dir runs/cohort_w0_mmd20_eval \
     --checkpoint epoch_002=runs/cohort_w0_mmd20/checkpoints/epoch_002.pt \
     --checkpoint final=runs/cohort_w0_mmd20 \
     --technical-batch-col Lane \
     --signature ifn18 --signature-adata /path/full_gene_query.h5ad \
     --signature-id-col observation_joinid --device cuda
   ```

The evaluator writes one CSV of core mapping metrics, support-aware
stratum/donor tables, optional scIB statistics, signature reports, and a UMAP
whose mapper is shared across all requested states.  Disease signatures are
post-hoc readouts only: neither training nor checkpoint selection sees them.

## Input contract

- Raw count matrix only; no normalized expression passed to the encoder.
- Exact model gene IDs and order; missing genes are an error, never zero-padded.
- Two or more healthy query donors, split by donor into A (MMD) and B
  (held-out).
- A tissue-matched healthy reference bank with labels in the same stratum
  vocabulary.
- Unique query cell IDs if expression signatures are evaluated across files;
  `prepare_external_cohort.py --cell-id-col` can establish one explicitly.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the rationale and the phase-II
boundary.
