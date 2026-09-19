---
name: external-matching-diagnostic
description: "2026-09-07 phase-0 of the hard-negative apparatus on lupus cohort 2 — alignment makes the external pool reachable, but per-donor retrieval matches a BATCH BLOCK, not a person."
metadata: 
  node_type: memory
  type: project
  originSessionId: 9e54ad71-2c38-4bf4-b67f-ff624399e3a4
  modified: 2026-09-07T12:23:51.994Z
---

Ran the feasibility diagnostic for the "Fixed MMD hard-negative matching" step from
CLAUDE.md, on lupus cohort 2 against the census healthy pool (1605 donors, zero
donor overlap with the cohort — verified). Scripts: `scenarios/external_pool_diag.py`,
`scenarios/gene_space_signature.py`, `scenarios/match_provenance.py`,
`scenarios/match_within_block.py`. All numbers in units of the median census–census
donor-centroid distance.

**The pool has exactly two blocks, and they are perfectly confounded:**
OneK1K 981 donors / 10x 3' v2 / European; AIDA 624 donors / 10x 5' v2 / Asian.
Cohort 2 is all 10x 3' v2, SLE 57 European / 63 Asian, but its 22 internal controls
are 21 European / 1 Asian — so **63 Asian SLE donors have no ancestry-matched
internal control**, which is the concrete gap external matching could fill. A match
can hold assay OR ancestry constant, never both.

**Alignment is the precondition, and it works.** base → final (anchor+MMD):
census–normal 1.39 → 0.95, census–sle 1.78 → 1.40, disease contrast normal–sle
1.25 → 1.18 (preserved). In the BASE latent the batch offset (1.39) exceeds the
disease contrast (1.25), so matching there would be matching on batch. Nearest
external healthy beats nearest internal control for 77% of SLE donors (91%
stratified) in `final` vs 37% in `base` — and this survives restricting candidates
to a single block (AIDA 77%, OneK1K 62%), so it is not just one block moving closer.

**But the retrieval itself has no demonstrable biological content.** Global
nearest-neighbour picks a block, not a person: gene space takes 100% of picks from
OneK1K, every latent arm ~75–81% from AIDA, both regardless of the SLE donor's own
ancestry (concordance lift 0.97–1.01). Within a fixed block, sex lift 0.91–1.04 and
age lift 0.97–1.07 — nothing. The strongest measurable within-block driver in
final/AIDA is |Δ log n_cells| (rho 0.266), i.e. cell yield. Alignment did kill the
composition dependence there (|Δmonocyte| rho 0.32 → 0.08), which is a real win from
the MMD work, but nothing biological took its place among what we can check.
Caveat: sex/age are weak axes in PBMC, so this is "not demonstrably biological",
not "demonstrably non-biological".

**Gene-space RFF-MMD is NOT the non-circular escape hatch.** Raw counts → shared
PCA(50) → RFF(512) → donor mean puts the whole cohort in one tight island far from a
spread-out census cloud (nearest external 0.63 vs nearest internal 0.28, external
closer for 0% of SLE donors) and agrees with the latent ranking only at per-donor
rho 0.05–0.25 / top-50 overlap ~11%. Non-circular and batch-dominated are the same
thing here. See [[unified-deconfound-method]], [[mmd-distribution-ruler]].

**In the 20-dim latent, RFF-MMD ≈ centroid**: pearson 0.985, per-donor rank rho 0.97,
same top-1 pick only ~60%. MMD earns its place only where the first moment is
structurally insufficient (per-compartment, multimodal donors), not as a drop-in
upgrade.

**Circularity to respect:** `final` was trained with an MMD term against census cells
and an anchor pinning census, so "census is close to the cohort" partly restates the
objective. The clean held-out pool is `data/allen_soundlife_immune_variation.h5ad`,
still untouched.

**How to apply:** never do global nearest-neighbour retrieval against this pool —
restrict candidates to one block first. Before trusting any matched set, run the
downstream A/B that actually decides it: AB-MIL with K *matched* external y=0 donors
vs K *random* external donors from the same block vs cohort-only. Prior from
[[retrieval-augmented-harmony]] says matched/farthest selection has lost to random
before. Also measure the batch shortcut directly (AUC of external-vs-cohort-normal);
adding external y=0 donors is only safe if that shortcut is closed.
Design settled with Vlad: external donors enter AB-MIL as plain `y=0` under BCE —
NOT an explicit contrastive term maximising latent distance, which has a degenerate
solution (inflate scale / re-import the batch axis). Distance is a diagnostic, not
a loss. See [[composition-as-offset]].
