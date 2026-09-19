---
name: project-direction
description: Overall goal and current strategic direction of the scRNA cross-dataset retrieval project
metadata: 
  node_type: memory
  type: project
  originSessionId: 929a453b-e262-4a29-a02b-97158a45aebf
---

Goal (stated 2026-06-10): ship a research **paper first, then turn it into a product/platform**. The intended relationship: the paper is the *trust/credibility layer* for the platform — it proves the retrieval engine finds biologically-similar patients, not batch-mates. Design ONE artifact that serves both (a validated patient-retrieval engine), don't build a benchmark paper disconnected from the product.

As of 2026-06-10 the user **abandoned the original PBMC IR-benchmark protocol** (deleted from the vault, along with the design-council and задача notes) and is redesigning the task toward **patient-level retrieval**: embed patients, query nearest similar patients across batches/cohorts; fast ANN index (FAISS) as the working interface. Platform vision: "throw in your cohort → get comparable patients from a large indexed public DB, batch-aware."

The Obsidian vault (`obsidian/Retrieval/`) was restructured by the user into his own concept notes (in his own words): `Батч эффект не может быть полностью отделен от биологического смысла`, `Матчить по метаданным = Разорвать циркулярность`, and `Лит-обзор — геометрия батч-vs-биология` (the related-work seed I built). The user takes notes himself; my role is to scaffold + wire links, not to write his conclusions for him.

Refined architecture conclusions (2026-06-10):
- **Do NOT embed metadata** — at patient scale (thousands, not millions) it's a queryable table + distance function; ANN/embedding is a cell-scale concern. Real vector DBs keep metadata as filter columns next to the vector. Embed ONLY expression.
- Expression embedding's role = the **measurement/payload** (molecular difference on a metadata-matched set), AND the **object of study**, NOT the retrieval key. Without trustworthy expression comparison the product is just a metadata search engine.
- The geometry/diagnostic mechanics (batch⊥biology orthogonality, probing, subspace disentanglement) are **already taken / actively published in 2025–26** — novelty must be the QUESTION (donor-level retrieval-as-IR + identifiability as predictor of retrieval success + runtime confidence signal), not the method. See lit-review note.

Hard-won design principles to carry into ANY reframing:
- **Match/retrieve on metadata + cell composition** (age, sex, brain-region cell distribution), but **study the expression embedding** — keeps the relevance label independent of the tested embedding, which breaks the circularity. (User's most valuable insight, reached independently.)
- Batch effect = an **identifiability** problem: unsolvable in general when batch is confounded with biology (information-theoretic, not an engineering gap). The contribution is to **characterize when retrieval reflects biology vs site**, NOT to "solve" batch. This is both the paper's thesis and the product's honest disclaimer.
- **Composition is matched-on (to neutralize as confound), not retrieved-by** (retrieving by composition is a shortcut).

Open questions blocking redesign (asked 2026-06-10, awaiting user):
- Actual dataset/domain unconfirmed. User keeps using Alzheimer's brain as the example → that would be **snRNA (single-nucleus)**, the track the [[design-council-pbmc-retrieval]] deferred to a second paper (different corpus, different batch effects). Need to know if real data is AD brain (snRNA), blood/PBMC (scRNA), other, or none yet.
- Product user / pain point unconfirmed.
