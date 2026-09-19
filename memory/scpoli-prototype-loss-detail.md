---
name: scpoli-prototype-loss-detail
description: Micro-note — scPoli's prototype loss shapes only the encoder (via z), never touches the decoder, since the decoder isn't in the prototype-loss computation path. Same training step as CVAE loss, joint backward, just no gradient path to decoder weights.
metadata:
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
  modified: 2026-07-20T15:39:50.168Z
---

2026-07-20. scPoli (De Donno et al., Nat Methods 2023) total loss = L_CVAE + eta*L_prototype,
one joint sum, one backward pass per step (not staged training). L_prototype = mean distance(z_i, p_k)
where p_k = mean z of same-cell-type cells (a learned centroid in latent space), used for label
transfer via nearest-prototype at query time. Vlad's question confirmed: prototype loss only reaches
the ENCODER (z is encoder output, prototypes are computed from z, so gradient flows back through the
encoder) — the decoder never appears in that loss term's computation graph, so it gets zero gradient
signal from L_prototype; decoder is shaped only by the reconstruction term in L_CVAE. Open detail not
verified from the paper: whether prototypes are recomputed per-minibatch or via running average, and
whether stop-gradient is applied to p_k itself when computing distances.

**How to apply:** relevant if/when we port our retrieval-augmented de-confounding to a scPoli backend
([[scvi-lever-result]] TODO list mentions scPoli/MrVI as candidate backends) — the sample-level
condition embedding s (not the prototype mechanism) is the piece we'd actually use for patient
signatures; prototypes are a separate, orthogonal cell-type-labeling mechanism, not directly relevant
to our diagnosis-level de-confounding work.
