---
name: scvi-corpus-donor-ambiguity
description: "Open data-hygiene issue in the scVI lung corpus: ~8 donor+assay labels recur across 2 datasets each — possible biological duplicates or reused generic IDs; decide/fix later."
metadata:
  node_type: memory
  type: project
  originSessionId: b4fec24d-2ca6-4da9-8741-caa144c2dd7f
---

2026-07-02. In the [[scset-pivot]] scVI corpus (data/scvi_corpus, keyed pid = dataset_id[:8]_donor_id_assay), **8 donor+assay pairs appear in >1 dataset**: NL1, NL2, NL3, Emp1, Emp2, Emp3, D496, D503 — all `10x 3' v3`. Two indistinguishable possibilities: (a) the SAME patient curated into both its own dataset and an integrated atlas → biological duplicate; or (b) DIFFERENT people, since donor_id is known to be reused across datasets (NL#/Emp# are generic "normal lung"/"emphysema" labels). We currently keep them **dataset-separated on purpose** (safe: never merge possibly-different people; files don't collide because dataset[:8] prefix differs).

Risk: only if they're truly the same person AND leak into both sides of a held-out split. Only 8/576 patients. **Plan: handle at analysis time** — when building held-out folds, forbid the same donor_id+assay from landing in both train and test (dedup or co-assign). Do NOT blindly merge by donor+assay (would wrongly fuse ID collisions). User: note it now, decide the fix later. Related: [[hlca-adeno-dataset-build]] (donor_id collisions were the reason we key by the dataset+donor+assay tuple).
