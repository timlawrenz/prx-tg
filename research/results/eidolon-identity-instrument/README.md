# Eidolon identity instrument — Step 0 (retrieval machinery validation)

**Date:** 2026-09-24
**Status:** `[EVAL-ONLY — REPRODUCED]`
**Purpose:** validate the identity *measurement* before any renderer exists, so
that Stage E's identity numbers rest on a checked instrument rather than a
plausible one. No model, no VAE and no face-detection in the loop.

Artefacts: `step0_replication.json` (raw), `scripts/eidolon_identity_replication.py`
(the producing code, git-tracked), run at seed 42.

## Result

| quantity | value |
|---|---|
| index | `averages/*.lda.npy` — 325 LDA-space persona averages, L2-normalized |
| queries | 321 (one corpus image per persona, resolved via `metadata.json`; 0 unresolved) |
| chance R@1 | 0.0031 (= 1/325) |
| **R@1** | **0.8910** (recorded 0.8879, delta **+0.0031 = exactly one query**) |
| **R@5** | **0.9502** — exact match to recorded |
| **R@10** | **0.9657** — exact match to recorded |
| cosine vs euclidean | identical, as required on L2-normalized vectors |

**Verdict: REPRODUCED.** The delta is one image out of 321 from a different
random draw of which image per persona to query.

## The number that matters most for Stage E: the margin

```
own-average distance     0.0494
nearest-other distance   0.0740
gap                      0.0246
frac(own nearer)         0.8910   -> ~10.9% of queries have a nearer WRONG persona
```

The correct persona is separable from the nearest wrong one by ~0.025 on a unit
scale, and ~11% of faithful per-image queries already retrieve the wrong persona.
Two consequences for the instrument:

1. **Identity must be measured as a rank/margin statistic with a persona-level
   bootstrap CI — never as an absolute cosine threshold.** Any fixed cosine cut
   would sit inside the noise.
2. **G6's reference point is 0.8910, not 1.0.** A renderer's cross-shoot R@1 has
   ~0.89 as the practical ceiling for faithful vectors, so a lower render score
   must be read against that ceiling, not against perfection.

## What was discarded, and why

An earlier attempt used the **321 corpus centroids** as the index with 1,284
arbitrary per-image queries and returned R@1 0.8427. That is not a failed
replication — it is a *different measurement* (wrong index artefact, wrong query
population). Discarded. Its companion `B_cross_shoot` figure (0.8201) is also
untrustworthy: its queries came from the whole v1 `lda` tree instead of
approved corpus images only. The cross-shoot ceiling must be re-measured with
corpus-restricted queries.

## Known residue, adjudicated upstream (recorded so it is not re-raised)

2,220 per-image LDA files in the v1 tree are stale (norm < 10, pre-refit basis).
Per the eidolon evidence README: **all 2,220 are tainted/non-approved** (2,219
`extraction_nonface`, 1 `contamination`) and therefore outside every consumed
path. A 289-file sample taken here found 2.77% stale vs a 0.75% global rate,
consistent with shoot-clustered taint. Queries in this measurement are drawn
only from approved corpus images (0 unresolved), so none can be stale.

## Provenance

- basis fingerprint `120e1c5a1dc4f423`; basis artifacts
  `auraface_lda.npz` `8ebcc47f6de6cecb`, `auraface_preprocess.npz` `6bbc0937b7b83d6a`
  (hashes recomputed independently and matching both directory stamps).
- Conventions: query = per-image LDA L2-normalized (raw norm ~153); index =
  L2-normalized (norm 1.0). Mixing them, or skipping the query normalization,
  makes the cosine meaningless (~150x scale mismatch).
- **Two fingerprints exist by design**: `basis_fingerprint` = sha256(basis
  artefacts) = the projection; `lda_basis_fingerprint` = sha256(corpus
  `averages/*.lda.npy`) = the content. Not an inconsistency.
- FFHQ cannot enter this measurement: 1 image per identity gives no
  cross-shoot pair.
- Replicates `docs/assets/exp/sapiens2-keypoints-study/evidence-20260923/
  test_average_discriminative.py` in the eidolon repo.

## Not yet done

- VAE round-trip floor, cross-shoot ceiling and pose-actually-moved control —
  all need eidolon's `extract_auraface_for_eval()` helper for the
  pixel -> AuraFace path (detection + alignment + embedding).
- **prx-tg-side gap:** `production/data_stratum.py` loads `auraface_lda.npy`
  without calling eidolon's `assert_basis`, so it can ingest a differently-basis'd
  identity slot silently. That is the guard whose absence produced the
  mixed-basis arms. Wiring it needs `ffhq/stratum`'s stamp status confirmed first.
