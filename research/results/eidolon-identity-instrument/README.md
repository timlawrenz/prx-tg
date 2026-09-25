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
2. **G6's reference point is ~0.82, not 0.8910 and not 1.0.** Step 0's 0.8910 is
   itself inflated by leakage (see the ceiling section below); a render score must
   be read against the cross-shoot ceiling, not against perfection.

## Cross-shoot ceiling — the number G6 is read against

Step 0's queries are leaky: one image per persona, matched against that persona's
average computed over ALL shots, query image included. That inflates the number,
and the inflation *is* the cross-shoot penalty. Re-measured with a
leave-one-shoot-out centroid (index built from the persona's OTHER shoots, queries
from a held-out shoot, corpus-restricted) via `scripts/eidolon_cross_shoot_ceiling.py`:

| centroid support | R@1 | 95% CI (persona bootstrap) | R@5 | R@10 | n |
|---|---|---|---|---|---|
| 30 vectors | 0.8039 | [0.7707, 0.8381] | 0.8970 | 0.9146 | 913 |
| ~69 vectors | **0.8180** | [0.7901, 0.8456] | 0.9271 | 0.9427 | 1467 |

The 0.014 spread is centroid-support noise, so the ceiling is **~0.82** — a slight
*lower* bound, since the full average uses ~100 vectors per persona. **Leakage
correction = 0.8910 − 0.8180 = 0.0730.**

Margin, stable across both runs and consistent with Step 0 once converted:
own 0.9985 vs best-other 0.9970 → gap **0.0015** in cosine (~0.05 Euclidean on
unit vectors). 313 of 321 corpus personas have ≥2 shoots, so 8 can never form a
cross-shoot pair; 7 of 31,711 corpus dirs do not resolve against the v1 lda tree
(0.02%).

**Discarded, not a failed replication:** an early attempt used the 321 corpus
centroids as the index with 1,284 arbitrary per-image queries (R@1 0.8427) — a
different measurement — plus a companion `B_cross_shoot` 0.8201 whose queries came
from the whole v1 tree rather than approved corpus images.

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

- **VAE round-trip floor** (`scripts/eidolon_vae_floor.py`): `a1` =
  cos(embed(pixel), embed(decode(latent))) isolates the VAE's identity cost; `a2`
  = cos(embed(pixel), persona centroid) checks that the centroid we *condition on*
  describes the image we *train on* — the corpus ships the persona centroid in
  every sample dir, not a per-image vector; control = cross-identity cosines.
  No GPU claim needed (CPU decode). Awaiting first run.
- **Step (b):** the ceiling measured *through* the VAE, against 0.8180.
- **Pose-actually-moved control** (DWPose yaw on decoded latents).
- eidolon's shared extractor landed at `e5390a4` (`tools/auraface`:
  `extract_auraface`, `extract_auraface_batch`, `describe_instrument`,
  `verify_basis`), so the pixel → AuraFace path exists. Input convention is
  (H,W,3) **BGR uint8** (= `cv2.imread` order) — an RGB tensor must be reversed
  before it is passed in.
- **prx-tg-side gap: closed.** `production/data_stratum.py` now refuses a
  mixed-basis identity slot (stamp match *and* measured unit-norm band), commit
  `1b5a03b`. It is deliberately a *local* guard rather than a call to eidolon's
  `assert_basis`: the loader must not import a sibling repo that is absent on
  Vast.ai, and `verify_basis()` guards the *extractor's* basis (auto-called before
  extraction) rather than the dataset directory's. Complementary, not duplicated.
