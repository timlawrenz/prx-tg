# eidolon-identity-renderer (Arm EIR)

**Status:** `[PLANNED — pre-registered, NOT yet run]`
**Mode:** `confirmatory` (gate frozen before the first run — see provenance.yaml)
**Branch:** `arm/eidolon-identity-renderer`

## Hypothesis

A from-scratch FLUX-VAE **latent** DiT (P1-shaped: 16ch, 128×128, patch 2) carrying the
Eidolon adapter — 64-d AuraFace-LDA identity via adaLN, `z_g` geometry via
cross-attention — learns a **generalizable identity mapping** from a two-source
LDA conditioning stream (FFHQ per-image vectors + hegre persona centroids), rather
than memorizing per-image identity keys.

Stated positively: conditioning on the vector of a person **never seen in
identity-training** should render *that person*, and a pose change should leave the
identity intact.

## What it is not

- **No text.** The conditioning is identity (LDA 64-d) + geometry (`z_g`). The
  identity vector is the only content control; `z_g` controls pose. (Hegre ships no
  `t5_hidden`; the Eidolon adapter is identity+geometry by design.)
- **No warm-start, no PP stage, no weight donor.** Fresh model, no lineage — so the
  warm-start taint question does not apply.
- **No REPA.** Hegre ships zero `dinov3_patches.npy`; REPA must be dropped.
- **Not pixel-space.** This is a latent arm; photorealism is not the claim here.

## Differs from

- `latent-first-pretrain` (P1): same latent/VAE shape, but the Eidolon adapter replaces
  text/DINO conditioning, and hegre joins FFHQ as a training source.
- `eidolon-conditioning`: that arm is **pixel-space** (`in_channels: 3`, `patch_size: 16`);
  this one trains in FLUX-AE latent space.
- `zg-token-basis-cfg-guard` (Arm O): Arm O is a **code reference, not a weight donor**.
  Its identity stream trained during the mixed-basis window, so its identity
  conditioning is untested and unsound; only its geometry result stands.

## Expected outcome

PASS on the pre-registered gate below, i.e. held-out identity retrieval well above
chance on both probes. The **falsification we actually expect to be informative** is
the memorization signature: a model that scores well on trained identities while
failing on held-out ones. That outcome is a FAIL even though its "did it learn
identity" number looks good, and it is the specific risk this arm is constructed to
expose (see `research/results/eidolon-identity-instrument/README.md`).

## The instrument references (measured BEFORE this arm, no model in the loop)

| probe | index | index size | reference | chance |
|---|---|---|---|---|
| FFHQ unseen-identity **fidelity** (1-shot) | held-out FFHQ identities | 6,996 | ~1.0 (VAE round trip `a1` = 0.9998) | 1.4e-4 |
| hegre unseen-persona **generalization** (cross-shoot) | held-out hegre personas | 32 | **0.9504** | 0.031 |
| hegre **persistence / recall** (cross-shoot) | all corpus personas | 313 | **0.8180** | 0.003 |

Three rules these numbers impose, all learned the hard way in the instrument work:

1. **The ceiling depends on the index size.** 0.9504 at 32 personas, 0.9005 at 96,
   0.8180 at 313. A renderer must be read against the ceiling measured **at the same
   index size** — comparing a 32-index score to 0.82 would flatter it.
2. **Rank/margin with a persona-level bootstrap CI, never an absolute cosine
   threshold.** The identity margin is ~0.0015 cosine and the own/wrong-centroid
   distributions overlap (`a2` = 0.9985 own vs control max 0.9981).
3. **Detection survival is a first-class number.** A face-filling image returns
   `no_face` at the corpus convention (`det_size` default 640) — 24/24 in a synthetic
   probe — so `batch.skipped` / `no_face` must be reported alongside every R@1, never
   silently dropped.

## Holdout (locked, hashed, committed before the split)

`experiment-configs/eidolon-identity-renderer/holdout.json` — builder `scripts/build_identity_holdout.py`,
seed `20260925`:

| set | unit | pool | holdout | excluded from training | sha256 (first 16) |
|---|---|---|---|---|---|
| FFHQ | identity (== image) | 69,960 | **6,996** (10%) | identity **and** imagery | `2967fafac71f2a75` |
| hegre | persona (multi-shot) | 313 | **32** (10%) | entirely (all shots) | `705936d1c0f3b805` |

The hegre pool is reproduced with the **ceiling's own eligibility rule** (corpus dirs
resolving into the LDA tree, ≥2 sets among their own corpus samples), not the looser
"≥2 set dirs exist" rule (320) — otherwise the 0.9504 reference would not be exact.
The holdout count is pinned to 32, the index size at which that ceiling was measured.

Cost, stated plainly: this removes 10% of FFHQ imagery and 32 of 313 hegre personas
from training. The hegre personas are the **only** source of the pose-invariance
signal, which is why the holdout stays at 10% rather than being enlarged for a more
discriminative index.

**Neither holdout is a never-touched test set** — both are monitored in-training and
used for the verdict. The plan's separate never-touched slice is a Stage B
reconciliation, deliberately not invented here.

## Open items before the gate can be considered frozen-in-full

1. **Dropout profile** — undecided: `.10/.20/.30` (identity-only marginals) vs the
   `eidolon-conditioning` profile. `p_identity_only` is the dangerous slice (AuraFace
   leaks pose; yaw R²≈0.46, pitch≈0.64) and needs the CFG guard or it is noise injection.
2. **Step budget** — 10k at 1k-segment granularity is the standing default.
3. **Model size / architecture** — inherits P1's shape; not yet frozen in this config.
4. **hegre latents** — encoding in progress (31,711 dirs). Step (b) of the instrument
   cannot run until it completes.
5. **`config.yaml` is PROVISIONAL** — it records what is decided and marks the rest
   `TBD`. No run may launch against it until those are resolved and it is frozen.
