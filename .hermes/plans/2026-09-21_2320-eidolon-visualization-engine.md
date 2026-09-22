# Research Plan — Eidolon Visualization Engine (identity vector → pixels)

**Date:** 2026-09-21 (v1)
**Status:** SUPERSEDED same-day by the v2 plan (P1/PP gated redesign). Reason: Arm PP cancelled as tainted — P1 had no official gate, and P1/PP were diagnosed as ignoring text conditioning while memorizing CLS tokens as dataset keys. v1 assumed PP as the backbone donor; that premise is dead. Kept for history — do not execute.
**Branch context:** written on `arm/pixel-posttrain` (Arm PP running, step ~2200/10000 at review time).

---

## 1. Goal

Build the renderer half of the Eidolon product: a model that takes the eidolon
conditioning vector — **AuraFace-LDA (64-d identity)** + **z_g (50-d transient
pose/expression)** — and renders photoreal pixels. Text is *out* of the render
path (Phase 5a conclusion: text→AuraFace-LDA Prior is a separate component that
produces the identity seed; z_g comes from user sliders). This is the "pure
renderer" of `docs/architecture.md` and the deliverable the whole
stratum-hq → prx-tg → eidolon pipeline points at.

## 2. Where we stand (review findings, 2026-09-21)

| Fact | Source |
|---|---|
| Arm O (`zg-token-basis-cfg-guard`) **PASS**: dim0→yaw binding holds at steps 3000+5000, zero collapse, recon LPIPS 0.819 @ 5k. First proof the identity-vector renderer architecture can bind geometry. | ledger, `docs/EXPERIMENT_TREE.md` |
| Arm O's limits: from-scratch pixel, 5k steps, pre-P1 stack, **trained on FFHQ + hegre mixed (w2.3/w1.0)** — see data pitfall D1 below. | prx-tg-development skill |
| P1 (`latent-first-pretrain`) completed 10k steps; checkpoint `checkpoint_step0010000.pt` (sha256 6bd10df2…). | NAS runs dir |
| Arm PP (`pixel-posttrain`) is **running now** (2026-09-20_1413, ~step 2200/10000): warm-start P1 body → pixel I/O, stratum conditioning. It is the future backbone donor. | `experiment-configs/pixel-posttrain/` |
| `vae-ceiling` (2026-09-10) **falsified** the FLUX AE as a photoreal decode path (luminance 0.44→0.86, contrast halved); precomputed latent set is 512², unusable. Pixel post-training is first-class, not optional. | `experiment-configs/vae-ceiling/README.md` |
| New tangential research (2026-09-21, Qwen-Image-2.1): the VAE ceiling has a published fix — **residual/skip-connection decoders at the same 0.25 floats/px budget** (Qwen VAE-2.0 f16c64; DC-AE f32, MIT). Reframes the AE question from "more bandwidth" to "better decoder". Cheap deterministic eval row. | `docs/tangential-research.md` |
| EidolonAdapter + per-dim `geo_basis` + CFG-guard code **is present** on the current branch (`production/adapters.py`, `production/model.py:410-442`). | verified on disk |
| FFHQ stratum (70k) has full eidolon satellite data: `auraface_lda.npy` (64,), `z_g.npy` (50,). Hegre corpus (31.7k) has identity+geometry only. | `docs/DATA_INVENTORY.md` |

## 3. The pitfalls this plan is built around

### D1 — Hegre must not be trained on (the "what data when" trap)

Prior eidolon arms (O, hegre-geometry) trained on FFHQ (w2.3) **+ hegre corpus
(w1.0)**. The eidolon project rule is explicit: **hegre is the gate instrument —
training on it contaminates every downstream identity-persistence evaluation.**
It is also the *only* cross-shoot ("different photo, same person") substrate that
exists; FFHQ is 1 image/identity. Once hegre is in the training set there is no
held-out identity-persistence measurement left, anywhere.

**Rule R1:** All renderer training is **FFHQ stratum only** in eidolon adapter
mode. Hegre is reserved for the cross-shoot generalization gate (Stage 3,
eval-only). This is a deliberate correction of Arms O/hegre-geometry — flag it
in `diff_summary`, don't hide it.

### D2 — Conditioning consistency (train what you eval, eval what you train)

- **R2 — pure renderer, no silent stream re-addition.** The eidolon loader
  (`_load_sample_eidolon`) fills zero stubs for T5/DINO/pose. That is the
  product definition: identity vector + geometry in, pixels out. Re-adding text
  or DINO "to help quality" is a *different arm*, not a tweak.
- **R3 — asymmetric CFG is load-bearing.** AuraFace leaks pose (yaw R²0.46,
  pitch R²0.64): without identity-stream dropout the DiT cheats pose from the
  identity vector and z_g binding dies (the "always frontal" failure). Keep the
  Arm O recipe: per-dim `geo_basis` tokens + CFG-guarded basis + identity CFG
  dropout. Every CFG combination used at eval must have non-zero probability in
  training (eval-regime = train-regime rule).
- **R4 — identity semantics per dataset differ; don't mix.** FFHQ auraface_lda
  is *per-image* (70k identities); hegre's is persona-averageable. Mixing
  per-image and persona-averaged identity vectors in one run is a confound —
  another reason for R1.
- **R5 — eval through the training-equivalent harness.** Eidolon validation uses
  **training weights, not EMA** (EMA lag gave false-bad readings). And all
  numbers must come from the trainer's own precision path — the quarantined
  step-10000 top-up (standalone fp32 vs trainer fp8+compile, "recon 0.52→0.81")
  is the standing warning: a parallel re-measurement is a different function.
- **R6 — frozen input definitions.** z_g comes only from
  `encoder_production.npz` (DWPose face slice [23:91], 3D-frontalized);
  auraface_lda from the frozen LDA basis (`auraface_lda.npz`). No recomputing
  conditioning vectors mid-arm (measurement provenance equivalence).

### D3 — the VAE spike, in perspective

`vae-ceiling` was crude by design (a feasibility spike, never pre-registered —
its README says so honestly). Its *conclusion* stands and is now sharper: the
question is not "more latent bandwidth" but "residual decoder at the same
budget". This matters to the visualization engine only in one place: **if P1 is
ever re-run or extended, the AE is the ceiling.** The renderer itself is
pixel-space (P2), so the VAE work is a cheap parallel eval row (Stage 0b), not
a training arm.

## 4. Arm plan

```
Stage 0a  Data + substrate audit (no GPU, ~1h)           [prerequisite]
Stage 0b  VAE decoder eval rows (deterministic, ~30min)  [parallel, optional]
Stage 1   Arm EV-0 — wiring spike (exploratory, 50-100 steps)
Stage 2   Arm EV-1 — eidolon-renderer (confirmatory, warm-start from PP)
Stage 3   Gate EV-G — hegre cross-shoot persistence (eval-only, no training)
```

### Stage 0a — Data and substrate audit (no GPU)

1. Measure FFHQ stratum `auraface_lda.npy` / `z_g.npy` coverage (expect ~70k;
   readdir-only listing per the NFS rule, never rglob). If coverage < 100%,
   the arm's `data_snapshot` records the real number and the scan gates on
   artifact existence (same discipline as `require_pose2`).
2. Spot-check 10 random dirs: auraface vector L2-norm sane, z_g finite, no
   all-zero rows (DWPose zero-keypoint pitfall poisons centroids and could
   poison conditioning here).
3. Verify eval-side AuraFace extraction path exists for the identity gate
   (render → re-embed → cosine): `fal/AuraFace-v1`, Apache 2.0.
4. Verify `EidolonAdapter` constructor kwargs match this branch's
   `train_production.py` injection (Pitfall 0 discipline).

### Stage 0b — VAE decoder ceiling, extended (deterministic, optional)

Extend `scripts/vae_ceiling_test.py` with two rows on the existing 16-image
harness: **DC-AE f32** (MIT, license-clean candidate) and **Qwen-Image-2.1 VAE**
(evaluation-only reference; Qwen Research License is non-commercial). Same
falsification frame as 2026-09-10: real FFHQ → encode → decode → LPIPS +
luminance/contrast vs the frozen band. Feasibility/exploratory — it informs the
next P1 decision, it is not evidence for any training arm.

### Stage 1 — Arm EV-0 `eidolon-renderer-spike` (exploratory wiring sanity)

**Purpose:** prove the *path*, not the hypothesis: eidolon adapter + warm-start
machinery + FFHQ-only eidolon data + all lifecycle hooks, end to end, before
any gate is registered on it (feasibility-before-registration rule).

- 50–100 steps, production precision/batch, `visual_debug_interval ≤ steps/2`,
  `save_every ≤ steps`, validation firing in-window (the step-250 lesson).
- Success = no crash, hooks fire, both artifact dirs non-empty, data loader
  serves FFHQ-only (assert no hegre paths in the scan log), yaw-sweep debug
  collage renders.
- Exploratory by definition; nothing here counts as evidence.

### Stage 2 — Arm EV-1 `eidolon-renderer` (confirmatory — the core arm)

**One-line:** warm-start the pixel backbone from **Arm PP's final checkpoint**,
swap the stratum adapter for the eidolon adapter, train FFHQ-only, and gate on
identity retention + geometry binding + disentanglement + quality.

- **mode:** confirmatory (gate frozen before first run)
- **differs_from:** `pixel-posttrain` (PP). **Scientific variable — exactly
  one:** conditioning scheme stratum (T5+DINO+pose) → eidolon
  (AuraFace-LDA adaLN + z_g cross-attn with per-dim basis). Backbone init (PP
  checkpoint), data (FFHQ stratum), budget, and schedule all inherited from PP.
  vs Arm O the changes are backbone init and FFHQ-only data — state both
  explicitly in `diff_summary` (D1 correction is deliberate).
- **hypothesis:** The PP-warm-started pixel body retains its photorealism
  trajectory under the pure identity+geometry conditioning, and binds the
  identity vector measurably better than Arm O did from scratch at matched
  steps.
- **falsified_if:** identity retention does not beat the mismatched-vector
  null by a pre-registered margin, OR yaw binding is absent, OR quality
  collapses vs PP at the matched step beyond the registered margin.
- **pre_registered_gate (draft — numbers to be frozen at registration):**
  - **G1 identity retention:** median cosine(AuraFace(gen), cond vector)
    exceeds the **mismatched-vector null** (cosine against random other FFHQ
    identities) by ≥ +0.15, on n≥100 fixed-seed samples. *Instrument ceiling
    measured first:* the same metric on PP/Arm-J reconstructions of real FFHQ
    with their own vectors — if the ceiling is below the gate, the gate is
    rewritten against the ceiling, not the hope (feasibility rule).
  - **G2 geometry binding:** z_g dim0 sweep → monotonic DWPose-measured yaw at
    ≥2 checkpoints (Arm O's gate, inherited).
  - **G3 disentanglement:** sweeping z_g (±2σ, 5 points) at fixed identity
    drifts AuraFace cosine by ≤ ε₁; swapping identity at fixed z_g drifts
    DWPose yaw by ≤ ε₂. (ε values frozen after the instrument probe.)
  - **G4 quality:** recon LPIPS ≤ PP@matched-step + 0.03 margin; ≥4/6 G0 gates
    in-band vs frozen calibration.
  - **G5 no collapse:** clean loss/grad, no NaN, non-degenerate full-res
    visuals (deterministic per-step pixel-statistics detector armed).
- **eval protocol:** fixed identity vectors + fixed z_g grid + fixed seeds
  across all checkpoints (intra-sample temporal tracking, user preference).
  Training weights, not EMA (R5).
- **budget:** 10,000 steps under the 1k-segment GPU policy; sanity EV-0 first;
  launcher armed only on a clean tree (Pitfall 19).

**Pivot paths (written before results):**
- G1 fails, G2 passes → identity stream too weak: arm variant raising identity
  CFG weight / lowering identity dropout; consider adding an identity
  cross-attention token alongside adaLN (single-variable follow-up).
- G4 fails, G1/G2 pass → adapter swap disturbed the body: freeze the body,
  train adapter-only (the isolation arm).
- G2 fails → regression vs Arm O: bisect warm-start vs data change first
  (that's why both are declared in `diff_summary`).

### Stage 3 — Gate EV-G `hegre-crossshoot-persistence` (eval-only)

The gate the whole product cares about, and the reason for R1: render from
**hegre persona centroids** (the `averages/*.auraface.npy` the eidolon project
produces), re-embed the renders, measure **cross-shoot retrieval R@1** against
held-out shoots of the same persona, persona-level bootstrap CI. No training,
no hegre pixels near the training loader. Verdict feeds the product question
"can I render *this person* from their vector", which no FFHQ metric can answer.

## 5. Explicit decision points for Tim

1. **R1 hegre exclusion** — corrects Arms O/hegre-geometry. Confirm FFHQ-only
   training + hegre as held-out gate is the agreed data contract.
2. **Sequencing:** EV-1 warm-starts from PP's *final* checkpoint → EV-1 cannot
   launch before PP completes (~8k steps remaining). Alternative (rejected):
   warm-start from P1 directly — duplicates PP's pixel-I/O learning under a
   different adapter and confounds the PP comparison.
3. **Budget:** 10k steps (~5 days of segments) inherited from PP for
   matched-step comparability — confirm before registration.
4. **Stage 0b (VAE eval)** is optional and off the critical path.

## 6. Governance checklist (when approved)

- [ ] `experiment-configs/eidolon-renderer/{README.md,config.yaml,provenance.yaml}`
      committed **before** EV-0 (trio first, `mode` set, gate frozen before EV-1)
- [ ] `docs/EXPERIMENT_TREE.md` registration `[ACTIVE]`
- [ ] new branch `arm/eidolon-renderer` (never reuse history)
- [ ] `python scripts/check_arm_records.py` → exit 0
- [ ] run the plan-skill pre-execution hygiene: clean tree before any launch
