# Arm: latent-first-pretrain

**Status:** P1 training complete (step 10000/10000, 2026-09-19 23:23) · **verdict: active** ·
final-step validation MISSING (see below) · no ledger entry yet.

Recovered into `experiment-configs/` on 2026-09-20. This arm ran from 2026-09-13 to
2026-09-19 with its config living **only** on the NAS behind the `experiments/`
symlink, so git could not see it (AGENTS.md §0.2 / §0.7 trap). The `config.yaml` here
is a copy of the frozen run-start config at
`experiments/latent-first-pretrain/runs/2026-09-13_2225/config.yaml` (identical apart
from a trailing newline).

## Hypothesis

Scheme B Phase 1: learning semantics in **FLUX-AE latent space** (16ch, 8×; 1024px →
128×128 latent, patch_size 2 → the same 64×64 token grid as the pixel ps16 champion)
reaches comparable structural/semantic fidelity **faster** than pixel-from-scratch, so
the remaining photorealism work can move to a cheaper Phase-2 pixel post-train.

## Falsified if

Latent-space pretraining fails to reach comparable or better structural fidelity than
the pixel-space champion at an equal step budget, or decoded latents show no semantic
gain over pixel-from-scratch (i.e. no reduction in the fidelity work left for Phase 2).

## Pre-registered gate

**None — retro-declared.** This arm was launched 2026-09-13 before AGENTS.md §0.2
existed; no numeric gate was frozen before the first run. The only contemporaneous
a-priori statement is the config header (written 2026-09-13 21:59, before the 22:25 run
start): *"semantic/structural fidelity faster than pixel-from-scratch; measured by G0
gates + blind review on decoded latents."* Per §0.7 rule 3, a result that was not
pre-registered is **exploratory, permanently** — `mode: exploratory` in
`provenance.yaml` reflects that, and should be confirmed rather than quietly upgraded.

## Differs from

`faces70k-fp8` (pixel-space champion). Same 768×18 body, same 70k FFHQ stratum data,
same conditioning stack (T5 + DINOv3 + pose). Changes:

| | champion | this arm |
|---|---|---|
| `in_channels` | 3 | 16 (FLUX-AE latent) |
| `patch_size` | 16 | 2 (keeps the 64×64 token grid) |
| `prediction_type` | — | `x_prediction` on the clean **latent** x0 |
| data | `pixel.npy` | `flux_latent.npy` |
| `latent_space` | false | true (samplers decode via FLUX AE) |
| `sampling.num_steps` | 50 | 24 |

Unchanged: Muon lr 3e-4 → min_lr 1e-6, REPA 0.5 with decay 4000→8000, TREAD
self-guidance 3.0, fp8, eff-batch 256 (4×64).

## Expected outcome

Structural/semantic fidelity reaching champion level in fewer steps, judged on G0
physics gates plus blind review of decoded latents. Photorealism is explicitly **not**
expected from P1 — it is the job of the Phase-2 pixel post-train.

## What actually happened

**Loss** (mean per 1000-step bucket) converged smoothly, no NaN/divergence:

```
1000-2000  1.7548      6000-7000  1.4363
2000-3000  1.6312      7000-8000  1.4156
3000-4000  1.5628      8000-9000  1.4050
4000-5000  1.5072      9000-10000 1.4034   (final 1.4091, grad 0.38, lr 1e-6)
5000-6000  1.4671
```

Grad norms 0.25–0.7 throughout; lr parked at `min_lr` 1e-6. Ran ~4.6 days cumulative
across resumes, at ~35 s/it (degrading to ~55–65 s/it whenever the BreastTissue-Net
labeler shared the 4090).

**In-train validation** (reconstruction / text-only LPIPS, 4-sample hook):

```
step     recon    text_only
7000     0.5094   0.8647
7500     0.5049   0.8592
8000     0.4981   0.8594   <- best recon
8500     0.5011   0.8528
9000     0.5161   0.8559
9500     0.5235   0.8743
```

## Known gaps and caveats (read before citing anything above)

1. **The final-step (10000) validation is MISSING.** The 1k-segment GPU policy SIGTERM'd
   the trainer at its final boundary *during* the end-of-run validation, so
   `validation/step0010000/results.json` was never written and `checkpoint_final.pt` was
   never produced. Details in `.../validation/NOTE.md`. Because
   `validation.interval_steps: 2500` but the hook actually fired every 500, step 9500 is
   the last complete number.

2. **A standalone re-run of the missing validation is NOT valid, and I tried it.**
   A driver that rebuilds the model and calls the same `create_validation_fn` produced
   recon LPIPS 0.8018 at step 10000 — and, run as a control against step 9500 (known
   in-train value 0.5235), it produced 0.8060. It fails its own control. Root cause:
   `torch.manual_seed` appears **only** in `run_divergence_test` (`validate.py:508`),
   which this arm skips because `self_guidance: true`. So reconstruction, text-only,
   dino-swap and text-manip all sample from the **global RNG state**, which is a function
   of the whole training history. Reproducing fp8 via `convert_to_float8_training`
   changed nothing (image correlation stayed +0.31…+0.49 vs in-train). The invalid
   outputs are quarantined at `validation/_INVALID_step0010000_driver-mismatch/`.
   **Do not read 0.8018 as a regression** — read naively it looks like a 53% collapse in
   the last 500 steps.

3. **The recon/text-only LPIPS trajectory is not seed-controlled.** Same root cause as
   (2): each validation call draws fresh noise from whatever the global RNG state is at
   that moment. The gentle upward drift after step 8000 (0.4981 → 0.5235, coinciding with
   the REPA weight reaching 0 at `decay_end_step: 8000`) is therefore **partly RNG drift,
   not purely model change**, and the per-step values are not a clean like-for-like
   comparison. Fixing this (seed the samplers per sample index) is the single highest-value
   change for this arm's evidence quality.

4. **Objective pixel statistics do not support a large visual improvement.** On the
   `visual_debug` images (fixed prompts, 4 samples/step, steps 7000→10000): Laplacian
   variance 367.0 → 382.9 (**+4.3%**), grain index 2.555 → 2.549 (**−0.2%, flat**),
   saturation 137.2 → 145.1 (+5.8%). A vision-model read claimed "graininess almost
   entirely eliminated" — the measurement does not support that. Sharpness improved
   modestly. A malformed-hands defect persists on the "fair-skinned woman" prompt.

## Runs

- `experiments/latent-first-pretrain/runs/2026-09-13_2225` — **canonical run** (this arm;
  resumed repeatedly, final segment 2026-09-19 14:00 → 23:25).
- Earlier aborted/partial attempts in the same arm dir: `2026-09-12_1046`,
  `2026-09-12_1254`, `2026-09-12_1313`, `2026-09-13_1400`.
- Sanity runs live under the separate slug `experiments/latent-first-pretrain-sanity/`.

`metadata.json` records `git_commit 0209a2634473319b34e4f1c9749e8c175b0d41a1`,
`git_dirty: false` at run start; the final resume segment recorded commit `086d9c0`
with `dirty: True`, so the tail of the run is **not** bit-reproducible from git alone.

## Provenance / reproducibility notes

- `git_dirty: false` at run start; the final resumed segment was dirty (see above).
- 21 of this project's 37 arms lost their provenance to the NAS-symlink trap; this arm is
  no longer one of them, but the recovery is retroactive documentation, not a real
  pre-registration. Treat every conclusion from it as exploratory.
