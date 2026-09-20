# Arm PP — `pixel-posttrain` (Phase-2 pixel post-train, warm-started from P1)

> **Status:** pre-registered, not yet run. Gate frozen 2026-09-20 before the first run.

## Hypothesis

Warm-starting pixel-space training from the **P1 latent-pretrain body** (182 of 188
tensors) beats the from-scratch pixel champion at the **same 10,000-step budget**,
because P1 already supplies semantic structure and leaves P2 only the texture /
photorealism work.

This is the bet the whole two-part program rests on. P1's README states its purpose
as reaching comparable structural/semantic fidelity *faster* than pixel-from-scratch
"so the remaining photorealism work can move to a cheaper Phase-2 pixel post-train."
Arm PP is the test of that sentence.

## Differs from

`faces70k-fp8` (Arm J — the pixel-space champion, `experiments/2026-06-04_2116`,
trained to step 35,000).

**Scientific difference — exactly one:**

- The model is **warm-started** from the P1 step-10,000 checkpoint via
  `--init-from` (`production/warm_start.py`). Every shape-matching tensor transfers;
  the three I/O projection modules (`x_embedder`, `final_proj`, `output_conv`) are
  re-initialised; the optimizer starts **fresh**; the EMA is **re-seeded** from the
  warm weights.

**Instrumentation / budget differences (not scientific variables):**

| Setting | Arm J | Arm PP | Why |
|---|---|---|---|
| `total_steps` | 40000 | 10000 | user budget decision |
| `checkpoint.save_every` | 2500 | 500 | a checkpoint must land exactly ON each 1k segment boundary |
| `validation.visual_debug_interval` | 0 (off) | 1000 | the segment policy releases the GPU at boundaries *so the visuals can be reviewed* |
| `validation.interval_steps` / `num_samples` | 500 / 25 | **verbatim** | keeps the step-10,000 validation like-for-like with Arm J's |
| `repa.decay_start_step` / `decay_end_step` | 8000 / 16000 | **verbatim** | see below |

**On the REPA window (deliberate, and a known consequence):** P2 is only 10k steps, so
keeping Arm J's 8000→16000 verbatim means REPA weight is still **0.375 at the gate
step** — P2 never fully sheds the REPA scaffold inside this budget, unlike P1 (which
used 4000→8000). That was chosen on purpose: it makes the step-10,000 comparison
against Arm J one-variable. Read the result with that in mind — if P2 clears the gate
anyway, the warm start is doing real work; if it does not, the REPA constraint is a
live confound and the follow-up is the same arm with a scaled window.

## Warm-start accounting (verified, not assumed)

From `production/warm_start.py`, checked on CPU against the real P1 checkpoint
(`checkpoint_step0010000.pt`, sha256 `6bd10df2…`):

```
transferred : 182
skipped     : 6
missing     : 0
```

The 6 re-initialised tensors are the weight+bias of the three I/O projections:

| tensor | P1 (latent) | PP (pixel) |
|---|---|---|
| `x_embedder.proj.weight` | (768, 16, 2, 2) | (768, 3, 16, 16) |
| `x_embedder.proj.bias` | (768,) | (768,) |
| `final_proj.weight` | (64, 768) | (768, 768) |
| `final_proj.bias` | (64,) | (768,) |
| `output_conv.weight` | (16, 16, 3, 3) | (3, 3, 3, 3) |
| `output_conv.bias` | (16,) | (3,) |

`x_embedder.proj.bias` is the one subtlety: it is `(768,)` in **both** spaces, so a
pure shape rule would transfer it and leave that module *half*-warm (new weight, old
bias). The warm start therefore takes an explicit `force_reinit` list of module
prefixes rather than relying on shape luck — the test asserts both behaviours.

## Expected outcome

At step 10,000, Arm PP beats Arm J's step-10,000 image metrics. Arm J was still
REPA-constrained and still buying structure at 10k; P1 has already paid for that. If
PP does **not** beat it, the latent pretrain bought nothing for photorealism at this
budget and the P1→P2 program is falsified.

## Frozen gate

Anchored on **Arm J's own recorded numbers at step 10,000** — matched step, same 70k
stratum data, same architecture, same validation protocol (n=25, interval 500):

| metric | Arm J @ 10k |
|---|---|
| recon LPIPS | 0.8264 |
| text_only LPIPS | 0.9375 |
| text_manip LPIPS diff | 0.3443 |
| aesthetic | 4.8183 |
| CLIP | 0.2193 |
| face conf | 0.8585 |

**PASS** if at step 10,000 **all** of:

1. recon LPIPS ≤ 0.8264 **and** text_only ≤ 0.9375 **and** text_manip ≤ 0.3443;
2. aesthetic ≥ 4.8183 **and** CLIP ≥ 0.2193 **and** face_conf ≥ 0.8585;
3. **≥ 4 of the 6 G0 gates in-band** vs the frozen real-FFHQ band
   (`research/avenues/gates_calibration.json`);
4. no collapse — clean loss/grad, no NaN through the first 1,000 steps,
   non-degenerate visuals.

**Falsified if** P2 does not beat Arm J's matched step-10,000 values on the
decoded-image metrics, or its G0 gates sit no closer to the real-FFHQ band.

### Note on criterion 3's baseline

Arm J kept only `collage.png` per step — no `prompt_*.png` — so its step-10k G0
cannot be read off disk. Its `checkpoint_step0010000.pt` survives, so the 10 prompt
images are regenerated from it and G0 is measured on those (a short GPU job).

Two measurement caveats, recorded now so the comparison is read honestly:

- The existing G0 records for `dip-conv-head` and `gamma2-noise-scale` aggregate
  `n: 11` — they include `collage.png`, which has borders and labels and does not
  belong in a physics aggregate. Arm PP's baseline and its own G0 are measured on
  `prompt_*.png` only (`n: 10`), so those older numbers are **not** exactly
  like-for-like with these.
- G0 is a photorealism-physics gate, and photorealism is a **Phase-2** property; the
  P1 arm's own LPIPS is not a validated proxy for final quality. Criterion 3 is
  therefore read as "is PP closer to real-image physics than a from-scratch pixel
  model at the same step", not as "PP is photorealistic".

## GPU policy

Runs under the **1k-segment policy** (user directive 2026-09-18): reserve the 4090
only to the next 1,000-step boundary, release at the boundary so other projects get
the GPU, and auto-continue if nothing else is queued. Requires
`save_every: 500` so a clean checkpoint exists on each boundary.

## Provenance

Init weights: `experiments/latent-first-pretrain/runs/2026-09-13_2225/checkpoints/checkpoint_step0010000.pt`
(2,863,122,225 bytes, sha256 `6bd10df2db93b7945b1b1fa0b8982ea9fb87811895eff7d5e38f598a6b89b955`),
source `ck['ema']['ema_params']`.
