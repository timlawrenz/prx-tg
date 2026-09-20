# Arm J (`faces70k-fp8`) — G0 gate baseline at step 10,000

This is the **criterion-3 anchor** for the frozen `pixel-posttrain` (Arm PP) gate:
the champion's G0 physics at the matched 10,000-step budget.

Measured 2026-09-20. Arm J's run is `experiments/2026-06-04_2116`.

## Result — 4 of 6 gates in band

| gate | value | n | band | verdict |
|---|---|---|---|---|
| `g0a_sensor_noise_floor_full` | −1.55285 | 10 | real-FFHQ p5..p95 | **OUT** |
| `g0b_spectral_slope` | −1.46603 | 10 | " | IN |
| `g0c_skin_texture_energy_full` | +0.00253 | 10 | " | **OUT** |
| `g0d_local_contrast_full_p5` | +0.01086 | 10 | " | IN |
| `g0d_local_contrast_full_p50` | +0.05212 | 10 | " | IN |
| `g0d_local_contrast_full_p95` | +0.19148 | 10 | " | IN |

Band source: `research/avenues/gates_calibration.json` (frozen real-FFHQ calibration).

## Why it had to be regenerated

June's run kept `validation_evals/step0010000/` but only the **collages** — no
`prompt_*.png`. G0 measures image statistics, so it cannot be computed from a
collage (borders, gutters and labels are not model output). The checkpoint
survived (`checkpoints/checkpoint_step0010000.pt`, 2.88 GB), so the images were
regenerated from it rather than substituting a proxy.

## How

1. `scripts/evaluate_checkpoint.py -c experiments/2026-06-04_2116/checkpoints/checkpoint_step0010000.pt
   --config experiments/2026-06-04_2116/config.yaml -o experiments/2026-06-04_2116/g0_step0010000/images`
   — the tool's 10 fixed prompts with fixed seeds (42, 142, … 942).
2. `measure_g0_prompts.py --glob 'prompt_*.png'` — reuses `run_gates.measure_image`
   (the frozen preprocessing; no reimplementation) and writes the same JSONL shape.

Images are hard data and live on the NAS:
`experiments/2026-06-04_2116/g0_step0010000/images/` (10 × `prompt_NN.png` + `collage.png`).

## Blocked on a real bug, then verified

The first attempt **failed to load the checkpoint at all**: Arm J (2026-06) stores
adapter tensors bare (`dino_proj.weight`) while the current model expects
`adapter.dino_proj.weight` — 15 tensors. The loader was a strict `load_state_dict`,
so no pre-adapter-wrapper checkpoint could be evaluated. Fixed in `910e1e0` by
re-namespacing only where the prefixed form is expected and the bare form is not,
and by refusing to evaluate when anything is missing (a `strict=False` load would
have silently dropped the learned `null_*` embeddings and still produced numbers).

**Independently verified as the right weights**, because the regeneration reproduces
June's recorded per-prompt scores exactly:

| | regenerated | June record |
|---|---|---|
| prompt_0 clip | 0.17251574993133545 | 0.17251574993133545 |
| prompt_0 aesthetic | 5.075051307678223 | 5.075051307678223 |
| prompt_1 clip | 0.23517698049545288 | 0.23517698049545288 |
| prompt_1 aesthetic | 4.7438273429870605 | 4.7438273429870605 |

Face confidence differs only at ~1e-5 (ONNX/DWPose run-to-run), and the summary
means match to the printed precision (aesthetic 4.82, CLIP 0.219, face-conf 0.858).

## Two caveats a reader must know

**1. n = 10 here, n = 11 in the older records.** The existing G0 records
(`gates/dip-conv-head/step0010000/`, `gates/gamma2-noise-scale/10000/`) globbed
`*.png` and therefore included `collage.png`. Re-measuring `dip-conv-head` at
n = 10 reproduced its recorded n = 11 values within ~0.005 **with identical
in/out verdicts** (3 in, 3 out), so the collage is immaterial to the verdict —
but the n = 10 measurement is the cleaner one and is what this baseline uses.

**2. These are RAW-weight numbers, not EMA.** `evaluate_checkpoint.py` intends to
apply EMA and does not (see the ledger entry for the finding), and the trainer's
own validation path likewise never applies the EMA it is handed. Both the anchor
and Arm PP's future numbers come from those same two paths, so the comparison is
like-for-like — but the "EMA" label on these metrics is false, and the release
path ships EMA weights, so the reported numbers do not describe the shipped
artifact. Recorded here so the caveat travels with the number.

## What clearing criterion 3 does and does not mean

Arm J — a from-scratch model that is **not** photoreal — already sits at 4/6. The
frozen gate asks Arm PP for ≥ 4/6, i.e. "physics no worse than the champion's".
So passing criterion 3 is **necessary but not sufficient** for photorealism: the
champion passes it too. Verdict language must not read a 4/6 (or 5/6) as
"photorealism achieved".
