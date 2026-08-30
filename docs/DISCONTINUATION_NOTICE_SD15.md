# Discontinuation Notice — Geometry Adapter on Pretrained SD 1.5

**Date:** 2026-07-17
**Verdict:** KILL
**Scope:** Thread-level — all four experiment arms (P, P2, P3, P4) in the "inject geometry tokens into a pretrained SD1.5-family cross-attention stream" research thread.

## What was attempted

A lightweight geometry adapter (~630K params) encodes z_g (50-dim disentangled geometry vector from prx-tg/eidolon) into 50 cross-attention tokens, injected alongside 77 CLIP text tokens at positions 77–126 into a pretrained SD 1.5 UNet. The goal: control face yaw (z_g[0]) via a per-dim learned token basis, leaving the UNet's text-conditioned face generation otherwise intact.

## What was tested

| Arm | Base model | Attention regime | Steps | Result |
|-----|-----------|-----------------|-------|--------|
| P | vanilla SD1.5 | Frozen | 10K | Tokens ignored → artifacts, no yaw |
| P2 | vanilla SD1.5 (pose-stripped captions) | Frozen | 10K | Same — pose stripping didn't help |
| P3 | AbsoluteReality v1.8.1 | Frozen | 2.3K killed | Garbled output immediately |
| P4 | vanilla SD1.5 | LoRA-unfrozen (rank-8, ~5M params) | 10K | Loss diverged +48%; collapsed to noise |

40,000+ cumulative training steps. Three base models. Two attention regimes. **Zero controlled yaw.**

## Why it didn't work

The original hypothesis was "frozen cross-attention can't attend to novel tokens." P4 was the direct falsification test: unfreeze attention via LoRA. It failed *worse* — the model degraded monotonically (loss minimum at step ~1K–2K, then +48% divergence). Per the pre-registered falsification clause in P4's provenance.yaml, this means the problem is **deeper than attention trainability**.

**Refined root cause** (code audit deleg_141d713e, confirmed by P4 tensorboard loss curve):

The ε-prediction denoising loss provides **no gradient signal that rewards geometry control**. The frozen text/CLIP path already denoises the face from the caption alone — the geometry tokens are redundant to the loss function. The adapter and LoRA weights receive only noisy, uninformative gradients. Over thousands of steps, the trainable weights either:

1. **Stay inert** (frozen attention, P/P2/P3): geometry tokens ignored by the UNet → stable-but-inert differential noise where all z_g values produce similar output.
2. **Random-walk into destruction** (LoRA, P4): ~5M K,V params diffuse away from their zero-init under unconstrained noisy gradients (no gradient clipping, ×2.0 LoRA scale, trainable null_geometry corrupting CFG baseline) → progressive UNet collapse to pure noise.

Secondary enablers (compounding, not causal): no gradient clipping, flat LR 1e-4 with no warmup/decay, weight_decay on LoRA, ×2.0 LoRA scale.

## What survived

- **The z_g data is valid.** FFHQ z_g[0] = yaw with R²=0.996 (from eidolon). The vectors are clean — the problem is using them as conditioning in a pretrained cross-attention stream.
- **The per-dim token basis is proven** (Arm N, from-scratch DiT, yaw binding at steps 2000–2500). It works in a model that co-evolves attention patterns with conditioning tokens from scratch. It does NOT work when injected into a pretrained model where the denoising loss doesn't reward it.
- **The diagnostic purpose was served.** The SD1.5 thread proved that prx-tg's from-scratch DiT collapse (Arm N) is NOT merely a from-scratch-training-dynamics problem — a frozen backbone failed the same injection. The collapse mechanism in Arm N is separate and was already diagnosed (CFG guard on per-dim basis).

## What NOT to try

- More SD1.5 base models (photorealistic, anime, etc.) — P3 ruled out base-model quality as the bottleneck.
- Caption engineering (pose-stripping, prompt rewriting) — P2 ruled out text/pose conflict as the bottleneck.
- Different LoRA ranks/scales, gradient clipping, cosine LR schedules — these address the *acceleration* of the P4 degradation but not the root cause (uninformative loss). Training longer or more carefully on the same loss surface will hit the same wall.

## What to redirect to

The from-scratch DiT path (eidolon Arm O, `zg-token-basis-cfg-guard`) is the natural home for geometry-conditioning research. In a from-scratch model, the attention patterns and the geometry tokens co-evolve, AND the denoising loss IS informative for geometry (there is no pretrained shortcut). Arm N proved the per-dim basis works at the architecture level; Arm O's CFG guard addresses the specific collapse mechanism.

## Collateral locations

- Ledger: `docs/EXPERIMENTS_AND_RESULTS.md` — Arms P/P2/P3/P4 entries + thread-level synthesis
- Tree: `docs/EXPERIMENT_TREE.md` — concluded series, refined root cause, project PIVOT recommendation
- Project status: `PROJECT_STATUS.md`
- Code audit: deleg_141d713e (full root-cause analysis)
- Experiment artifacts: `experiments/geometry-adapter-sd15/`, `experiments/geo-adapter-bugfix-restart/`, `experiments/geo-adapter-absreal/`, `experiments/geo-adapter-lora-crossattn/`

## Disposition of collateral

- **Checkpoints:** Keep (40K+ steps of negative evidence). Standard retention — last 10 per run.
- **TensorBoard:** Keep (P4 loss divergence curve is the key quantitative evidence).
- **Eval sweeps:** Keep. In-domain sweeps at P4 steps 1000/2000; cross-model (AbsoluteReality) sweeps at P4 steps 1000/2000/3000/5000/10000 annotated as confounded.
- **Code:** Keep under `experiments/geometry-adapter-sd15/src/`. The production change in `production/data_stratum.py` (`sd15_geometry` adapter mode + `_strip_pose_words`) is thread-scoped and isolated.

## Contact / point of decision

Tim (project owner). This notice formalizes the thread KILL; redirection to Arm O is recommended but not pre-decisional.
