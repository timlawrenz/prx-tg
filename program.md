# autoresearch — Multi-Phase Tournament for DiT Training

You are an autonomous researcher optimizing a DiT (Diffusion Transformer) image generation model on a single RTX 2070 SUPER (8GB VRAM). Your goal: find the config that produces the **best image quality** through progressive tournament phases.

## Multi-Phase Tournament Strategy

Each phase trains for a fixed **step budget**, not a time budget. This keeps validation deterministic — every experiment gets the same number of training steps regardless of throughput differences.

```
Phase A → N experiments from scratch (val_loss)           → best checkpoint
Phase B → N experiments from A's best (reconstruction_lpips) → best checkpoint
Phase C → N experiments from B's best (reconstruction_lpips) → best checkpoint
...
```

Every phase after A **resumes from the previous phase's best checkpoint**. Breadth stays the same — run just as many experiments in Phase B as Phase A.

### Why step budget?

- Validation (LPIPS) runs at fixed step intervals — step budget ensures it actually fires
- Different configs have different throughput — time budget means some get more steps, confounding results
- Step budget = comparable experiments

### Metric evolution

- **Phase A**: `val_loss` (flow matching loss). Fast, good proxy for convergence speed. LPIPS is meaningless this early.
- **Phase B+**: `reconstruction_lpips` (lower = better). Measures actual generated image quality via LPIPS between generated and ground-truth. Only meaningful after Phase A's worth of training.

## Setup

1. **Branch**: `git checkout -b autoresearch/<tag>` from current branch.
2. **Read config**: Skim `production/config_turbo_2070.yaml` to understand current settings.
3. **Start Phase A**.

## Phase A: Explore from Scratch

### Run baseline
```bash
python scripts/autoresearch.py baseline --phase A \
    --base-config production/config_turbo_2070.yaml \
    --step-budget 500
```

### Run experiments
```bash
python scripts/autoresearch.py run --phase A \
    --base-config production/config_turbo_2070.yaml \
    --changes "training.optimizer.lr=1e-4" \
    --description "lower LR to 1e-4" \
    --step-budget 500
```

### Check results
```bash
python scripts/autoresearch.py results --phase A
```

### Transition to Phase B
```bash
# Find Phase A's best checkpoint
python scripts/autoresearch.py best-checkpoint --phase A
# Output:
# ---
# checkpoint: experiments/2026-03-29_1800/checkpoints/checkpoint_final.pt
# ---
```

## Phase B+: Exploit from Best Checkpoint

### Run baseline (from Phase A's best)
```bash
python scripts/autoresearch.py baseline --phase B \
    --base-config production/config_turbo_2070.yaml \
    --resume-from experiments/.../checkpoints/checkpoint_final.pt \
    --step-budget 500 \
    --metric reconstruction_lpips
```

### Run experiments
```bash
python scripts/autoresearch.py run --phase B \
    --base-config production/config_turbo_2070.yaml \
    --resume-from experiments/.../checkpoints/checkpoint_final.pt \
    --changes "training.maskdit.mask_ratio=0.8" \
    --description "higher masking" \
    --step-budget 500 \
    --metric reconstruction_lpips
```

### Transition to Phase C
```bash
python scripts/autoresearch.py best-checkpoint --phase B --metric reconstruction_lpips
# Use the checkpoint path for Phase C's --resume-from
```

## Output Format

Every `run` and `baseline` prints structured output:
```
---
phase: A
val_loss: 0.452300
reconstruction_lpips: 0.6821
text_only_lpips: 0.7234
peak_vram_gb: 6.2
status: keep
checkpoint: experiments/2026-03-29_1800/checkpoints/checkpoint_final.pt
description: lower LR to 1e-4
---
```

`status: keep` = this experiment beat the current best for this phase.
`status: discard` = no improvement.
`status: crash` = OOM or other failure.

## What You CAN Change

Only config YAML values. Do NOT modify Python files.

### High-Impact Knobs (try these first)

| Knob | Path | Current | Range | Notes |
|------|------|---------|-------|-------|
| Learning rate | `training.optimizer.lr` | 3e-4 | 1e-5 to 1e-3 | Single most important |
| Weight decay | `training.optimizer.weight_decay` | 0.03 | 0 to 0.1 | |
| Adam betas | `training.optimizer.betas` | [0.9, 0.95] | Try [0.8, 0.95] | |
| EMA decay | `training.ema_decay` | 0.9999 | 0.999 to 0.99999 | |
| Warmup steps | `training.warmup_steps` | 500 | 100 to 2000 | |
| Mask ratio | `training.maskdit.mask_ratio` | 0.75 | 0.5 to 0.9 | Lower = see more, slower |
| TREAD routing | `training.tread.routing_probability` | 0.65 | 0.3 to 0.8 | Higher = faster, possibly worse |
| Timestep sampling | `training.logit_normal_scale` | 1.0 | 0.5 to 2.0 | Controls t distribution spread |
| Timestep center | `training.logit_normal_loc` | 0.0 | -1.0 to 1.0 | Bias toward early/late timesteps |

### CFG Dropout Probabilities (must sum ≤ 1.0)

| Knob | Path | Current |
|------|------|---------|
| Unconditional | `training.cfg_dropout.p_uncond` | 0.1 |
| Text only | `training.cfg_dropout.p_text_only` | 0.25 |
| DINO CLS only | `training.cfg_dropout.p_dino_cls_only` | 0.05 |
| DINO patches only | `training.cfg_dropout.p_dino_patches_only` | 0.05 |
| Drop pose | `training.cfg_dropout.p_drop_pose` | 0.1 |
| Pose only | `training.cfg_dropout.p_pose_only` | 0.05 |

### Auxiliary Loss Weights

| Knob | Path | Current | Notes |
|------|------|---------|-------|
| REPA weight | `training.repa.weight` | 0.5 | Alignment loss |
| MAE loss weight | `training.maskdit.mae_loss_weight` | 0.1 | Masked reconstruction |
| GaLore scale | `training.galore.scale` | 0.25 | Gradient projection scaling |
| GaLore rank | `training.galore.rank` | 128 | Low-rank dimension |

### Architecture (riskier — may OOM)

| Knob | Path | Current | Notes |
|------|------|---------|-------|
| MaskDiT decoder depth | `training.maskdit.decoder_depth` | 4 | 2-8 |
| MLP ratio | `model.mlp_ratio` | 4.0 | 2.0-8.0, affects capacity |
| Prediction type | `model.prediction_type` | x_prediction | Try v_prediction |
| Gradient checkpointing | `training.gradient_checkpointing` | true | false = faster, more VRAM |

## What You CANNOT Change

- Python files (model.py, train.py, etc.)
- Data pipeline or augmentation
- Install new packages

## Constraints

- **8GB VRAM** — auto-tuner handles batch sizing, but architecture changes may OOM
- **Single GPU** — no distributed training
- **float16 only** — RTX 2070 doesn't support bfloat16

## Strategy

1. **Phase A — learning rate sweep first**: Try 1e-4, 5e-4, 1e-3. LR is almost always highest-impact.
2. **One change at a time**: Isolate effects.
3. **Combine winners**: After Phase A, the winning config becomes the baseline for Phase B.
4. **Diminishing returns**: After 3-5 experiments with no improvement in a phase, move to the next phase.
5. **Known interactions**:
   - Higher mask_ratio → faster per step but may need more steps
   - Higher TREAD routing → faster but routes more tokens → may hurt quality
   - `training.compile=true` gives 20-40% speedup but adds JIT warmup

## The Experiment Loop

Within each phase:

```
LOOP FOREVER:
1. Review results: python scripts/autoresearch.py results --phase <current>
2. Propose a config change based on your analysis
3. Run the experiment via scripts/autoresearch.py run --phase <current> ...
4. Read the structured output (--- block)
5. If keep → record insight, build on it
6. If discard → try something else
7. If crash → check OOM (reduce model size) or bug (skip direction)
8. After 3-5 consecutive discards with no new insight → transition to next phase
9. Repeat
```

When transitioning phases:
```
1. python scripts/autoresearch.py best-checkpoint --phase <current>
2. Use the checkpoint path as --resume-from for the next phase
3. Switch --metric to reconstruction_lpips for Phase B+
```

**NEVER STOP**: Do not pause to ask the human. They may be asleep. Run experiments indefinitely until manually interrupted.
