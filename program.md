# autoresearch — DiT Experiment Program

You are an autonomous researcher optimizing a DiT (Diffusion Transformer) image generation model on a single RTX 2070 SUPER (8GB VRAM). Your goal: find the config that achieves the **lowest validation loss** within a fixed training time budget.

## Setup

1. **Verify the branch**: `git checkout -b autoresearch/<tag>` from current branch.
2. **Read context**: Skim `production/config_turbo.yaml` to understand the current config.
3. **Run baseline**: Establish reference val_loss:
   ```
   python scripts/autoresearch.py baseline --base-config production/config_turbo.yaml --time-budget 15
   ```
4. **Check results**: `python scripts/autoresearch.py results`
5. **Confirm and go**.

## Running Experiments

```bash
python scripts/autoresearch.py run \
    --base-config production/config_turbo.yaml \
    --changes "training.maskdit.mask_ratio=0.8" \
    --description "increase mask ratio from 0.75 to 0.8" \
    --time-budget 15
```

The script automatically:
- Creates a temporary config with your changes
- Auto-tunes throughput (batch_size, grad_accum) for this specific config
- Trains for the time budget
- Evaluates validation loss
- Reports keep/discard

### Output format

```
---
val_loss: 0.452300
peak_vram_gb: 6.2
status: keep
description: increase mask ratio from 0.75 to 0.8
---
```

## What you CAN change

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
| REPA weight | `training.repa.weight` | 0.5 | Alignment loss (if enabled) |
| MAE loss weight | `training.maskdit.mae_loss_weight` | 0.1 | Masked reconstruction |
| GaLore scale | `training.galore.scale` | 0.25 | Gradient projection scaling |
| GaLore rank | `training.galore.rank` | 128 | Low-rank dimension |

### Architecture (riskier — may OOM)

| Knob | Path | Current | Notes |
|------|------|---------|-------|
| MaskDiT decoder depth | `training.maskdit.decoder_depth` | 4 | 2-8, affects reconstruction quality |
| MLP ratio | `model.mlp_ratio` | 4.0 | 2.0-8.0, affects model capacity |
| Prediction type | `model.prediction_type` | x_prediction | Try v_prediction |
| Gradient checkpointing | `training.gradient_checkpointing` | true | false = faster but more VRAM |

## What you CANNOT change

- Python files (model.py, train.py, etc.)
- Data pipeline or augmentation
- The evaluation metric (validation flow matching loss)
- Install new packages

## Constraints

- **8GB VRAM** — the auto-tuner handles batch sizing, but architectural changes that increase memory (e.g., deeper MaskDiT decoder, larger MLP ratio) may cause OOM
- **Single GPU** — no distributed training
- **float16 only** — RTX 2070 doesn't support bfloat16

## Strategy Suggestions

1. **Start with learning rate**: Try 1e-4, 5e-4, 1e-3. LR is almost always the highest-impact knob.
2. **One change at a time**: Isolate the effect of each change.
3. **Combine winners**: After finding individual improvements, try combining them.
4. **Diminishing returns**: After 3-5 experiments with no improvement, try a more radical change.
5. **Known interactions**:
   - Higher mask_ratio → trains faster per step but may need more steps → adjust time budget
   - Higher TREAD routing → faster but routes more tokens around blocks → may hurt quality
   - Enabling `training.compile=true` gives 20-40% speedup but adds JIT warmup time

## The Experiment Loop

LOOP FOREVER:

1. Review results.tsv — what worked, what didn't
2. Propose a config change based on your analysis
3. Run the experiment via `scripts/autoresearch.py run`
4. Read the structured output (`---` block)
5. If `status: keep` — good, record the insight and build on it
6. If `status: discard` — the change didn't help, try something else
7. If `status: crash` — check if it's OOM (reduce model size) or a bug (skip this direction)
8. Repeat

**NEVER STOP**: Do not pause to ask the human. They may be asleep. Run experiments indefinitely until manually interrupted. If you run out of ideas, re-read this file, try combining previous near-misses, or try more radical changes.
