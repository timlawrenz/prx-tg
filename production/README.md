# Production Training

Production-scale training for the dual-conditioned DiT model.

## Quick Start

```bash
# Train with default config
python -m production.train_production

# Train with custom config
python -m production.train_production --config my_config.yaml

# Resume from checkpoint
python -m production.train_production --resume checkpoints/checkpoint_step010000.pt
```

## Configuration

All hyperparameters are controlled via `config.yaml`. See `config.yaml` for full documentation.

Key sections:
- `model`: Architecture (depth, hidden_size, patch_size)
- `training`: Optimization, scheduling, CFG dropout
- `data`: Shards, buckets, augmentation
- `validation`: Frequency, tests to run
- `checkpoint`: Save frequency, retention

## Architecture

Copied from `validation/` with config-driven initialization:
- `model.py`: NanoDiT architecture (proven in validation)
- `train.py`: Trainer + ProductionTrainer (config wrapper)
- `data.py`: WebDataset loader
- `sample.py`: Euler sampler for dual CFG
- `config_loader.py`: YAML config loading

## Status

**Stage 1 (Foundation): In Progress**
- ✅ Directory structure
- ✅ Config system (YAML + dataclasses)
- ✅ ProductionTrainer wrapper
- ✅ Basic dataloader (single bucket)
- ⏳ Testing initial training run
- 📝 TODO: Multi-bucket sampling (Stage 3)
- 📝 TODO: Gradient accumulation (Stage 2)
- 📝 TODO: EMA warmup (Stage 2)

## Differences from validation/

- **Config-driven**: All params from YAML, not hardcoded
- **Scalable**: Supports 12-28 layers, 384-1024 hidden
- **Production features** (coming in Stage 2):
  - Gradient accumulation for large effective batch
  - Logit-normal timestep sampling
  - EMA warmup schedule
  - Advanced monitoring

## Training Recipes

### Small (Validation)
```yaml
model:
  depth: 12
  hidden_size: 384
training:
  batch_size: 8
  grad_accumulation_steps: 1
```

### Medium (Intermediate)
```yaml
model:
  depth: 24
  hidden_size: 512
training:
  batch_size: 8
  grad_accumulation_steps: 16  # Effective: 128
```

### Production (Full Scale)
```yaml
model:
  depth: 28
  hidden_size: 1024
training:
  batch_size: 4
  grad_accumulation_steps: 64  # Effective: 256
  gradient_checkpointing: true
```

## Implementation Plan

See `docs/implementation-plan.md` for detailed 5-stage plan.

**Current Stage:** Stage 1 (Foundation)
