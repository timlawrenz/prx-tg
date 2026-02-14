## Why

Before committing to full-scale training (400M+ parameters, 60k images, weeks of GPU time), we must prove the architecture works on a tiny scale. Training a "Nano" DiT model on 100 images validates that DINOv3 conditioning, T5 cross-attention, and flow matching loss function correctly. This catches architectural bugs, normalization issues, and conditioning failures early, saving GPU hours and preventing false starts.

## What Changes

- Add Nano DiT architecture (12 layers, 384 hidden dim, 6 heads) for validation testing
- Create 100-image overfit dataset loader using WebDataset validation shards (512x512 downsampling)
- Implement dual conditioning (DINOv3 → adaLN-Zero, T5 → Cross-Attention) with independent CFG dropout
- Implement flow matching training loop with logit-normal timestep sampling
- Add EMA model tracking (decay warmup from 0 → 0.9999)
- Add three validation objectives: reconstruction fidelity, DINO swap test, text prompt manipulation
- Training configuration: 5k steps, batch size 8-16 with gradient accumulation, AdamW optimizer
- Output: Validation checkpoint proving architecture correctness before scaling to 400M model

## Capabilities

### New Capabilities
- `nano-dit-model`: Nano-scale DiT architecture (12L/384H/6A) with dual conditioning for validation testing
- `overfit-dataloader`: WebDataset loader for 100-image validation set with 512x512 resizing and aspect bucketing
- `flow-matching-training`: Training loop implementing rectified flow with logit-normal sampling and EMA tracking
- `validation-objectives`: Three-metric validation system (reconstruction, DINO injection, text control) to verify architecture correctness

### Modified Capabilities
<!-- No existing capability requirements are changing -->

## Impact

- **New directory**: `validation/` - Nano model definition, training script, validation dataset loader
- **New dependencies**: `torch`, `diffusers` (for Flux VAE decoder inference), `webdataset`, `numpy`
- **Uses validation shards**: `data/shards/validation/` created by create_webdataset_shards.py (100 samples)
- **Training time**: ~2-4 hours on 4090 (5k steps @ 512x512)
- **Storage**: ~500MB (checkpoints + logs)
- **Validates before**: Part C full-scale training (saves weeks if architecture is broken)
