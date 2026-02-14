## Nano DiT Validation

Validation training for the Nano DiT architecture before full-scale training. This proves the dual conditioning (DINOv3 + T5) and flow matching implementation work correctly on a 100-image overfit dataset in just **15-20 minutes** on an RTX 4090.

> **⚠️ IMPORTANT:** Checkpoints created before commit `329c404` (2026-02-14) are **invalid**. 
> They were trained with broken CFG dropout semantics and must be discarded and retrained.
> See "CFG Dropout Fix" section below.

### Purpose

Before committing weeks of GPU time to train the 400M+ parameter production model, we validate the architecture on a minimal scale in just 15-20 minutes:
- **Model**: 12 layers, 384 hidden dim, 6 heads (~30-50M parameters)
- **Data**: 100 images at 512×512 resolution  
- **Training**: 5,000 steps (~10 minutes pure training, ~15-20 min with validation)
- **Validation**: 3 tests (reconstruction, DINO swap, text manipulation)

### Hardware Requirements

- **GPU**: 24GB VRAM minimum (tested on RTX 4090)
- **RAM**: 32GB system RAM recommended
- **Disk**: ~2GB (checkpoints, validation outputs, logs)
- **Training time**: 15-20 minutes for 5,000 steps (including validation)

### Installation

```bash
# Install dependencies
cd validation
pip install -r requirements.txt

# Verify data exists
ls ../data/shards/100/bucket_*/shard-*.tar
# Should show 5 shard files (100 samples total)
```

### Usage

#### Basic Training

```bash
# Train from scratch
python -m validation.run_validation

# With custom settings
python -m validation.run_validation \
  --batch-size 16 \
  --total-steps 5000 \
  --peak-lr 3e-4 \
  --checkpoint-dir my_checkpoints
```

#### Resume Training

```bash
# Resume from checkpoint
python -m validation.run_validation \
  --resume checkpoints/checkpoint_step0003000.pt
```

#### Skip Validation (Faster Testing)

```bash
# Skip validation tests (saves ~5 min total)
python -m validation.run_validation --no-validation
```

### Configuration

Default hyperparameters (defined in `config.py`):

**Model:**
- Layers: 12
- Hidden size: 384
- Attention heads: 6
- Patch size: 2

**Training:**
- Batch size: 8
- Total steps: 5,000
- Peak LR: 3e-4 (linear warmup over 5k steps)
- Min LR: 1e-6 (cosine decay)
- Weight decay: 0.03
- Gradient clipping: 1.0
- EMA decay: 0.9999 (with warmup)

**CFG Dropout:**
- Both signals dropped: 10%
- Text-only (DINO dropped): 10%
- DINO-only (text dropped): 10%
- Both signals active: 70%

**Validation:**
- Frequency: Every 1,000 steps
- Sampling: 50 Euler steps
- CFG scales: text=3.0, DINO=2.0

### Validation Metrics

#### Test 1: Reconstruction Fidelity
- Generates all 100 images from original caption + DINO embedding
- Measures LPIPS distance to ground truth
- **Success criterion**: Mean LPIPS < 0.2 (indicates successful memorization)

#### Test 2: DINO Embedding Swap
- Swaps DINO embeddings between 5 image pairs
- Keeps original captions
- **Visual inspection**: Generated image should adopt composition/lighting of swapped DINO source

#### Test 3: Text Manipulation
- Modifies captions (e.g., "left" → "right", "sitting" → "standing")
- Keeps original DINO embedding
- **Visual inspection**: Generated image should follow text changes

### Output Structure

```
checkpoints/
  checkpoint_step0001000.pt  # Every 1,000 steps
  checkpoint_step0002000.pt
  ...
  checkpoint_final.pt
  training_log.jsonl         # Loss, gradient norm, LR per step

validation/
  step0001000/
    reconstruction/          # All 100 regenerated images
    dino_swap/               # 5 pairs with swapped DINO
    text_manip/              # 5 samples with modified captions
    results.json             # LPIPS scores, metadata
  step0002000/
    ...
```

### Interpreting Results

#### Success Indicators (Go for Part C)
- ✅ Loss decreases monotonically
- ✅ Gradient norm stays in range [0.1, 10.0]
- ✅ Reconstruction LPIPS < 0.2 at step 5,000
- ✅ DINO swap shows composition transfer (visual)
- ✅ Text manipulation shows spatial control (visual)

#### Failure Indicators (Debug Architecture)
- ❌ Loss plateaus or increases
- ❌ Gradient norm explodes (>100) or vanishes (<0.01)
- ❌ Reconstruction LPIPS > 0.3 at step 5,000
- ❌ Generated images look identical regardless of conditioning
- ❌ Images are mean-gray or collapsed to noise

### Troubleshooting

**Out of Memory:**
- Reduce batch size: `--batch-size 4`
- Check GPU memory: `nvidia-smi`

**Slow Training:**
- Ensure CUDA is available: `torch.cuda.is_available()`
- Check GPU utilization: `nvidia-smi dmon`
- Disable validation for testing: `--no-validation`

**Validation Fails:**
- Ensure validation shards exist: `ls data/shards/validation/`
- Check LPIPS installation: `pip list | grep lpips`
- Check diffusers installation (for VAE): `pip list | grep diffusers`

**Checkpoint Loading Fails:**
- Verify checkpoint path exists
- Ensure checkpoint matches current model architecture
- Check PyTorch version compatibility

### Development

Run unit tests for individual components:

```bash
# Test model forward pass
python -m validation.model

# Test dataloader
python -m validation.data

# Test training loop
python -m validation.train

# Test sampler
python -m validation.sample

# Test configuration
python -m validation.config
```

### Next Steps

After successful validation:
1. Review validation outputs in `validation/step0005000/`
2. Verify LPIPS < 0.2 and visual quality
3. Document results in `validation/RESULTS.md`
4. **If successful**: Proceed to Part C (full-scale training)
5. **If failed**: Debug architecture, re-run validation

### CFG Dropout Fix (2026-02-14)

**Issue:** Initial implementation used independent Bernoulli masks for CFG dropout, causing overlapping conditions:
- Some samples had both text AND DINO dropped simultaneously
- Model never learned clean unconditional/text-only/DINO-only modes
- Dual CFG sampling expects distinct modes that weren't properly trained

**Fixed in commit `329c404`:** Now uses mutually exclusive categorical sampling:
- Exactly 70% both conditionings present
- Exactly 10% unconditional (both dropped)
- Exactly 10% text-only (DINO dropped)
- Exactly 10% DINO-only (text dropped)

**Impact:** Checkpoints trained before this fix are **invalid** and must be discarded. The model's conditioning behavior was corrupted, explaining high text sensitivity in validation (LPIPS diff 0.64).

**Action:** Delete old checkpoints and retrain:
```bash
rm checkpoints/checkpoint_*.pt
python -m validation.run_validation
```

### Reference

Architecture based on:
- DiT (Diffusion Transformer): Peebles & Xie, 2023
- Rectified Flow: Liu et al., 2022
- adaLN-Zero: Peebles & Xie, 2023
- Independent CFG: Multiple conditioning signals with separate dropout
- Flux VAE: Black Forest Labs, 2024
