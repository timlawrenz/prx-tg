# PRX-TG: Text-to-Image Training on Consumer Hardware

**Proving that high-quality text-to-image models can be trained on small, vertical datasets using consumer hardware.**

This project demonstrates that you don't need massive datasets (millions of images) or datacenter infrastructure to train a functional text-to-image diffusion model. By focusing on a curated vertical dataset and leveraging pre-computed embeddings, we achieve practical training on a single consumer GPU.

## Project Goals

1. **Vertical Dataset Viability**: Train a text-to-image model on a small, curated dataset (thousands, not millions)
2. **Consumer Hardware**: Run entirely on consumer-grade GPUs (RTX 4090, etc.)
3. **Pre-computed Embeddings**: Avoid expensive forward passes during training by pre-computing all conditioning signals
4. **Flow Matching**: Use modern rectified flow formulation for stable, efficient training
5. **Quad Conditioning**: Combine text (T5), visual (DINOv3), and pose (DWPose) signals for precise control

## Architecture

### Model: NanoDiT (Diffusion Transformer)

- **Production**: 768 hidden, 18 layers, 12 heads (~400M parameters)
- **Patch-based**: 2×2 patches in pixel space for high detail
- **Pixel-space training**: Model predicts x0 (RGB pixels) directly — no VAE encoder/decoder
- **Quad conditioning**: 
  - Text via T5-Large embeddings (1024-dim, 512 tokens)
  - Visual style via DINOv3 CLS token (1024-dim)
  - Spatial layout via DINOv3 patch embeddings (~4000 tokens × 1024-dim)
  - Pose via DWPose whole-body keypoints (133 joints × 3-dim [x, y, confidence])
- **Dynamic positional encoding**: Supports variable aspect ratios

### Training: Rectified Flow Matching

- **Algorithm**: Flow matching (continuous-time diffusion)
- **Prediction type**: x_prediction (model outputs x0 directly)
- **Loss**: MSE between predicted and target pixels
- **Timestep sampling**: Logit-normal distribution (focuses on mid-diffusion)
- **Timestep Encoding**: High-frequency 1000x scaled sinusoidal embeddings for stable flow
- **CFG strategy**: 
  - **Training dropout** (7 mutually exclusive categories): 10% uncond, 25% text-only, 5% DINO-CLS-only, 5% DINO-patches-only, 10% drop-pose, 5% pose-only, 40% all present
  - **Self-guidance** (default with TREAD): 2 passes — dense (all tokens) vs routed (50% tokens). Single `guidance_scale` (default 3.0)
  - **Dual CFG** (fallback): 3 passes — unconditional + text-only + DINO-only

### Pose Conditioning: DWPose

- **133 whole-body keypoints**: 17 body + 6 feet + 68 face + 42 hands (COCO-WholeBody format)
- **Projection**: MLP (3→768→768 with GELU) + learned per-joint type embeddings
- **Confidence masking**: Joints below threshold (0.05) replaced with learned [NULL_POSE] token
- **CFG dropout**: Dedicated pose dropout categories (independent of text/DINO dropout)
- **Cross-attention**: Pose tokens appended to DINO patches in conditioning sequence

## Dataset

### Source
A curated collection of images from a vertical domain (specific subject matter, consistent quality).

### Processing Pipeline

All preprocessing is done via `scripts/generate_approved_image_dataset.py`:

#### 1. **Aspect Ratio Bucketing**
Images are assigned to 7 buckets (~1 megapixel each) to enable variable aspect ratio training without distortion:

```
1024×1024  (1.00)  - Square
1216×832   (1.46)  - Landscape  
832×1216   (0.68)  - Portrait
1280×768   (1.67)  - Wide landscape
768×1280   (0.60)  - Tall portrait
1344×704   (1.91)  - Very wide
704×1344   (0.52)  - Very tall
```

**Bucketing algorithm**:
1. Assign image to closest aspect ratio bucket
2. Resize-to-cover (scale to fill bucket, no black bars)
3. Center-crop to exact bucket dimensions
4. All subsequent processing uses this bucketed view

#### 2. **Caption Generation**
Dense, objective image descriptions generated using Google Gemma3:27b:
- Physical attributes, composition, lighting, colors
- No subjective interpretation
- ~150-200 words per image
- Example: "A fair-skinned woman with a slender build and visible clavicle is positioned in a close-up, frontal portrait. Her dark, curly hair frames her face..."

#### 3. **Embedding Extraction**

All embeddings are pre-computed and stored to avoid expensive forward passes during training:

##### T5-Large Text Embeddings
- **Model**: Google T5-Large
- **Output**: 512 tokens × 1024 dimensions (fp16)
- **Storage**: `data/derived/t5/*.npy` (~1 MB per image)
- Caption → T5 encoder → fixed-length sequence embeddings

##### DINOv3 Visual Embeddings
- **Model**: Facebook DINOv3-ViT-L/16 (304M parameters)
- **CLS token**: Single 1024-dim global feature vector
  - Captures color palette, style, composition
  - **Storage**: `data/derived/dinov3/*.npy` (~4 KB per image)
- **Patch embeddings** (variable spatial patches):
  - ~4000 × 1024 dimensions for spatial conditioning
  - **Storage**: `data/derived/dinov3_patches/*.npy` (~0.78 MB per image)

##### Pixel Images
- **Format**: RGB float16, range [0, 1]
- **Output**: 3 channels × H × W (bucket-specific dimensions)
- **Storage**: `data/derived/image/*.npy` (varies by bucket)

##### DWPose Keypoints
- **Model**: DWPose (ONNX) — YOLOx-L detector + DW-LL pose estimator
- **Output**: 133 joints × 3 dimensions [x_norm, y_norm, confidence] (float16)
- **Normalization**: x_norm = (2x/W) - 1, y_norm = (2y/H) - 1 (relative to bucket)
- **Storage**: `data/derived/pose/*.npy` (~1.6 KB per image)

#### 4. **WebDataset Sharding**

Embeddings are packed into tar shards for efficient streaming during training:

```
data/shards/10000/
  bucket_1024x1024/
    shard-000000.tar  # Contains .npy files for each modality
    shard-000001.tar
  bucket_1216x832/
    ...
```

Each shard contains:
- `{image_id}.image.npy` - RGB pixel data (3, H, W)
- `{image_id}.dinov3.npy` - DINOv3 CLS token
- `{image_id}.dinov3_patches.npy` - DINOv3 spatial patches
- `{image_id}.t5h.npy` - T5 text embeddings
- `{image_id}.t5m.npy` - T5 attention mask
- `{image_id}.pose.npy` - DWPose keypoints (133, 3)
- `{image_id}.json` - Metadata (bucket, dimensions, caption)

### Storage Requirements

**Per image**:
- Pixel data: ~6 MB (depends on bucket)
- T5 embeddings: ~1 MB
- DINOv3 CLS: ~4 KB
- DINOv3 patches: ~0.78 MB
- DWPose keypoints: ~1.6 KB
- **Total**: ~8 MB per image

**Full dataset**:
- Embeddings + original images
- Storage requirements scale linearly with dataset size

## Training Process

### Hardware Requirements

**Baseline (384 hidden, 12 layers)**:
- GPU: RTX 4090 (24 GB VRAM) or similar
- RAM: 32 GB system memory
- Storage: Sufficient SSD for embeddings
- Training speed: ~1.6 it/s with gradient accumulation

**Production (768 hidden, 18 layers)**:
- GPU: RTX 4090 or similar (24 GB VRAM)
- RAM: 64 GB system memory recommended
- Storage: Sufficient SSD for embeddings
- Training speed: ~1.0 it/s with gradient accumulation + gradient checkpointing

### Training Configuration

```yaml
model:
  hidden_size: 768
  depth: 18
  num_heads: 12
  patch_size: 2
  in_channels: 3              # RGB pixels
  prediction_type: x_prediction  # Pixel-space

training:
  total_steps: 6000     # Optimizer steps (accumulated)
  batch_size: 1
  grad_accumulation_steps: 256  # Effective batch = 256
  learning_rate: 3e-4 → 1e-6 (cosine decay)
  warmup_steps: 500
  mixed_precision: bfloat16
  gradient_checkpointing: true  # Saves ~3-4× memory
  optimizer:
    type: Muon             # Hybrid Muon + AdamW (or "AdamW" for pure AdamW)
    muon:
      momentum: 0.95
      nesterov: true
      ns_steps: 5
      adjust_lr_fn: match_rms_adamw
  cfg_dropout:
    p_uncond: 0.10
    p_text_only: 0.25
    p_dino_cls_only: 0.05
    p_dino_patches_only: 0.05
    p_drop_pose: 0.10
    p_pose_only: 0.05
  resolution_schedule:             # Train at lower res first (optional)
    - until_step: 3000
      scale: 0.5                   # Half resolution (4× fewer tokens)
    - until_step: 6000
      scale: 1.0                   # Full resolution
  repa:
    enabled: true
    weight: 0.5           # REPA loss weight
    block_index: -1        # -1 = depth // 2 (block 9)
    loss_type: cosine      # Cosine similarity alignment
  tread:
    enabled: true          # Token routing for throughput
    routing_probability: 0.5
    self_guidance: true    # Use self-guidance instead of dual CFG
    guidance_scale: 3.0
```

### Optimization Techniques

1. **Gradient Accumulation**: Effective batch size of 256 with batch_size=1
2. **FlashAttention (SDPA)**: Uses memory-efficient `scaled_dot_product_attention` to avoid OOM on large cross-attention sequences
3. **Mixed Precision**: bfloat16 training (required for flow matching stability)
4. **Gradient Checkpointing**: Trade 20-30% speed for 3-4× memory savings
5. **Bucket-aware Batching**: Sample from aspect ratio buckets proportionally
6. **EMA**: Exponential moving average of weights (decay=0.9999) with 500-step warmup
7. **REPA (REPresentation Alignment)**: Auxiliary loss aligning transformer hidden states with DINOv3 patch features at the middle block, improving convergence and representation quality (weight=0.5, cosine similarity)
8. **TREAD (Token Routing)**: Randomly routes 50% of latent tokens past middle blocks (1→depth-2), effectively halving compute for 16 of 18 blocks. Parameter-free — adds zero new weights. Pairs with self-guidance sampling (2 passes instead of 3-pass dual CFG)
9. **Muon Optimizer**: Hybrid Muon + AdamW — all 2D weight matrices (~237M params) use Muon's Newton-Schulz orthogonalization for geometry-aware updates; non-2D params (convs, biases, ~0.1M) stay on AdamW. Uses `adjust_lr_fn="match_rms_adamw"` so both optimizers share the same LR schedule.
10. **Resolution Scheduling**: Train at lower resolution first (e.g., 0.5× spatial scale = 4× fewer tokens), then transition to full resolution.
11. **DWPose Conditioning**: Whole-body pose keypoints with dedicated CFG dropout, confidence masking, and learned [NULL_POSE] embeddings.

### Data Augmentation

- **Horizontal flip**: 50% probability (applied to pixel images)
- **Note**: T5 embeddings are NOT flipped (known limitation - would require caption rewriting)

## Validation

Comprehensive validation suite runs periodically during training, but can also be executed on demand for specific checkpoints.

### On-Demand Checkpoint Validation
If training is interrupted during a validation phase, or if you want to back-test an older checkpoint with new validation logic, you can run the validation suite standalone:

```bash
python scripts/run_checkpoint_validation.py \
  --config experiments/2026-02-22_1227/config.yaml \
  --checkpoint experiments/2026-02-22_1227/checkpoints/checkpoint_step001200.pt
```
This loads the model and EMA weights, runs the full validation suite (Reconstruction, DINO Swap, CFG Divergence, Text Manipulation), and saves the LPIPS scores and images to the corresponding `validation_outputs/step{N}/` directory within the experiment folder.

### Validation Suite Tests:

#### 1. Reconstruction Test
- **Goal**: Test model's ability to reconstruct training images
- **Method**: Use original image's DINO + T5 embeddings as "perfect" conditioning
- **Metric**: LPIPS (perceptual similarity, lower is better)
- **Expected**: <0.3 excellent, 0.3-0.5 good, 0.5-0.7 blurry, >0.7 poor

### 2. DINO Swap Test
- **Goal**: Test visual style transfer
- **Method**: Swap DINO embeddings between image pairs, keep original text
- **Expected**: Generated image should match swapped DINO's style but original text's content

### 3. Text Manipulation Test
- **Goal**: Test text conditioning strength
- **Method**: Modify text embeddings (e.g., "brunette" → "blonde"), keep same DINO
- **Expected**: Changes in generation should match text modifications

### 4. Divergence Analysis
- **Goal**: Detect overfitting or mode collapse
- **Method**: Track distribution of LPIPS scores across validation set
- **Expected**: Stable distribution, no sudden spikes

### Validation Dataset

- **Size**: 25 images per test
- **Sampling**: Deterministic (seeded at 42) for reproducibility across runs
- **Source**: Separate holdout set from training data
- **Outputs**: Saved to `validation_outputs/step{N}/`

## Monitoring

### TensorBoard Metrics

- **Training**:
  - Loss (flow matching MSE + REPA alignment)
  - REPA loss (cosine alignment with DINOv3 patches)
  - LPIPS loss (perceptual quality on decoded crops, when enabled)
  - Gradient norm
  - Velocity norm (RMS of predicted velocity vectors, should be ~1.0)
  - Learning rate
- **Validation**:
  - LPIPS (reconstruction, DINO swap, text manipulation)
  - Per-test statistics

### Visual Debugging

Quick 4-image generation every 100 steps:
- Fixed deterministic samples for consistency
- Saved to `experiments/{timestamp}/visual_debug/step{N}/`
- Useful for spotting training issues early

### Checkpoints

- Saved every 500 steps to `experiments/{timestamp}/checkpoints/`
- Includes:
  - Model weights
  - Optimizer state
  - EMA weights
  - RNG states (Python, NumPy, PyTorch CPU/CUDA)
  - Training step and epoch
- Checkpoint size: ~3.6 GB (768 hidden model)

## Current Status

### Production Training (768 hidden, 18 layers)
- ✅ **Pixel-Space Training**: Direct RGB prediction (x_prediction) — no VAE encoder/decoder needed
- ✅ **Quad Conditioning**: Text (T5-Large) + DINOv3 CLS + DINOv3 patches + DWPose keypoints
- ✅ **DWPose Integration**: 133 whole-body keypoints with confidence masking, dedicated CFG dropout, learned [NULL_POSE] embeddings
- ✅ **Memory Optimized**: FlashAttention (SDPA) and Gradient Checkpointing enable training at 1024px on 24GB GPUs
- ✅ **REPA Alignment**: Auxiliary loss aligns hidden states with DINOv3 teacher features for faster convergence
- ✅ **TREAD Token Routing**: Routes 50% of tokens past middle blocks for ~2× throughput, with self-guidance sampling
- ✅ **Muon Optimizer**: Hybrid Muon + AdamW for geometry-aware weight updates
- ✅ **Resolution Scheduling**: Multi-phase training at lower resolution first for faster convergence
- 🚧 In progress: Training on curated dataset, scaling to 70k images

### Known Limitations

1. **Caption Flipping**: Horizontal flip augmentation doesn't modify T5 embeddings
   - Left/right mentions in captions don't swap with image
   - Would require NLP caption rewriting (future work)

2. **Training Data Size**: Preparing 70k image dataset.

## Repository Structure

```
prx-tg/
├── production/              # Core training code
│   ├── train.py            # Training loop (optimizer-step based)
│   ├── model.py            # NanoDiT with SDPA & high-freq timesteps
│   ├── data.py             # WebDataset loading, bucket-specific stats
│   ├── validate.py         # Validation suite
│   ├── visual_debug.py     # Quick sample generation
│   ├── sample.py           # Inference sampler (Euler)
│   └── config.yaml         # Training configuration (14-day schedule)
│
├── scripts/
│   ├── generate_approved_image_dataset.py  # Data preprocessing
│   └── test_*.py           # Diagnostic scripts
│
├── data/
│   ├── approved/           # Original images
│   ├── derived/            # Pre-computed embeddings
│   └── shards/15000/       # WebDataset tar shards
│
├── experiments/            # Unified experiment tracking (checkpoints, logs, visuals)
└── docs/                   # Documentation and plans
```

## Usage

### 1. Data Preparation

```bash
# Generate all embeddings
python -m scripts.generate_approved_image_dataset \
  --device cuda \
  --pass-filter all \
  --verbose

# This will:
# - Caption images with Gemma3:27b
# - Extract T5-Large text embeddings
# - Extract DINOv3 visual embeddings (CLS + patches)
# - Extract DWPose keypoints (133 whole-body joints)
# - Save bucketed RGB pixel images
# - Create WebDataset shards
```

### 2. Training

```bash
# Start fresh training
python -m production.train_production

# Resume from checkpoint
python -m production.train_production \
  --resume checkpoints/checkpoint_step010000.pt
```

### 3. Inference

```bash
# Generate images (implementation in progress)
python -m production.sample \
  --checkpoint checkpoints/checkpoint_latest.pt \
  --prompt "Your text prompt here" \
  --reference-image path/to/style.jpg \
  --output output.png
```

## Dependencies

- Python 3.13+
- PyTorch 2.6+ with CUDA
- transformers (HuggingFace)
- webdataset
- lpips
- Pillow
- numpy

See full environment in `.venv/` (not committed).

## Future Work

See [docs/prx-part3-analysis.md](docs/prx-part3-analysis.md) for a detailed analysis of techniques from Photoroom's PRX Part 3.

1. **Larger Dataset**: Expand dataset size towards 70k target.

2. **Latent REPA**: Replace DINOv3 teacher alignment with self-supervised latent alignment (future exploration).

## License

(To be determined)

## Acknowledgments

- **T5-Large**: Google (text encoder)
- **DINOv3**: Facebook AI Research (visual encoder)
- **DWPose**: Open-source whole-body pose estimation (ONNX)
- **Gemma3:27b**: Google (caption generation)
- **Flow Matching**: Inspired by Stable Diffusion 3 and Flux.1 training methodology

---

*This is a research/hobby project demonstrating that high-quality generative models can be trained on consumer hardware with curated vertical datasets. Not intended for commercial use.*
