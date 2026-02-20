# PRX-TG: Text-to-Image Training on Consumer Hardware

**Proving that high-quality text-to-image models can be trained on small, vertical datasets using consumer hardware.**

This project demonstrates that you don't need massive datasets (millions of images) or datacenter infrastructure to train a functional text-to-image diffusion model. By focusing on a curated vertical dataset and leveraging pre-computed embeddings, we achieve practical training on a single consumer GPU.

## Project Goals

1. **Vertical Dataset Viability**: Train a text-to-image model on ~15k curated images (not millions)
2. **Consumer Hardware**: Run entirely on consumer-grade GPUs (RTX 4090, etc.)
3. **Pre-computed Embeddings**: Avoid expensive forward passes during training by pre-computing all conditioning signals
4. **Flow Matching**: Use modern rectified flow formulation for stable, efficient training
5. **Dual Conditioning**: Combine text (T5) and visual (DINOv3) embeddings for better control

## Architecture

### Model: NanoDiT (Diffusion Transformer)

- **Current (baseline)**: 384 hidden, 12 layers, 6 heads (~90M parameters)
- **Target (production)**: 768 hidden, 18 layers, 12 heads (~400M parameters)
- **Patch-based**: 2×2 patches in latent space for high detail
- **Dual conditioning**: 
  - Text via T5-XXL embeddings (4096-dim)
  - Visual style via DINOv3 CLS token (1024-dim)
- **Dynamic positional encoding**: Supports variable aspect ratios

### Training: Rectified Flow Matching

- **Algorithm**: Flow matching (continuous-time diffusion)
- **Loss**: MSE between predicted and target velocity vectors
- **Timestep sampling**: Logit-normal distribution (focuses on mid-diffusion)
- **CFG strategy**: 
  - 10% unconditional (drop both text + DINO)
  - 30% text-only (drop DINO, reduce over-reliance on visual conditioning)
  - 5% DINO-only (drop text)
  - 55% dual-conditioned (both signals)

### VAE: Flux.1 Latent Space

- Pre-trained VAE from Black Forest Labs
- 16 channels, 8× spatial compression
- All training happens in latent space (no pixel-level operations)

## Dataset

### Source
~15,690 curated images from a vertical domain (specific subject matter, consistent quality).

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
Dense, objective image descriptions generated using Florence-2 (Microsoft):
- Physical attributes, composition, lighting, colors
- No subjective interpretation
- ~150-200 words per image
- Example: "A fair-skinned woman with a slender build and visible clavicle is positioned in a close-up, frontal portrait. Her dark, curly hair frames her face..."

#### 3. **Embedding Extraction**

All embeddings are pre-computed and stored to avoid expensive forward passes during training:

##### T5-XXL Text Embeddings
- **Model**: Google T5-XXL (11B parameters)
- **Output**: 512 tokens × 4096 dimensions (fp16)
- **Storage**: `data/derived/t5/*.npy` (~4.2 MB per image)
- Caption → T5 encoder → fixed-length sequence embeddings

##### DINOv3 Visual Embeddings
- **Model**: Facebook DINOv3-ViT-L/14 (304M parameters)
- **CLS token**: Single 1024-dim global feature vector
  - Captures color palette, style, composition
  - **Storage**: `data/derived/dinov3/*.npy` (~4 KB per image)
- **Patch embeddings** (196 spatial patches, 14×14 grid):
  - 196 × 1024 dimensions for spatial conditioning
  - **Storage**: `data/derived/dinov3_patches/*.npy` (~0.78 MB per image)
  - *Currently extracted but not used in training* (future spatial conditioning)

##### Flux VAE Latents
- **Model**: Black Forest Labs Flux.1 VAE
- **Output**: 16 channels × H/8 × W/8 (fp16)
- **Storage**: `data/derived/vae/*.npy` (varies by bucket)
- Original pixels are never stored - training operates purely on latents

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
- `{image_id}.vae_latent.npy` - VAE latents
- `{image_id}.dinov3.npy` - DINO CLS token
- `{image_id}.t5_hidden.npy` - T5 text embeddings
- `{image_id}.t5_attention_mask.npy` - T5 attention mask
- `{image_id}.json` - Metadata (bucket, dimensions, caption)

### Storage Requirements

**Per image**:
- VAE latent: ~300-500 KB (depends on bucket)
- T5 embeddings: ~4.2 MB
- DINOv3 CLS: ~4 KB
- DINOv3 patches: ~0.78 MB (optional)
- **Total**: ~5-6 MB per image

**Full dataset (15,690 images)**:
- Embeddings: ~80 GB
- Original images: ~40 GB
- **Total**: ~120 GB

## Training Process

### Hardware Requirements

**Baseline (384 hidden, 12 layers)**:
- GPU: RTX 4090 (24 GB VRAM) or similar
- RAM: 32 GB system memory
- Storage: 120 GB SSD
- Training speed: ~1.6 it/s with gradient accumulation

**Production (768 hidden, 18 layers)**:
- GPU: RTX 4090 or similar (24 GB VRAM)
- RAM: 64 GB system memory recommended
- Storage: 150 GB SSD
- Training speed: ~1.0 it/s with gradient accumulation + gradient checkpointing

### Training Configuration

```yaml
model:
  hidden_size: 768
  depth: 18
  num_heads: 12
  patch_size: 2

training:
  total_steps: 250,000
  batch_size: 1
  grad_accumulation_steps: 256  # Effective batch = 256
  learning_rate: 3e-4 → 1e-6 (cosine decay)
  warmup_steps: 7,500
  mixed_precision: bfloat16
  gradient_checkpointing: true  # Saves ~3-4× memory
```

### Optimization Techniques

1. **Gradient Accumulation**: Effective batch size of 256 with batch_size=1
2. **Mixed Precision**: bfloat16 training (required for flow matching stability)
3. **Gradient Checkpointing**: Trade 20-30% speed for 3-4× memory savings
4. **Pre-computed Embeddings**: No T5/DINO forward passes during training
5. **Bucket-aware Batching**: Sample from aspect ratio buckets proportionally
6. **EMA**: Exponential moving average of weights (decay=0.9999) for stable inference

### Data Augmentation

- **Horizontal flip**: 50% probability (applied to VAE latents)
- **Note**: T5 embeddings are NOT flipped (known limitation - would require caption rewriting)

## Validation

Comprehensive validation suite runs every 5,000 steps:

### 1. Reconstruction Test
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
  - Loss (flow matching MSE)
  - Gradient norm
  - Velocity norm (RMS of predicted velocity vectors)
  - Learning rate
- **Validation**:
  - LPIPS (reconstruction, DINO swap, text manipulation)
  - Per-test statistics

### Visual Debugging

Quick 4-image generation every 1,000 steps:
- Fixed deterministic samples for consistency
- Saved to `visual_debug/step{N}/`
- Useful for spotting training issues early

### Checkpoints

- Saved every 5,000 steps to `checkpoints/`
- Includes:
  - Model weights
  - Optimizer state
  - EMA weights
  - RNG states (Python, NumPy, PyTorch CPU/CUDA)
  - Training step and epoch
- Checkpoint size: ~3.6 GB (768 hidden model)

## Current Status

### Baseline Training (384 hidden, 12 layers)
- ✅ Completed: 250,000 steps on 5,141 images
- Results: "Watercolor blobs" - correct colors, poor spatial coherence
- Conclusion: Model capacity too small for this task

### Production Training (768 hidden, 18 layers)
- 🚧 In progress: Started fresh training
- Expected: Better spatial structure and detail
- Architecture limitation: Using only DINO CLS token (no spatial info)
  - This explains "blobs in wrong places" - model has color/style but no spatial layout
  - Future: Integrate 196 DINOv3 patch embeddings for spatial conditioning

### Known Limitations

1. **Spatial Layout**: Current architecture uses only global DINO CLS token
   - No spatial conditioning → poor layout accuracy
   - DINOv3 patches extracted but not yet integrated into architecture
   
2. **Caption Flipping**: Horizontal flip augmentation doesn't modify T5 embeddings
   - Left/right mentions in captions don't swap with image
   - Would require NLP caption rewriting (future work)

3. **Training Data Size**: 15k images is small by modern standards
   - Adequate for proof-of-concept and vertical domains
   - Larger datasets would improve generalization

## Repository Structure

```
prx-tg/
├── production/              # Core training code
│   ├── train.py            # Training loop, optimizer, EMA
│   ├── model.py            # NanoDiT architecture
│   ├── data.py             # WebDataset loading, bucketing
│   ├── validate.py         # Validation suite
│   ├── visual_debug.py     # Quick sample generation
│   ├── sample.py           # Inference sampler (Euler)
│   └── config.yaml         # Training configuration
│
├── scripts/
│   ├── generate_approved_image_dataset.py  # Data preprocessing
│   └── test_*.py           # Diagnostic scripts
│
├── data/
│   ├── approved/           # Original images
│   ├── derived/            # Pre-computed embeddings
│   │   ├── dinov3/         # DINO CLS tokens
│   │   ├── dinov3_patches/ # DINO spatial patches
│   │   ├── vae/            # VAE latents
│   │   └── t5/             # T5 text embeddings
│   └── shards/10000/       # WebDataset tar shards
│
├── checkpoints/            # Training checkpoints
├── validation_outputs/     # Validation results by step
├── visual_debug/           # Quick debug samples
└── experiments/            # Experiment tracking
```

## Usage

### 1. Data Preparation

```bash
# Generate all embeddings and create shards
python scripts/generate_approved_image_dataset.py \
  --device cuda \
  --pass-filter all \
  --verbose

# This will:
# - Caption images with Florence-2
# - Extract T5 text embeddings
# - Extract DINOv3 visual embeddings
# - Encode VAE latents
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
  --checkpoint checkpoints/checkpoint_step250000.pt \
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

1. **Spatial Conditioning**: Integrate DINOv3 patch embeddings into architecture
   - Add cross-attention to spatial features
   - Should dramatically improve layout accuracy

2. **Larger Dataset**: Expand to 50k+ images
   - Better generalization
   - More diverse compositions

3. **Caption Augmentation**: Implement left/right swapping for flipped images
   - Requires NLP caption rewriting
   - More coherent flip augmentation

4. **Multi-GPU Training**: Scale to multiple GPUs
   - Faster iteration
   - Larger effective batch sizes

5. **Latent Consistency Models**: Distill to few-step sampler
   - 1-4 step inference instead of 25
   - Better for production deployment

## License

(To be determined)

## Acknowledgments

- **Flux VAE**: Black Forest Labs (Flux.1-dev)
- **T5-XXL**: Google (text encoder)
- **DINOv3**: Facebook AI Research (visual encoder)
- **Florence-2**: Microsoft (caption generation)
- **Flow Matching**: Inspired by Stable Diffusion 3 and Flux.1 training methodology

---

*This is a research/hobby project demonstrating that high-quality generative models can be trained on consumer hardware with curated vertical datasets. Not intended for commercial use.*
