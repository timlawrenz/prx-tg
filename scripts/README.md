# Scripts

## sync_approved_photos.py

Creates/updates an extension-correct symlinked view of the approved photo list:

- Source of truth: `https://crawlr.lawrenz.com/photos.json?page=N` (paginate until empty)
- Raw files: `data/raw/<filename>` (no extension)
- Output symlinks: `data/approved/<filename>.<ext>` pointing to `../raw/<filename>`

Pruning is **not** performed in this change (stale entries in `data/approved/` are left as-is).

### Usage

```bash
python3 scripts/sync_approved_photos.py --help

# Smoke test
python3 scripts/sync_approved_photos.py --dry-run --end-page 1 --limit 20

# Create just one symlink for quick verification
python3 scripts/sync_approved_photos.py --stop-after-links 1 --verbose

# Disable progress output
python3 scripts/sync_approved_photos.py --progress-every 0
```

## generate_approved_image_dataset.py

Generates/enriches a Stage 2 hybrid dataset from `data/approved/`: lightweight JSONL metadata + external .npy embedding files for training efficiency.

**Key features:**
- **Stage 2 hybrid format**: Metadata JSONL (~5-10MB) + separate .npy files for embeddings (~18GB total)
- **Three-pass architecture**: DINOv3 extraction, VAE latent encoding, T5 hidden state encoding run independently
- **Idempotent by default**: Automatically detects existing .npy files and skips completed work
- **Smart resumability**: Stage 1→2 migration, partial embedding completion, crash recovery
- **Incremental progress saving**: Writes to `.tmp` file during processing, Ctrl+C preserves partial work
- **Verification mode**: Check data integrity (missing/corrupt .npy files)

### Output Structure

Stage 2 uses a hybrid format with separate directories per embedding type:

```
data/derived/
  approved_image_dataset.jsonl           # Metadata only (~5-10MB)
  dinov3/
    o2jdtq9uz05whadn9jc8y4mz46xg.npy    # 1024 floats, float32 (~4KB each)
  vae_latents/
    o2jdtq9uz05whadn9jc8y4mz46xg.npy    # 16×H//8×W//8, float16 (~131KB each for 1024×1024)
  t5_hidden/
    o2jdtq9uz05whadn9jc8y4mz46xg.npy    # 77×1024, float16 (~158KB each)
```

**Total storage for 60k images:** ~18GB (240MB DINOv3 + 8GB VAE + 9.5GB T5 + 10MB JSONL)

### JSONL Format (Stage 2)

Each line contains one JSON object with metadata fields (no inline embeddings):

```json
{
  "image_id": "o2jdtq9uz05whadn9jc8y4mz46xg",
  "image_path": "data/approved/o2jdtq9uz05whadn9jc8y4mz46xg",
  "caption": "A woman in a red dress...",
  "t5_attention_mask": [1, 1, 1, ..., 0, 0],
  "height": 1024,
  "width": 768,
  "aspect_bucket": "832x1216",
  "format_version": 2
}
```

**Key differences from Stage 1:**
- ✅ Added: `image_id`, `aspect_bucket`, `format_version`
- ❌ Removed: `dinov3_embedding` (now external .npy file)
- ✅ External: VAE latents and T5 hidden states in separate .npy files

### Install

```bash
python3 -m pip install -r scripts/requirements-approved-image-embeddings.txt
pip install diffusers  # For Flux VAE

# If needed, install torch using the PyTorch installer for your platform/CUDA/ROCm:
# https://pytorch.org/get-started/locally/
```

### Usage

#### For High-VRAM Systems (>40GB VRAM recommended)

**Process everything in one go** (all models loaded simultaneously):
```bash
# Default behavior: loads all models, processes each image through all stages
python3 scripts/generate_approved_image_dataset.py \
  --output data/derived/approved_image_dataset.jsonl \
  --progress-every 100
```

**Benefits:**
- ✅ Fastest approach (no model loading overhead between passes)
- ✅ Simplest workflow for incremental updates
- ✅ Each new image gets fully processed in one go
- ⚠️ Requires ~30-40GB VRAM (DINOv3 + Gemma 27B + FLUX VAE + T5-Large)

#### For Low-VRAM Systems (8-24GB VRAM)

**Run passes separately** to avoid OOM errors:
```bash
# Pass 1: Extract DINOv3 and generate captions (~10-15GB VRAM)
python3 scripts/generate_approved_image_dataset.py --pass dinov3

# Pass 2: Generate VAE latents (~3-4GB VRAM)
python3 scripts/generate_approved_image_dataset.py --pass vae

# Pass 3: Generate T5 hidden states (~5-8GB VRAM)
python3 scripts/generate_approved_image_dataset.py --pass t5

# Pass 4: JSONL migration only (no models loaded)
python3 scripts/generate_approved_image_dataset.py --pass migrate
```

**Benefits:**
- ✅ Works on consumer GPUs (8-24GB VRAM)
- ✅ Can spread work across multiple days
- ✅ Easier to debug individual stages
- ⚠️ Slower overall (model loading overhead)

#### Common Commands

**Verify data integrity:**
```bash
python3 scripts/generate_approved_image_dataset.py --verify
```

**Force regeneration from scratch:**
```bash
python3 scripts/generate_approved_image_dataset.py --no-resume
# Warning: Deletes output.jsonl and all .npy directories!
```

**Smoke test (process 2 images):**
```bash
python3 scripts/generate_approved_image_dataset.py \
  --limit 2 \
  --output data/derived/test.jsonl \
  --verbose
```

### Three-Pass Architecture (Optional)

The `--pass` flag allows processing in independent stages. This is **optional** - by default, the script loads all models and processes everything in one go.

**When to use separate passes:**
- Low VRAM systems (8-24GB) that can't load all models simultaneously
- Want to spread work across multiple runs
- Debugging specific pipeline stages

**When NOT needed:**
- High VRAM systems (>40GB) - just run without `--pass` flag for best performance
- Systems with sufficient memory can handle all models at once (~30-40GB VRAM total)

#### Pass Details

| Pass | Models Loaded | VRAM | What It Does | Time (60k images) |
|---|---|---|---|---|
| `dinov3` | DINOv3 + Gemma + T5 tokenizer | ~10-15GB | Extract inline embeddings to .npy, generate captions for new images | ~5-8 hours |
| `vae` | Flux VAE encoder | ~3-4GB | Encode images to VAE latent space | ~8-10 hours |
| `t5` | T5-Large encoder + tokenizer | ~5-8GB | Encode captions to T5 hidden states | ~3-4 hours |
| `migrate` | T5 tokenizer only | <1GB | Update JSONL to Stage 2 format | ~1 minute |
| **default** (no --pass) | All models simultaneously | **~30-40GB** | **Full pipeline in one go (fastest)** | **~16-22 hours** |

**Memory comparison:**
- **Separate passes**: Peak 10-15GB VRAM (one model at a time)
- **Default (all-in-one)**: Peak 30-40GB VRAM (all models loaded)
- **Recommendation**: Use default unless VRAM is constrained

### Progress Tracking

The script tracks four operation types:

```
progress: 120 migrated, 980 enriched, 50 extracted, 3850 skipped (total: 5000) rate=2.3/s
```

| Counter | Meaning |
|---|---|
| **migrated** | Stage 1 records converted to Stage 2 with all embeddings generated |
| **enriched** | Existing Stage 2 records with missing VAE/T5 embeddings filled in |
| **extracted** | Stage 1 records where only DINOv3 extraction happened |
| **skipped** | Fully complete Stage 2 records (format_version=2, all .npy files exist) |

### Migration Guide (Stage 1 → Stage 2)

If you have existing Stage 1 data (`dinov3_embedding` inline):

**Step 1: Backup your data**
```bash
cp data/derived/approved_image_dataset.jsonl data/derived/approved_image_dataset.jsonl.stage1.backup
```

**Step 2: Run DINOv3 extraction (safe, fast)**
```bash
python3 scripts/generate_approved_image_dataset.py --pass dinov3
# This extracts inline embeddings to data/derived/dinov3/*.npy
```

**Step 3: Generate VAE latents (slow)**
```bash
python3 scripts/generate_approved_image_dataset.py --pass vae
# Creates data/derived/vae_latents/*.npy
```

**Step 4: Generate T5 hidden states**
```bash
python3 scripts/generate_approved_image_dataset.py --pass t5
# Creates data/derived/t5_hidden/*.npy
```

**Step 5: Migrate JSONL format**
```bash
python3 scripts/generate_approved_image_dataset.py --pass migrate
# Updates JSONL: adds image_id, aspect_bucket, format_version; removes dinov3_embedding
```

**Verification:**
```bash
python3 scripts/generate_approved_image_dataset.py --verify
# Should show: all valid, 0 invalid, 0 missing
```

### Storage Requirements

| Component | Size per Image | Total (60k images) |
|---|---|---|
| DINOv3 embedding | ~4KB (1024 float32) | ~240MB |
| VAE latent | ~131KB (16×64×64 float16) | ~8GB |
| T5 hidden state | ~158KB (77×1024 float16) | ~9.5GB |
| JSONL metadata | ~170 bytes | ~10MB |
| **Total** | ~293KB | **~18GB** |

**Disk space check:** Script aborts if < 20GB available at start.

### Crash Recovery

The script writes progress incrementally to `output.jsonl.tmp`:
- **During processing**: Records written to `.tmp` file immediately after completion
- **On Ctrl+C**: Gracefully merges `.tmp` into `output.jsonl` before exiting
- **On unexpected crash**: Partial work preserved in `.tmp` file
- **On restart**: Automatically loads both `output.jsonl` and `output.jsonl.tmp`, merges them
- **On success**: `.tmp` merged into final output atomically, then deleted

**This means:**
- Ctrl+C is **always safe** - it triggers a clean merge before exit
- `output.jsonl` is **always up-to-date** after Ctrl+C
- You never lose work from interrupted runs
- `.npy` files are written atomically (safe to Ctrl+C mid-encoding)

### Troubleshooting

**Issue: "insufficient disk space"**
- Solution: Free up 20GB+ or use `--output-base-dir /path/to/large/disk`

**Issue: VAE encoding fails with "not found" error**
- Solution: Ensure `diffusers` installed: `pip install diffusers`
- Check model ID is correct (should auto-download on first run)

**Issue: ROCm kernel compilation hangs**
- Solution: First run compiles kernels (~15-30 min wait). Disable tuning for smoke tests:
  ```bash
  PYTORCH_TUNABLEOP_TUNING=0 python3 scripts/generate_approved_image_dataset.py --limit 2
  ```

**Issue: T5 hidden states have wrong shape**
- Solution: Rerun with `--pass t5` - script validates shape automatically

**Issue: Missing .npy files after crash**
- Solution: Run `--verify` to identify missing files, then rerun with appropriate `--pass` flag

**Issue: JSONL still has inline dinov3_embedding**
- Solution: Run `--pass migrate` to update format (removes inline embedding, adds format_version=2)

### Notes

- **Do not enable** `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` on ROCm systems (causes hardware exceptions with Gemma 3)
- **First run downloads large models** (~55GB for Gemma-3-27B, ~1.5GB for DINOv3, ~2GB for Flux VAE, ~3GB for T5-Large)
- **VAE latents are encoded at original resolution** (resizing happens in Stage 3 dataloader for training)
- **Aspect buckets use 7 predefined resolutions** at 1024px equivalent area (all dims divisible by 64)

## create_webdataset_shards.py

Creates WebDataset tar shards from Stage 2 embeddings for efficient sequential streaming during training.

**Purpose:** Package "ready" samples (caption + attention mask + DINOv3/VAE/T5 .npy files) into tar files following WebDataset naming conventions, grouped by aspect bucket.

### Output Structure

```
data/shards/
  bucket_1024x1024/
    shard-000000.tar  # ~1000 samples, ~300MB
    shard-000001.tar
    ...
  bucket_832x1216/
    shard-000000.tar
    ...
```

**Each tar contains WebDataset entries:**
```
{image_id}.json         # Metadata (image_id, aspect_bucket, caption, etc.)
{image_id}.dinov3.npy   # DINOv3 embedding (1024 floats, float32, ~4KB)
{image_id}.vae.npy      # VAE latent (16×H//8×W//8, float16, ~131KB for 1024×1024)
{image_id}.t5h.npy      # T5 hidden states (77×1024, float16, ~158KB)
{image_id}.t5m.npy      # T5 attention mask (77 ints, uint8, ~77 bytes)
```

### Usage

#### Smoke Test (Verify Readiness)
```bash
# Dry-run with 2 samples - fast validation without writing files
python3 scripts/create_webdataset_shards.py --limit 2 --dry-run
```

#### Create Validation Shard (100 samples, shuffled)
```bash
# For Part B validation testing
python3 scripts/create_webdataset_shards.py \
  --limit 100 \
  --shuffle \
  --output-dir data/shards/validation
```

#### Create Full Training Shards
```bash
# All ready samples, grouped by bucket
python3 scripts/create_webdataset_shards.py \
  --output-dir data/shards/train
```

#### Bucket-Specific Shards
```bash
# Only create shards for 1024×1024 images
python3 scripts/create_webdataset_shards.py \
  --bucket 1024x1024 \
  --output-dir data/shards/square_only
```

### Command-Line Options

| Flag | Default | Description |
|---|---|---|
| `--input-jsonl` | `data/derived/approved_image_dataset.jsonl` | Stage 2 metadata JSONL |
| `--derived-dir` | `data/derived` | Base directory with dinov3/, vae_latents/, t5_hidden/ |
| `--output-dir` | `data/shards` | Output directory for shards |
| `--limit N` | `0` (all) | Write at most N ready samples total |
| `--shard-size N` | `1000` | Max samples per tar file |
| `--shuffle` | off | Shuffle ready samples before sharding (enables diverse validation sets) |
| `--seed N` | `1337` | RNG seed when --shuffle is used |
| `--bucket NAME` | (all) | Only include specific bucket (e.g., 832x1216). Repeatable. |
| `--overwrite` | off | Overwrite existing shard tar files |
| `--dry-run` | off | Simulate without writing files |
| `--progress-every N` | `500` | Print progress every N ready records (0=disable) |

### Ready Sample Requirements

A sample is "ready" when:
- ✅ JSONL record has `caption` (non-empty string)
- ✅ JSONL record has `t5_attention_mask` (77-length list of 0s/1s)
- ✅ JSONL record has `image_id` and `aspect_bucket`
- ✅ All three .npy files exist: `dinov3/{id}.npy`, `vae_latents/{id}.npy`, `t5_hidden/{id}.npy`

Incomplete samples are skipped with `skipped_incomplete` counter incremented.

### Notes

- **No WebDataset dependency required** - script uses Python's built-in `tarfile` module
- **Preserves .npy files verbatim** - no re-encoding, float16/float32 precision maintained
- **Default shard size (1000)** balances file handle efficiency with shuffling granularity (~300MB per tar)
- **Bucket grouping** ensures training dataloader can stream one bucket's shards sequentially
- **Progress is cheap** - scanning JSONL + filesystem checks for 60k samples takes <10 seconds
- **Storage:** ~90GB for 60k images (metadata + embeddings packed into tars)
