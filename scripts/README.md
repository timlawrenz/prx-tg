# Scripts

## sync_approved_photos.py

Creates/updates an extension-correct symlinked view of the approved photo list:

- Source of truth: `https://crawlr.lawrenz.com/photos.json?page=N` (paginate until empty)
- Raw files: `data/raw/<filename>` (no extension)
- Output symlinks: `data/approved/<filename>.<ext>` pointing to `../raw/<filename>`
- **Automatic download**: If raw file is missing, downloads from `exportable_url` in JSON

Pruning is **not** performed in this change (stale entries in `data/approved/` are left as-is).

### Features

- **Downloads missing files**: Automatically fetches raw files from CDN if not present locally
- **Dry-run support**: Use `--dry-run` to preview actions without downloading or modifying filesystem
- **Progress tracking**: Shows download counts, missing files, symlink operations
- **Error handling**: Skips and continues on download failures (logs warnings)
- **Resume-friendly**: Can be interrupted and rerun; only processes what's needed

### Usage

```bash
python3 scripts/sync_approved_photos.py --help

# Dry-run to see what would be downloaded/created
python3 scripts/sync_approved_photos.py --dry-run --end-page 1 --limit 20

# Process first 100 photos with verbose output
python3 scripts/sync_approved_photos.py --limit 100 --verbose

# Create just one symlink for quick verification
python3 scripts/sync_approved_photos.py --stop-after-links 1 --verbose

# Full sync (downloads missing files, creates all symlinks)
python3 scripts/sync_approved_photos.py

# Disable progress output
python3 scripts/sync_approved_photos.py --progress-every 0
```

### How It Works

1. **Fetch photo list**: Paginates through `photos.json` API
2. **Check raw file**: Looks for `data/raw/<filename>`
3. **Download if missing**: Fetches from `exportable_url` in JSON (CDN)
4. **Detect type**: Reads magic bytes to determine extension (jpg/png/webp/gif)
5. **Create symlink**: `data/approved/<filename>.<ext>` → `../raw/<filename>`

### Output Example

```
page 1: fetching https://crawlr.lawrenz.com/photos.json?page=1
page 1: 14 items
downloading: abc123xyz from https://crawlr-assets.lawrenz.com/abc123xyz
downloaded: abc123xyz (847.3 KB)
downloading: def456uvw from https://crawlr-assets.lawrenz.com/def456uvw
downloaded: def456uvw (1203.5 KB)
progress: processed=1000 downloaded=45 missing_raw=2 download_failed=1 unknown_type=0 created=997 updated=0

Summary:
  processed:        2500
  downloaded:       118
  download failed:  3
  missing raw:      5
  unknown type:     0
  symlink created:  2492
  symlink updated:  0
  symlink unchanged:0
```

### Notes

- **Downloads are automatic**: Missing files are fetched from CDN without user confirmation
- **Dry-run skips downloads**: Use `--dry-run` to preview without downloading
- **Failed downloads are skipped**: Script continues on errors, logs warnings to stderr
- **Resume-friendly**: Rerun anytime; only downloads missing files, updates changed symlinks
- **No authentication**: CDN URLs are public (exportable_url)

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

---

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

Each tar contains WebDataset-formatted files:
```
shard-000000.tar:
  000000.json          # {"caption": "...", "t5_attention_mask": [...]}
  000000.dinov3.npy    # 1024 float32 values
  000000.vae.npy       # 16×H/8×W/8 float16 latents
  000000.t5.npy        # 77×1024 float16 hidden states
  000001.json
  000001.dinov3.npy
  ...
```

### Usage

```bash
# Create shards from Stage 2 data (default: 1000 samples per shard)
python3 scripts/create_webdataset_shards.py

# Custom shard size and output location
python3 scripts/create_webdataset_shards.py \
  --shard-size 500 \
  --output-dir /path/to/shards

# Only shard specific aspect buckets
python3 scripts/create_webdataset_shards.py \
  --bucket 1024x1024 \
  --bucket 832x1216

# Shuffle samples before sharding (loads all into RAM)
python3 scripts/create_webdataset_shards.py --shuffle --seed 42

# Dry-run to see what would be created
python3 scripts/create_webdataset_shards.py --dry-run

# Limit to first N ready samples (testing)
python3 scripts/create_webdataset_shards.py --limit 100
```

### How It Works

1. **Scans JSONL**: Reads `data/derived/approved_image_dataset.jsonl`
2. **Validates completeness**: Checks each record has:
   - Valid `caption` string
   - Valid `t5_attention_mask` (77 integers, 0s and 1s)
   - Existing `dinov3/<image_id>.npy` file
   - Existing `vae_latents/<image_id>.npy` file
   - Existing `t5_hidden/<image_id>.npy` file
   - Valid `aspect_bucket` string
3. **Groups by bucket**: Separates samples by aspect ratio
4. **Creates shards**: Writes tar files with WebDataset naming
   - Default: 1000 samples per shard (~300MB)
   - Each sample gets sequential ID within shard (000000, 000001, etc.)
   - JSON metadata + three .npy files per sample

### Output Example

```
page 1: fetching JSONL...
progress: scanned_total=5000 ready=4850 skipped_incomplete=150
progress: scanned_total=10000 ready=9730 skipped_incomplete=270

ready_samples=29450 buckets=7 shard_size=1000 dry_run=False output_dir=data/shards

bucket 1024x1024: samples=12340
  writing shard-000000.tar (1000 samples)
  writing shard-000001.tar (1000 samples)
  ...
  writing shard-000012.tar (340 samples)
  
bucket 832x1216: samples=8920
  writing shard-000000.tar (1000 samples)
  ...

done: total_records=31000 ready_records=29450 skipped_incomplete=1550 
      written_samples=29450 written_shards=30
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--input-jsonl` | Stage 2 JSONL path | `data/derived/approved_image_dataset.jsonl` |
| `--derived-dir` | Base dir with .npy subdirs | `data/derived` |
| `--output-dir` | Output directory for shards | `data/shards` |
| `--shard-size` | Max samples per shard | `1000` |
| `--bucket` | Filter to specific bucket(s) | All buckets |
| `--shuffle` | Shuffle before sharding | `False` |
| `--seed` | RNG seed for shuffle | `1337` |
| `--limit` | Max samples to write | `0` (all) |
| `--overwrite` | Overwrite existing shards | `False` |
| `--dry-run` | Report without writing | `False` |
| `--progress-every` | Progress frequency | `500` |

### What Gets Skipped

Records are skipped if:
- Missing or invalid `image_id`
- Missing or invalid `aspect_bucket`
- Missing or empty `caption`
- Invalid `t5_attention_mask` (not 77 ints or contains non-0/1 values)
- Any of the three .npy files don't exist
- Bucket filtered out by `--bucket` flag

### Storage Requirements

| Component | Size per Sample | Total (30k samples) |
|---|---|---|
| JSON metadata | ~200 bytes | ~6MB |
| DINOv3 .npy | ~4KB | ~120MB |
| VAE .npy | ~131KB | ~4GB |
| T5 .npy | ~158KB | ~5GB |
| **Per shard (1000)** | **~293MB** | - |
| **Total shards** | - | **~9GB** |

Note: Tar overhead is minimal (~1%). Shards are uncompressed for fast streaming.

### WebDataset Integration

Shards are compatible with the [WebDataset](https://github.com/webdataset/webdataset) library:

```python
import webdataset as wds

dataset = wds.WebDataset("data/shards/bucket_1024x1024/shard-{000000..000012}.tar")
dataset = dataset.decode()  # Automatically decodes .npy files

for sample in dataset:
    caption = sample["json"]["caption"]
    dinov3 = sample["dinov3.npy"]    # numpy array (1024,)
    vae = sample["vae.npy"]           # numpy array (16, H/8, W/8)
    t5 = sample["t5.npy"]             # numpy array (77, 1024)
    # ... training code
```

### Notes

- **No WebDataset dependency required** - script uses Python's built-in `tarfile` module
- **Preserves .npy files verbatim** - no re-encoding, float16/float32 precision maintained
- **Default shard size (1000)** balances file handle efficiency with shuffling granularity (~300MB per tar)
- **Bucket grouping** ensures training dataloader can stream one bucket's shards sequentially
- **Progress is cheap** - scanning JSONL + filesystem checks for 60k samples takes <10 seconds
- **Storage:** ~9GB for 30k images (metadata + embeddings packed into tars)

---

## prune_dataset.py

Removes orphaned data when images are deleted for quality reasons. Cleans up:
- Broken symlinks in `data/approved/` (pointing to missing raw files)
- Orphaned JSONL records (no corresponding symlink in approved/)
- Orphaned `.npy` files in derived directories

### When to Use

Run this script after removing images from `data/raw/` or `data/approved/` to clean up derived data:

1. **After deleting raw files**: Broken symlinks remain in `data/approved/`
2. **After deleting symlinks**: JSONL records and .npy files are orphaned
3. **Periodic cleanup**: Remove stale data from dataset

### Usage

**IMPORTANT: Always run with `--dry-run` first to preview deletions!**

```bash
# Preview what would be deleted (safe, no changes)
python3 scripts/prune_dataset.py --dry-run --verbose

# Actually delete orphaned data (permanent!)
python3 scripts/prune_dataset.py

# Custom paths
python3 scripts/prune_dataset.py \
  --approved-dir /path/to/approved \
  --derived-dir /path/to/derived \
  --jsonl /path/to/dataset.jsonl
```

### What It Does

1. **Scans for broken symlinks** in `data/approved/`
   - Finds symlinks pointing to non-existent files in `data/raw/`
   
2. **Finds orphaned JSONL records**
   - Records with no corresponding symlink in `data/approved/`
   
3. **Deletes broken symlinks** from `data/approved/`

4. **Removes orphaned .npy files**
   - `data/derived/dinov3/<image_id>.npy`
   - `data/derived/vae_latents/<image_id>.npy`
   - `data/derived/t5_hidden/<image_id>.npy`

5. **Prunes JSONL file**
   - Removes records for deleted images
   - Uses atomic file replacement (safe, no corruption risk)

### Output Example

```
Dataset Pruning Script - DELETION MODE
Approved dir: data/approved
Derived dir:  data/derived
JSONL file:   data/derived/approved_image_dataset.jsonl

⚠️  WARNING: This will permanently delete data!

Phase 1: Discovering orphaned data...

Discovery Results:
  Broken symlinks:    23
  Orphaned records:   5
  Total unique IDs:   28

Phase 2: Deleting orphaned data...

Summary:
  Broken symlinks deleted:     23
  JSONL records removed:       28
  DINOv3 embeddings deleted:   28
  VAE latents deleted:         28
  T5 hidden states deleted:    28
  Total .npy files deleted:    84

✅ Pruning complete.
```

### Safety Features

- **Dry-run mode**: Test before deleting (`--dry-run`)
- **Verbose logging**: See exactly what's being deleted (`--verbose`)
- **Atomic JSONL updates**: Writes to temp file, then renames (no corruption)
- **Graceful error handling**: Continues on individual failures, reports errors
- **Validation**: Checks directories exist before starting

### Integration with Workflow

**Recommended workflow when removing images:**

1. Delete images from `data/raw/` or symlinks from `data/approved/`
2. Run prune script in dry-run mode:
   ```bash
   python3 scripts/prune_dataset.py --dry-run
   ```
3. Review the output, verify counts look correct
4. Run prune script to actually delete:
   ```bash
   python3 scripts/prune_dataset.py
   ```
5. Verify dataset integrity:
   ```bash
   python3 scripts/generate_approved_image_dataset.py --verify
   ```

**Note:** Do not run pruning while dataset generation is running (may cause race conditions).

### Troubleshooting

**Issue: "warning: approved directory does not exist"**
- Solution: Check that `data/approved/` exists and path is correct

**Issue: "warning: JSONL file does not exist"**
- Solution: Check that JSONL file exists at specified path

**Issue: Dry-run shows unexpected deletions**
- Solution: Review the image IDs carefully - ensure they are actually orphaned
- Verify symlinks are truly broken: `ls -l data/approved/<filename>`

**Issue: "error: failed to delete symlink"**
- Solution: Check file permissions - may need write access to `data/approved/`

**Issue: JSONL corruption after pruning**
- Solution: Script uses atomic updates - if interrupted, `.jsonl.prune-tmp` file may remain
- Recovery: Restore from backup or rerun prune (will clean up temp file)
