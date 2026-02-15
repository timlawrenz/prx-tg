# Data Directory Structure

This directory contains the image datasets and generated embeddings for the diffusion model training pipeline.

**⚠️ Not committed to git** - this directory is in `.gitignore` due to size (180GB+ expected).

## Directory Layout

```
data/
├── raw/               # Original images (no file extensions)
│                      # ~31k images from crawlr system
│                      # Populated by sync_approved_photos.py (auto-downloads from CDN)
├── approved/          # Symlinked approved images with correct extensions
│                      # Populated by scripts/sync_approved_photos.py
│                      # Symlinks point to ../raw/<filename>
└── derived/           # Generated Stage 2 hybrid format
    ├── approved_image_dataset.jsonl      # Lightweight metadata
    ├── dinov3/                            # DINOv3 embeddings (.npy)
    ├── vae_latents/                       # Flux VAE latents (.npy)
    └── t5_hidden/                         # T5-Large hidden states (.npy)
```

## How to Populate

### 1. Sync Approved Photos (Downloads + Symlinks)
Run the sync script to download missing raw files and create symlinks:
```bash
# Dry-run to preview
python3 scripts/sync_approved_photos.py --dry-run --limit 100 --verbose

# Full sync (downloads missing files, creates symlinks)
python3 scripts/sync_approved_photos.py
```

This script:
- Fetches approved photo list from `https://crawlr.lawrenz.com/photos.json`
- Downloads missing files from CDN to `data/raw/`
- Creates symlinks in `data/approved/` with correct extensions

### 2. Generate Stage 2 Dataset
```bash
# Install dependencies
pip install -r scripts/requirements-approved-image-embeddings.txt

# High-VRAM systems (>40GB) - process everything in one go
python3 scripts/generate_approved_image_dataset.py --progress-every 100

# Low-VRAM systems (8-24GB) - run passes separately
python3 scripts/generate_approved_image_dataset.py --pass dinov3
python3 scripts/generate_approved_image_dataset.py --pass vae
python3 scripts/generate_approved_image_dataset.py --pass t5
python3 scripts/generate_approved_image_dataset.py --pass migrate
```

See `scripts/README.md` for full documentation.

## Stage 2 Format (Current)

### JSONL Record
Each line in `derived/approved_image_dataset.jsonl`:
```json
{
  "image_path": "data/approved/abc123.jpg",
  "caption": "A woman in a red dress...",
  "width": 1024,
  "height": 768,
  "t5_attention_mask": [1, 1, 1, ..., 0, 0],
  "image_id": "abc123",
  "aspect_bucket": "1024x1024",
  "format_version": 2
}
```

**Note:** `dinov3_embedding` is NO LONGER inline - now stored as external `.npy` file.

### External Embeddings (.npy files)
- `derived/dinov3/<image_id>.npy` - 1024 float32 values (~4KB each)
- `derived/vae_latents/<image_id>.npy` - 16×H/8×W/8 float16 (~131KB for 1024×1024)
- `derived/t5_hidden/<image_id>.npy` - 77×1024 float16 (~158KB each)

## Storage Estimates

| Component | Size per Image | Total (31k images) | Notes |
|---|---|---|---|
| Raw images | ~700KB | ~22GB | Downloaded from CDN |
| Approved symlinks | negligible | ~1MB | Symlinks only |
| JSONL metadata | ~200 bytes | ~6MB | Lightweight (no inline embeddings) |
| DINOv3 .npy | ~4KB | ~124MB | 1024 float32 |
| VAE latents .npy | ~131KB | ~4GB | 16×H/8×W/8 float16 |
| T5 hidden .npy | ~158KB | ~5GB | 77×1024 float16 |
| **Subtotal (Stage 2)** | - | **~31GB** | Current implementation |
| WebDataset shards (future) | - | ~35GB | Stage 3: tar archives |
| **Total (all stages)** | - | **~65GB** | With sharding |

**Note:** Estimates based on 31k images currently in dataset. Full dataset may reach 60k images (~120GB total).

## Cleanup

To remove orphaned data when images are deleted:
```bash
# Preview what would be deleted
python3 scripts/prune_dataset.py --dry-run --verbose

# Actually delete orphaned data
python3 scripts/prune_dataset.py
```

This removes:
- Broken symlinks in `data/approved/`
- Orphaned JSONL records
- Orphaned .npy files in all three directories
