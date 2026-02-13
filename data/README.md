# Data Directory Structure

This directory contains the image datasets and generated embeddings for the diffusion model training pipeline.

**⚠️ Not committed to git** - this directory is in `.gitignore` due to size (100GB+ expected).

## Directory Layout

```
data/
├── raw/               # Original images (no file extensions)
│                      # ~60k images from crawlr system
├── approved/          # Symlinked approved images
│                      # Populated by crawling https://crawlr.lawrenz.com/photos.json
└── embeddings/        # Generated JSONL files with DINOv3 + Gemma captions
    └── approved.jsonl # Output from scripts/generate_approved_image_dataset.py
```

## How to Populate

### 1. Raw Images
Place raw images (without file extensions) in `data/raw/`.

### 2. Approved Symlinks
Run the symlink script to populate `data/approved/` from the approved photos API:
```bash
# TODO: Document the symlink script once implemented
```

### 3. Generate Embeddings
```bash
# Install dependencies
pip install -r scripts/requirements-approved-image-embeddings.txt

# Generate JSONL with DINOv3 embeddings + Gemma captions
python scripts/generate_approved_image_dataset.py \
  --input-dir data/approved \
  --output data/embeddings/approved.jsonl \
  --progress-every 100
```

See `scripts/README.md` for full documentation.

## JSONL Format

Each line in `embeddings/approved.jsonl`:
```json
{
  "image_path": "data/approved/abc123.jpg",
  "dinov3_embedding": [0.123, -0.456, ...],  // 1024-dim float array
  "caption": "A woman in a red dress..."     // Single paragraph, no lists
}
```

## Storage Estimates

- Raw images: ~40GB (60k images @ ~700KB each)
- Approved symlinks: negligible (symlinks only)
- Embeddings JSONL: ~2GB (60k × 32KB per entry)
- VAE latents (future): ~30GB (60k × 16ch × 64×64 × float16)
- T5 embeddings (future): ~20GB (60k × 77 tokens × 1024-dim × float16)
- WebDataset shards (future): ~90GB (combined, compressed)

**Total: ~180GB**
