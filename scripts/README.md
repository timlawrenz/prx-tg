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

Generates a JSONL dataset from `data/approved/` containing DINOv3 embeddings + Gemma captions.

### Install

```bash
python3 -m pip install -r scripts/requirements-approved-image-embeddings.txt

# If needed, install torch using the PyTorch installer for your platform/CUDA/ROCm:
# https://pytorch.org/get-started/locally/
```

### Usage

```bash
python3 scripts/generate_approved_image_dataset.py --help

# Smoke test (will download large models)
python3 scripts/generate_approved_image_dataset.py --limit 2 --output data/derived/approved_image_dataset.jsonl
```

**Note:** Do not enable `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` when running this script on ROCm systems, as it causes hardware exceptions with the Gemma 3 model.
