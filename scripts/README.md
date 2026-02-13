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

Generates/enriches a JSONL dataset from `data/approved/` containing DINOv3 embeddings, Gemma captions, and metadata (T5 attention masks, image dimensions).

**Key features:**
- **Idempotent by default**: Automatically enriches existing records with missing fields
- **Smart resumability**: Skips expensive operations (DINOv3, Gemma) when data already exists
- **Incremental progress saving**: Writes to `.tmp` file during processing, so Ctrl+C preserves partial work
- **Atomic commits**: Final output only updated on successful completion

### Output Format

Each line in the JSONL file contains one JSON object with 6 fields:

```json
{
  "image_path": "data/approved/abc123.jpg",
  "dinov3_embedding": [0.123, -0.456, ...],  // 1024-dim float array
  "caption": "A woman in a red dress...",     // Single paragraph description
  "t5_attention_mask": [1, 1, 1, ..., 0, 0], // 77 ints (1=valid, 0=padding)
  "height": 1024,                             // Image height in pixels
  "width": 768                                // Image width in pixels
}
```

### Install

```bash
python3 -m pip install -r scripts/requirements-approved-image-embeddings.txt

# If needed, install torch using the PyTorch installer for your platform/CUDA/ROCm:
# https://pytorch.org/get-started/locally/
```

### Usage

**First run (generate new dataset):**
```bash
python3 scripts/generate_approved_image_dataset.py \
  --output data/derived/approved_image_dataset.jsonl \
  --progress-every 100
```

**Enrich existing dataset (add missing fields):**
```bash
# Same command - script automatically detects existing records and enriches them
python3 scripts/generate_approved_image_dataset.py \
  --output data/derived/approved_image_dataset.jsonl
```

**Force regeneration from scratch:**
```bash
python3 scripts/generate_approved_image_dataset.py \
  --output data/derived/approved_image_dataset.jsonl \
  --no-resume
```

**Smoke test (process 2 images):**
```bash
python3 scripts/generate_approved_image_dataset.py \
  --limit 2 \
  --output data/derived/test.jsonl \
  --verbose
```

### Enrichment Behavior

The script intelligently determines what to compute for each image:

| Existing Fields | Operation | What's Computed |
|---|---|---|
| None (new image) | **Process new** | DINOv3 + Gemma + metadata |
| Has embeddings/caption | **Enrich** | Only T5 mask + dimensions (fast) |
| All 6 fields present | **Skip** | Nothing (already complete) |

Progress output shows:
```
progress: 150 new, 4800 enriched, 100 skipped (total: 5050) rate=2.5/s
```

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

### Notes

- **Do not enable** `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` on ROCm systems (causes hardware exceptions with Gemma 3)
- **First run downloads large models** (~55GB for Gemma-3-27B, ~1.5GB for DINOv3)
- **ROCm kernel compilation** may cause 15-30 min wait on first GPU inference (disable `PYTORCH_TUNABLEOP_TUNING=1` for smoke tests)
