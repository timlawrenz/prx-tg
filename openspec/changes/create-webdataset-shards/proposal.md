## Why

Stage 2 embeddings are stored as individual .npy files (60k+ files across three directories). Reading individual files during training causes severe I/O bottlenecks due to filesystem overhead and random seeks. WebDataset tar shards enable sequential streaming (10-100× faster), batch loading ~1000 samples per file handle, and cloud-native distribution. This is required before starting validation/training runs.

## What Changes

- Add `scripts/create_webdataset_shards.py` to package Stage 2 derived embeddings into WebDataset tar format
- Group samples by aspect ratio bucket and shard within each bucket
- Support configurable shard size (default 1000 samples/shard) and sample limits for validation dataset creation
- Output structure: `data/shards/bucket_{WxH}/shard-NNNNNN.tar`
- Each tar entry contains: `{image_id}.json`, `{image_id}.dinov3.npy`, `{image_id}.vae.npy`, `{image_id}.t5h.npy`, `{image_id}.t5m.npy`

## Capabilities

### New Capabilities
- `webdataset-shards`: Packaging Stage 2 embeddings (metadata JSONL + .npy files) into WebDataset tar shards grouped by aspect bucket for efficient sequential streaming during training

### Modified Capabilities
<!-- No existing capability requirements are changing -->

## Impact

- **New script**: `scripts/create_webdataset_shards.py`
- **New output directory**: `data/shards/bucket_{WxH}/` (not committed to git, added to .gitignore)
- **Dependencies**: Requires `numpy` (already present); WebDataset library itself is only needed during training (dataloader), not during shard creation
- **Storage**: ~90GB for 60k images (metadata + embeddings packed into tars)
- **Enables**: Part B (Minimal Validation) and Part C (Full Scale Training) from the-plan.md
