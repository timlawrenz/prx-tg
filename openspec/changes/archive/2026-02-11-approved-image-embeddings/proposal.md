## Why

We need a machine-readable dataset derived from the approved photo set (`data/approved/`) that includes both semantic embeddings (for similarity/search/training) and consistent dense captions (for text-to-image training and analysis).

## What Changes

- Add a pipeline/script that iterates all images in `data/approved/`.
- For each image:
  - Compute a DINOv3 embedding using Hugging Face model `facebook/dinov3-vitl16-pretrain-lvd1689m`.
  - Generate a ~300-token caption using Hugging Face model `google/gemma-3-27b-it` with the provided prompt.
  - Write one JSONL record containing: image path, embedding, and caption.
- Ensure the pipeline can be resumed/re-run without reprocessing completed items.

## Capabilities

### New Capabilities
- `approved-image-embeddings`: Produce a JSONL dataset for `data/approved/` images containing (a) DINOv3 embeddings and (b) Gemma-generated dense captions.

### Modified Capabilities

## Impact

- Compute: GPU/accelerator is strongly preferred for DINOv3 + Gemma at scale.
- Dependencies: introduces Hugging Face `transformers` (and likely `torch`) + image loading.
- Storage: JSONL size will be large (embedding vectors for all approved images).
- Runtime: long-running batch job; needs progress reporting and resume semantics.
