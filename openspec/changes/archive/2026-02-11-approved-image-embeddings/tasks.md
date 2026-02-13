## 1. Setup

- [x] 1.1 Add Python dependency pinning for `torch`, `transformers`, and `pillow` (and document install options)
- [x] 1.2 Create script skeleton `scripts/generate_approved_image_dataset.py` with CLI args (input dir, output path, limit, device, progress interval, resume)

## 2. Input enumeration + resume

- [x] 2.1 Implement deterministic enumeration of `data/approved/` images (sorted filenames)
- [x] 2.2 Implement resume by reading existing output JSONL (or a sidecar index) and skipping already-emitted `image_path`s
- [x] 2.3 Implement unreadable-image handling (catch decode errors, count skipped, continue)

## 3. DINOv3 embeddings

- [x] 3.1 Load `facebook/dinov3-vitl16-pretrain-lvd1689m` + image processor; support selecting device (cpu/cuda)
- [x] 3.2 Implement embedding extraction to a fixed-length vector per image and serialize into JSON-compatible form

## 4. Gemma captions

- [x] 4.1 Load `google/gemma-3-27b-it` tokenizer/model for generation; configure ~300 tokens via `max_new_tokens`
- [x] 4.2 Implement caption generation using the exact prompt; post-process to enforce a single paragraph (no blank lines)

## 5. JSONL output + progress

- [x] 5.1 Write append-only JSONL records `{image_path, dinov3_embedding, caption}` and flush periodically
- [x] 5.2 Add progress reporting (processed/skipped counts + current image) at a configurable interval

## 6. Smoke test

- [x] 6.1 Run the script with `--limit` on a small subset and verify JSONL records are produced with required fields
