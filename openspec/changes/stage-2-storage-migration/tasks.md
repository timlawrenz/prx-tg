## 1. Add Flux VAE Model Support

- [x] 1.1 Add Flux VAE model ID constant (determine exact model path: `black-forest-labs/FLUX.1-dev` or VAE-specific)
- [x] 1.2 Create `load_flux_vae()` function that loads VAE encoder component only (not full diffusion model)
- [x] 1.3 Add VAE preprocessing function to prepare images for encoding (resize, normalize to VAE input range)
- [x] 1.4 Create `encode_vae_latent(image_path: Path, vae_encoder) -> np.ndarray` function that returns (16, H//8, W//8) float16 array
- [x] 1.5 Add error handling for VAE encoding failures (log and skip image on error)
- [x] 1.6 Test VAE encoding on single sample image, verify output shape and dtype

## 2. Add T5-Large Encoder Support

- [x] 2.1 Add T5_MODEL_ID constant (value: "t5-large") if not already present
- [x] 2.2 Create `load_t5_encoder()` function that loads full T5-Large encoder model (not just tokenizer)
- [x] 2.3 Create `compute_t5_hidden_states(caption: str, tokenizer, encoder) -> np.ndarray` function that returns (77, 1024) float16 array
- [x] 2.4 Add validation to ensure hidden state shape is exactly (77, 1024)
- [x] 2.5 Test T5 encoding on sample caption, verify output shape and dtype

## 3. Add Aspect Ratio Bucketing

- [x] 3.1 Define ASPECT_BUCKETS constant as list of tuples: [(1024, 1024), (832, 1216), (1216, 832), (768, 1280), (1280, 768), (704, 1344), (1344, 704)]
- [x] 3.2 Create `compute_aspect_ratio(width: int, height: int) -> float` helper function
- [x] 3.3 Create `assign_aspect_bucket(width: int, height: int) -> str` function that returns closest bucket as string (e.g., "1024x1024")
- [x] 3.4 Add unit test or validation for bucket assignment with edge cases (square, portrait, landscape, extreme ratios)

## 4. Add Image ID Field

- [x] 4.1 Create `compute_image_id(image_path: Path) -> str` function that returns basename without extension
- [x] 4.2 Add `image_id` field to all new JSONL records
- [x] 4.3 Update existing record loading to compute `image_id` if missing (migration path)

## 5. Implement .npy File I/O

- [x] 5.1 Create output directories: `data/derived/dinov3/`, `data/derived/vae_latents/`, `data/derived/t5_hidden/` at script start
- [x] 5.2 Create `save_npy(array: np.ndarray, path: Path, dtype: np.dtype)` helper function with type conversion
- [x] 5.3 Create `load_npy(path: Path) -> np.ndarray` helper with error handling (return None if missing/corrupt)
- [x] 5.4 Add `extract_dinov3_to_npy(record: dict, output_dir: Path)` function that saves inline embedding to file
- [x] 5.5 Add `save_vae_latent(latent: np.ndarray, image_id: str, output_dir: Path)` function
- [x] 5.6 Add `save_t5_hidden(hidden: np.ndarray, image_id: str, output_dir: Path)` function

## 6. Modify JSONL Record Format

- [x] 6.1 Add `format_version` field (value: 2) to all new/migrated records
- [x] 6.2 Add `aspect_bucket` field computation using assign_aspect_bucket()
- [x] 6.3 Remove `dinov3_embedding` field from records after extracting to .npy file (Stage 1 → Stage 2 migration)
- [x] 6.4 Ensure all required fields are present: `image_id`, `image_path`, `caption`, `t5_attention_mask`, `height`, `width`, `aspect_bucket`, `format_version`

## 7. Implement Three-Pass Architecture

- [x] 7.1 Add `--pass` CLI argument with choices: ["all", "dinov3", "vae", "t5", "migrate"] (default: "all")
- [x] 7.2 Create `needs_dinov3_extraction(record: dict, dinov3_dir: Path) -> bool` function (checks for inline embedding and missing .npy file)
- [x] 7.3 Create `needs_vae_latent(record: dict, vae_dir: Path) -> bool` function (checks for missing .npy file)
- [x] 7.4 Create `needs_t5_hidden(record: dict, t5_dir: Path) -> bool` function (checks for missing .npy file)
- [x] 7.5 Create `needs_migration(record: dict) -> bool` function (checks if format_version != 2)
- [x] 7.6 Implement pass filtering logic in main loop (only execute operations for selected pass)

## 8. Refactor Main Processing Loop

- [x] 8.1 Modify main() to load models conditionally based on selected pass (only load VAE if pass includes "vae")
- [x] 8.2 Update batch processing to handle three embedding types independently (DINOv3 extraction, VAE encoding, T5 encoding)
- [x] 8.3 Add logic to skip records where all required .npy files exist and format_version == 2
- [x] 8.4 Implement explicit model unloading and cache clearing between passes (del model; torch.cuda.empty_cache())
- [x] 8.5 Update flush_batch() to write Stage 2 format records to .tmp file

## 9. Update Progress Tracking

- [x] 9.1 Add four progress counters: `migrated`, `enriched`, `extracted`, `skipped`
- [x] 9.2 Increment `migrated` when converting Stage 1 record to Stage 2 with all embeddings
- [x] 9.3 Increment `enriched` when adding missing VAE/T5 embeddings to existing Stage 2 record
- [x] 9.4 Increment `extracted` when only extracting DINOv3 from Stage 1 record (format_version == 2 but .npy missing)
- [x] 9.5 Increment `skipped` when record is complete (format_version == 2 and all .npy files exist)
- [x] 9.6 Update progress output format: "progress: X migrated, Y enriched, Z extracted, W skipped (total: N) rate=R/s"

## 10. Add Disk Space and Validation Checks

- [x] 10.1 Add disk space check at script start (abort if < 20GB available in data/derived/)
- [x] 10.2 Add `--verify` flag that scans JSONL records and checks for missing/corrupt .npy files
- [x] 10.3 Implement verification logic: load each .npy file, check shape and dtype match expectations
- [x] 10.4 Report verification results: count of valid/invalid/missing files per embedding type

## 11. Update CLI Arguments and Help Text

- [x] 11.1 Update script help text to document Stage 2 hybrid format
- [x] 11.2 Add `--pass` argument documentation with pass descriptions
- [x] 11.3 Add `--verify` argument documentation
- [x] 11.4 Add `--output-base-dir` argument to allow custom output location (default: data/derived/)
- [x] 11.5 Update `--no-resume` flag behavior to delete .npy directories in addition to JSONL

## 12. Handle Stage 1 to Stage 2 Migration

- [x] 12.1 Implement logic to detect Stage 1 records (inline `dinov3_embedding` present, no `format_version` field)
- [x] 12.2 Extract inline DINOv3 embedding to `dinov3/{image_id}.npy` file
- [x] 12.3 Add `image_id` and `aspect_bucket` fields to record
- [x] 12.4 Set `format_version = 2`
- [x] 12.5 Remove `dinov3_embedding` field from record
- [x] 12.6 Preserve all other fields unchanged (`caption`, `t5_attention_mask`, `height`, `width`)

## 13. Update Documentation

- [x] 13.1 Update scripts/README.md with Stage 2 format documentation
- [x] 13.2 Document output structure (JSONL + three .npy directories)
- [x] 13.3 Add example of Stage 2 JSONL record format
- [x] 13.4 Document three-pass architecture and `--pass` flag usage
- [x] 13.5 Add migration guide (Stage 1 → Stage 2) with example commands
- [x] 13.6 Document storage requirements (18GB for 60k images)
- [x] 13.7 Add troubleshooting section (common errors, verification steps)

## 14. Testing and Validation

- [ ] 14.1 Test DINOv3 extraction pass on existing Stage 1 data (10 sample records)
- [ ] 14.2 Verify extracted .npy files have correct shape (1024,) and dtype (float32)
- [ ] 14.3 Test VAE encoding pass on sample images (verify shape (16, H//8, W//8) and dtype float16)
- [ ] 14.4 Test T5 encoding pass on sample captions (verify shape (77, 1024) and dtype float16)
- [ ] 14.5 Test full migration on 10-image subset (Stage 1 → Stage 2 with all embeddings)
- [ ] 14.6 Verify JSONL format v2 records have all required fields and no inline embeddings
- [ ] 14.7 Test idempotency: run script twice on same data, verify no regeneration on second run
- [ ] 14.8 Test crash recovery: interrupt script mid-processing, restart, verify correct resumption
- [ ] 14.9 Test aspect bucket assignment on various image ratios (square, portrait, landscape, extreme)
- [ ] 14.10 Verify `--verify` flag correctly identifies missing/corrupt .npy files
- [ ] 14.11 Run full migration on complete dataset (1444+ existing records)
- [ ] 14.12 Verify total storage size matches expectations (~18GB for full dataset)
