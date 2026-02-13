## 1. Add T5 Tokenizer Support

- [x] 1.1 Add T5-Large tokenizer loading to script initialization (load once, reuse)
- [x] 1.2 Create `compute_t5_attention_mask(caption: str) -> list[int]` function that tokenizes caption with max_length=77, padding="max_length", truncation=True
- [x] 1.3 Add unit test or inline verification that attention mask is list of 1s and 0s with length 77

## 2. Add Image Dimension Reading

- [x] 2.1 Create `get_image_dimensions(image_path: Path) -> tuple[int, int]` function using PIL header reading (returns width, height)
- [x] 2.2 Handle errors gracefully (if image can't be opened, return None or skip)

## 3. Implement Two-Pass Architecture

- [x] 3.1 Create `load_existing_records(output_path: Path) -> dict[str, dict]` function that reads JSONL and returns `{image_path: record}` dict
- [x] 3.2 Handle malformed JSONL lines gracefully (skip invalid lines, log warning)
- [x] 3.3 Modify main loop to check if image_path exists in existing_records dict before processing

## 4. Implement Field Detection and Enrichment Logic

- [x] 4.1 Create `needs_field(record: dict, field_name: str) -> bool` helper to check if field is missing or null/empty
- [x] 4.2 Modify processing logic to conditionally compute DINOv3 embedding only if `needs_field(record, "dinov3_embedding")`
- [x] 4.3 Modify processing logic to conditionally compute Gemma caption only if `needs_field(record, "caption")`
- [x] 4.4 Add logic to compute `t5_attention_mask` if missing (always when enriching old records)
- [x] 4.5 Add logic to compute `height` and `width` if missing (always when enriching old records)
- [x] 4.6 Ensure all six fields are present in final record: `image_path`, `dinov3_embedding`, `caption`, `t5_attention_mask`, `height`, `width`

## 5. Implement Atomic Write

- [x] 5.1 Replace direct write to output_path with write to temporary file (use `output_path.with_suffix(".tmp")`)
- [x] 5.2 Write all records to temp file (one JSON object per line)
- [x] 5.3 After successful write, atomically rename temp file to output_path using `Path.rename()`
- [x] 5.4 Ensure original file is preserved if script crashes before rename

## 6. Update Progress Reporting

- [x] 6.1 Track three counters: `processed_new`, `enriched`, `skipped`
- [x] 6.2 Increment `processed_new` when generating full record (DINOv3 + Gemma + metadata)
- [x] 6.3 Increment `enriched` when adding only metadata to existing record (has embeddings/caption)
- [x] 6.4 Increment `skipped` when all fields already present
- [x] 6.5 Update periodic progress output to show: "Processing: X new, Y enriched, Z skipped (total: N/M)"
- [x] 6.6 Add verbose logging that shows which operation is performed for each image

## 7. Remove Sidecar .idx File Logic

- [x] 7.1 Remove `read_completed_paths()` function that reads `.idx` file
- [x] 7.2 Remove code that writes to `.idx` file after each record
- [x] 7.3 Update `--no-resume` flag behavior to delete output JSONL (not `.idx` file)

## 8. Update CLI Arguments and Documentation

- [x] 8.1 Update script help text to document new behavior (automatic enrichment, idempotent by default)
- [x] 8.2 Update `--no-resume` flag help text to clarify it now deletes output and regenerates everything
- [x] 8.3 Update scripts/README.md with enrichment examples and behavior explanation

## 9. Testing and Validation

- [ ] 9.1 Test on empty directory (new dataset generation) - verify all 6 fields present
- [ ] 9.2 Test on existing JSONL with old format (3 fields) - verify enrichment adds 3 new fields without modifying old ones
- [ ] 9.3 Test on partially complete dataset (some have 6 fields, some have 3) - verify mixed behavior works
- [ ] 9.4 Test `--no-resume` flag - verify it clears output and regenerates from scratch
- [ ] 9.5 Verify progress reporting shows correct counts for processed_new, enriched, skipped
- [ ] 9.6 Validate output JSONL format: all records have exactly 6 fields with correct types
