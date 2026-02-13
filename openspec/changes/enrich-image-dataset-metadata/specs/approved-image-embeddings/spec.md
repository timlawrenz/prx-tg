## MODIFIED Requirements

### Requirement: Emit one JSONL record per successfully processed image
The system SHALL write output to a JSONL file with one JSON object per image containing at minimum `image_path`, `dinov3_embedding`, `caption`, `t5_attention_mask`, `height`, and `width`.

#### Scenario: Output record contains required fields
- **WHEN** an image is successfully processed
- **THEN** the corresponding JSON object contains `image_path`, `dinov3_embedding`, `caption`, `t5_attention_mask`, `height`, and `width`

#### Scenario: T5 attention mask is computed correctly
- **WHEN** a caption is tokenized with T5-Large tokenizer
- **THEN** the `t5_attention_mask` field contains an array of 1s (valid tokens) and 0s (padding)

#### Scenario: Image dimensions are captured
- **WHEN** an image is successfully processed
- **THEN** the `height` field contains the image height in pixels
- **AND** the `width` field contains the image width in pixels

### Requirement: Resume without reprocessing completed images
The system SHALL support resuming a partially completed run without recomputing embeddings/captions for images already written to the output. When existing records are missing fields, the system SHALL enrich them by adding the missing fields without regenerating existing data.

#### Scenario: Restart resumes from existing output
- **WHEN** the generator is re-run with an existing output JSONL that already contains a record for `image_path = X`
- **THEN** the generator does not emit a duplicate record for `X`

#### Scenario: Existing records are enriched with missing fields
- **WHEN** the generator is re-run with an existing output JSONL that contains records missing `t5_attention_mask`, `height`, or `width`
- **THEN** the generator reads each existing record, adds the missing fields, and writes the enriched record back

#### Scenario: Expensive operations are not recomputed during enrichment
- **WHEN** an existing record has `dinov3_embedding` and `caption` but is missing metadata fields
- **THEN** the generator does not recompute the DINOv3 embedding or Gemma caption
- **AND** only computes the missing `t5_attention_mask`, `height`, and `width`

#### Scenario: Fully complete records are skipped
- **WHEN** an existing record has all required fields (`image_path`, `dinov3_embedding`, `caption`, `t5_attention_mask`, `height`, `width`)
- **THEN** the generator skips reprocessing that image entirely

### Requirement: Progress reporting
The system SHALL report progress periodically, including at least the count of images processed and the current image (or current position). Progress reporting SHALL distinguish between new processing, enrichment, and skipping.

#### Scenario: Periodic progress output
- **WHEN** the generator processes many images
- **THEN** it prints periodic progress updates without waiting for completion

#### Scenario: Progress distinguishes operation types
- **WHEN** the generator processes a mix of new images, incomplete records, and complete records
- **THEN** progress output indicates whether each image is being "processed new", "enriched", or "skipped"
