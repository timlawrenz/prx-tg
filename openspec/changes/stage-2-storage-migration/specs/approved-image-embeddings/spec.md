## ADDED Requirements

### Requirement: Generate VAE latents per image
For each approved image, the system SHALL encode the image to latent space using the Flux VAE encoder and save the result as a separate `.npy` file.

#### Scenario: VAE latent is generated for each image
- **WHEN** an image is successfully processed in Stage 2
- **THEN** a VAE latent file exists at `data/derived/vae_latents/{image_id}.npy`

#### Scenario: VAE latent uses Flux VAE encoder
- **WHEN** generating VAE latents
- **THEN** the system uses the VAE component from Flux model (`black-forest-labs/FLUX.1-dev` or equivalent)

#### Scenario: VAE latent encoding preserves original resolution
- **WHEN** encoding an image to VAE latent space
- **THEN** the latent is generated from the image at its original dimensions (not bucket dimensions)

### Requirement: Generate T5 hidden states per caption
For each caption, the system SHALL encode it using T5-Large encoder to produce hidden states (not just pooled embeddings) and save as a separate `.npy` file.

#### Scenario: T5 hidden states are generated for each caption
- **WHEN** a caption is successfully processed in Stage 2
- **THEN** a T5 hidden state file exists at `data/derived/t5_hidden/{image_id}.npy`

#### Scenario: T5 uses full encoder model
- **WHEN** generating T5 hidden states
- **THEN** the system loads the full T5-Large encoder model (not just tokenizer)
- **AND** extracts the `last_hidden_state` from the encoder output

#### Scenario: T5 hidden state is sequence not pooled
- **WHEN** a T5 hidden state is generated
- **THEN** the output shape is (77, 1024) representing 77 tokens with 1024-dimensional embeddings
- **AND** the output is NOT a single pooled vector

### Requirement: Assign aspect ratio buckets
The system SHALL analyze each image's aspect ratio and assign it to the closest predefined bucket from a set of seven standard resolutions.

#### Scenario: Aspect ratio is computed correctly
- **WHEN** an image has width W and height H
- **THEN** the aspect ratio is computed as W / H

#### Scenario: Closest bucket is selected
- **WHEN** an image's aspect ratio is 0.65
- **THEN** the system assigns it to bucket `"832x1216"` (ratio 0.68, closest match)

#### Scenario: Bucket assignment is recorded in metadata
- **WHEN** an image is assigned to an aspect bucket
- **THEN** the JSONL record contains an `aspect_bucket` field with the bucket name (e.g., `"1024x1024"`)

## MODIFIED Requirements

### Requirement: Generate a DINOv3 embedding per image
For each approved image, the system SHALL compute a fixed-length embedding using Hugging Face model `facebook/dinov3-vitl16-pretrain-lvd1689m` from the image pixels and save it as a separate `.npy` file.

#### Scenario: Embedding is saved to external file
- **WHEN** an image is successfully processed in Stage 2
- **THEN** a DINOv3 embedding file exists at `data/derived/dinov3/{image_id}.npy`

#### Scenario: Embedding is not duplicated in JSONL
- **WHEN** an image is successfully processed in Stage 2
- **THEN** the JSONL record does NOT contain a `dinov3_embedding` field

### Requirement: Emit one JSONL record per successfully processed image
The system SHALL write output to a JSONL file with one JSON object per image containing `image_id`, `image_path`, `caption`, `t5_attention_mask`, `height`, `width`, `aspect_bucket`, and `format_version`. Embedding arrays SHALL be stored in external `.npy` files.

#### Scenario: Output record contains required metadata fields
- **WHEN** an image is successfully processed in Stage 2
- **THEN** the corresponding JSON object contains `image_id`, `image_path`, `caption`, `t5_attention_mask`, `height`, `width`, `aspect_bucket`, and `format_version`

#### Scenario: Output record does not contain inline embeddings
- **WHEN** an image is successfully processed in Stage 2
- **THEN** the JSON object does NOT contain `dinov3_embedding`, `vae_latent`, or `t5_hidden_state` fields

#### Scenario: Image ID enables cross-referencing
- **WHEN** a JSONL record has `image_id = "abc123xyz"`
- **THEN** the corresponding embedding files can be located at:
  - `data/derived/dinov3/abc123xyz.npy`
  - `data/derived/vae_latents/abc123xyz.npy`
  - `data/derived/t5_hidden/abc123xyz.npy`

#### Scenario: Format version identifies record structure
- **WHEN** an image is successfully processed in Stage 2
- **THEN** the `format_version` field equals `2`
- **AND** this indicates the hybrid storage format with external embeddings

#### Scenario: T5 attention mask is computed correctly
- **WHEN** a caption is tokenized with T5-Large tokenizer
- **THEN** the `t5_attention_mask` field contains an array of 1s (valid tokens) and 0s (padding)

#### Scenario: Image dimensions are captured
- **WHEN** an image is successfully processed
- **THEN** the `height` field contains the image height in pixels
- **AND** the `width` field contains the image width in pixels

### Requirement: Resume without reprocessing completed images
The system SHALL support resuming a partially completed run without recomputing embeddings/captions for images already written to the output. When existing records are missing fields or external files, the system SHALL enrich them by adding the missing components without regenerating existing data. The system SHALL detect completion by checking both JSONL format version and existence of external `.npy` files.

#### Scenario: Restart resumes from existing output
- **WHEN** the generator is re-run with an existing output JSONL that already contains a record for `image_path = X` with `format_version = 2`
- **THEN** the generator does not emit a duplicate record for `X`

#### Scenario: Existing Stage 1 records are migrated to Stage 2
- **WHEN** the generator encounters a record with inline `dinov3_embedding` field (Stage 1 format)
- **THEN** the generator extracts the embedding to an external `.npy` file
- **AND** updates the record to Stage 2 format (adds `image_id`, `aspect_bucket`, `format_version`, removes inline embedding)

#### Scenario: Existing records are enriched with missing embeddings
- **WHEN** the generator is re-run with a Stage 2 record that is missing one or more `.npy` files
- **THEN** the generator creates only the missing embedding files
- **AND** does not regenerate embeddings that already exist

#### Scenario: Expensive operations are not recomputed during enrichment
- **WHEN** an existing record has DINOv3 and caption but is missing VAE or T5 files
- **THEN** the generator does not recompute the DINOv3 embedding or Gemma caption
- **AND** only computes the missing VAE latent and/or T5 hidden state

#### Scenario: Fully complete records are skipped
- **WHEN** an existing record has `format_version = 2` and all three `.npy` files exist (DINOv3, VAE, T5)
- **THEN** the generator skips reprocessing that image entirely

### Requirement: Progress reporting
The system SHALL report progress periodically, including at least the count of images processed and the current image (or current position). Progress reporting SHALL distinguish between new processing, migration, enrichment, and skipping.

#### Scenario: Periodic progress output
- **WHEN** the generator processes many images
- **THEN** it prints periodic progress updates without waiting for completion

#### Scenario: Progress distinguishes operation types
- **WHEN** the generator processes a mix of new images, Stage 1 records, incomplete Stage 2 records, and complete Stage 2 records
- **THEN** progress output indicates whether each image is being "migrated" (Stage 1→2), "enriched" (missing embeddings added), or "skipped" (already complete)

## REMOVED Requirements

None - all existing requirements are preserved or enhanced in Stage 2.
