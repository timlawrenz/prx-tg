## ADDED Requirements

### Requirement: Enumerate approved images deterministically
The system SHALL iterate over the image files in `data/approved/` in a deterministic order and process each image at most once per dataset generation run.

#### Scenario: Stable iteration order
- **WHEN** the dataset generator is run twice against an unchanged `data/approved/` directory
- **THEN** the sequence of `image_path` values emitted is identical between runs

### Requirement: Generate a DINOv3 embedding per image
For each approved image, the system SHALL compute a fixed-length embedding using Hugging Face model `facebook/dinov3-vitl16-pretrain-lvd1689m` from the image pixels.

#### Scenario: Embedding is present in output record
- **WHEN** an image is successfully processed
- **THEN** its JSONL record contains a non-empty `dinov3_embedding` value

### Requirement: Generate a dense caption per image using Gemma
For each approved image, the system SHALL generate a caption using Hugging Face model `google/gemma-3-27b-it` with the prompt:
`You are generating a caption for a text-to-image training dataset. Write exactly one dense paragraph in a dry, descriptive tone (no flowery language, no lists). Describe only what is visible in the image; do not guess or invent details. Include (when visible): subject, pose, clothing/accessories, lighting, background, composition/framing, and camera angle.`

#### Scenario: Caption is a single paragraph
- **WHEN** an image is successfully processed
- **THEN** its `caption` field contains exactly one paragraph (no blank lines)

### Requirement: Emit one JSONL record per successfully processed image
The system SHALL write output to a JSONL file with one JSON object per image containing at minimum `image_path`, `dinov3_embedding`, and `caption`.

#### Scenario: Output record contains required fields
- **WHEN** an image is successfully processed
- **THEN** the corresponding JSON object contains `image_path`, `dinov3_embedding`, and `caption`

### Requirement: Resume without reprocessing completed images
The system SHALL support resuming a partially completed run without recomputing embeddings/captions for images already written to the output.

#### Scenario: Restart resumes from existing output
- **WHEN** the generator is re-run with an existing output JSONL that already contains a record for `image_path = X`
- **THEN** the generator does not emit a duplicate record for `X`

### Requirement: Progress reporting
The system SHALL report progress periodically, including at least the count of images processed and the current image (or current position).

#### Scenario: Periodic progress output
- **WHEN** the generator processes many images
- **THEN** it prints periodic progress updates without waiting for completion

### Requirement: Error handling for unreadable images
If an input image cannot be decoded/read, the system SHALL skip it and continue processing remaining images.

#### Scenario: Unreadable image is skipped
- **WHEN** an image file in `data/approved/` is unreadable or not decodable
- **THEN** the generator skips that file and continues processing subsequent images
