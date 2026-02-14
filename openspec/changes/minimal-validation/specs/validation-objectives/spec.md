## Purpose

Define three validation objectives to verify the Nano DiT architecture learns correctly: reconstruction fidelity, DINO embedding control, and text prompt manipulation.

## ADDED Requirements

### Requirement: Reconstruction fidelity test
The system SHALL generate images from the validation set using their original caption and DINOv3 embedding, then measure visual similarity to verify the model has memorized the 100-image dataset.

#### Scenario: Generate from validation sample
- **WHEN** validation runs with a sample's caption and DINO embedding
- **THEN** the system generates an image and computes reconstruction similarity score

#### Scenario: High reconstruction similarity indicates success
- **WHEN** all 100 validation samples are generated
- **THEN** average LPIPS distance < 0.2 indicates successful memorization

#### Scenario: Validation runs every 1000 steps
- **WHEN** training reaches step 1000, 2000, 3000, 4000, 5000
- **THEN** reconstruction fidelity test is executed and logged

### Requirement: DINO embedding swap test
The system SHALL swap DINOv3 embeddings between two images while keeping their original captions, then verify the generated image adopts the composition/lighting of the swapped DINO source.

#### Scenario: Generate with swapped DINO embedding
- **WHEN** validation swaps DINO of Image A with Image B while keeping caption A
- **THEN** the generated image should resemble B's composition but follow caption A's description

#### Scenario: DINO swap test runs on fixed pairs
- **WHEN** validation runs DINO swap test
- **THEN** it uses 5 predefined image pairs for consistency across runs

#### Scenario: Visual inspection confirms DINO control
- **WHEN** DINO swap test completes
- **THEN** generated images are saved to validation output directory for manual inspection

### Requirement: Text prompt manipulation test
The system SHALL modify captions (e.g., "looking left" → "looking right") while keeping the same DINO embedding, then verify the generated image follows the text change.

#### Scenario: Modify directional prompts
- **WHEN** validation changes "left" to "right" in a caption
- **THEN** the generated image should show the subject looking right instead of left

#### Scenario: Text manipulation test runs on fixed samples
- **WHEN** validation runs text manipulation test
- **THEN** it uses 5 predefined caption modifications for consistency

#### Scenario: Modifications cover spatial and attribute changes
- **WHEN** text manipulation test runs
- **THEN** test cases include directional changes (left/right), attribute changes (red dress → blue dress), and pose changes

#### Scenario: Visual inspection confirms text control
- **WHEN** text manipulation test completes
- **THEN** generated images are saved to validation output directory for manual inspection

### Requirement: Sampling uses dual CFG
The system SHALL generate validation samples using dual classifier-free guidance with text_scale=3.0 and dino_scale=2.0.

#### Scenario: CFG formula uses both scales
- **WHEN** sampling for validation
- **THEN** final velocity v = v_uncond + text_scale × (v_text - v_uncond) + dino_scale × (v_dino - v_uncond)

#### Scenario: Three forward passes per sampling step
- **WHEN** one denoising step is executed
- **THEN** model runs three times: unconditional, text-only, dino-only

### Requirement: Sampling uses 50 denoising steps
The system SHALL use 50 Euler steps for validation sampling to balance quality and speed.

#### Scenario: Timestep schedule is uniform
- **WHEN** sampling begins
- **THEN** timesteps are uniformly spaced from t=1.0 to t=0.0 in 50 steps

#### Scenario: Euler integration updates latent
- **WHEN** each denoising step is executed
- **THEN** latent is updated using z_{t-1} = z_t + v_pred × dt

### Requirement: Decode VAE latents to images
The system SHALL decode final latents to RGB images using the Flux VAE decoder and save to PNG files.

#### Scenario: Latents are decoded to 512x512 images
- **WHEN** a 16×64×64 latent is decoded
- **THEN** output is a 512×512 RGB image

#### Scenario: Images are saved with descriptive names
- **WHEN** validation generates an image for sample with image_id="abc123" and test="reconstruction"
- **THEN** output file is named "step5000_abc123_reconstruction.png"

#### Scenario: Generated images are organized by step
- **WHEN** validation runs at step 3000
- **THEN** all generated images are saved to "validation/step3000/" directory

### Requirement: Validation output is organized
The system SHALL save validation outputs to a structured directory with logs and images organized by step and test type.

#### Scenario: Directory structure is created
- **WHEN** validation runs for the first time
- **THEN** directory structure "validation/step{N}/{test_type}/" is created

#### Scenario: Validation log is written
- **WHEN** each validation run completes
- **THEN** metrics (LPIPS scores, test results) are written to "validation/step{N}/results.json"

#### Scenario: Images are grouped by test type
- **WHEN** validation completes
- **THEN** reconstruction images are in "reconstruction/", DINO swap in "dino_swap/", text manipulation in "text_manip/"
