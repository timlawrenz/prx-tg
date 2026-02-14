## 1. Project Structure

- [x] 1.1 Create validation/ directory in repository root
- [x] 1.2 Create validation/__init__.py (empty package marker)
- [x] 1.3 Create validation/requirements.txt with dependencies (torch, webdataset, lpips, tqdm)

## 2. Model Architecture

- [x] 2.1 Create validation/model.py with DiT block definition
- [x] 2.2 Implement patch embedding layer (16×H×W → 16×64×64)
- [x] 2.3 Implement positional embedding (RoPE or learned 2D)
- [x] 2.4 Implement adaLN-Zero modulation (DINOv3 → scale/shift)
- [x] 2.5 Implement cross-attention layer (T5 as K/V)
- [x] 2.6 Implement self-attention layer with 6 heads
- [x] 2.7 Implement feedforward MLP (384 → 1536 → 384)
- [x] 2.8 Stack 12 DiT blocks with residual connections
- [x] 2.9 Implement final linear projection (384 → 16 channels)
- [x] 2.10 Implement zero-initialization for adaLN gates
- [x] 2.11 Add independent CFG dropout masks for DINOv3 and T5

## 3. Dataloader

- [x] 3.1 Create validation/data.py
- [x] 3.2 Implement WebDataset loading from data/shards/validation/
- [x] 3.3 Implement .npy loading for DINOv3, VAE, T5 hidden, T5 mask
- [x] 3.4 Implement bucket grouping (load all buckets, batch within bucket)
- [x] 3.5 Implement 512×512 resize (VAE latents → 64×64)
- [x] 3.6 Implement horizontal flip augmentation
- [x] 3.7 Implement caption left/right swap regex (\\bleft\\b ↔ \\bright\\b)
- [x] 3.8 Implement numpy → torch tensor conversion
- [x] 3.9 Implement infinite iterator with shuffling
- [x] 3.10 Add batch collation with padding for variable sequence lengths

## 4. Training Loop

- [x] 4.1 Create validation/train.py
- [x] 4.2 Implement flow matching loss calculation (MSE on velocity prediction)
- [x] 4.3 Implement logit-normal timestep sampling (mean=0, std=1)
- [x] 4.4 Implement forward diffusion (add noise: z_t = (1-t) × z_0 + t × ε)
- [x] 4.5 Implement velocity target calculation (v = ε - z_0)
- [x] 4.6 Implement independent CFG dropout logic (10% both, 10% text, 10% DINO, 70% full)
- [x] 4.7 Initialize AdamW optimizer (β1=0.9, β2=0.95, wd=0.03, lr=3e-4)
- [x] 4.8 Implement linear warmup schedule (0 → 3e-4 over 5k steps)
- [x] 4.9 Implement cosine decay schedule (3e-4 → 1e-6 after warmup)
- [x] 4.10 Implement gradient clipping (clip_norm=1.0)
- [x] 4.11 Implement EMA model (decay warmup from 0 → 0.9999 over 5k steps)
- [x] 4.12 Implement checkpoint saving (every 1000 steps)
- [x] 4.13 Implement loss logging to console and file
- [x] 4.14 Add gradient norm logging for stability monitoring

## 5. Sampling Infrastructure

- [x] 5.1 Create validation/sample.py
- [x] 5.2 Implement Euler solver (50 uniform timesteps from t=1 → t=0)
- [x] 5.3 Implement dual CFG guidance (text_scale=3.0, dino_scale=2.0)
- [x] 5.4 Implement three forward passes per step (uncond, text-only, dino-only)
- [x] 5.5 Implement CFG combination formula (v = v_uncond + text_scale × Δv_text + dino_scale × Δv_dino)
- [x] 5.6 Implement VAE decoder loading (Flux VAE)
- [x] 5.7 Implement latent → image decoding (16×64×64 → 512×512 RGB)
- [x] 5.8 Implement image saving to PNG with descriptive filenames

## 6. Validation Tests

- [x] 6.1 Create validation/validate.py
- [x] 6.2 Implement reconstruction test (generate from original caption + DINO)
- [x] 6.3 Implement LPIPS metric calculation for reconstruction
- [x] 6.4 Implement DINO swap test (5 fixed pairs, swap DINO embeddings)
- [x] 6.5 Implement text manipulation test (5 fixed samples with caption mods)
- [x] 6.6 Create fixed test sample lists (reconstruction: all 100, DINO swap: 5 pairs, text manip: 5 samples)
- [x] 6.7 Implement validation directory structure (validation/step{N}/{test_type}/)
- [x] 6.8 Implement results.json logging (LPIPS scores, sample metadata)
- [x] 6.9 Add validation trigger every 1000 training steps
- [x] 6.10 Add EMA model loading for validation (use smooth weights, not training weights)

## 7. Caption Modifications

- [x] 7.1 Create text manipulation test cases (left → right, up → down)
- [x] 7.2 Create attribute swap test cases (red dress → blue dress)
- [x] 7.3 Create pose change test cases (sitting → standing)
- [x] 7.4 Verify test cases exist in 100-sample validation set
- [x] 7.5 Document expected visual outcomes for manual inspection

## 8. Configuration

- [x] 8.1 Create validation/config.py with hyperparameters
- [x] 8.2 Add model config (12L, 384H, 6A, patch_size=2)
- [x] 8.3 Add training config (batch_size=8, steps=5000, warmup=5000)
- [x] 8.4 Add optimizer config (AdamW parameters)
- [x] 8.5 Add sampling config (CFG scales, num_steps=50)
- [x] 8.6 Add paths (data_dir, output_dir, checkpoint_dir)
- [x] 8.7 Add validation config (frequency=1000, test sample indices)

## 9. Entry Point

- [x] 9.1 Create validation/run_validation.py (main entry point)
- [x] 9.2 Add argument parser for config overrides
- [x] 9.3 Add CUDA device selection and memory logging
- [x] 9.4 Add training loop orchestration
- [x] 9.5 Add validation orchestration
- [x] 9.6 Add graceful interrupt handling (save checkpoint on Ctrl+C)
- [x] 9.7 Add resume from checkpoint option

## 10. Documentation

- [x] 10.1 Create validation/README.md with usage instructions
- [x] 10.2 Document hardware requirements (24GB GPU, 32GB RAM)
- [x] 10.3 Document expected runtime (2-4 hours on 4090)
- [x] 10.4 Document validation metrics and success criteria
- [x] 10.5 Document directory structure and output locations
- [x] 10.6 Add example command line invocations
- [x] 10.7 Document how to interpret results.json

## 11. Testing

- [x] 11.1 Run smoke test (10 steps, verify no crashes)
- [x] 11.2 Verify dataloader produces correct batch shapes
- [x] 11.3 Verify model forward pass produces correct output shape
- [x] 11.4 Verify loss calculation is finite and non-negative
- [x] 11.5 Verify EMA update runs without errors
- [x] 11.6 Verify checkpoint save/load roundtrip
- [x] 11.7 Verify sampling produces valid images
- [x] 11.8 Run full 5k step training and validate metrics

## 12. Final Validation

- [x] 12.1 Run complete training (5k steps)
- [x] 12.2 Verify loss curve decreases monotonically
- [x] 12.3 Verify gradient norm stays in healthy range (0.1 - 10.0)
- [x] 12.4 Verify reconstruction LPIPS < 0.2 at step 5000
- [x] 12.5 Visual inspection: DINO swap shows composition transfer
- [x] 12.6 Visual inspection: Text manipulation shows spatial control
- [x] 12.7 Document results in validation/RESULTS.md
- [x] 12.8 Make go/no-go decision for Part C (full-scale training)
