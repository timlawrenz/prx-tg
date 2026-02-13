# The Plan

Here is the technical roadmap split into Data Pipeline, Validation, and Production Training.

## Part A: Completing the Data Pipeline (The "Strix Halo" Phase)

**Goal:** Convert raw images and JSONs into a high-performance, streamable format so the 4090 never waits for I/O.

### 1. Aspect Ratio Bucketing (The First Sort)

Since you are not cropping, you must mathematically group your images.

- **Analysis:** Scan all 100k images. Calculate their aspect ratios (W/H).
- **Clustering:** Define a set of target buckets (e.g., 1024x1024, 832x1216, 1216x832). Map every image to the closest bucket.
- **Resizing:** Resize images to their bucket resolution. Do not crop. If the aspect ratio doesn't match perfectly, you have two choices: pad with noise (easiest) or slight crop (risky for framing). Given your "high quality" requirement, smart resizing to the nearest 64-pixel modulus is preferred.

### 2. Latent Encoding (Flux VAE)

- **Action:** Run all 100k images through the Flux VAE Encoder.
- **Normalization (Vital):** VAE latents are not naturally normally distributed. You must calculate the mean and standard deviation of your specific dataset's latents.
  - **Why?** If you don't, your model will spend the first 5,000 steps just learning to shift the pixel values, rather than learning structure.
- **Output:** Save the latents as float16 or bfloat16.

### 3. Text & Vision Encoding

- **Text:** Run captions through T5-Large. Save the `last_hidden_state` (the sequence, not just the pooled vector) and the `attention_mask`. You need the sequence for spatial control.
- **Vision:** Ensure your DINOv3 embeddings are normalized (L2 norm) if the model expects unit vectors, though usually, raw features are fine if the projection layer handles it.

### 4. Sharding (WebDataset/Tar)

- **The Problem:** Reading 100k individual .npy files kills training speed due to inode lookups.
- **The Solution:** Pack your data into Shards (tar files).
  - Each shard should contain ~1,000 samples (Latent + Text Emb + DINO Emb).
  - This allows the data loader to read one large file and stream data to the GPU.

## Part B: Minimal Validation (The "Sanity Check")

**Goal:** Prove the architecture works before burning electricity on the full run. Use the 4090.

### 1. The "Overfit" Dataset

- **Selection:** Select exactly 100 images from your dataset. Choose diverse poses but consistent lighting.
- **Resolution:** Force resize these to 512x512 (Latent 64×64). This makes iteration instant.

### 2. The "Toy" Model

**Architecture:** Initialize a "Nano" version of your DiT.

- **Layers:** 12
- **Hidden Size:** 384
- **Heads:** 6
- **Conditioning:** Hook up the DINOv3 projection and T5 Cross-Attention.

### 3. The Validation Objective

Train for ~5,000 steps.

- **Success Metric 1 (Reconstruction):** Can the model memorize these 100 images? If you prompt for "Image #42" (using its specific caption), does it reproduce it perfectly?
- **Success Metric 2 (DINO Injection):** If you swap the DINO embedding of Image A with Image B, does the generation take on the composition of A but the lighting/vibe of B?
- **Success Metric 3 (Text Control):** If you change the prompt from "looking left" to "looking right," does the blob move?

**If this fails, do not proceed to Part C.** Debug your layers, normalization, or loss function.

## Part C: Full Scale Training (The Production Run)

**Goal:** Train the 400M+ parameter model on the full 100k dataset.

### 1. The "Real" Model Architecture

- **Size:** ~400M - 600M parameters.
- **Patch Size:** Use Patch Size 2.
  - **Why?** Flux VAE is powerful. If you use Patch Size 4 (to save compute), you throw away the high-frequency detail the VAE preserved. Patch Size 2 is heavy but necessary for hair/skin texture.

### 2. Training Strategy: Rectified Flow

- **Loss:** Flow Matching Loss (predicting the velocity vector v).
- **Timestep Sampling:** Logit-Normal sampling (focuses training on the "middle" of the noise process where structure is formed).
- **Precision:** bfloat16 (Brain Floating Point).
  - **Strict Rule:** Never use float16 for Flow Matching. The vector fields require the dynamic range of bfloat16 to avoid exploding gradients.

### 3. The "EMAgic" (Exponential Moving Average)

**Setup:** Maintain two copies of the model weights.

- **Model A (Train):** Updates every step. Weights will look noisy.
- **Model B (EMA):** Updates as: `W_ema = 0.9999 × W_ema + 0.0001 × W_train`

**Usage:** Only save and visualize Model B. Training from scratch produces high-variance weights; EMA smooths this out into a coherent model.

### 4. Batch Size & Gradient Accumulation

- **Hardware Reality:** On a 24GB 4090 with a 400M model and 16-channel latents, your native batch size will likely be small (e.g., 8 or 16).
- **Target:** You need an effective batch size of roughly 256 to stabilize the gradient for a "from scratch" run.
- **Math:** Accumulate gradients for 256/16=16 steps before running `optimizer.step()`.

### 5. Checkpointing Strategy

- Save every 5,000 steps.
- **Monitor:** Watch the Gradient Norm.
  - **If it spikes and stays high:** Your learning rate is too high.
  - **If it approaches zero:** Your model has collapsed (likely generating a mean-gray image).
  - **Healthy training** shows a "hairy" gradient norm graph that slowly trends downward.

