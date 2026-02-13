# **The Plan**

Here is the technical roadmap split into Data Pipeline, Validation, Production Training, and Technical Specifications.

## **Part A: Completing the Data Pipeline (The "Strix Halo" Phase)**

**Goal:** Convert raw images and JSONs into a high-performance, streamable format so the 4090 never waits for I/O.

### **1\. Aspect Ratio Bucketing (The First Sort)**

Since you are not cropping, you must mathematically group your images.

* **Analysis:** Scan all 100k images. Calculate their aspect ratios (W/H).  
* **Clustering:** Define a set of target buckets (e.g., 1024x1024, 832x1216, 1216x832). Map every image to the closest bucket.  
* **Resizing:** Resize images to their bucket resolution. Do not crop. If the aspect ratio doesn't match perfectly, you have two choices: pad with noise (easiest) or slight crop (risky for framing). Given your "high quality" requirement, smart resizing to the nearest 64-pixel modulus is preferred.

### **2\. Latent Encoding (Flux VAE)**

* **Action:** Run all 100k images through the Flux VAE Encoder.  
* **Normalization (Vital):** VAE latents are not naturally normally distributed. You must calculate the mean and standard deviation of your specific dataset's latents.  
  * **Why?** If you don't, your model will spend the first 5,000 steps just learning to shift the pixel values, rather than learning structure.  
* **Output:** Save the latents as float16 or bfloat16.

### **3\. Text & Vision Encoding**

* **Text:** Run captions through T5-Large. Save the last\_hidden\_state (the sequence, not just the pooled vector) and the attention\_mask. You need the sequence for spatial control.  
* **Vision:** Ensure your DINOv3 embeddings are normalized (L2 norm) if the model expects unit vectors, though usually, raw features are fine if the projection layer handles it.

### **4\. Sharding (WebDataset/Tar)**

* **The Problem:** Reading 100k individual .npy files kills training speed due to inode lookups.  
* **The Solution:** Pack your data into Shards (tar files).  
  * Each shard should contain \~1,000 samples (Latent \+ Text Emb \+ DINO Emb).  
  * This allows the data loader to read one large file and stream data to the GPU.

## **Part B: Minimal Validation (The "Sanity Check")**

**Goal:** Prove the architecture works before burning electricity on the full run. Use the 4090\.

### **1\. The "Overfit" Dataset**

* **Selection:** Select exactly 100 images from your dataset. Choose diverse poses but consistent lighting.  
* **Resolution:** Force resize these to 512x512 (Latent 64×64). This makes iteration instant.

### **2\. The "Toy" Model**

**Architecture:** Initialize a "Nano" version of your DiT.

* **Layers:** 12  
* **Hidden Size:** 384  
* **Heads:** 6  
* **Conditioning:** Hook up the DINOv3 projection and T5 Cross-Attention.

### **3\. The Validation Objective**

Train for \~5,000 steps.

* **Success Metric 1 (Reconstruction):** Can the model memorize these 100 images? If you prompt for "Image \#42" (using its specific caption), does it reproduce it perfectly?  
* **Success Metric 2 (DINO Injection):** If you swap the DINO embedding of Image A with Image B, does the generation take on the composition of A but the lighting/vibe of B?  
* **Success Metric 3 (Text Control):** If you change the prompt from "looking left" to "looking right," does the blob move?

**If this fails, do not proceed to Part C.** Debug your layers, normalization, or loss function.

## **Part C: Full Scale Training (The Production Run)**

**Goal:** Train the 400M+ parameter model on the full 100k dataset.

### **1\. The "Real" Model Architecture**

* **Size:** \~400M \- 600M parameters.  
* **Patch Size:** Use Patch Size 2\.  
  * **Why?** Flux VAE is powerful. If you use Patch Size 4 (to save compute), you throw away the high-frequency detail the VAE preserved. Patch Size 2 is heavy but necessary for hair/skin texture.

### **2\. Training Strategy: Rectified Flow**

* **Loss:** Flow Matching Loss (predicting the velocity vector v).  
* **Timestep Sampling:** Logit-Normal sampling (focuses training on the "middle" of the noise process where structure is formed).  
* **Precision:** bfloat16 (Brain Floating Point).  
  * **Strict Rule:** Never use float16 for Flow Matching. The vector fields require the dynamic range of bfloat16 to avoid exploding gradients.

### **3\. The "EMAgic" (Exponential Moving Average)**

**Setup:** Maintain two copies of the model weights.

* **Model A (Train):** Updates every step. Weights will look noisy.  
* **Model B (EMA):** Updates as: W\_ema \= 0.9999 × W\_ema \+ 0.0001 × W\_train

**Usage:** Only save and visualize Model B. Training from scratch produces high-variance weights; EMA smooths this out into a coherent model.

### **4\. Batch Size & Gradient Accumulation**

* **Hardware Reality:** On a 24GB 4090 with a 400M model and 16-channel latents, your native batch size will likely be small (e.g., 8 or 16).  
* **Target:** You need an effective batch size of roughly 256 to stabilize the gradient for a "from scratch" run.  
* **Math:** Accumulate gradients for 256/16=16 steps before running optimizer.step().

### **5\. Checkpointing Strategy**

* Save every 5,000 steps.  
* **Monitor:** Watch the Gradient Norm.  
  * **If it spikes and stays high:** Your learning rate is too high.  
  * **If it approaches zero:** Your model has collapsed (likely generating a mean-gray image).  
  * **Healthy training** shows a "hairy" gradient norm graph that slowly trends downward.

## **Part D: Technical Specifications & Hyperparameters**

**Goal:** Ensure mathematical stability and conditioning coherence.

### **1\. Optimization Hyperparameters**

* **Optimizer:** AdamW  
  * **Betas:** (0.9, 0.95) (Standard 0.999 is often too slow to adapt for generative models from scratch).  
  * **Weight Decay:** 0.03 (Standard regularization).  
  * **Epsilon:** 1e-8.  
* **Learning Rate Schedule:**  
  * **Peak LR:** 3e-4 (Higher than fine-tuning 1e-5, because we are training from scratch).  
  * **Warmup:** Linear warmup for **5,000 steps**. (Crucial to stabilize the adaLN layers early on).  
  * **Decay:** Cosine decay down to 1e-6.  
* **Gradient Clipping:** Clip norm at 1.0 or 2.0 to prevent explosions during the initial phase.

### **2\. Noise & Sampling**

* **Formulation:** Rectified Flow (Velocity prediction).  
* **Timestep Sampling:** **Logit-Normal Sampling** (m=0.0, s=1.0).  
  * *Why:* Uniform sampling \[0, 1\] over-samples easy noise levels. Logit-Normal concentrates training density in the "middle" of the process where the image structure is actually formed.

### **3\. Smart Data Augmentation**

Since we have only 100k images, we must augment to prevent overfitting.

* **Strategy:** Horizontal Flip (p=0.5).  
* **The "Smart" Part:** We must keep spatial captions correct.  
  * **Code Logic:** If flip \== True:  
    1. Flip image tensor.  
    2. Run Regex on caption: Swap \\bleft\\b ↔ \\bright\\b.  
  * *Note:* Do not use random cropping (conflicts with bucketing) or color jitter (destroys the lighting/quality you are trying to learn).

### **4\. Dual Conditioning Architecture (How to Blend Signals)**

To ensure the model listens to both DINOv3 (Vibe/Global) and Text (Spatial/Details), we use specific injection points and dropout strategies.

**A. The Architecture (Injection Points)**

The signals should enter via orthogonal mechanisms so they don't compete:

1. **DINOv3 (Global Context) → adaLN-Zero**  
   * The DINO vector is projected to regress the scale (![][image1]) and shift (![][image2]) parameters of the LayerNorm.  
   * **Effect:** This modulates the *amplitude* of the features globally. It tells the block: "Everything you process in this layer should be dark and moody."  
2. **Text Embeddings (Spatial Context) → Cross-Attention**  
   * The Text tokens act as Keys/Values (K, V) in the attention blocks.  
   * **Effect:** This routes information spatially. It tells the block: "Move the attention to the top-left corner for the 'raised hand'."

**B. The Guarantee (Conditioning Dropout)**

To force the model to learn both mechanisms effectively, we use **Independent Classifier-Free Guidance (CFG) Dropout**:

During training, for each batch:

* **10%** of the time: Drop **BOTH** DINO and Text (Learn unconditional distribution).  
* **10%** of the time: Drop **TEXT ONLY** (Force model to rely purely on DINO for structure).  
* **10%** of the time: Drop **DINO ONLY** (Force model to infer style purely from Text).  
* **70%** of the time: Keep **BOTH**.

This ensures the model cannot ignore one signal in favor of the other. It *must* learn to generate valid images using either signal alone, which makes the combination robust.

### **5\. Inference & Guidance Strategy**

Because we used **Independent Conditioning Dropout** during training, we can decouple the guidance scales at inference time to fine-tune the results.

**A. The Formula (Dual CFG)**

We calculate the final velocity field (![][image3]) by combining the unconditional prediction with the separate conditional directions:

![][image4]**B. Recommended Scales**

* **Text Scale (![][image5]):** **2.0 \- 4.5**  
  * *Purpose:* Controls spatial adherence (pose, hands, gaze).  
  * *Note:* Flow Matching models typically use lower scales than Diffusion models. Going above 5.0 often "burns" the image.  
* **DINO Scale (![][image6]):** **1.5 \- 3.0**  
  * *Purpose:* Controls identity and lighting match.  
  * *Note:* DINO embeddings are very strong. A scale of 1.0 (no guidance boost) or 1.5 is often enough. High values here might rigidify the image into the exact pose of the reference, ignoring the text.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAAA00lEQVR4XmNgGAV0A6Kiojzy8vKzgfgDEL8E4ihkeSB/saKiojyIzSwnJ7cTKPAfiM8C8UcQGyjmApKUkZFRBfLXgnUBBeMUFBRWAgWlQXwtLS02oFgHEK+AyvcD5R1gVrSCnAHmQAFUwy2gIg6g/F5kOawAqGgJEJcANaWiy2EAoMJaIH4ANF0AXQ4DAE3MA5mOLo4VQJ3ggy6OFQCt3wTUoIgujgFUVFTYgQpfAZnM6HIYAGi9DVDxYXRxrADkOaAzGtHFsQKgqT3AGFVBFx8kAADPYilkO1+X3gAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAYCAYAAADOMhxqAAABNUlEQVR4XtWQvUoDQRSFJ7FQUllZLOz/oi+QJlgpYmGrkEpCRLG0EiHYhXQGwc5KfALFH1DEwkYQGwufIIWYLmBpod8Ma7h7YbHOgcPM/e65OztjzMSoGoZhD9/iZ3zieV5Nh8YKguCY0LKoL6mPZGasKIraNLclo+4y9CSZU5Zl0zTu2VYkZ+ARfiaZE19v4aZkBBcZ+GRNJHcifI5n88uO8Dv+8H2/rrNO9nL5ukKwg7fwm72XzpokSeZp9jVneM8Oaf73OuuaM7AD/2ZbLTQYOE3TdK4AjXuhPh5qbgeuNEMVwq+ccleg+cuMjDqW4Ab8Bzckt8eu0XxhPczRFPUm9Rcf2y2ErWj04jheIHTA/oF1wHrN+6/qrBPNG81KxZEzDFxoXirCS+Lf/xfh/dJ/nQz9AqpbQiH2Yqf/AAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACUAAAAYCAYAAAB9ejRwAAACQUlEQVR4Xu2UT0hUURTGh9JSEjL0MTp/3puZJpTnJpqknZHkthatbKHSSlcTLZQCEUUUjcSFFYQaFYkEgVBG4CKjhVY7XRTMuhiYdQRB6O+8uc9uFxejjLjwffBxz/nOn3vnzL0vFAoQIECAAPtDIpG45DjOc7gej8dbRYtEInH8nG3bWTP/wBGNRmNsvhIOh09xuC/YL0RPp9MW9k+4apQcPNj0HtO4yoTOYm/BO36MQ/bhP9Hzyw32XpJ9Y7FY1IzJ4UbNIIfqhr16XrmRTCab2PebqXsgsAk/GNo7+Xt1rdyQfwbOmnpI7pNMiVEO+Zr8Avynyq3AHofv1V86ApfVfbyB9gwOSz7+G6lVPdrw5+AksTG/N7mdaG/hA/TPrLf82H8gkIcPxSbxDIWvZFV+h3qh6/Cmyh+E99HvigY3sWtZf5PfwnoFLmcymcpUKnUa+6XUcXevYX/FrFD6FnXNOwfRQeA6CQXZmKaP8c/7MRpFLMuqIfaLV3lSNPkBcFZ9OmbkkP+6efFV+iyyjrAOyEMSPVF84d5k5POD/UOv2xNodpkGK76PvYbWo2Ib+O07ycW4NzFdc133BPpfHtM58Yln8eepb9DzSgYN+mnwSNku9ifM46yN8A8bVev5aN+ZxEWx1TdvEvMYa4FDVAnp81Gm5qhrs2dQ+FomBQdpNs0lr1N6u+i75F+AC3AKznF/bNGp7eJAE2gzrLed4qT6zPqSQHGe+1Nv6ocG9YHLmfqhQV6eekV5xtxpxo8MtgGLq4+jq26NnwAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,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>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAYCAYAAACfpi8JAAABwUlEQVR4Xu2UzStEYRTGGZQIGxqmuffOjJtRdsbSdxYUpSwlf4OFlKJ8lsHkDzBLYqGUZjUbNZSVhRL/gGykNElW/E7zXt25zXfNRvepp/ec5z3zvOfe886tqXHhwoWL/whN0wYMw0jAJEzpuj7jrKk6AoFAL4e/mqbZITnxKnx01lUdHLoMP2moU3LWJTjrqBkPhUI9dq0SFPRhDBMU/Cg+0cQecq29Bu2B8Q3ZtUpQ1IcGpuA2vFUNzVt7MjLydCQSabD/plwU9OFtnEmXdo3id/QFFe/IPusLTMp9Ej0YDA6Tx2GU2l3RZLTkp/BYNBkx8QXsyufzB8Q3GLNydcAN3bdampiirVk58RhMyJMx7zbiE1V3FA6HW3w+n4b27ff7TdY0nkYunyywOS2m8Aruwy0xt9eg3fEEo7b8Wt4k6ybrCjPvFl0aUPtzMGnVW3D6lAWv19uMwQcGjRzUjlRH/kUDfc5aC+wfWk8u45I1l0/Wj4qBAwcxSKl4naWe/Fk+gKKpCxgl9Kh/hHwY74knlbaRzydzQomg+yZ5zRguwhHRyPuNzKWMwTij1JE9cpC6Bwesl9Sfq72cPi5clIpfE+h9+oLJUSwAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACQAAAAYCAYAAACSuF9OAAACLUlEQVR4Xu2UTUiUURSGZyY3QaX9jMj8fvMDo7gKLZDAhbgRjZKESKtFFG5alGCEixCqhYlC4EIRjFDURQtDoqSFkLVxpxDhRjdBRJK0iJb5vMy5NHyQDCKuvhde7vl57znn3u/OhEIBAgQIEOD/SCaTjel0+g18D1dSqdRFv+bQ4HleLUN8y+fzUfnYA/CzX3dooHk//M1gNfJZ78PLfp2gONpNbnBcfqFQOI6/kUgk4n7tvkHxVor+NX6h6RDhsF/ngGaaPdfNjWDfCe2h3xdo0gafwE82WI9fYwiT22Zoz584EHC6eWqvl8Zo+JP4DefH4/HTuhU4BifhlumuwCnYLz+TyRSwZyx2lxoPWJf5wZwrqV0HX8ER8gusZ13OCbbhqPMp2oz/kQd+wkIRBl5lc5fpR+E0b+Yo6219Lg2gHLqHNI/h/6HOeYs915uUzcES5L6yp95qdcJF61MEyY508ee+CIfh42w2W+nyFLtEbDNkb0SnItarx8xahb+kwsrlcrlq078t2f9BMbOHyM26nA205vyyoAEp9NLcCP4OQ2SJ1dhtfNdtahjTj8AB2bFY7Az2r2g0eow9JzUc/q2S2s+kd35ZoMg9Nj2Vbbe5xdqgwvoUXvGTXCV2wfSr5JpMf1OHwW+BbfgvWLuV07AaUDf9r1sZ4K2cosiYXXcfnMOegFXwGk1esw6a/Aj5H6wVcrDb4XKq+J8V5kZz2O/gI316cnWuT4AAB4FdnK2L6HG7JgsAAAAASUVORK5CYII=>