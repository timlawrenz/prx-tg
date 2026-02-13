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

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAAA00lEQVR4XmNgGAV0A6Kiojzy8vKzgfgDEL8E4ihkeSB/saKiojyIzSwnJ7cTKPAfiM8C8UcQGyjmApKUkZFRBfLXgnUBBeMUFBRWAgWlQXwtLS02oFgHEK+AyvcD5R1gVrSCnAHmQAFUwy2gIg6g/F5kOawAqGgJEJcANaWiy2EAoMJaIH4ANF0AXQ4DAE3MA5mOLo4VQJ3ggy6OFQCt3wTUoIgujgFUVFTYgQpfAZnM6HIYAGi9DVDxYXRxrADkOaAzGtHFsQKgqT3AGFVBFx8kAADPYilkO1+X3gAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAYCAYAAADOMhxqAAABNUlEQVR4XtWQvUoDQRSFJ7FQUllZLOz/oi+QJlgpYmGrkEpCRLG0EiHYhXQGwc5KfALFH1DEwkYQGwufIIWYLmBpod8Ma7h7YbHOgcPM/e65OztjzMSoGoZhD9/iZ3zieV5Nh8YKguCY0LKoL6mPZGasKIraNLclo+4y9CSZU5Zl0zTu2VYkZ+ARfiaZE19v4aZkBBcZ+GRNJHcifI5n88uO8Dv+8H2/rrNO9nL5ukKwg7fwm72XzpokSeZp9jVneM8Oaf73OuuaM7AD/2ZbLTQYOE3TdK4AjXuhPh5qbgeuNEMVwq+ccleg+cuMjDqW4Ab8Bzckt8eu0XxhPczRFPUm9Rcf2y2ErWj04jheIHTA/oF1wHrN+6/qrBPNG81KxZEzDFxoXirCS+Lf/xfh/dJ/nQz9AqpbQiH2Yqf/AAAAAElFTkSuQmCC>