# Tangential Research

External papers, articles, and releases that are not directly about prx-tg but contain ideas worth recording for potential future application. Each entry notes the source, date reviewed, what was learned, and its relevance to prx-tg.

---

## 2026-08-02 — AMD Instella-MoE-16B-A3B

**Source:** https://www.marktechpost.com/2026/08/01/amd-instella-moe-16b-a3b-fully-open-mixture-of-experts-llm/
**Also:** [ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/instella-moe/README.html) · [GitHub](https://github.com/AMD-AGI/Instella-MoE) · [Hugging Face collection](https://huggingface.co/collections/amd/instella-moe)
**Domain:** LLM (decoder-only MoE), trained on AMD Instinct MI300X/MI325X

### Summary

16B total / 2.8B active MoE LLM, fully open (ResearchRAIL weights, MIT training code). 27 layers, hidden 2048, 2 shared + 6-of-64 routed experts. Uses Gated Multi-head Latent Attention (Gated MLA) and FarSkip-Collective (overlapping expert-parallel comm with compute) — 12.7% pretraining speedup, 39.2% lower TTFT. Trained on 7.1T tokens. Post-training: SFT → DPO → RLVR + multi-teacher on-policy distillation. Base averages 76.7 on benchmarks, leading fully open peers.

### Relevance to prx-tg

prx-tg is a 238.8M dense pixel-space DiT — fundamentally different from a 16B MoE LLM. Most of this work (MoE routing, FarSkip-Collective, router bias, load-balancing loss, DPO/RLVR, YaRN) is not applicable. A few ideas are transferable:

#### 1. Gated cross-attention outputs — **most actionable**

Gated MLA adds a lightweight learned output gate to attention: a linear projection derives an input-conditioned gate, applied multiplicatively before the output projection.

**prx-tg application:** The StratumAdapter uses 7 conditioning streams (T5 text, DINOv3 CLS/patches, DWPose 133kp) via cross-attention. A learned per-token gate on cross-attention outputs would let the model dynamically down-weight uninformative conditioning streams per spatial location — e.g., suppress pose keypoints in background regions, or suppress DINOv3 patches when text is the primary driver.

- **Effort:** Low — one linear projection per cross-attention stream, applied multiplicatively before output projection.
- **Isolation:** Fits the one-variable-per-arm rule (Arm candidate: `gated-xattn`).
- **Risk:** Minimal — the gate can initialize to identity (gate=1), so it degrades gracefully to current behavior.
- **When to consider:** After stratum2 integration is stable and pose2 arm is baselined. Not before.

#### 2. Weight averaging (model soup) across data variants — **free to try later**

AMD averages checkpoints from three mid-training data mixture variants instead of picking one.

**prx-tg application:** If multiple config-variant arms are trained (e.g., different stratum2 artifact subsets, different LR schedules), averaging their EMA checkpoints could yield a more robust final model at zero training cost. This is a post-hoc technique — only relevant once multiple comparable arms exist.

#### 3. Multi-Token Prediction (MTP) as auxiliary objective — **speculative**

They use MTP during pretraining and mid-training.

**prx-tg application:** The diffusion analogue would be multi-scale or multi-noise-level prediction — predicting denoised targets at multiple noise levels or spatial resolutions simultaneously. Conceptually adjacent to REPA alignment (auxiliary representation matching). Not something to adopt now, but if sample efficiency stalls, it's a research direction worth investigating. Would require a new arm and careful isolation.

#### 4. ROCm production viability — **confirms existing capability**

They trained end-to-end on Instinct MI300X/MI325X with ROCm, Primus, and Miles; served with SGLang.

**prx-tg application:** Confirms the ROCm stack is production-viable for large models. The Strix Halo 128GB (MI300-class APU) is a real inference/eval path for prx-tg, not just a fallback. Already known and used for Sapiens2 inference — this validates the approach at scale.

### Not relevant to prx-tg

- **MoE routing / FarSkip-Collective** — distributed MoE-specific, no path into a dense single-GPU DiT.
- **Router bias / load-balancing loss** — MoE-specific.
- **DPO / RLVR / distillation** — LLM post-training toolchain; diffusion alignment is a different space.
- **YaRN / long-context extension** — text-sequence-specific.
- **7.1T token data mixture** — text corpus; no overlap with image training data.

---

## 2026-08-02 — MiniMax H3 (Omni-modal Video Generation)

**Source:** https://www.minimax.io/blog/minimax-h3
**Domain:** Omni-modal video generation (text/image/video/audio → 2K video with native stereo audio, 15s). Weights planned to open.

### Summary

General-purpose omni-modal generation model. Four core technical pillars:

1. **Contextual Omni Representation** — heavy captioning investment: ~100K tokens of multimodal VLM inference per source sample, distilled to ~4K-token captions that describe *relationships between conditioning elements and target* (not just the target). Language is treated as the generalizable bridge that unifies tasks.
2. **H3-VAE overhaul** — complete tokenizer rewrite; 4× gain in effective sequence length; key enabler of native 2K output.
3. **H3-Omni Transformer** — deliberately dropped the architecturally-fancier Hailuo-02 design because it added complexity without serving task generalization. Separates understanding (encoding) vs. generation (denoising) workloads during training, tuning hardware utilization per stage → +30% training throughput.
4. **In-Context Regeneration** — no dedicated super-resolution module; the base model regenerates its own low-resolution output at high res, re-reading the original multimodal context. Recovers details (small text, fine structure) that conventional SR "guesses" at.

Design philosophy: "architecture should serve the task" — task generalization over architectural tricks. Fuse diverse data/tasks as early as possible; the mixing ratio is the key hyperparameter.

### Relevance to prx-tg

prx-tg is a single-task T2I pixel-space DiT — the omni-modal, multi-shot, and audio aspects are out of scope. But several ideas transfer:

#### 1. In-context regeneration instead of super-resolution — **most actionable**

H3 replaces the conventional dedicated SR module with the base model itself: generate at low res, then condition on the low-res output **plus the original conditioning** and regenerate at high res. The regeneration pass re-attends to the original context, which is where fine detail is anchored.

**prx-tg application:** prx-tg trains pixel-space at 1024×1024 — the conventional wisdom is that pixel-space can't scale to high res. H3's approach maps directly:
- Train at 512 → regenerate at 1024 with the 512 output as an additional conditioning stream alongside T5/DINOv3/pose.
- The regeneration pass re-attends to pose/DINOv3 conditioning — the same argument H3 makes about recovering small text applies to fine facial detail (eyes, fingertips, hair strands).
- Fits the existing StratumAdapter cross-attention design: low-res image tokens become one more context stream with CFG dropout.

- **Effort:** Moderate — new conditioning stream, two-stage sampling procedure, config flag for low/high res.
- **Isolation:** Clean single-variable arm against the current 1024-direct baseline (`incontext-regen` candidate).
- **Risk:** Doubles inference cost; training pipeline must handle paired low/high-res targets.
- **When to consider:** After the stratum2 pose2 arm is baselined. This is a candidate for the next arm after that, not before.

#### 2. Captioning philosophy — **validates and sharpens the stratum2 plan**

H3's single largest engineering investment was caption quality — and specifically *relational* captions: describing how context elements relate to the target, not just the target itself. Their pipeline spends ~100K tokens of VLM inference per sample to produce ~4K captions.

**prx-tg application:** This is direct evidence for the stratum2 integration plan (pose2/seg2/pointmap/normal2/matting). But H3's lesson goes further: the **text caption should verbalize the same spatial/structural facts** that the structured conditioning encodes — e.g., not "a woman posing" but "left hand raised toward camera, foreshortened, partially occluding face." When the FFHQ stratum2 captioning pass is designed, adding a VLM step that derives spatial facts from pose2/seg2 and writes them into the text stream would align text and structured conditioning — H3's argument is that this agreement is what drives instruction-following.

#### 3. "Architecture should serve the task" — **discipline reminder, not a new technique**

MiniMax explicitly dropped the Hailuo-02 architecture despite its prior advantages because it complicated task generalization. They chose the simpler design.

**prx-tg application:** prx-tg stacks several architectural tricks (TREAD routing, REPA, AsymFlow, Muon). The ablation program already tests them individually — H3's experience is a reminder that "simpler wins" is a legitimate experimental outcome, and arms should be allowed to fail honestly rather than keeping a trick because it's clever. No action; a framing to keep in mind when reading ablation results.

#### 4. Sequence-length efficiency via tokenizer — **partial analog: patch size**

H3-VAE's 4× sequence gain is their 2K enabler. prx-tg is pixel-space (no VAE), but the analogous knob is **patch size** — p=8 vs p=16 at 1024 trades sequence length the same way H3 traded VAE compression ratio.

**prx-tg application:** If throughput or memory becomes the binding constraint, a patch-size ablation is one of the highest-leverage efficiency axes. Not currently planned; flagged as a candidate arm if compute pressure appears.

#### 5. Heterogeneous workload separation (+30% throughput) — **check current pipeline**

H3 separates understanding (encoding context) from generation (denoising) and tunes hardware per stage. prx-tg's analog: T5, DINOv3, and DWPose encoders are frozen — if they run on-the-fly in the training loop, precomputing their outputs offline is free throughput with zero ablation risk.

**prx-tg application:** Verify whether the current training pipeline encodes conditioning on-the-fly or from cache. If on-the-fly, offline precompute is a zero-risk optimization worth doing before the next long run.

### Not relevant to prx-tg

- **Omni-modal (audio, video) generation** — prx-tg is stills only.
- **Native multi-shot modeling** — video-specific.
- **Task unification via language as the bridge** — prx-tg has a single task (T2I stills); the multi-task generalization argument does not transfer.
- **Scale/price claims (2K, per-second pricing)** — no signal for a 238.8M single-GPU model.
- **H3-VAE specifics** — prx-tg is pixel-space by design; the VAE overhaul itself has no direct path in.

---

## 2026-08-06 — Hunyuan3D-Buffalo 1.0 (Unified 3D Multimodal: Gen + Understand + Edit)

**Source:** https://huggingface.co/papers/2608.02711 (arXiv 2608.02711)
**Also:** [Project page](https://tencent-hunyuan.github.io/Hunyuan3D-Buffalo1.0/) · [arXiv HTML](https://arxiv.org/html/2608.02711v1)
**Domain:** Unified 3D generation / understanding / editing (VLM semantic engine + 3D-DiT generator)

### Summary

Tencent unifies 3D understanding, text-to-3D generation, instruction-guided editing, and text-grounded part generation in one architecture: a 3D-aware VLM (Qwen-VL backbone + 133 special tokens incl. quantized box coordinates) acts as the semantic engine; a 3D-DiT (initialized from Hunyuan3D-2.1) is the generator; a lightweight MLP-connector projects VLM hidden states into the DiT via cross-attention.

Data engine at the heart of the scale-up: an 87M-sample corpus — 25M understanding, 50M text-to-3D, 12M editing pairs. Editing pairs are built by Nano3D-v2 (VLM anchor-view selection → 2D edit → learned 3D edit-region localizer → voxel-level FlowEdit with the box exterior frozen → sub-voxel + texture refinement → VLM annotation/filtering). Captioning emits 6 monotonic tiers plus a calibrated geometry-quality score.

Results: 56.6% human preference vs. ~18.4% best baseline (Omni123) on text-to-3D; SOTA on edit benchmarks. Core findings: (1) generating improves editing — editing ability emerges from more text-to-3D pairs with zero extra edit data (3M→15M→50M pairs = 8.4%→28.6%→57.5% preference); (2) understanding improves editing (3D-VLM conditioning beats CLIP conditioning: edit CD 0.0158→0.0091, F1 0.6336→0.6515). Training is staged: VLM instruction-tuned then frozen; text-to-3D pretrain; omni pretrain with text-to-3D:(edit+part) at 1:1 sampling with editing data repeated 4×.

### Relevance to prx-tg

prx-tg is a single-task pixel-space T2I DiT — the 3D representation, editing, and part-generation abilities are out of scope. But several ideas transfer:

#### 1. Source-object conditioning: concat the source into the noisy latent — **most actionable**

For editing/part tasks they concatenate the source 3D representation directly into the noisy latent fed to the DiT's self-attention (on top of VLM cross-attention). The denoiser can *attend to the original geometry* during denoising, so unedited regions are preserved with no mask loss.

**prx-tg application:** The same mechanism applies to conditioning a generation model on a structural/stochastic prior, not just editing — e.g., the low-res draft in a future in-context-regeneration arm, or a reference-image stream. Concretely: instead of only cross-attending a conditioning stream, concatenate its tokens into the latent map input to self-attention, letting the model directly copy structure from it. This is the natural complement to the cross-attention-only StratumAdapter, and it's cheap to test against the current design.

- **Effort:** Low–moderate — a concat along the sequence axis into the DiT input; conditional only for the new stream.
- **Isolation:** Clean single-variable arm vs. the cross-attention-only baseline.
- **Risk:** Changes the latent input shape (sequence length), so it must be conditioned on a real conditioning stream to be testable; initialize so the new path can be gated off.
- **When to consider:** Alongside or right after an in-context-regen arm (H3 entry above) — the two ideas compose (concat the low-res draft into the latent, cross-attend the rest).

#### 2. Cheap data scales the expensive capability — **validates the stratum enrichment bet**

The paper's sharpest empirical result: editing ability emerges from scaling *generation* data, which is far cheaper to construct than edit pairs. They frame it as a deliberate strategy — "to improve editing, maximize text-to-3D."

**prx-tg application:** Direct supporting evidence for the existing plan of complete FFHQ/stratum2 enrichment: unlabeled/understanding-style data is cheap relative to curated paired passes, and the marginal gains from more data (28.6%→57.5% preference from 15M→50M) are real. Worth keeping in mind when deciding between another labeled pass vs. enriching/enlarging the base corpus.

#### 3. Task mixing recipe: 1:1 sampling + 4× repeat of scarce data — **reference if prx-tg ever goes multi-stream**

They balance text-to-3D against (edit+part) at exactly 1:1 within each batch, repeating the scarce editing data 4× to match, and keep 50% generation data mixed into edit/part fine-tuning to preserve quality.

**prx-tg application:** prx-tg is single-task today, so this is not directly usable. But if a future arm benefits from mixing understanding/supervision data into generation training (e.g., segmentation or pose prediction as an auxiliary task), this is a concrete, published ratio+repeat recipe to start from instead of guessing the mix.

#### 4. Frozen semantic engine, train connector + generator — **confirms current design**

The VLM is fully frozen after instruction tuning; all downstream work is the connector + DiT. This validates prx-tg's frozen T5/DINOv3/DWPose encoder setup and the idea of training only the injected-conditioning path.

**prx-tg application:** No action; confirms the frozen-encoder approach is the mainstream recipe for VLM-conditioned DiTs. Also note their 3D-VLM > CLIP conditioning result as one more data point that rich semantic conditioning beats a weak text encoder.

#### 5. Multi-tier monotonic captions + calibrated quality score — **design input for the FFHQ captioning pass**

They caption with the rendered *untextured geometry as source of truth*, emit 6 caption tiers that are monotonic (shorter tiers only drop info, never add), and assign a [0,10] geometry-quality score with calibrated anchors and a directive to use the full range. Filtering keeps only the quality-10 tier split.

**prx-tg application:** Directly useful when the FFHQ stratum2 captioning pass is designed (aligns with the H3 caption lesson above): (a) caption from the actionable ground truth, not the prompt; (b) monotonic tiers give the sampler caption-length variety for free; (c) a score with calibrated anchors (not default-to-high) is what makes the quality filter actually cut.

### Not relevant to prx-tg

- **3D representation stack** — VecSet encoder, Q-Former→512 tokens, voxel/TRELLIS latents, LATTICE sub-voxel refinement, point clouds: all 3D-specific, no path into a pixel-space DiT.
- **Nano3D-v2 editing-pair construction** — 3D editing data engine; prx-tg has no editing task.
- **Instruction-guided editing / part generation / grounding** — 3D task families out of scope for a T2I stills model.
- **Autoregressive VLM token prediction (boxes, captions)** — LLM-side machinery, not the generation path.
- **Score distillation / multi-stage geometry** — optimization-based 3D priors; irrelevant to single-stage image DiT.
- **Benchmark numbers themselves** — UniPart-Bench, text-to-3D human evals are 3D-specific; only the transfer findings carry over.

---

## 2026-08-10 — Subword tokenization's "routes not taken" (PIXEL / ByT5 / Charformer / phonetic/IPA)

**Source:** LLM-generated op-ed pasted by Tim (author/origin unknown), reviewed 2026-08-10
**Domain:** Text tokenization and conditioning encoders for text-to-image

### Summary

Op-ed arguing standard subword tokenization (BPE / WordPiece / SentencePiece — what T5 uses) is a morphological-blind "ugly hack" adopted purely for sequence compression, and surveys four alternative encoders:
1. **PIXEL** — render text as pixels, encode with a ViT (typography-aware, OOV-proof).
2. **ByT5 / CANINE** — feed raw UTF-8 bytes, no tokenizer; heavy early downsampling to recover sequence length.
3. **Charformer hashing** — hash character n-grams into continuous vectors, no embedding table, learned weighting of n-gram lengths.
4. **Phonetic/IPA** — encode spelling→phonemes so homophones ("knight"/"night") share representation.

Author's bottom line: easiest drop-in pivot for a T5 architecture is **ByT5**.

### Relevance to prx-tg

**Verdict: all four routes answer questions prx-tg doesn't have.** The framing presupposes an end-to-end fine-tunable tokenizer; prx-tg's T5 is a *frozen, offline pre-encoder* — captions are encoded once at data-gen time (`Gemma-3-27B` caption → `t5-large` → `t5_hidden.npy` (512, 1024) on the NAS) and the DiT only ever sees continuous embeddings. No tokenizer runs at train/val/inference time.

#### 1. **Verify the actual text encoder is t5-large (VERIFIED 2026-08-10)** — **most actionable / free fix**

Confirmed on-disk: `ffhq/stratum/*/t5_hidden.npy` is `(512, 1024) float16` — d_model 1024 = **T5-large**. T5-XXL would be 4096-wide. Every encoder instantiation (`production/validate.py:694`, `validation/validate.py:280`, `hf_space/inference.py:91`, `scripts/txt2img.py:130`, `scripts/generate_approved_image_dataset.py:26`, `T5_MODEL_ID = "t5-large"`) loads `t5-large`. The config field `paths.t5_path: models/t5xxl_fp16.safetensors` (in `production/config.yaml:159`) is **dead** — defined in `config_loader.py` but consumed by zero code paths.

**Why this is a clean, high-leverage upgrade and not a tokenizer swap:** upgrading the offline encoder (large → XXL, or → a caption-native multimodal tower like SigLIP/UFLIP) is a pure data-regen win for the text stream — re-encode 70k captions once on the NAS, no architecture change except the adapter's text-projection input dim (1024 → 4096 for XXL). Compare to ByT5, which would balloon text cross-attention ~3× (512 tokens → ~1400 bytes) on the *weakest* of the 7 conditioning streams (text-only generation is the known-collapse pitfall), and would inherit a *weaker* fixed model you don't get to fine-tune. Real lever > tokenizer purity.

#### 2. Caption distribution is the text-stream ceiling — **validates H3 lesson**

The four routes all trade web-scale caption-semantic prior for character-level robustness (OOV, typos, Unicode) — every one of which is a symptom of a finite vocabulary. FFHQ captions are short, grammatical, in-distribution (Gemma-3-27B output): OOV/emoji almost never occur. The ceiling on text conditioning is caption *distribution*, not tokenization — same conclusion as the H3 (MiniMax) captioning entry above.

#### 3. No ablation arm justified — **governance note**

A tokenizer swap changes sequence length, embedding space, cross-attention cost, and pretrained weights simultaneously — violates the one-variable-per-arm rule. Because the encoder is frozen+offline, it's also not an architecture ablation at all; it's a data-regeneration decision.

### Not relevant to prx-tg

- **PIXEL (text-as-pixels)** — headline benefit (legible in-image text) is irrelevant for a *face* generator; you don't want attribute conditioning to be font-specific. Only matters if prx-tg is ever tasked with rendering text (shirt/tattoo/sign) — that's a decoder/data capability add, and PIXEL-the-conditioning is a niche tool there, not a replacement.
- **ByT5 / bytes** — sequence explosion on the weakest stream; ByT5 wins (multilingual, typography) are low-value for English FFHQ captions.
- **Charformer hashing** — GBST is learned-subword, not vocabulary-free; lost to BPE at scale; the "24.5M embedding table" concern is trivial next to a frozen multi-billion-param encoder.
- **Phonetic/IPA** — homophone reasoning matters only for models that must reason about *sound*; G2P would inject a new failure mode into a working stream.
- **Acoustic/lyric use-cases generally** — no path into a stills face T2I model.

---

## 2026-08-18 — Z-Image-Turbo Pixel: Empirical Study of Pixel-Space T2I Training (Alibaba)

**Source:** https://arxiv.org/abs/2608.16887 (arXiv:2608.16887v1)
**Also:** builds on L2P (arXiv:2605.12013), DiP (CVPR 2026), JiT (CVPR 2026); compares against AsymFlow (arXiv:2605.12964) and FLUX2-klein-AsymFlow
**Domain:** Pixel-space text-to-image diffusion (flow matching) at industrial scale — the same field prx-tg trains in.

### Summary

Alibaba systematically studies how to train pixel-space T2I diffusion models that match latent-space quality, using the Z-Image backbone (single-stream DiT, Qwen3-4B text stack) and >20B image-text pairs. Headline result: **pixel-space pretraining converges substantially slower than latent-space pretraining** under identical settings (pixel stays behind through 150K iterations). Their answer is a latent-to-pixel recipe: pretrain in latent space to acquire structure and text-image alignment cheaply, then post-train in pixel space. Component ablations of that transition:

1. **Weight init:** latent-pretrained init ≫ from scratch (from-scratch produces degraded, unstable samples for much longer).
2. **Data:** self-generated samples from the *same* latent model = fastest convergence (low distribution shift, acts as a bridge); real images alone = slow; **mixing self-generated + real 1:1 = best trade-off** (corrects inherited artifacts such as malformed typography).
3. **Prediction target:** x-prediction consistently beats v-prediction in pixel space, even when initialized from a v-prediction latent model.
4. **Decoder head:** DiP (lightweight conv U-Net, 10.09M params, 834 GFLOPs) is the best quality-efficiency trade-off. The JiT linear head shows visible grid-like patch-boundary artifacts (their Fig. 9); PiT (Transformer head, 2.25B params) scores highest GenEval but is impractical; Deco and the VAE lose.
5. **Noise scale:** x_t = t·x0 + (1−t)^γ·ε — **γ=2 wins** on GenEval and DPG; γ=1, 4, 8 all produce noticeable color shifts; the resolution-derived γ=8 is suboptimal (empirical calibration required).
6. **Progressive patch adaptation:** ps16→ps32 (4× token reduction, ~comparable quality) by initializing the expanded input projection with **W′ = ½[W,W,W,W]** (preserves activation variance across the 4× fan-in) plus a 5K-step LR warmup on transferred params (expanded input projection exempt). Direct ps32 training from the latent checkpoint converges slowly with local artifacts; ps64 degrades quality.
7. **Step distillation:** Decoupled-DMD → DMDR (Z-Reward) yields a **4-NFE, no-CFG** pixel model: 0.20 s per 1024² image on H800 vs 0.95 s for the latent distilled counterpart (4.75× end-to-end).

The recipe transfers across two model families (Z-Image and FLUX2-klein): 3.18–4.75× end-to-end speedups at comparable or better benchmark scores than the latent counterparts, and better than the L2P / AsymFlow pixel baselines on most benchmarks.

### Relevance to prx-tg

prx-tg IS a pixel-space model — this is the first large-scale empirical map of exactly the design space prx-tg trains in. Three components of the winning recipe are already in place: x_prediction (`production/config.yaml:12`), ps16 token grid (`patch_size: 16`), logit-normal timestep sampling (`train.py:220`). The gaps are actionable:

#### 1. Noise scale γ=2 — cheapest experiment, one line in `flow_matching_loss`

**prx-tg application:** `train.py:227` computes zt = (1−t)·x0 + t·z1 — plain rectified flow, i.e. γ=1, which the paper flags among the settings with visible color shifts in pixel space. Switching to zt = (1−t)·x0 + t^γ·z1 with the matching velocity target v = γ·t^(γ−1)·z1 − x0 (and the corresponding path in the sampler) is a scoped ablation worth testing before any architectural change.

- **Effort:** Very low — noise-scale factor + velocity conversion in one function, config flag.
- **Isolation:** Single-variable arm (`gamma2`) against the current γ=1 baseline. Keep AsymFlow ON in both (the one interaction to watch: the rank-8 z1 projection; the paper's γ result was obtained without it).
- **Risk:** Low-moderate — the paper's γ=2 was calibrated in a latent→pixel post-training setting, not from-scratch pixel training, so the optimum may differ for prx-tg. They swept {1, 2, 4, 8}; running the same small sweep is the safe version.
- **When to consider:** Any time — no new data, no architecture change, runs on the existing 70k FFHQ pipeline.

#### 2. DiP-style conv head — prx-tg's linear head is the paper's "JiT" archetype

**prx-tg application:** `model.py:436-440`: head = final LayerNorm → Linear(hidden→ps²·3) → unpatchify → single 3×3 conv. The paper shows the linear-head family produces grid-like patch-boundary artifacts; DiP (a small conv U-Net head) removes them. prx-tg's single 3×3 conv is a weak version of the same fix — deepening it is a direct quality lever.

- **Effort:** Moderate — replace the Linear head with a 2–3 stage conv stack (down/up), config-gated, head trained from scratch on a checkpointed backbone.
- **Isolation:** Single-variable arm (`dip-head`) vs the linear+1conv baseline.
- **Risk:** Low, but with a NanoDiT-sized caveat: at their 6B scale the DiP head (834 GFLOPs) is a small fraction of backbone compute; at prx-tg's 240M scale a full DiP could dominate FLOPs at 1024². Use a shallower variant and profile.
- **When to consider:** Alongside or after the γ experiment; touches only the output, composes with any arm.

#### 3. Progressive patch adaptation ps16→ps32 — concrete recipe for the already-flagged "patch size" axis

**prx-tg application:** The H3 entry above flagged patch size as the highest-leverage efficiency axis; this paper supplies the missing recipe. Do NOT train ps32 from scratch (slow convergence + local artifacts). Instead: adapt a converged ps16 checkpoint (Arm J / current best qualifies) by expanding the patch conv (`model.py:64`, stride 16→32) with W′ = ½[W,W,W,W], 5K-step warmup on all transferred params, full LR on the new projection from step 0. At 1024²: 4096 → 1024 tokens, ~4× training throughput AND ~4× inference speedup with a small quality gap.

- **Effort:** Low — weight-replication init + warmup exception ≈ 30 lines; config flag for patch size.
- **Isolation:** Clean single-variable arm (`ps32-adapt`) seeded from a frozen ps16 checkpoint.
- **Risk:** Low-moderate — quality gap vs ps16 is small but real; ps64 is confirmed-bad, don't over-reach.
- **When to consider:** Strongest near-term candidate — it directly serves the inference-speed priority (current 10-step single-pass ~0.5 s) and reuses existing checkpoints instead of a fresh from-scratch run.

#### 4. Latent-first pretraining — the headline result; a roadmap question, not an arm

**prx-tg application:** This quantifies prx-tg's observed slow pixel-from-scratch convergence and the quality gap vs latent models: the controlled comparison shows pixel-from-scratch losing through 150K iterations at 20B-pair scale. Their prescription — latent pretraining (FLUX-AE 8× compression, latent ps2 → the same 64×64 token grid prx-tg already uses, i.e. identical sequence length/cost) followed by pixel post-training with this recipe — is the first-principles answer for any next-generation prx-tg (e.g., the proposed 0.8B backbone scale-up). For the current 240M generation it argues against another from-scratch pixel arm where a transfer path exists.

- **Effort:** Large (pipeline change) — a decision to make before the next model generation is designed, not an arm now.
- **When to consider:** When scoping the 0.8B/next-gen model, or if quality plateaus again after the cheap experiments.

#### 5. Self-generated + real 1:1 data mix — the documented recipe for any transition stage

**prx-tg application:** Self-generated data = fast low-shift adaptation; real data = artifact correction; 1:1 mix = best. This is the exact data config to use IF prx-tg runs a transition stage — `ps32-adapt` (generate from the ps16 checkpoint, mix with real FFHQ) or a future latent→pixel post-training. Not justified as a standalone arm on the current pipeline.

#### 6. Step distillation (Decoupled-DMD → DMDR) — 4 steps, no CFG

**prx-tg application:** Matches tim's preferred no-CFG default exactly and would cut the 10-step default to 4 at better quality than naive few-step. Implementation is substantial (teacher sampling, on-policy fake data, reward stage) — park it. After quality stabilizes, this is the path to a sub-0.3 s default.

### Not relevant to prx-tg

- **GenEval/DPG/OneIG/LongText numbers** — general T2I benchmarks, not face-domain; only relative comparisons carry over.
- **H800 latency figures / 20B-pair data scale** — prx-tg is a single consumer GPU with 70k curated images.
- **Qwen3-4B text stack / FLUX-AE internals** — prx-tg uses t5-large offline embeddings and no VAE in the pixel path.
- **L2P / full-AsymFlow method comparisons** — prx-tg uses only the low-rank AsymFlow noise projection (rank 8, `train.py:6`), not the full AsymFlow model. The paper's comparison is mixed (their γ=2 recipe beats AsymFlow on most benchmarks, AsymFlow wins GenEval on FLUX2-klein), so it is not a verdict on prx-tg's cheap variant — test γ on top of it rather than ripping it out.

---

## 2026-09-21 — Qwen-Image-2.1 + Qwen-Image-VAE-2.0 (unified T2I/editing, 7B single-stream DiT)

**Source:** https://huggingface.co/Qwen/Qwen-Image-2.1 (released 2026.09.20)
**Also:** [GitHub](https://github.com/QwenLM/Qwen-Image-2.1) · [Blog](https://qwen.ai/blog?id=qwen-image-2.1) · [Qwen-Image-VAE-2.0, arXiv:2605.13565](https://arxiv.org/abs/2605.13565) · [DC-AE, arXiv:2410.10733](https://arxiv.org/abs/2410.10733)
**Domain:** Latent-space text-to-image + editing (7B single-stream DiT). The *generation* side is not prx-tg's architecture; the **VAE** and **noise-schedule** components are directly load-bearing for prx-tg's P1→P2 plan.

### Summary

7B visual generator (32 single-stream DiT layers, hidden 4096 = 32×128-head, `mlp_ratio 3`, `patch_size 1`, 3-axis RoPE `[16,56,56]`), text encoder **Qwen3-VL 8B** (`context_in_dim 4096`, encodes text *and* condition images into one representation), **64-channel RGBA VAE at 16× spatial compression** (`z_dim 64`, `base_dim 96`, `decoder_base_dim 144`, `is_residual: true`, per-channel `latents_mean`/`latents_std`), FlowMatch-Euler-discrete scheduler with **dynamic shifting** (`base_shift 0.5`, `max_shift 0.9`, exponential, `base_image_seq_len 256`, `max_image_seq_len 8192`). 2K native (2048²), 40 steps. Headline 2.0→2.1 deltas: native RGBA, up to 10 reference images, mask/annotation local edits, mixed-granularity attention (token-level causal for text, chunk-level bidirectional per image) + prefix-KV-cache reuse, and — listed as a first-class improvement — **"realistic textures, improved typography, portrait lighting, fine details."**

### Relevance to prx-tg

#### 1. The VAE ceiling has a *published* fix, and it is not "more latent bandwidth" — **most actionable**

**prx-tg application:** `vae-ceiling` (2026-09-10) concluded that the FLUX AE's encode→decode of a *real* photo is washed/mushy (luminance 0.437→0.855, contrast 0.239→0.154) and that the AE, not the latent model, is the binding constraint. Its queued follow-up leaves the replacement open ("FLUX AE ... **or a better AE**"). Qwen-Image-2.1's VAE is the productized descendant of **Qwen-Image-VAE-2.0**, whose entire subject is reconstruction bottlenecks under high spatial compression; its two named techniques are **Global Skip Connections** (residual autoencoding) and **expanded latent channels**, trained at billion-image scale with a synthetic rendering engine.

The non-obvious part, and the reason this reframes the search: **Qwen 2.1's latent has exactly the same raw bandwidth as the FLUX AE.** f16 with 64 channels = 64/16² = **0.25 floats per pixel**; FLUX f8 with 16 channels = 16/8² = **0.25 floats per pixel**. Qwen moved the *same* bit budget to 4× fewer spatial positions and 4× more channels, and reports better reconstruction. Their VAE-2.0 report claims the f16 variant beats FLUX's f8 VAE outright (f16c128: SSIM 0.9706, docs-legibility NED 0.9617 vs FLUX.1-dev's 0.9546), and even f32c192 matches established f8 VAE quality. **So the vae-ceiling follow-up should not be scoped as "find a higher-capacity AE" but as "test a residual/skip-connection decoder at the same token budget."** DC-AE (`mit-han-lab`, arXiv:2410.10733) is the independent parallel result — Residual Autoencoding + expressive latent, same two ideas, and its diffusers card is **MIT** (permissive), unlike the Qwen line.

- **Effort:** Very low — extend the existing `scripts/vae_ceiling_test.py` harness with two more `autoencoders:` rows (Qwen 2.1 VAE; DC-AE f32c32) and re-run its already-queued 16-image pass (~6 min, batch 1).
- **Isolation:** Deterministic, no training — same falsification frame as the 2026-09-10 run (real photo → encode → decode, LPIPS + luminance/contrast). Single variable: the AE.
- **Token-neutral, if it ever graduates:** at 1024² the Qwen f16 VAE yields a **64×64 latent**; at `patch_size 1` that is exactly prx-tg's current 64×64 token grid (pixel ps16, and P1's FLUX-AE ps2 on 128×128). A latent-backend swap therefore does **not** change sequence length or per-step compute — only the decode fidelity. Re-encoding 70k FFHQ latents is a short GPU job.
- **Risk / license caveat:** Qwen-Image-2.1 is under the **Qwen Research License (non-commercial only, §2a)**; §4b additionally requires "Built with Qwen" attribution if the materials are used to train or improve a model that is distributed. Treat the Qwen VAE as an **evaluation-only** reference row, not a production latent backend, until a commercial license is considered. DC-AE (MIT) is the license-clean candidate to carry forward.
- **Also note:** `vae-ceiling` has no ledger entry, branch, or tag yet, and `research/results/` is gitignored — per AGENTS.md §0.1 that soft data must reach the repo.

#### 2. Resolution-conditional noise shifting — **cheap, pre-registerable, independent of γ**

**prx-tg application:** Qwen's scheduler sets `use_dynamic_shifting: true` (exponential time-shift, `base_shift 0.5 → max_shift 0.9`, interpolated on `base_image_seq_len 256 → max_image_seq_len 8192`). This is the SD3/Flux-family recipe: the noise schedule is reparametrised by **sequence length**, so high-resolution/long-sequence inputs are not systematically under- or over-noised relative to short ones. prx-tg trains **multiple aspect-ratio buckets** (`1024×1024`, `1216×832`, …) with a **single global, resolution-independent** schedule (`timestep_sampling: logit_normal`, `logit_normal_loc 0.0`, `scale 1.0`). Whether prx-tg's buckets see consistent effective SNR is currently unmeasured.

- **Distinct from `gamma2-noise-scale`:** γ changes the *interpolant exponent* (`z_t = (1−t)·x0 + t^γ·z1`); dynamic shift reparametrises the *timestep distribution*. Different mechanism, can be tested independently, and γ≠1 does not substitute for it.
- **Effort:** Low — a shift function on `t` in the sampler/training loop plus a config block; no new data, no architecture change.
- **Isolation:** Single-variable arm (`dynamic-shift`) vs the current static logit-normal, ideally parked until the γ question is settled so the two schedule changes don't confound each other.
- **Risk:** Low-moderate — the constants are calibrated for *latent* space at 2K; prx-tg is pixel-space at 1024², so the shift constants need their own calibration rather than copying `base_shift/max_shift`.

#### 3. Strong VLM text encoder is the frontier norm — **confirms the already-flagged encoder upgrade**

**prx-tg application:** Qwen 2.1 uses a **Qwen3-VL 8B** encoder emitting a 4096-dim unified text + condition-image representation; the 2026-08-18 Z-Image entry noted the same stack choice (Qwen3-4B). prx-tg's text stream is T5, 1024-dim, 512 tokens, precomputed offline. This is now a **second independent data point** (after Hunyuan3D-Buffalo) that a rich semantic/multimodal tower beats a weak fixed text encoder, and it lands on the stream the 2026-08-10 entry already identified as prx-tg's weakest leg and the known text-only-collapse pitfall. The re-encode is a pure offline data-regen job (70k captions once); the only code change is the text-projection input dim.

#### 4. Prompt rewriting shipped as a first-class 9B component — **caption detail is load-bearing**

**prx-tg application:** Qwen ships two fine-tuned **Qwen3.5-VL 9B** rewriters (T2I and edit) as official pipeline parts, not extras: the release explicitly recommends them because short prompts underperform. That is strong evidence that caption/description *detail* is a quality lever big enough to be worth a dedicated model. Two carry-overs: (a) supporting evidence for the planned stratum2 captioning pass (see the H3 "verbalize the same spatial facts" and Buffalo "monotonic tiers + calibrated quality score" lessons); (b) directly usable at inference time as a prompt-expansion step in prx-tg's sampling path. Their rewriter also **predicts an aspect ratio** (`wh_ratio`) — i.e. bucket selection is treated as a learnable decision, relevant to prx-tg's multi-aspect sampling.

#### 5. Failure-mode convergence — **the last mile is texture/lighting, not semantics**

**prx-tg application:** The 2.0→2.1 delta is explicitly *textures, typography, portrait lighting, fine detail* — plus a new VAE — with no change in size or unified-gen+edit design. prx-tg's own logged failure signature is the same fight from the other side: G0b spectral slope **OUT (too steep)** and G0a noise floor **OUT (too smooth)** at step 10000, verdict "painterly". Two independent projects landing on "high-frequency texture fidelity is the binding constraint" is useful corroboration that prx-tg is attacking the right gate, and that the productive levers are the **texture path** (output head, latent/decode fidelity, pixel post-train phase) rather than more semantic conditioning. It also reinforces the existing P1/P2 split: Qwen keeps a *separate, heavily engineered* decode path alive rather than trusting the latent to carry texture.

#### 6. Prefix KV cache — **no action; prx-tg is already structurally equivalent**

Qwen's efficiency win comes from the condition prefix (text + reference images) being static across denoising steps and therefore encoded and cached once. prx-tg's conditioning (T5, DINOv3, DWPose) is precomputed offline into `.npy` sidecars and never re-encoded per step, so the benefit is already banked by construction. Nothing to adopt.

### Not relevant to prx-tg

- **Native RGBA / transparent-layer generation and editing** — prx-tg outputs opaque face photographs; alpha output is a compositing capability with no path in. (stratum2's matting pass is for conditioning/filter selection, not output format.)
- **Up to 10 reference images / identity-preserving composition** — prx-tg generates *novel* identities; multi-reference composition is an editing capability for *given* subjects. Task scope differs (false-competition rule).
- **Mask / circle / painted-annotation local editing** — editing, not generation.
- **Qwen-Image-Bench, typography and text-rendering numbers** — face-domain irrelevant; only relative signals carry.
- **FlagOS multi-chip / 2K serving latency** — not prx-tg's regime. Mild aside: Qwen lists day-0 **AMD Radeon ROCm + Diffusers** support with accuracy aligned across platforms, which is one more data point that the ROCm path (Strix Halo) is viable for image-model inference.
- **`mlp_ratio 3`** — a 7B-scale parameter-budget choice; at 240M the FFN saving is ~12% and not independently motivated.

