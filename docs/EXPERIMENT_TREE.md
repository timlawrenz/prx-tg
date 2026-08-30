# Experiment Tree — prx-tg

Living map of all active, planned, and concluded workstreams. Each entry links to the experiment arm directory under `experiments/{slug}/`. Status tags: `[ACTIVE]`, `[CONCLUDED]`, `[TBD]`, `[DISCONTINUED]`.

Last updated: 2026-08-30

---

## Active

* **[ACTIVE] Phase 1b Research Loop** — photorealism gates + blind review + avenue registry
  (`.hermes/plans/2026-08-30_photorealism-gates-blind-review.md`). M1 shipped 2026-08-30;
  M2 (calibration) + M3 (tick state machine) next. No training arms active.

---

## Concluded

### Geometry Adapter for Pretrained SD 1.5 Series (Arms P, P2, P3, P4) — `[CONCLUDED — KILL]`

Four-arm series testing whether a lightweight geometry adapter (~630K params) can steer face yaw in a pretrained SD 1.5 UNet by injecting z_g geometry tokens alongside CLIP text embeddings. All four arms failed — zero controlled yaw across 40K+ cumulative steps.

* **Arm P — `geometry-adapter-sd15`** (`experiments/geometry-adapter-sd15/`) `[CONCLUDED — FAIL]`
  * Vanilla SD1.5, frozen attention. 10K steps. Adapter learns (differential output) but produces artifacts, not yaw. Bug 1 (basis leak) contaminated training. Code review found 8 bugs.
* **Arm P2 — `geo-adapter-bugfix-restart`** (`experiments/geo-adapter-bugfix-restart/`) `[CONCLUDED — FAIL]`
  * Frozen attention + bug fixes + pose-stripped captions. 10K steps. Same artifacts, no yaw control.
* **Arm P3 — `geo-adapter-absreal`** (`experiments/geo-adapter-absreal/`) `[CONCLUDED — KILL]`
  * Frozen attention, AbsoluteReality v1.8.1. Killed at step 2340 (budget-justified: P/P2 exhausted 10K identically). Garbled output.
* **Arm P4 — `geo-adapter-lora-crossattn`** (`experiments/geo-adapter-lora-crossattn/`) `[CONCLUDED — KILL]`
  * LoRA-UNFROZEN cross-attention (rank-8 K,V, ~5M params). 10K steps on vanilla SD1.5. Loss diverged +48% from step 2K→10K; output collapsed to pure noise. Falsified the "frozen attention" root cause — unfreezing attention made it *worse*. ⚠️ Cross-model eval confound (trained vanilla, 4/6 sweeps on AbsoluteReality).

**Refined root cause (supersedes earlier "frozen attention can't attend"):** The ε-prediction denoising loss provides no gradient that rewards geometry control — the frozen text/CLIP path already denoises the face, so the adapter is redundant to the loss. Trainable weights either stay inert (frozen attention: P/P2/P3) or random-walk into destruction (LoRA: P4). Not data-dependent, not base-model-quality-dependent, not attention-freeze-dependent. **Thread-level KILL. Project → PIVOT to from-scratch DiT (Arm O).**


### Ablation Series (Arms A–D) — `[CONCLUDED — GO]`

Foundational 4-arm sweep isolating TREAD, Muon, and REPA contributions on 7k FFHQ subset. All 5,000 steps on same quad-GPU Vast.ai node.

* **Arm A — `minimal-baseline`**: AdamW only, no TREAD, no REPA. Baseline reference.
* **Arm B — `tread-adamw`**: TREAD + AdamW. **Reconstruction collapsed at step 3500** — TREAD's token sparsity incompatible with AdamW's per-parameter adaptive rates.
* **Arm C — `tread-muon`**: TREAD + Muon. **Recovered stability.** Muon's orthogonalized updates prevent learning-rate inflation under sparse gradients. Best text controllability (Text Manip delta 0.546).
* **Arm D — `full-stack-baseline`**: TREAD + Muon + REPA. **Best overall quality** — Recon LPIPS 0.927, Text LPIPS 0.922. REPA loss scale not comparable to A/B/C.
  * 📄 Detailed writeup: `docs/ablation_a_through_d/ablation_study.md`
  * ⚠️ Note: Arm D directories (`full-stack-baseline`, `minimal-baseline`, `tread-adamw`, `tread-muon`) not yet created under `experiments/` — artifacts live in legacy timestamp-named dirs.

### Scaling & Optimization Arms (E–J)

* **Arm E — `seg-weight-spatial`** `[CONCLUDED — FAIL (CONFOUNDED)]`
  * Seg weight map (face 2×, bg 0.5×) vs Arm D. Confounded variables: effective batch size 128 vs 256, REPA warmdown not matched.
  * 📄 Compliance review: `docs/part-e-compliance-review.md`

* **Arm F — `seg-weight-clean-ablation`** `[CONCLUDED — PAUSED]`
  * Clean re-run of Arm E with batch size and REPA warmdown matched. ⏸️ Paused.

* **Arm G — `asym-flow-ablation`** `[CONCLUDED — GO]`
  * Asymmetric Flow Matching (rank 8). Recon LPIPS 0.938, Text-only LPIPS 0.914. Convergence acceleration verified. No quality regression vs baseline.

* **Arm H — `shared-adaln-lora`** `[CONCLUDED — KILL]`
  * Shared adaLN + per-block LoRA (rank 8). **Failed to converge at 5k steps.** Recon LPIPS 0.982, Text-only 1.006. Visually noise/dithering. Per-block modulation is load-bearing.
  * 💀 Tombstone: `docs/DISCONTINUATION_NOTICE.md`

* **Arm I — `fp8-native`** (`experiments/arm_i_fp8/`) `[CONCLUDED — GO]`
  * Native FP8 via torchao. **2×+ speedup (26h vs 56h BF16)** with negligible quality loss (Recon LPIPS 0.906 vs 0.900 BF16). Text controllability actually improved (Text Manip Diff 0.504 vs 0.485).

* **Arm J — `faces70k-fp8`** (`experiments/faces70k-fp8/`) `[CONCLUDED — INCOMPLETE]`
  * Scaled FP8 to 70k portraits (40k steps w/ REPA linear decay). 🔄 Stopped at 34k steps.

### Quality Metric Arms (K–M) — `[CONCLUDED — GO]`

3-arm series testing DINO patch token contribution to generation quality. All use quality metrics (CLIP + aesthetic + DWPose face confidence) alongside LPIPS.

* **Arm K — `spatial-window-baseline`** (`experiments/spatial-window-baseline/`) `[CONCLUDED]`
  * Full DINO patches. Best Recon LPIPS (0.937) but worst generation quality metrics.

* **Arm L — `spatial-window-2`** (`experiments/spatial-window-2/`) `[CONCLUDED — GO]`
  * DINO patch spatial window (r=2) via 2×2 AvgPool2d. **Degraded reconstruction (0.999 vs 0.937) without matching the quality of full removal.** 155× FLOP reduction doesn't justify quality trade-off.

* **Arm M — `no-dino-patch-ablation`** (`experiments/no-dino-patch-ablation/`) `[CONCLUDED — GO]`
  * DINO patch cross-attention disabled entirely. **Best generation quality on 4 of 5 metrics** (CLIP 0.204, Aesthetic 4.44, Face Conf 0.790, Text-only LPIPS 0.907).
  * 🏆 Key finding: DINO patches buy reconstruction fidelity (+0.043 LPIPS) at the cost of CLIP score, aesthetics, face confidence, and text-following. DINO CLS token alone is sufficient style conditioning.

### Eidolon Conditioning (branch `exp/eidolon-conditioning`)

* **Eidolon Conditioning baseline** (`experiments/eidolon-conditioning/`) `[CONCLUDED — GO]`
  * Neutral DiT + `EidolonAdapter` (AuraFace-LDA identity via adaLN + z_g geometry via cross-attention). T5 text + DINO patches dropped. Renders all faces frontal (root cause identified: z_g tokens are permutation-symmetric).

* **Arm O — `zg-token-basis-cfg-guard`** (`experiments/zg-token-basis-cfg-guard/`) `[CONCLUDED — PASS]` ✅
  * CFG-guarded per-dim basis. **PASS on pre-registered gate:** monotonic dim0→yaw at steps 3000 AND 5000, zero collapse through 5,000 steps, loss 0.0099, recon LPIPS 0.819. First arm to hold controlled yaw without collapse — validates the Arm N collapse diagnosis. Re-scoring under the G-gate suite pending (M2/M3).
  * Release artifact: `release/prx_tg_armo_cfgguard_step5000_bf16.safetensors`.

* **Hegre Geometry** (`experiments/hegre-geometry/`) `[CONCLUDED — GO]`
  * Added hegre corpus data (w1.0) + geometry-only CFG dropout (p=0.30). Shared MLP (no per-dim basis). Still renders all faces frontal. Proved data augmentation alone doesn't fix the architectural bug.

* **Arm N — `zg-token-basis`** (`experiments/zg-token-basis/`) `[CONCLUDED — KILL]`
  * **Per-dim learned embedding** so cross-attention can bind z_g dim0 → head yaw. ✅ **Proved yaw binding**: dim0 controlled head yaw at steps 2000/2500 (first time ever). ❌ **Mode collapse at step 3500**: output darkness, no gradient/loss spike.
  * Root cause: basis added on all steps (~30% CFG-dropped), making it a pure noise injection (~1500 events) that accumulated and pushed output into clipping.
  * 💀 Tombstone: `docs/DISCONTINUATION_NOTICE.md`

---

## TBD

* **[TBD] Single-Stream DiT** — Remove cross-attention, concatenate T5 text + DINO CLS into self-attention sequence. Validated by Ideogram 4 at 9.3B scale. Good arm after eidolon branch stabilizes.
* **[TBD] iREPA Upgrade** — Replace `nn.Linear` REPA projection with Conv2d + spatial normalization (per Singh et al.). ~4 lines each, proven FID gains.
* **[TBD] Resolution-Aware Logit-Normal Schedule** — Adjust mu per bucket resolution (per Ideogram 4). 3-line change in `train.py`.
* **[TBD] DINO-Patch-Free Architecture** — Redesign conditioning around DINO CLS + DWPose + T5 text only, dropping DINO patches permanently per Arms K–M findings.
* **[TBD] Structured Face Attribute Captions** — Convert caption pipeline to JSON attribute templates for denser supervision (per Ideogram 4).
* **[TBD] SpectraReward RL Post-Training** — Freeze MLLM reward model, score rollout groups, apply advantage-weighted matching for face attribute binding.
* **[TBD] QK-RMSNorm** — Add RMSNorm to Q and K separately for attention stability (per Ideogram 4 / Llama 3).
* **[TBD] Sapiens2 0.8B Backbone** — Scale up backbone after winning config found (from breast-tissue-net notes).
