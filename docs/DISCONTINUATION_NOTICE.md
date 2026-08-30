# Project Discontinuation Notices — prx-tg

Tombstones for KILLed approaches. These convert sunk cost into permanent, reusable knowledge and prevent blind re-attempt.

---

## Arm H: Shared adaLN + Per-Block LoRA

**Date killed:** 2026-05-25
**Arm:** H — `shared-adaln-lora`
**Status:** DISCONTINUED

### Summary

Attempted to replace per-block adaLN modulation with a single shared adaLN module plus per-block LoRA adapters (rank 8). Reduced parameter count from 237.7M to 178.8M (59M savings). Failed to converge at 5,000 steps — generated noise/dithering throughout.

### What We Learned

#### Successful components ✅
- LoRA adapters successfully reduced parameter count without architectural changes to attention or FFN blocks.
- Shared adaLN module correctly computed conditioning signals from the global conditioning vector.

#### Failed components ❌
- **Recon LPIPS: 0.9823** (vs 0.9267 full stack). Effectively no reconstruction capability.
- **Text-only LPIPS: 1.0062** (vs 0.9219). Worse than random.
- Visually indistinguishable from noise at all checkpoints.

### Root Cause

**Per-block modulation is load-bearing, not redundant.** Each DiT block requires independent conditioning injection because the representation at each layer depth encodes fundamentally different information (early blocks: structure/layout; middle blocks: features/texture; late blocks: high-frequency detail). A single shared adaLN module cannot produce modulation signals appropriate for all depths simultaneously. The LoRA adapters (rank 8) lack sufficient capacity to compensate for the loss of per-block modulation.

### Why We're Sharing This

Future attempts to reduce DiT parameter count should not target the adaLN modulation path. Parameter savings from shared modulation come at the cost of convergence — the 59M params saved are not "redundant." They encode depth-specific conditioning that is essential for the diffusion process.

### Salvage
- The shared adaLN module code may be reusable as a baseline for other conditioning experiments.
- The finding that per-block modulation is load-bearing informed the `EidolonAdapter` design (identity goes through adaLN, geometry through separate cross-attention — maintaining depth-specific pathways).
- ⚠️ No dedicated experiment directory exists under `experiments/shared-adaln-lora/`. Artifacts may be in legacy timestamp-named dirs.

---

## Arm N: z_g Token Basis (Eidolon Conditioning)

**Date killed:** 2026-07-10
**Arm:** N — `zg-token-basis`
**Status:** DISCONTINUED (superseded by Arm O)

### Summary

Added a per-dimension learned token embedding (`geo_basis`) to `EidolonAdapter` so cross-attention can bind specific z_g axes to geometric attributes (e.g., dim0 → head yaw). **Proved the core hypothesis**: dim0 controlled head yaw at steps 2000/2500 — the first time ever in this architecture. But the model mode-collapsed into output darkness at step 3500 with no gradient or loss spike.

### What We Learned

#### Successful components ✅
- **Per-dim basis works.** z_g dim0 cleanly controlled head yaw at steps 2000/2500. This validates the root cause diagnosis — the "always frontal" failure is an architecture bug (permutation-symmetric tokens), not a data or z_g-vector problem.
- **The pre-run diagnostics were correct.** Full-corpus audit (z_g→yaw R²=0.996), identity leak test (AUC 0.65), and CFG sweep all correctly identified the problem before the run.

#### Failed components ❌
- **Mode collapse at step 3500.** Output became completely dark with no gradient/loss spike — a silent failure.
- **Wrong application pattern.** The basis was added on ALL steps, including ~30% of steps where geometry was CFG-dropped (zeroed). This turned the basis into a pure noise injection — approximately 1,500 noise events over 5,000 steps.

### Root Cause

The `geo_basis` embedding was computed unconditionally in `EidolonAdapter.forward()` — it wasn't gated on whether geometry conditioning was actually live. During CFG dropout steps (p_geometry_only=0.30), the geometry input is zeroed, so the basis adds structured noise to the zero-vector. This noise accumulated across 1,500+ steps, gradually pushing output into clipping/saturation.

A secondary possibility: the gain path through the basis embedding may need downscale initialization to prevent early over-amplification.

### Why We're Sharing This

The per-dim basis is the right fix — it produced the first-ever yaw control. The failure mode is a simple gating bug, not a conceptual failure. Arm O implements the fix: the basis only activates when `z_g` is live (nonzero). A second Arm O failure would point to the gain-path hypothesis.

### Salvage

- The per-dim basis code itself is sound and carried forward into Arm O.
- Checkpoints at steps 2000 and 2500 demonstrate yaw binding — these are valuable reference artifacts.
- The diagnostic pipeline (full-corpus audit, CFG sweep, identity leak test) should be run on every arm touching geometry conditioning.
- Arm directory: `experiments/zg-token-basis/` (config ✓, README ✓, provenance ✓, checkpoints ✓)
