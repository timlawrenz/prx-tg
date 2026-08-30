# Phase 6: Neutral DiT + Adapter Architecture (v4 — verified against prx-tg @ aafde3b)
> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.
> **Repo-location note:** This plan file lives in the `eidolon` repo but 100% of its edits target `prx-tg`. Copy it to `/home/tim/source/activity/prx-tg/.hermes/plans/` and run implementation from there so the execution repo owns its own plan (per prx-tg AGENTS.md convention).
> **Line-number policy:** Any `line NNN` reference below is an as-of-`aafde3b` hint, NOT a guarantee. The subagent MUST locate edit points by grepping for the named symbol/string, not by seeking to a line number. Line numbers drift on every commit.

**Goal:** Refactor prx-tg's conditioning pathway into a role-based neutral interface with pluggable adapters, plus the Eidolon adapter (AuraFace-LDA identity + z_g geometry). Covers all call sites: model, training loop, config, data loader, sampling, mask decoder, validation, FP8 filter, attribute logging, and CFG inference spec.
**Architecture:** `DiTBlock` receives `global_cond` (adaLN) + `sequence_cond` (cross-attn KV) + a single pre-assembled `sequence_mask`. Adapters produce `ConditioningOutput`. Training loop is the single dispatch point. Two adapters: `StratumAdapter` (regression gate, exact DINO+T5+pose replica) and `EidolonAdapter` (identity→global, z_g→sequence, no T5/patches). `isinstance` dispatch NOT used anywhere — training loop builds kwargs dicts.
**Tech Stack:** PyTorch, existing prx-tg codebase (FP8/Muon/REPA/TREAD stack partially refactored for adapter awareness).
**Repo:** `/home/tim/source/activity/prx-tg`
**Branch:** `exp/eidolon-conditioning` (new; branched from current HEAD, e.g. `exp/spatial-window-ablation` or `main` — confirm base before branching).
**Config:** New file at `experiments/eidolon-conditioning/config.yaml`. Do NOT modify `production/config.yaml`.
**Data pre-req:** AuraFace-LDA + z_g `.npy` files must exist in stratum tree. **This is now a hard GATE — see Task -1, which runs FIRST.** WebDataset NOT supported for eidolon (documented).
**Checkpoint compatibility:** Old checkpoints incompatible — projectors moved to `adapter.*`. Documented in Task 16.

---

## Task -1 (GATE — RUN FIRST): Verify AuraFace-LDA + z_g `.npy` on NAS
**Why first:** prx-tg AGENTS.md mandates verifying data availability before any work that depends on it. If these files are missing, Tasks 1–19 build an untrainable adapter. Do NOT branch or write code until this passes (or until a decision is made to generate the files first).

```bash
# Confirm the .npy files exist in the stratum tree for a representative sample of item_ids.
ls "$STRATUM_DIR" | head
# For a handful of item dirs, both files must be present:
#   <item_id>/auraface_lda.npy   (expected shape (64,))
#   <item_id>/z_g.npy            (expected shape (50,))
```
**Exit criteria:** Both `.npy` files exist and load with the expected shapes for sampled items. If missing → STOP and run the extraction pipelines (formerly Task 14 body) before proceeding. Record the verified stratum path + a coverage count (how many item_ids have both files) in the experiment README.

**(This supersedes the old Task 14, which has been demoted to a back-reference.)**

---

## Task 0: Branch and verify clean state
**Files:** None yet.

```bash
cd /home/tim/source/activity/prx-tg
git checkout -b exp/eidolon-conditioning
mkdir -p experiments/eidolon-conditioning
touch experiments/eidolon-conditioning/.gitkeep
python3.11 -m pytest tests/ -v 2>&1 | tail -20
git add experiments/eidolon-conditioning/
git commit -m "Branch exp/eidolon-conditioning — Phase 6 neutral DiT + adapters"
```

---

## Task 1: ConditioningAdapter base class (`production/adapters.py`, new)
**Design:** Uniform `**kwargs` interface. Base class provides `apply_cfg_drop_source` static method used by all adapters. No dead `null_global`/`null_sequence_token` — adapters use source-space nulls exclusively.

```python
"""Neutral conditioning adapters for the DiT."""

import torch, torch.nn as nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class ConditioningOutput:
    global_cond: torch.Tensor          # (B, hidden_size)
    sequence_cond: torch.Tensor        # (B, S, hidden_size)
    sequence_mask: Optional[torch.Tensor]  # (B, S) or None

class ConditioningAdapter(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    @staticmethod
    def apply_cfg_drop_source(source_emb, drop_mask, null_emb):
        if drop_mask is not None:
            null = null_emb.expand(source_emb.shape[0], *source_emb.shape[1:])
            source_emb = torch.where(
                drop_mask.view(-1, *([1] * (source_emb.ndim - 1))), null, source_emb)
        return source_emb

    def forward(self, **kwargs) -> ConditioningOutput:
        raise NotImplementedError
```

```bash
git add production/adapters.py && git commit -m "Add ConditioningAdapter base class"
```

---

## Task 2: StratumAdapter — regression gate
**Objective:** Exact replica of current DINO+T5+pose behavior. `dino_pool_factor` raises `NotImplementedError` (pooling not ported). Tested for shape, CFG behavior, and pooling guard.

**Constructor params (these MOVE here from `NanoDiT.__init__`):** `hidden_size`, `dino_dim=1024`, `dino_patch_dim=1024`, `text_dim=1024`, `dino_patches_enabled`, `num_pose_joints=133`, `pose_dim=3`, `pose_confidence_threshold`, `dino_pool_factor`. The StratumAdapter owns `dino_proj`, `dino_patch_proj`, `text_proj`, `pose_proj`, `pose_joint_embed`, and all four source-space null params (`null_dino`, `null_dino_patch_token`, `null_text`, `null_pose`). See Task 9 for how these arrive via `AdapterConfig`.

**⚠️ CRITICAL — mask assembly moves here.** The old `DiTBlock._forward_impl` builds the combined cross-attention sequence AND its mask internally: it concatenates `[c_text, c_dino_cls_token, (c_patches)]` and assembles `cross_mask` from `text_mask` + a synthesized all-ones CLS mask + `patches_mask` (current model.py `_forward_impl`, the `combined_context` / `cross_mask` blocks). In the neutral design DiTBlock only sees `sequence_cond` + a single `sequence_mask`, so **StratumAdapter is now responsible for producing that concatenated context AND the combined mask (including synthesizing the CLS-token all-ones column).** Do not lose the CLS-mask synthesis — it is easy to drop and will silently corrupt cross-attention. The adapter returns `ConditioningOutput(global_cond, sequence_cond=combined_context, sequence_mask=cross_mask)`.

Implementation: DINO-CLS → global_cond (with `t_emb` added, matching current `dino_cond = self.dino_proj(dino_emb) + t_emb`), T5/CLS/patches/pose → concatenated sequence_cond, combined mask → sequence_mask. Source-space nulls. Test file: `tests/test_adapters.py` with 3 tests (shape, CFG, pooling-guard). **Add a 4th regression test: the combined `sequence_mask` for a batch with a shorter text_mask matches the mask the old DiTBlock would have produced (guards the CLS-column synthesis).**

```bash
git add production/adapters.py tests/test_adapters.py
git commit -m "Add StratumAdapter — DINO+T5+pose via neutral interface"
```

---

## Task 3: EidolonAdapter — AuraFace-LDA (global) + z_g (sequence)
**Critical fix:** `geometry_proj` uses `Linear(1, hidden)` + `geometry_emb.unsqueeze(-1)` → 50 tokens (one per z_g component), NOT a single pooled vector.

```python
class EidolonAdapter(ConditioningAdapter):
    def __init__(self, hidden_size: int, identity_dim: int = 64, z_g_dim: int = 50):
        super().__init__(hidden_size)
        self.identity_proj = nn.Linear(identity_dim, hidden_size, bias=True)
        self.geometry_proj = nn.Sequential(
            nn.Linear(1, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.null_identity = nn.Parameter(torch.zeros(1, identity_dim))
        self.null_geometry = nn.Parameter(torch.zeros(1, z_g_dim))
        self.z_g_dim = z_g_dim

    def forward(self, **kwargs) -> ConditioningOutput:
        identity_emb = kwargs["identity_emb"]
        geometry_emb = kwargs["geometry_emb"]
        t_emb = kwargs.get("t_emb")

        identity_emb = self.apply_cfg_drop_source(
            identity_emb, kwargs.get("cfg_drop_identity"), self.null_identity)
        geometry_emb = self.apply_cfg_drop_source(
            geometry_emb, kwargs.get("cfg_drop_geometry"), self.null_geometry)

        identity_cond = self.identity_proj(identity_emb)
        global_cond = identity_cond + (t_emb if t_emb is not None else 0)
        geometry_cond = self.geometry_proj(geometry_emb.unsqueeze(-1))  # (B, 50, hidden)
        B = geometry_cond.shape[0]

        return ConditioningOutput(
            global_cond=global_cond,
            sequence_cond=geometry_cond,
            sequence_mask=torch.ones(B, self.z_g_dim, device=geometry_cond.device),
        )
```

Tests: shape + CFG drop. 5 total adapter tests.

```bash
git add production/adapters.py tests/test_adapters.py
git commit -m "Add EidolonAdapter — AuraFace-LDA (global) + z_g (sequence), 50 tokens"
```

---

## Task 4-7 note: Intermediate Broken States
Tasks 4 through 6 leave the model in progressively broken states that don't fully work until Task 7 completes. Do not attempt to test model functionality between these commits. The order is: DiTBlock (4) → __init__ (5) → forward (6) → MaskDiTDecoder (7). Full functionality is restored at Task 7.

---

## Task 4: DiTBlock → neutral `global_cond + sequence_cond`
**Removed:** internal concat of `[c_text, c_dino_cls_token, c_patches]` AND the combined-mask assembly (the `combined_context` block + the `cross_mask` block inside `_forward_impl`). This logic does NOT disappear — it moves into StratumAdapter (Task 2). DiTBlock now trusts that `sequence_cond` is already concatenated and `sequence_mask` is already assembled. All old source-specific params (`c_dino, c_text, text_mask, c_dino_cls_token, c_patches, patches_mask`) removed from both `_forward_impl` and `forward()`. Grep for `def _forward_impl` and `combined_context` to find the edit region (was ~lines 234–273 @ aafde3b — verify).

```python
def _forward_impl(self, x, global_cond, sequence_cond, sequence_mask=None, x_mask=None):
    shift_msa, scale_msa, shift_ca, scale_ca, shift_mlp, scale_mlp = \
        self.adaLN_modulation(global_cond).chunk(6, dim=1)
    x = x + self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask=x_mask)
    x = x + self.cross_attn(
        modulate(self.norm2(x), shift_ca, scale_ca),
        context=sequence_cond, mask=sequence_mask)
    x = x + self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
    return x

def forward(self, x, global_cond, sequence_cond, sequence_mask=None, x_mask=None):
    if self.use_checkpoint and self.training:
        return torch.utils.checkpoint.checkpoint(
            self._forward_impl, x, global_cond, sequence_cond,
            sequence_mask, x_mask, use_reentrant=False)
    else:
        return self._forward_impl(x, global_cond, sequence_cond,
                                  sequence_mask, x_mask)
```

```bash
git add production/model.py
git commit -m "Refactor DiTBlock to neutral global_cond + sequence_cond"
```

---

## Task 5: NanoDiT.__init__ → adapter
**Removed:** `self.dino_proj`, `self.text_proj`, `self.pose_proj`, `self.pose_joint_embed`, `self.dino_patch_proj`, `self.null_dino`, `self.null_text`, `self.null_dino_patch_token`, `self.null_pose`. All live in the adapter now.

```python
from production.adapters import StratumAdapter, EidolonAdapter
adapter_name = adapter_kwargs.pop("name", "stratum")
if adapter_name == "stratum":
    self.adapter = StratumAdapter(
        hidden_size=hidden_size,
        dino_patches_enabled=dino_patches_enabled,
        dino_pool_factor=kwargs.pop("dino_pool_factor", None),
        pose_confidence_threshold=pose_confidence_threshold,
        **adapter_kwargs)
elif adapter_name == "eidolon":
    self.adapter = EidolonAdapter(hidden_size=hidden_size, **adapter_kwargs)
else:
    raise ValueError(f"Unknown adapter: {adapter_name}")
```

**Callers updated — this is MORE than a 5-line swap.** The real `NanoDiT(...)` call in `train_production.py` (grep for `model = NanoDiT(`, was ~line 313 @ aafde3b) currently threads these flat kwargs that must ALL be rerouted into `adapter_kwargs` (because their projectors now live in the adapter): `num_pose_joints`, `pose_confidence_threshold`, `dino_patches_enabled`, and `dino_pool_factor`. Note `dino_pool_factor`'s real source is **`config.training.dino_pool_factor`** (a `TrainingConfig` field, passed today via `getattr(config.training, 'dino_pool_factor', None)`) — NOT `config.model`. Decide whether to keep reading it from `config.training` and inject into `adapter_kwargs`, or migrate it into the `adapter:` config block; document the choice. `scripts/auto_tune.py` has its own `NanoDiT(...)` constructor call that needs the same treatment (grep it). Both call sites should end up passing `adapter_kwargs={"name": config.adapter.name, ...stratum-specific fields...}`.

```bash
git add production/model.py production/train_production.py scripts/auto_tune.py
git commit -m "NanoDiT.__init__ uses adapter — old projectors + nulls removed"
```

---

## Task 6: NanoDiT.forward → adapter(**kwargs)
**No isinstance dispatch.** `forward()` calls `self.adapter(**adapter_kwargs, t_emb=t_emb)`, receives `ConditioningOutput`. All old CFG/concat logic removed, **but KEEP Dynamic Tensor Masking (DTM) padding logic**.
- Change `forward` signature to: `def forward(self, x, t, return_repa_hidden=False, tread_enabled=None, maskdit_enabled=None, **adapter_kwargs):`
- After calling the adapter, explicitly apply the `pad_ctx` (Rule of 16) to `cond.sequence_cond` and `cond.sequence_mask` to prevent `torch.compile` graph breaks.
- Updated call sites (grep for `self.blocks[` and `block(x,` and `self.maskdit_decoder(` — line hints are @ aafde3b, verify):
- Non-TREAD loop (`for i, block in enumerate(self.blocks)`, ~line 764): `block(x, global_cond=…, sequence_cond=…, sequence_mask=…, x_mask=…)`.
- TREAD routing — THREE sites at `self.blocks[i](...)` (~lines 744, 750, 760): same rename; note each currently passes a different `x_mask` (`x_mask`, `visible_x_mask`, `x_mask`) — preserve that.
- MaskDiTDecoder call (`self.maskdit_decoder(...)`, ~lines 771–775): now passes the `ConditioningOutput` (or its `global_cond`/`sequence_cond`/`sequence_mask`) instead of `dino_cond, text_cond, text_mask, dino_cls_token, patches_cond, dino_patches_mask`. Decoder updated in Task 7.
- Note: `dino_cond`, `text_cond`, `patches_cond`, `dino_cls_token` are LOCALS built earlier in the current `forward()` from the projectors — that construction block is what the adapter replaces. Remove it.

```bash
git add production/model.py
git commit -m "NanoDiT.forward — adapter-driven, TREAD + MaskDiT call sites updated"
```

---

## Task 7: MaskDiTDecoder → ConditioningOutput
Decoder's `forward` currently takes `(x_visible, visible_idx, masked_idx, N_total, pos_embed, c_dino, c_text, text_mask, c_dino_cls_token, c_patches, patches_mask=None)` and loops `block(x_full, c_dino, c_text, text_mask, c_dino_cls_token, c_patches, patches_mask)`. Refactor its signature to receive `cond: ConditioningOutput` (replacing the six trailing conditioning args) and pass `cond.global_cond`/`cond.sequence_cond`/`cond.sequence_mask` to its neutral DiTBlocks. Update the call site in `NanoDiT.forward` (Task 6) to match.

```bash
git add production/model.py
git commit -m "Update MaskDiTDecoder to neutral DiTBlock — receives ConditioningOutput"
```

---

## Task 8: train.py — kwargs dispatch + CFG logic
**Single dispatch point.** Builds kwargs dict per adapter type.
- Safely unpack `batch.get('identity_emb')` and `batch.get('geometry_emb')` to `self.device` so Eidolon tensors don't cause CPU/GPU crashes.
- Updates CFG dropout logic for 2-stream eidolon scheme (`p_uncond`, `p_identity_only`, `p_geometry_only`) using `config.adapter.cfg_dropout`.

```bash
git add production/train.py
git commit -m "Update training loop — adapter-aware kwargs dict + eidolon CFG dropout"
```

---

## Task 9: config_loader.py — AdapterConfig
**Schema-gap fix:** `AdapterConfig` must carry BOTH the eidolon fields AND the Stratum-specific fields that StratumAdapter's constructor (Task 2/5) needs — otherwise Task 5 cannot build the StratumAdapter from config. When `adapter:` is absent from a legacy YAML, the loader MUST default `name="stratum"` and populate the stratum fields from the existing `model:`/`training:` values so old configs keep working unchanged (backward-compat gate).

```python
@dataclass
class AdapterConfig:
    name: str = "stratum"
    # --- eidolon fields ---
    identity_dim: int = 64
    z_g_dim: int = 50
    cfg_dropout: Optional[dict] = None
    # --- stratum fields (mirror what NanoDiT used to receive flat) ---
    dino_dim: int = 1024
    dino_patch_dim: int = 1024
    text_dim: int = 1024
    dino_patches_enabled: bool = True
    num_pose_joints: int = 133
    pose_dim: int = 3
    pose_confidence_threshold: float = 0.05
    dino_pool_factor: Optional[int] = None   # sourced from config.training today; see Task 5
```
Parsed from YAML `adapter:` block. Exposed to model constructor. **Backward-compat rule:** if no `adapter:` block, synthesize `AdapterConfig(name="stratum", dino_patches_enabled=config.model...or default, num_pose_joints=config.model.num_pose_joints, pose_confidence_threshold=config.model.pose_confidence_threshold, dino_pool_factor=config.training.dino_pool_factor)`. Add a loader test asserting a legacy config with no `adapter:` block still instantiates a working StratumAdapter with the correct field values.

```bash
git add production/config_loader.py
git commit -m "Add AdapterConfig to config loader"
```

---

## Task 10: Experiment config — new file, don't touch base
`experiments/eidolon-conditioning/config.yaml`:
```yaml
adapter:
  name: "eidolon"
  identity_dim: 64
  z_g_dim: 50
  cfg_dropout:
    p_uncond: 0.10
    p_identity_only: 0.20
    p_geometry_only: 0.15
model:
  dino_patches_enabled: false
data:
  source: "stratum"
```
*(Note: This adapter drops sequence length from ~4,634 tokens to just 50, resulting in a ~99% reduction in cross-attention FLOPs. A massive drop in `sys/active_tokens` is expected and correct.)*

```bash
git add experiments/eidolon-conditioning/config.yaml
git commit -m "Add eidolon experiment config — does not touch base config"
```

---

## Task 11: sample.py — kwargs dispatch + eidolon CFG (deferred)
Update kwargs dispatch (same pattern as Task 8). Note: actual eidolon CFG inference (Task 19) is documented but implemented in a future pass — training-only scope for Phase 6.

```bash
git add production/sample.py
git commit -m "Update sampling to pass adapter-appropriate kwargs"
```

---

## Task 12: validate.py — kwargs dispatch + eidolon tests
**eidolon-specific checks:**
- Reconstruction: identity_emb + geometry_emb from same image → render → AuraFace cosine.
- Identity swap: same geometry_emb, different identity_emb → AuraFace cosine distinguishes.
- Geometry sweep: fix identity_emb, sweep one z_g dimension → DWPose tracks pose change, AuraFace holds.
- Text manipulation: **NOT applicable** to eidolon (no T5). Skip when `adapter_name == "eidolon"`.

```bash
git add production/validate.py
git commit -m "Adapt validation to adapter interface + eidolon test specs"
```

---

## Task 13: data_stratum.py — load AuraFace-LDA + z_g
Stratum-only for Phase 6. WebDataset out of scope. Add `adapter_name` parameter to `get_production_dataloader()` (defaults to `"stratum"` for backward compatibility) and pass it through to the `StratumDataset` collation logic.

```python
if adapter_name == "eidolon":
    identity_emb = np.load(stratum_dir / item_id / "auraface_lda.npy")
    geometry_emb = np.load(stratum_dir / item_id / "z_g.npy")
    return {"image": img, "identity_emb": identity_emb, "geometry_emb": geometry_emb}
```

```bash
git add production/data_stratum.py
git commit -m "Extend data loader for eidolon — AuraFace-LDA + z_g .npy"
```

---

## Task 14: (SUPERSEDED — see Task -1 GATE at top)
Data verification moved to **Task -1** and promoted to a hard gate that runs FIRST. If Task -1 found the `.npy` files missing, the extraction pipelines must run here-equivalent (before any code work). This slot is retained only as a back-reference so downstream numbering is stable.

---

## Task 15: Smoke test — forward pass + loss drop
Reads from experiment config. 100-step synthetic loss check.

```bash
git add scripts/smoke_test_eidolon.py
git commit -m "Smoke test: EidolonAdapter forward pass + 100-step loss decrease"
```

---

## Task 16: Document checkpoint incompatibility
`experiments/eidolon-conditioning/README.md`: old checkpoints won't load (state dict keys moved to `adapter.*`).

```bash
git add experiments/eidolon-conditioning/README.md
git commit -m "Document checkpoint incompatibility for Phase 6"
```

---

## Task 17: FP8 filter → adapter-agnostic
Replace hardcoded projector names with `"adapter."` catch-all:
```python
exclude_keywords = ["adapter.", "t_embedder", "adaLN_modulation", "final_proj"]
```
Applies to `production/train_production.py` and `scripts/auto_tune.py`.

```bash
git add production/train_production.py scripts/auto_tune.py
git commit -m "Make FP8 filter adapter-agnostic — exclude adapter.*"
```

---

## Task 18: Redirect train.py logging + fix validation scripts
- `train.py`: Update `hasattr` checks for ALL projectors (`dino_patch_proj`, `text_proj`, `dino_proj`, `null_dino_patch_token`) to point to `model.adapter.*` or `model.module.adapter.*`.
- `scripts/validate_training_flow.py`, `scripts/validate_patch_gradients.py`: Add early exits if `not hasattr(model.adapter, "dino_patch_proj")` rather than letting them crash on Eidolon.

```bash
git add production/train.py scripts/validate_*.py
git commit -m "Redirect projector logging to model.adapter.*; fix validation crashes"
```

---

## Task 19: Document eidolon CFG inference formulation
Append to README:
```markdown
## CFG at inference (future)
v = v_uncond + identity_scale*(v_identity - v_uncond) + geometry_scale*(v_geometry - v_uncond)
```
Dual-source formulation. Implementation deferred to Phase 6 inference pass.

```bash
git add experiments/eidolon-conditioning/README.md
git commit -m "Document eidolon inference CFG formulation"
```

---

## Verification Checklist
- [ ] **AuraFace-LDA + z_g `.npy` on NAS verified (Task -1 GATE — before any code)**
- [ ] All adapter tests pass (5 eidolon/stratum + StratumAdapter mask-regression test)
- [ ] StratumAdapter regression: existing tests still pass
- [ ] Legacy config (no `adapter:` block) still builds a working StratumAdapter
- [ ] Forward pass with EidolonAdapter: correct shape
- [ ] 100-step loss decrease (no NaN, no OOM)
- [ ] Config loads and instantiates EidolonAdapter
- [ ] Data loader returns `identity_emb` (64,) + `geometry_emb` (50,)
- [ ] FP8 filter adapter-agnostic
- [ ] train.py logging → `model.adapter.*`
- [ ] Validation scripts don't crash

---

## Task summary (20 tasks — Task -1 gate added, Task 14 superseded)

| # | Task | Critical? |
|---|---|---|
| **-1** | **Verify AuraFace-LDA + z_g `.npy` on NAS** | 🔴 **GATE — RUN FIRST** |
| 0 | Branch + clean state | — |
| 1 | `ConditioningAdapter` base class | — |
| 2 | `StratumAdapter` + tests (owns mask assembly) | Regression gate 🔴 |
| 3 | `EidolonAdapter` + tests | Core deliverable |
| 4 | DiTBlock → neutral | 🔴 |
| 5 | NanoDiT.__init__ → adapter (reroute flat kwargs) | 🔴 |
| 6 | NanoDiT.forward → adapter(**kwargs) | 🔴 |
| 7 | MaskDiTDecoder → ConditioningOutput | 🔴 |
| 8 | `train.py` — kwargs dispatch + CFG | 🔴 |
| 9 | `config_loader.py` — AdapterConfig (schema-gap fix) | 🔴 |
| 10 | Experiment config (new file) | — |
| 11 | `sample.py` — kwargs dispatch | — |
| 12 | `validate.py` — kwargs + eidolon tests | — |
| 13 | `data_stratum.py` — load AF-LDA + z_g | 🔴 |
| 14 | ~~Verify data files on NAS~~ (SUPERSEDED by Task -1) | — |
| 15 | Smoke test — forward + loss | — |
| 16 | Document checkpoint incompat | — |
| 17 | FP8 filter → adapter-agnostic | 🔴 |
| 18 | Redirect logging + fix validation scripts | 🔴 |
| 19 | Document eidolon CFG formulation | — |
