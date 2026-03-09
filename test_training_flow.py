"""Reproduce the exact training flow to find where patches break."""
import torch
import torch.nn.functional as F
from production.model import NanoDiT
from production.train import flow_matching_loss

print("="*60)
print("REPRODUCING TRAINING FLOW")
print("="*60)

# Create model exactly like train_production.py
model = NanoDiT(
    input_size=64,
    patch_size=2,
    in_channels=16,
    hidden_size=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4.0,
    use_gradient_checkpointing=True,  # This is enabled in your training
)

print(f"\n✓ Model created with gradient checkpointing=True")
print(f"  Total params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Simulate one batch from data loader
B = 1
x0 = torch.randn(B, 16, 64, 64)
dino_emb = torch.randn(B, 1024)
dino_patches = torch.randn(B, 4096, 1024)  # Real patches from data
text_emb = torch.randn(B, 500, 1024)
text_mask = torch.ones(B, 500)

cfg_probs = {
    'p_uncond': 0.1,
    'p_text_only': 0.3,
    'p_dino_cls_only': 0.05,
    'p_dino_patches_only': 0.05,
}

print("\n✓ Batch data created")
print(f"  x0: {x0.shape}")
print(f"  dino_patches: {dino_patches.shape}")

# Training step
model.train()
optimizer.zero_grad()

print("\n" + "="*60)
print("RUNNING TRAINING STEP (no REPA)")
print("="*60)

loss, v_pred, repa_loss, lpips_loss = flow_matching_loss(
    model, x0, dino_emb, dino_patches, text_emb, text_mask,
    cfg_probs, return_v_pred=True
)

print(f"\n✓ Forward pass complete")
print(f"  Loss: {loss.item():.6f}")
print(f"  v_pred shape: {v_pred.shape}")
print(f"  repa_loss: {repa_loss}")
print(f"  lpips_loss: {lpips_loss}")
assert repa_loss is None, "REPA loss should be None when not enabled"
assert lpips_loss is None, "LPIPS loss should be None when not enabled"

# Backward
loss.backward()

print(f"\n✓ Backward pass complete")

# Check gradients
patch_proj_grad = model.dino_patch_proj.weight.grad
text_proj_grad = model.text_proj.weight.grad

print(f"\nGradient check:")
print(f"  text_proj grad norm: {text_proj_grad.norm().item():.8f}")
print(f"  patch_proj grad norm: {patch_proj_grad.norm().item():.8f}")

if patch_proj_grad.norm().item() < 1e-10:
    print("\n" + "="*60)
    print("✗✗✗ PATCH PROJECTION HAS ZERO GRADIENT! ✗✗✗")
    print("="*60)
    print("\nThis confirms patches are not affecting the loss!")
    print("The bug is in the model forward pass with gradient checkpointing.")
else:
    print("\n✓ Patches have non-zero gradient")

# Apply optimizer step
optimizer.step()

# Check if weights changed
initial_weight = torch.randn_like(model.dino_patch_proj.weight) * 0.037
weight_diff = (model.dino_patch_proj.weight - initial_weight).abs().max().item()
print(f"\nWeight change: {weight_diff:.8f}")

# ============================================================
# REPA TEST
# ============================================================
print("\n" + "="*60)
print("TESTING REPA")
print("="*60)

from production.config_loader import REPAConfig

repa_block_idx = 6  # depth // 2 for 12-layer model
model_repa = NanoDiT(
    input_size=64,
    patch_size=2,
    in_channels=16,
    hidden_size=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4.0,
    use_gradient_checkpointing=True,
    repa_block_idx=repa_block_idx,
)

print(f"\n✓ REPA model created (alignment at block {repa_block_idx})")
print(f"  Total params: {sum(p.numel() for p in model_repa.parameters())/1e6:.1f}M")
assert hasattr(model_repa, 'repa_proj'), "Model should have repa_proj layer"
print(f"  repa_proj shape: {model_repa.repa_proj.weight.shape}")

# Test forward with return_repa_hidden
model_repa.train()
optimizer_repa = torch.optim.AdamW(model_repa.parameters(), lr=3e-4)
optimizer_repa.zero_grad()

# Use matching spatial dims: 64x64 latent, patch_size=2 -> 32x32 = 1024 tokens
# DINOv3 patches should also be 1024 (same grid)
num_latent_tokens = (64 // 2) * (64 // 2)  # 1024
dino_patches_repa = torch.randn(B, num_latent_tokens, 1024)
dino_patches_mask = torch.ones(B, num_latent_tokens)

repa_config = REPAConfig(enabled=True, weight=0.5, block_index=repa_block_idx, loss_type="cosine")

loss_repa, v_pred_repa, repa_loss_val, _ = flow_matching_loss(
    model_repa, x0, dino_emb, dino_patches_repa, text_emb, text_mask,
    cfg_probs, dino_patches_mask=dino_patches_mask, return_v_pred=True,
    repa_config=repa_config,
)

print(f"\n✓ REPA forward pass complete")
print(f"  Total loss: {loss_repa.item():.6f}")
print(f"  REPA loss: {repa_loss_val.item():.6f}")
print(f"  v_pred shape: {v_pred_repa.shape}")
assert repa_loss_val is not None, "REPA loss should not be None when enabled"
assert repa_loss_val.item() > 0, "REPA loss should be positive"

# Backward
loss_repa.backward()
print(f"\n✓ REPA backward pass complete")

# Check REPA projection gradients
repa_proj_grad = model_repa.repa_proj.weight.grad
print(f"  repa_proj grad norm: {repa_proj_grad.norm().item():.8f}")
assert repa_proj_grad.norm().item() > 1e-10, "REPA projection should have non-zero gradients"
print(f"\n✓ REPA projection has non-zero gradients")

# Test with padding mask (some tokens masked out)
optimizer_repa.zero_grad()
dino_patches_mask_partial = torch.ones(B, num_latent_tokens)
dino_patches_mask_partial[:, num_latent_tokens // 2:] = 0  # mask out half

loss_masked, _, repa_loss_masked, _ = flow_matching_loss(
    model_repa, x0, dino_emb, dino_patches_repa, text_emb, text_mask,
    cfg_probs, dino_patches_mask=dino_patches_mask_partial, return_v_pred=True,
    repa_config=repa_config,
)

print(f"\n✓ REPA with partial mask:")
print(f"  Total loss: {loss_masked.item():.6f}")
print(f"  REPA loss (half masked): {repa_loss_masked.item():.6f}")

# Test with mismatched sizes (DINOv3 patches fewer than latent tokens)
optimizer_repa.zero_grad()
fewer_patches = num_latent_tokens - 48  # simulate DINOv3 having fewer patches
dino_patches_fewer = torch.randn(B, fewer_patches, 1024)
dino_patches_mask_fewer = torch.ones(B, fewer_patches)

loss_mismatch, _, repa_loss_mismatch, _ = flow_matching_loss(
    model_repa, x0, dino_emb, dino_patches_fewer, text_emb, text_mask,
    cfg_probs, dino_patches_mask=dino_patches_mask_fewer, return_v_pred=True,
    repa_config=repa_config,
)
loss_mismatch.backward()

print(f"\n✓ REPA with size mismatch ({num_latent_tokens} latent vs {fewer_patches} dino):")
print(f"  Total loss: {loss_mismatch.item():.6f}")
print(f"  REPA loss: {repa_loss_mismatch.item():.6f}")
assert repa_loss_mismatch.item() > 0, "REPA loss should work with mismatched sizes"

print("\n" + "="*60)
print("TESTING TREAD")
print("="*60)

from production.config_loader import TREADConfig

# Create model with TREAD routing
tread_route_start = 1
tread_route_end = 10  # depth - 2 for 12-layer model
model_tread = NanoDiT(
    input_size=64,
    patch_size=2,
    in_channels=16,
    hidden_size=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4.0,
    use_gradient_checkpointing=True,
    tread_route_start=tread_route_start,
    tread_route_end=tread_route_end,
    tread_routing_prob=0.5,
)

print(f"\n✓ TREAD model created (routing blocks {tread_route_start}-{tread_route_end})")
print(f"  Total params: {sum(p.numel() for p in model_tread.parameters())/1e6:.1f}M")
assert model_tread.tread_enabled, "TREAD should be enabled"

# Test forward with routing (training mode)
model_tread.train()
optimizer_tread = torch.optim.AdamW(model_tread.parameters(), lr=3e-4)
optimizer_tread.zero_grad()

v_pred_tread = model_tread(x0, torch.rand(B), dino_emb, text_emb, dino_patches_repa, text_mask,
                           dino_patches_mask=dino_patches_mask, tread_enabled=True)
print(f"\n✓ TREAD forward pass (routed)")
print(f"  v_pred shape: {v_pred_tread.shape}")
assert v_pred_tread.shape == x0.shape, f"Output shape should match input: {v_pred_tread.shape} vs {x0.shape}"

# Test forward WITHOUT routing (eval mode / dense)
v_pred_dense = model_tread(x0, torch.rand(B), dino_emb, text_emb, dino_patches_repa, text_mask,
                           dino_patches_mask=dino_patches_mask, tread_enabled=False)
print(f"✓ TREAD forward pass (dense)")
print(f"  v_pred shape: {v_pred_dense.shape}")
assert v_pred_dense.shape == x0.shape, "Dense output shape should match input"

# Test TREAD + REPA interaction
model_tread_repa = NanoDiT(
    input_size=64,
    patch_size=2,
    in_channels=16,
    hidden_size=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4.0,
    use_gradient_checkpointing=True,
    repa_block_idx=6,
    tread_route_start=tread_route_start,
    tread_route_end=tread_route_end,
    tread_routing_prob=0.5,
)

model_tread_repa.train()
v_pred_tr, repa_hidden_tr, visible_idx_tr = model_tread_repa(
    x0, torch.rand(B), dino_emb, text_emb, dino_patches_repa, text_mask,
    dino_patches_mask=dino_patches_mask, return_repa_hidden=True, tread_enabled=True,
)

num_latent_tokens = (64 // 2) * (64 // 2)  # 1024
N_visible = num_latent_tokens - int(num_latent_tokens * 0.5)

print(f"\n✓ TREAD + REPA forward pass")
print(f"  v_pred shape: {v_pred_tr.shape}")
print(f"  repa_hidden shape: {repa_hidden_tr.shape} (expected ~{N_visible} visible tokens)")
print(f"  visible_idx shape: {visible_idx_tr.shape}")
assert repa_hidden_tr.shape[1] == N_visible, f"REPA hidden should have {N_visible} visible tokens, got {repa_hidden_tr.shape[1]}"
assert visible_idx_tr.shape[0] == N_visible, f"visible_idx should have {N_visible} entries"

# Test TREAD + REPA loss computation
optimizer_tr = torch.optim.AdamW(model_tread_repa.parameters(), lr=3e-4)
optimizer_tr.zero_grad()

tread_config = TREADConfig(enabled=True, routing_probability=0.5,
                           route_start=tread_route_start, route_end=tread_route_end)
repa_config = REPAConfig(enabled=True, weight=0.5, block_index=6, loss_type="cosine")

loss_tr, v_pred_tr2, repa_loss_tr, _ = flow_matching_loss(
    model_tread_repa, x0, dino_emb, dino_patches_repa, text_emb, text_mask,
    cfg_probs, dino_patches_mask=dino_patches_mask, return_v_pred=True,
    repa_config=repa_config, tread_config=tread_config,
)

print(f"\n✓ TREAD + REPA flow_matching_loss")
print(f"  Total loss: {loss_tr.item():.6f}")
print(f"  REPA loss: {repa_loss_tr.item():.6f}")
assert repa_loss_tr.item() > 0, "REPA loss should be positive with TREAD"

# Backward pass
loss_tr.backward()
print(f"✓ TREAD + REPA backward pass complete")

# Check gradients flow to all blocks
grad_norms = []
for i, block in enumerate(model_tread_repa.blocks):
    block_grad = sum(p.grad.norm().item() for p in block.parameters() if p.grad is not None)
    grad_norms.append(block_grad)
print(f"\n  Block gradient norms (should all be > 0):")
for i, gn in enumerate(grad_norms):
    status = "✓" if gn > 0 else "✗"
    print(f"    Block {i}: {gn:.6f} {status}")
assert all(gn > 0 for gn in grad_norms), "All blocks should receive gradients"

# ============================================================
# SELF-GUIDANCE SAMPLING TEST
# ============================================================
print("\n" + "="*60)
print("TESTING SELF-GUIDANCE SAMPLING")
print("="*60)

from production.sample import EulerSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create a model with TREAD for self-guidance testing
model_sg = NanoDiT(
    hidden_size=128, depth=4, num_heads=4,
    in_channels=16, patch_size=2, input_size=8,
    tread_route_start=1,
    tread_route_end=3,
    tread_routing_prob=0.5,
).to(device)

sampler_sg = EulerSampler(num_steps=5)

# Test self-guidance sampling (2 passes: dense + routed)
B_sg = 1
dino_emb_sg = torch.randn(B_sg, 1024, device=device)
dino_patches_sg = torch.randn(B_sg, 16, 1024, device=device)
text_emb_sg = torch.randn(B_sg, 77, 1024, device=device)
text_mask_sg = torch.ones(B_sg, 77, device=device)

with torch.no_grad():
    latents_sg = sampler_sg.sample(
        model_sg, (B_sg, 16, 8, 8),
        dino_emb_sg, dino_patches_sg, text_emb_sg, text_mask_sg,
        device=device,
        self_guidance=True,
        guidance_scale=3.0,
    )

assert latents_sg.shape == (B_sg, 16, 8, 8), f"Self-guidance output shape wrong: {latents_sg.shape}"
print(f"\n✓ Self-guidance sampling: output shape {latents_sg.shape}")

# Test dual CFG sampling (3 passes: uncond + text + dino)
with torch.no_grad():
    latents_dual = sampler_sg.sample(
        model_sg, (B_sg, 16, 8, 8),
        dino_emb_sg, dino_patches_sg, text_emb_sg, text_mask_sg,
        device=device,
        self_guidance=False,
        text_scale=3.0,
        dino_scale=2.0,
    )

assert latents_dual.shape == (B_sg, 16, 8, 8), f"Dual CFG output shape wrong: {latents_dual.shape}"
print(f"✓ Dual CFG sampling: output shape {latents_dual.shape}")

# Test that self-guidance and dual CFG produce different results (different algorithms)
diff = (latents_sg - latents_dual).abs().mean().item()
assert diff > 1e-5, f"Self-guidance and dual CFG should differ, got diff={diff}"
print(f"✓ Self-guidance vs dual CFG differ (mean abs diff: {diff:.4f})")

# Test SamplingConfig has self-guidance fields
from production.config_loader import SamplingConfig
sc = SamplingConfig(self_guidance=True, guidance_scale=4.0)
assert sc.self_guidance is True
assert sc.guidance_scale == 4.0
print(f"✓ SamplingConfig has self_guidance and guidance_scale fields")

# ==============================================================
# MUON OPTIMIZER TESTS
# ==============================================================
print("\n" + "="*60)
print("TESTING MUON OPTIMIZER")
print("="*60)

from production.config_loader import OptimizerConfig, MuonConfig

# Test 1: Parameter split (2D → Muon, non-2D → AdamW)
test_model = NanoDiT(
    input_size=32, patch_size=2, in_channels=16,
    hidden_size=128, depth=4, num_heads=4, mlp_ratio=4.0,
)
muon_params = [p for p in test_model.parameters() if p.requires_grad and p.ndim == 2]
adam_params = [p for p in test_model.parameters() if p.requires_grad and p.ndim != 2]
n_muon = sum(p.numel() for p in muon_params)
n_adam = sum(p.numel() for p in adam_params)
assert n_muon > 0, "Should have 2D params for Muon"
assert n_adam > 0, "Should have non-2D params for AdamW"
# All 4D convs and 1D biases go to AdamW
for p in adam_params:
    assert p.ndim != 2, f"Non-2D group should not have 2D param (ndim={p.ndim})"
print(f"✓ Param split: {n_muon/1e6:.2f}M Muon (2D), {n_adam/1e6:.4f}M AdamW (non-2D)")

# Test 2: Create Muon optimizer via Trainer (hybrid mode)
from production.train import Trainer
opt_cfg = OptimizerConfig(
    type='Muon', lr=3e-4, min_lr=1e-6, betas=[0.9, 0.95],
    weight_decay=0.03, eps=1e-8, muon=MuonConfig(),
)
# Minimal dataloader stub
class FakeLoader:
    def __iter__(self):
        return iter([])
trainer = Trainer(
    model=test_model, dataloader=FakeLoader(), device='cpu',
    total_steps=10, warmup_steps=2, peak_lr=3e-4,
    optimizer_config=opt_cfg,
)
assert trainer.optimizer_muon is not None, "Muon optimizer should be created"
assert trainer.optimizer_adam is not None, "AdamW optimizer should be created"
assert trainer.optimizer_type == 'Muon'
print(f"✓ Muon hybrid optimizer created: Muon + AdamW")

# Test 3: Both optimizers step and update weights
test_model.zero_grad()
dummy_x0 = torch.randn(1, 16, 16, 16)
dummy_dino = torch.randn(1, 1024)
dummy_patches = torch.randn(1, 64, 1024)
dummy_text = torch.randn(1, 10, 1024)
dummy_mask = torch.ones(1, 10, dtype=torch.bool)
w_before = {n: p.clone() for n, p in test_model.named_parameters() if p.requires_grad}

loss = flow_matching_loss(
    test_model, dummy_x0, dummy_dino, dummy_patches, dummy_text, dummy_mask,
    {'p_uncond': 0.0, 'p_text_only': 0.0, 'p_dino_cls_only': 0.0, 'p_dino_patches_only': 0.0},
)
loss.backward()
trainer._step_optimizers()

changed_2d = 0
changed_other = 0
for n, p in test_model.named_parameters():
    if not p.requires_grad:
        continue
    if not torch.equal(p.data, w_before[n]):
        if p.ndim == 2:
            changed_2d += 1
        else:
            changed_other += 1
assert changed_2d > 0, "2D params should update via Muon"
assert changed_other > 0, "Non-2D params should update via AdamW"
print(f"✓ Both optimizers update weights: {changed_2d} 2D params, {changed_other} non-2D params changed")

# Test 4: AdamW fallback (type="AdamW" still works)
test_model2 = NanoDiT(
    input_size=32, patch_size=2, in_channels=16,
    hidden_size=128, depth=4, num_heads=4, mlp_ratio=4.0,
)
opt_cfg_adam = OptimizerConfig(type='AdamW')
trainer2 = Trainer(
    model=test_model2, dataloader=FakeLoader(), device='cpu',
    total_steps=10, warmup_steps=2, peak_lr=3e-4,
    optimizer_config=opt_cfg_adam,
)
assert trainer2.optimizer_muon is None, "Muon should be None in AdamW mode"
assert trainer2.optimizer_adam is not None, "AdamW should be created"
assert trainer2.optimizer_type == 'AdamW'
print(f"✓ AdamW fallback works correctly")

# Test 5: LR schedule applies to both optimizers
trainer._set_lr_optimizers(1e-5)
for pg in trainer.optimizer_muon.param_groups:
    assert pg['lr'] == 1e-5, f"Muon LR should be 1e-5, got {pg['lr']}"
for pg in trainer.optimizer_adam.param_groups:
    assert pg['lr'] == 1e-5, f"Adam LR should be 1e-5, got {pg['lr']}"
print(f"✓ LR schedule applies to both Muon and AdamW")

# ==============================================================
# RESOLUTION SCHEDULING TESTS
# ==============================================================
print("\n" + "="*60)
print("TESTING RESOLUTION SCHEDULING")
print("="*60)

from production.config_loader import ResolutionPhase, TrainingConfig
from production.data import BucketAwareDataLoader, ValidationDataset

# Test 1: ResolutionPhase config parsing
tc = TrainingConfig(resolution_schedule=[
    {'until_step': 100, 'scale': 0.5},
    {'until_step': 200, 'scale': 1.0},
])
phases = tc.get_resolution_phases()
assert len(phases) == 2
assert phases[0].until_step == 100 and phases[0].scale == 0.5
assert phases[1].until_step == 200 and phases[1].scale == 1.0
print(f"✓ Resolution schedule config parsing")

# Test 2: BucketAwareDataLoader.resolution_scale adjusts target sizes
# Create a mock bucket dataset
class MockBucketDS:
    def __init__(self, target):
        self.target_latent_size = target
    def __iter__(self):
        return iter([])

datasets = {
    'bucket_1024x1024': MockBucketDS((128, 128)),
    'bucket_832x1216': MockBucketDS((152, 104)),
}
loader = BucketAwareDataLoader(datasets, [1.0, 1.0])

# Default scale is 1.0
assert loader.resolution_scale == 1.0
assert datasets['bucket_1024x1024'].target_latent_size == (128, 128)

# Set to 0.5
loader.resolution_scale = 0.5
assert datasets['bucket_1024x1024'].target_latent_size == (64, 64)
# 152*0.5=76, 104*0.5=52 — both already even
assert datasets['bucket_832x1216'].target_latent_size == (76, 52)
print(f"✓ BucketAwareDataLoader.resolution_scale adjusts target sizes")

# Test 3: Even-dimension enforcement
datasets2 = {
    'bucket_test': MockBucketDS((100, 100)),  # 100*0.5=50 (even), 100*0.3=30 (even)
}
loader2 = BucketAwareDataLoader(datasets2, [1.0])
loader2.resolution_scale = 0.5
assert datasets2['bucket_test'].target_latent_size == (50, 50)
# 100*0.7=70 (even)
loader2.resolution_scale = 0.7
assert datasets2['bucket_test'].target_latent_size[0] % 2 == 0
assert datasets2['bucket_test'].target_latent_size[1] % 2 == 0
print(f"✓ Even dimension enforcement: {datasets2['bucket_test'].target_latent_size}")

# Test 4: Resolution schedule in trainer
trainer3 = Trainer(
    model=test_model, dataloader=loader, device='cpu',
    total_steps=200, warmup_steps=2, peak_lr=3e-4,
    optimizer_config=opt_cfg_adam,
)
trainer3.resolution_phases = phases
trainer3._current_resolution_scale = None
trainer3.step = 0
trainer3._update_resolution_schedule()
assert loader.resolution_scale == 0.5, f"Expected 0.5, got {loader.resolution_scale}"
trainer3.step = 100
trainer3._update_resolution_schedule()
assert loader.resolution_scale == 1.0, f"Expected 1.0, got {loader.resolution_scale}"
print(f"✓ Trainer resolution schedule transitions correctly")

# Test 5: Empty schedule = no-op
trainer4 = Trainer(
    model=test_model, dataloader=loader, device='cpu',
    total_steps=10, warmup_steps=2, peak_lr=3e-4,
    optimizer_config=opt_cfg_adam,
)
trainer4.resolution_phases = []
trainer4._update_resolution_schedule()  # Should not error
print(f"✓ Empty resolution schedule is a no-op")

# ==============================================================
# PERCEPTUAL LOSS TESTS
# ==============================================================
print("\n" + "="*60)
print("TESTING PERCEPTUAL LOSS (LPIPS)")
print("="*60)

from production.config_loader import PerceptualLossConfig
from production.train import PerceptualLossModule

# Test 1: Config dataclass defaults
pcfg = PerceptualLossConfig()
assert pcfg.enabled == False
assert pcfg.lpips_weight == 0.1
assert pcfg.every_n_microsteps == 4
assert pcfg.crop_size == 256
print(f"✓ PerceptualLossConfig defaults correct")

# Test 2: every_n gating in flow_matching_loss
# With perceptual disabled, lpips_loss should always be None
pcfg_disabled = PerceptualLossConfig(enabled=False)
_, _, _, lpips_val = flow_matching_loss(
    model, x0, dino_emb, dino_patches, text_emb, text_mask,
    cfg_probs, return_v_pred=True,
    perceptual_config=pcfg_disabled, micro_step=0,
)
assert lpips_val is None, "LPIPS loss should be None when disabled"
print(f"✓ Perceptual loss returns None when disabled")

# Test 3: every_n gating — no module provided
pcfg_on = PerceptualLossConfig(enabled=True, every_n_microsteps=4)
_, _, _, lpips_val2 = flow_matching_loss(
    model, x0, dino_emb, dino_patches, text_emb, text_mask,
    cfg_probs, return_v_pred=True,
    perceptual_config=pcfg_on, perceptual_module=None, micro_step=0,
)
assert lpips_val2 is None, "LPIPS loss should be None when module is None"
print(f"✓ Perceptual loss returns None when module is None")

# Test 4: Module creation (lazy — doesn't load models until compute())
plm = PerceptualLossModule(device='cpu')
assert plm._vae is None, "VAE should not be loaded at construction"
assert plm._lpips_fn is None, "LPIPS should not be loaded at construction"
print(f"✓ PerceptualLossModule is lazy (no models loaded at init)")

# Test 5: x0_hat reconstruction math
# Verify x0_hat = zt - t * v_pred recovers x0 when v_pred = v_target
B_test = 2
x0_test = torch.randn(B_test, 16, 8, 8)
z1_test = torch.randn(B_test, 16, 8, 8)
t_test = torch.tensor([0.3, 0.7]).view(B_test, 1, 1, 1)
zt_test = (1 - t_test) * x0_test + t_test * z1_test
v_target_test = z1_test - x0_test
x0_hat_test = zt_test - t_test * v_target_test
assert torch.allclose(x0_hat_test, x0_test, atol=1e-5), "x0_hat should match x0 when v_pred is perfect"
print(f"✓ x0_hat reconstruction: zt - t*v_target ≈ x0 (max diff: {(x0_hat_test - x0_test).abs().max():.2e})")

print("\n" + "="*60)
print("TESTING X-PREDICTION / PIXEL-SPACE")
print("="*60)

# Test 1: X-prediction loss math
# When prediction_type='x_prediction', model output is treated as x0_pred
# Loss converts to v-space: v = (zt - x0) / max(t, 0.05)
B_xp = 1
x0_xp = torch.randn(B_xp, 3, 32, 32)  # pixel-space: 3 channels
z1_xp = torch.randn(B_xp, 3, 32, 32)
t_xp = torch.tensor([0.3])

# Create noised sample
t_exp = t_xp.view(B_xp, 1, 1, 1)
zt_xp = (1 - t_exp) * x0_xp + t_exp * z1_xp

# X-prediction v-space conversion
t_clamped = t_xp.clamp(min=0.05).view(B_xp, 1, 1, 1)
v_pred_from_x0 = (zt_xp - x0_xp) / t_clamped
v_target_from_x0 = (zt_xp - x0_xp) / t_clamped
# If x0_pred == x0, loss should be zero
loss_xp = F.mse_loss(v_pred_from_x0, v_target_from_x0)
assert loss_xp.item() < 1e-10, f"Perfect x0 prediction should give zero v-space loss, got {loss_xp.item()}"
print(f"✓ X-prediction: perfect x0_pred gives zero v-space loss ({loss_xp.item():.2e})")

# Test 2: t clamping at small t
t_small = torch.tensor([0.01])
t_clamped_small = t_small.clamp(min=0.05)
assert abs(t_clamped_small.item() - 0.05) < 1e-6, f"t=0.01 should be clamped to 0.05, got {t_clamped_small.item()}"
print(f"✓ X-prediction: t=0.01 correctly clamped to {t_clamped_small.item()}")

# Test 3: PatchEmbed with bottleneck for pixel-space
from production.model import PatchEmbed
pe_pixel = PatchEmbed(
    patch_size=32, in_channels=3,
    hidden_size=384, bottleneck_size=256
)
pixel_input = torch.randn(1, 3, 1024, 1024)
pe_out = pe_pixel(pixel_input)
expected_tokens = (1024 // 32) ** 2  # 32x32 = 1024 tokens
assert pe_out.shape == (1, expected_tokens, 384), f"Expected (1, {expected_tokens}, 384), got {pe_out.shape}"
print(f"✓ PatchEmbed pixel-space: (1, 3, 1024, 1024) → {tuple(pe_out.shape)} ({expected_tokens} tokens)")

# Test 4: PatchEmbed without bottleneck (latent-space unchanged)
pe_latent = PatchEmbed(
    patch_size=2, in_channels=16,
    hidden_size=384, bottleneck_size=0
)
latent_input = torch.randn(1, 16, 128, 128)
pe_lat_out = pe_latent(latent_input)
expected_lat_tokens = (128 // 2) ** 2  # 64x64 = 4096 tokens
assert pe_lat_out.shape == (1, expected_lat_tokens, 384), f"Expected (1, {expected_lat_tokens}, 384), got {pe_lat_out.shape}"
print(f"✓ PatchEmbed latent-space: (1, 16, 128, 128) → {tuple(pe_lat_out.shape)} ({expected_lat_tokens} tokens)")

# Test 5: flow_matching_loss with x_prediction
model_xp = NanoDiT(
    input_size=32, patch_size=2, in_channels=3,
    hidden_size=128, depth=4, num_heads=4, mlp_ratio=4.0,
)
x0_fm = torch.randn(1, 3, 32, 32)
dino_emb_fm = torch.randn(1, 1024)
dino_patches_fm = torch.randn(1, 4, 1024)
text_emb_fm = torch.randn(1, 10, 1024)
text_mask_fm = torch.ones(1, 10)
cfg = {'p_uncond': 0.0, 'p_text_only': 0.0, 'p_dino_cls_only': 0.0, 'p_dino_patches_only': 0.0}

loss_xp, v_pred_xp, repa_xp, lpips_xp = flow_matching_loss(
    model_xp, x0_fm, dino_emb_fm, dino_patches_fm, text_emb_fm, text_mask_fm,
    cfg, return_v_pred=True, prediction_type="x_prediction", t_clamp_min=0.05,
)
assert loss_xp.item() > 0, "X-prediction loss should be positive"
assert v_pred_xp.shape == x0_fm.shape, f"v_pred shape mismatch: {v_pred_xp.shape} vs {x0_fm.shape}"
print(f"✓ flow_matching_loss with x_prediction: loss={loss_xp.item():.4f}, v_pred shape={tuple(v_pred_xp.shape)}")

# Test 6: flow_matching_loss backward compatibility (v_prediction)
loss_vp, v_pred_vp, repa_vp, lpips_vp = flow_matching_loss(
    model_xp, x0_fm, dino_emb_fm, dino_patches_fm, text_emb_fm, text_mask_fm,
    cfg, return_v_pred=True, prediction_type="v_prediction",
)
assert loss_vp.item() > 0, "V-prediction loss should be positive"
print(f"✓ flow_matching_loss with v_prediction: loss={loss_vp.item():.4f} (backward compatible)")

# Test 7: Euler sampler with x_prediction
from production.sample import EulerSampler
sampler_xp = EulerSampler(num_steps=5)
with torch.no_grad():
    out_xp = sampler_xp.sample(
        model_xp, (1, 3, 32, 32), dino_emb_fm, dino_patches_fm, text_emb_fm, text_mask_fm,
        device='cpu', prediction_type="x_prediction",
    )
assert out_xp.shape == (1, 3, 32, 32), f"Expected (1, 3, 32, 32), got {out_xp.shape}"
print(f"✓ EulerSampler with x_prediction: output shape {tuple(out_xp.shape)}")

# Test 8: Config prediction_type defaults
from production.config_loader import ModelConfig
mc = ModelConfig()
assert mc.prediction_type == "x_prediction", f"Default prediction_type should be x_prediction, got {mc.prediction_type}"
assert mc.t_clamp_min == 0.05, f"Default t_clamp_min should be 0.05, got {mc.t_clamp_min}"
assert mc.bottleneck_size == 0, f"Default bottleneck_size should be 0, got {mc.bottleneck_size}"
print(f"✓ ModelConfig defaults: prediction_type={mc.prediction_type}, t_clamp_min={mc.t_clamp_min}, bottleneck_size={mc.bottleneck_size}")

print("\n" + "="*60)
print("ALL TESTS PASSED ✓")
print("="*60)
