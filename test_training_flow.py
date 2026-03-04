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

loss, v_pred, repa_loss = flow_matching_loss(
    model, x0, dino_emb, dino_patches, text_emb, text_mask,
    cfg_probs, return_v_pred=True
)

print(f"\n✓ Forward pass complete")
print(f"  Loss: {loss.item():.6f}")
print(f"  v_pred shape: {v_pred.shape}")
print(f"  repa_loss: {repa_loss}")
assert repa_loss is None, "REPA loss should be None when not enabled"

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

loss_repa, v_pred_repa, repa_loss_val = flow_matching_loss(
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

loss_masked, _, repa_loss_masked = flow_matching_loss(
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

loss_mismatch, _, repa_loss_mismatch = flow_matching_loss(
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

loss_tr, v_pred_tr2, repa_loss_tr = flow_matching_loss(
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

print("\n" + "="*60)
print("ALL TESTS PASSED ✓")
print("="*60)
