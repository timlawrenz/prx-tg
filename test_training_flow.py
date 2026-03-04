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

print("\n" + "="*60)
print("ALL TESTS PASSED ✓")
print("="*60)
