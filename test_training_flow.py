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
    'p_drop_both': 0.1,
    'p_drop_text': 0.05,
    'p_drop_dino': 0.3,
}

print("\n✓ Batch data created")
print(f"  x0: {x0.shape}")
print(f"  dino_patches: {dino_patches.shape}")

# Training step
model.train()
optimizer.zero_grad()

print("\n" + "="*60)
print("RUNNING TRAINING STEP")
print("="*60)

loss, v_pred = flow_matching_loss(
    model, x0, dino_emb, dino_patches, text_emb, text_mask,
    cfg_probs, return_v_pred=True
)

print(f"\n✓ Forward pass complete")
print(f"  Loss: {loss.item():.6f}")
print(f"  v_pred shape: {v_pred.shape}")

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
