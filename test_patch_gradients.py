"""Test if gradients actually flow to dino_patch_proj."""
import torch
import torch.nn.functional as F
from production.model import NanoDiT

print("Creating model...")
model = NanoDiT(
    input_size=64,
    patch_size=2,
    in_channels=16,
    hidden_size=384,
    depth=12,
    num_heads=6,
)
model.train()

print("\nInitial patch projection weights:")
initial_weight = model.dino_patch_proj.weight.data.clone()
initial_bias = model.dino_patch_proj.bias.data.clone()
print(f"  weight: mean={initial_weight.mean():.6f}, std={initial_weight.std():.6f}")
print(f"  bias: mean={initial_bias.mean():.6f}, std={initial_bias.std():.6f}")

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Simulate one training step
B = 1
x = torch.randn(B, 16, 64, 64)
t = torch.rand(B)
dino_emb = torch.randn(B, 1024)
dino_patches = torch.randn(B, 4096, 1024)  # Real patches
text_emb = torch.randn(B, 500, 1024)
text_mask = torch.ones(B, 500)

# Forward pass
v_pred = model(x, t, dino_emb, text_emb, dino_patches, text_mask)

# Compute loss
v_target = torch.randn_like(v_pred)
loss = F.mse_loss(v_pred, v_target)

print(f"\nLoss: {loss.item():.6f}")

# Backward
optimizer.zero_grad()
loss.backward()

# Check gradients
if model.dino_patch_proj.weight.grad is not None:
    grad_norm = model.dino_patch_proj.weight.grad.norm().item()
    print(f"\n✓ Gradient exists!")
    print(f"  grad norm: {grad_norm:.6f}")
    print(f"  grad mean: {model.dino_patch_proj.weight.grad.mean():.6f}")
    print(f"  grad std: {model.dino_patch_proj.weight.grad.std():.6f}")
else:
    print(f"\n✗ NO GRADIENT!")

# Apply update
optimizer.step()

# Check if weights changed
new_weight = model.dino_patch_proj.weight.data
new_bias = model.dino_patch_proj.bias.data
weight_diff = (new_weight - initial_weight).abs().max().item()
bias_diff = (new_bias - initial_bias).abs().max().item()

print(f"\nAfter 1 optimizer step:")
print(f"  weight max change: {weight_diff:.8f}")
print(f"  bias max change: {bias_diff:.8f}")

if weight_diff > 1e-6:
    print("\n✓ Weights ARE changing - gradient flow is working!")
else:
    print("\n✗ Weights NOT changing - gradient flow is BROKEN!")
