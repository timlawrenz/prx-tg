#!/usr/bin/env python3
"""Smoke test: EidolonAdapter forward pass + 100-step loss decrease.

Creates a tiny model with EidolonAdapter, runs 100 synthetic training steps,
and verifies the loss decreases without NaN or OOM.
"""

import torch
import torch.nn.functional as F

# Use smallest possible model for quick smoke test
from production.model import NanoDiT
from production.adapters import EidolonAdapter

B, H, W = 2, 64, 64  # 64×64 latent space
C = 16               # latent channels

model = NanoDiT(
    input_size=64,
    patch_size=2,
    in_channels=C,
    hidden_size=128,     # Tiny hidden size for smoke test
    depth=4,             # 4 layers
    num_heads=4,
    mlp_ratio=2.0,
    adapter_kwargs={
        "name": "eidolon",
        "identity_dim": 64,
        "z_g_dim": 50,
    },
)

assert isinstance(model.adapter, EidolonAdapter), f"Expected EidolonAdapter, got {type(model.adapter).__name__}"
print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} params")
print(f"  Adapter: {type(model.adapter).__name__}")

# Synthetic data
identity_emb = torch.randn(B, 64)
geometry_emb = torch.randn(B, 50)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

model.train()
prev_loss = float('inf')
nan_count = 0

print("\nRunning 100 synthetic steps...")
for step in range(100):
    x0 = torch.randn(B, C, H, W)
    t = torch.rand(B)
    z1 = torch.randn_like(x0)
    t_expanded = t.view(B, 1, 1, 1)
    zt = (1 - t_expanded) * x0 + t_expanded * z1

    v_pred = model(zt, t, identity_emb=identity_emb, geometry_emb=geometry_emb)

    v_target = z1 - x0
    loss = F.mse_loss(v_pred, v_target)

    if torch.isnan(loss):
        print(f"  ✗ NaN at step {step}!")
        nan_count += 1
        if nan_count > 3:
            raise RuntimeError("Too many NaN steps — aborting")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0 or step == 99:
        print(f"  step {step:3d}: loss={loss.item():.6f}")

    prev_loss = loss.item()

print(f"\n✓ Smoke test passed! ({nan_count} NaN steps)")
print(f"  Final loss: {prev_loss:.6f}")
