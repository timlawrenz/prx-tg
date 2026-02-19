"""Test model forward pass to check for obvious bugs."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))

from model import NanoDiT

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = NanoDiT(
        hidden_size=384,
        depth=12,
        num_heads=6,
        patch_size=2,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Create dummy inputs
    B, C, H, W = 2, 16, 128, 128  # 1024x1024 image latent
    
    x = torch.randn(B, C, H, W, device=device)
    t = torch.rand(B, device=device)
    dino_emb = torch.randn(B, 1024, device=device)
    text_emb = torch.randn(B, 512, 1024, device=device)
    text_mask = torch.ones(B, 512, device=device)
    
    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  t: {t.shape}")
    print(f"  dino_emb: {dino_emb.shape}")
    print(f"  text_emb: {text_emb.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        v_pred = model(x, t, dino_emb, text_emb, text_mask)
    
    print(f"\nOutput shape: {v_pred.shape}")
    print(f"Expected shape: {x.shape}")
    print(f"Shape match: {v_pred.shape == x.shape}")
    
    # Check for NaNs
    print(f"\nNaNs in output: {torch.isnan(v_pred).any().item()}")
    print(f"Infs in output: {torch.isinf(v_pred).any().item()}")
    
    # Check output statistics
    print(f"\nOutput statistics:")
    print(f"  Mean: {v_pred.mean().item():.4f}")
    print(f"  Std: {v_pred.std().item():.4f}")
    print(f"  Min: {v_pred.min().item():.4f}")
    print(f"  Max: {v_pred.max().item():.4f}")
    
    # Check if output has spatial structure (not uniform noise)
    # Compare variance within patches vs across patches
    patch_size = 2
    h_patches = H // patch_size
    w_patches = W // patch_size
    
    # Reshape to patches
    v_patches = v_pred.view(B, C, h_patches, patch_size, w_patches, patch_size)
    v_patches = v_patches.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, h_patches, w_patches, C, patch_size, patch_size)
    
    # Variance within patches
    within_var = v_patches.var(dim=(3, 4, 5)).mean().item()
    
    # Variance across patches
    patch_means = v_patches.mean(dim=(3, 4, 5))  # (B, h_patches, w_patches)
    across_var = patch_means.var().item()
    
    print(f"\nSpatial structure:")
    print(f"  Variance within patches: {within_var:.6f}")
    print(f"  Variance across patches: {across_var:.6f}")
    print(f"  Ratio (across/within): {across_var / max(within_var, 1e-8):.4f}")
    print(f"  (Higher ratio = more spatial structure, lower = more uniform)")
    
    print("\n✓ Model forward pass successful")

if __name__ == '__main__':
    main()
