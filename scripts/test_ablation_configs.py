
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from production.model import NanoDiT

def test_ablation_configs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    B = 2
    H, W = 1024, 1024
    patch_size = 16
    hidden_size = 384
    dino_dim = 1024
    text_dim = 1024
    num_patches = 4096
    
    x = torch.randn(B, 3, H, W, device=device)
    t = torch.rand(B, device=device)
    dino = torch.randn(B, dino_dim, device=device)
    text = torch.randn(B, 512, text_dim, device=device)
    text_mask = torch.ones(B, 512, device=device, dtype=torch.long)
    dino_patches = torch.randn(B, num_patches, dino_dim, device=device)
    pose = torch.randn(B, 133, 3, device=device)
    pose[:, :, 2] = torch.rand(B, 133, device=device)
    
    print("Testing Base Configuration (with DINO patches)...")
    model_base = NanoDiT(
        input_size=1024, patch_size=patch_size, in_channels=3,
        hidden_size=hidden_size, depth=4, num_heads=6,
        dino_patches_enabled=True,
        dino_pool_factor=None,
        tread_route_start=1, tread_route_end=2, tread_routing_prob=0.5
    ).to(device)
    
    with torch.no_grad():
        out = model_base(x, t, dino, text, text_mask=text_mask, dino_patches=dino_patches, pose_kpts=pose)
    print(f"  ✓ Base forward pass succeeded. Output shape: {out.shape}")
    
    print("\nTesting Arm L Configuration (DINO pooling 2x2)...")
    model_pool = NanoDiT(
        input_size=1024, patch_size=patch_size, in_channels=3,
        hidden_size=hidden_size, depth=4, num_heads=6,
        dino_patches_enabled=True,
        dino_pool_factor=2,
        tread_route_start=1, tread_route_end=2, tread_routing_prob=0.5
    ).to(device)
    
    with torch.no_grad():
        out = model_pool(x, t, dino, text, text_mask=text_mask, dino_patches=dino_patches, pose_kpts=pose)
    print(f"  ✓ Pooled forward pass succeeded. Output shape: {out.shape}")
    
    print("\nTesting Arm M Configuration (No DINO patches)...")
    model_no_dino = NanoDiT(
        input_size=1024, patch_size=patch_size, in_channels=3,
        hidden_size=hidden_size, depth=4, num_heads=6,
        dino_patches_enabled=False,
        dino_pool_factor=None,
        tread_route_start=1, tread_route_end=2, tread_routing_prob=0.5
    ).to(device)
    
    with torch.no_grad():
        out = model_no_dino(x, t, dino, text, text_mask=text_mask, dino_patches=None, pose_kpts=pose)
    print(f"  ✓ No-DINO-patch forward pass succeeded. Output shape: {out.shape}")
    
    print("\nAll tests passed!")

if __name__ == '__main__':
    test_ablation_configs()
