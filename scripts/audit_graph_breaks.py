#!/usr/bin/env python3
"""Graph break audit for torch.compile on the NanoDiT model.

Runs a single forward + backward pass with TORCH_LOGS=graph_breaks to identify
where the compiler falls back to eager execution. Each graph break reduces the
effectiveness of operation fusion and hurts bandwidth-constrained architectures.

Usage:
    # Full audit (shows all graph breaks with source locations):
    TORCH_LOGS="graph_breaks" python scripts/audit_graph_breaks.py

    # Verbose mode (also show generated Triton code):
    TORCH_LOGS="graph_breaks,output_code" python scripts/audit_graph_breaks.py

    # Quick check (just count breaks):
    python scripts/audit_graph_breaks.py --quick
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_dummy_batch(config, device, batch_size=2):
    """Create a synthetic batch matching the production data format."""
    import torch

    model_cfg = config.model
    hidden = model_cfg.hidden_size
    patch_size = model_cfg.patch_size

    # Use 256x256 for fast audit (smallest resolution)
    H, W = 256, 256
    h_patches = H // patch_size
    w_patches = W // patch_size
    num_patches = h_patches * w_patches

    B = batch_size
    # Conditioning dims match NanoDiT defaults (not hidden_size)
    dino_dim = 1024
    dino_patch_dim = 1024
    text_dim = 1024
    return {
        'image_data': torch.randn(B, 3, H, W, device=device),
        'dino_embedding': torch.randn(B, dino_dim, device=device),
        'dinov3_patches': torch.randn(B, num_patches, dino_patch_dim, device=device),
        'dinov3_patches_mask': torch.ones(B, num_patches, dtype=torch.long, device=device),
        't5_hidden': torch.randn(B, 512, text_dim, device=device),
        't5_mask': torch.ones(B, 512, dtype=torch.long, device=device),
        'pose_keypoints': torch.randn(B, 133, 3, device=device),
    }


def run_audit(config_path: str, quick: bool = False, device: str = 'cuda'):
    import torch
    from production.config_loader import load_config
    from production.model import NanoDiT
    from production.train import flow_matching_loss

    config = load_config(config_path)
    model_cfg = config.model
    training_cfg = config.training

    print(f"=== Graph Break Audit ===")
    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    hip = getattr(torch.version, 'hip', None)
    if hip:
        print(f"ROCm/HIP: {hip}")
    print()

    # Build model
    model = NanoDiT(
        in_channels=model_cfg.in_channels,
        hidden_size=model_cfg.hidden_size,
        depth=model_cfg.depth,
        num_heads=model_cfg.num_heads,
        patch_size=model_cfg.patch_size,
        mlp_ratio=model_cfg.mlp_ratio,
        use_gradient_checkpointing=False,  # Never checkpoint during audit
        bottleneck_size=model_cfg.bottleneck_size,
        num_pose_joints=model_cfg.num_pose_joints,
        pose_confidence_threshold=model_cfg.pose_confidence_threshold,
    ).to(device)

    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # Compile
    print(f"\nCompiling with fullgraph=True (strict mode to detect ALL breaks)...")
    t0 = time.time()
    try:
        compiled = torch.compile(model, dynamic=True, fullgraph=True, mode='default')
        print(f"  fullgraph=True compilation accepted in {time.time()-t0:.1f}s")
        fullgraph_ok = True
    except Exception as e:
        print(f"  fullgraph=True FAILED: {e}")
        fullgraph_ok = False
        compiled = torch.compile(model, dynamic=True, fullgraph=False, mode='default')
        print(f"  Falling back to fullgraph=False")

    # Create dummy data
    batch = create_dummy_batch(config, device)

    # Forward pass
    print(f"\nRunning forward pass (watch for graph_breaks in logs above)...")
    cfg_probs = training_cfg.cfg_dropout.to_dict()
    tread_cfg = training_cfg.tread if training_cfg.tread.enabled else None

    dtype = torch.bfloat16 if training_cfg.precision == 'bfloat16' else torch.float16

    t0 = time.time()
    with torch.amp.autocast('cuda', dtype=dtype):
        loss = flow_matching_loss(
            compiled,
            batch['image_data'],
            batch['dino_embedding'],
            batch['dinov3_patches'],
            batch['t5_hidden'],
            batch['t5_mask'],
            cfg_probs,
            dino_patches_mask=batch['dinov3_patches_mask'],
            pose_kpts=batch['pose_keypoints'],
            prediction_type=model_cfg.prediction_type,
            tread_config=tread_cfg,
        )
    fwd_time = time.time() - t0
    print(f"  Forward: {fwd_time:.2f}s, loss={loss.item():.4f}")

    # Backward pass
    print(f"\nRunning backward pass...")
    t0 = time.time()
    loss.backward()
    bwd_time = time.time() - t0
    print(f"  Backward: {bwd_time:.2f}s")

    # Summary
    print(f"\n=== Summary ===")
    print(f"fullgraph=True: {'PASS ✓' if fullgraph_ok else 'FAIL ✗ (graph breaks detected)'}")
    print(f"Forward:  {fwd_time:.2f}s (includes JIT compilation)")
    print(f"Backward: {bwd_time:.2f}s (includes JIT compilation)")
    print()
    if not fullgraph_ok:
        print("To see graph break locations, run with:")
        print("  TORCH_LOGS='graph_breaks' python scripts/audit_graph_breaks.py")
        print()
        print("Common fixes:")
        print("  - Replace Python if/else with torch.where()")
        print("  - Replace torch.randperm with deterministic index tensors")
        print("  - Wrap non-compilable sections with torch.compiler.disable()")
    else:
        print("The model compiles cleanly with fullgraph=True!")
        print("torch.compile will fuse all operations into optimized kernels.")

    if not quick:
        # Run a second pass to get warm (cached) timings
        print(f"\n=== Warm Pass (cached kernels) ===")
        model.zero_grad()
        t0 = time.time()
        with torch.amp.autocast('cuda', dtype=dtype):
            loss = flow_matching_loss(
                compiled,
                batch['image_data'],
                batch['dino_embedding'],
                batch['dinov3_patches'],
                batch['t5_hidden'],
                batch['t5_mask'],
                cfg_probs,
                dino_patches_mask=batch['dinov3_patches_mask'],
                pose_kpts=batch['pose_keypoints'],
                prediction_type=model_cfg.prediction_type,
                tread_config=tread_cfg,
            )
        fwd_warm = time.time() - t0
        t0 = time.time()
        loss.backward()
        bwd_warm = time.time() - t0
        print(f"  Forward (warm):  {fwd_warm:.4f}s")
        print(f"  Backward (warm): {bwd_warm:.4f}s")
        print(f"  Speedup vs cold: {fwd_time/max(fwd_warm,0.001):.1f}x fwd, {bwd_time/max(bwd_warm,0.001):.1f}x bwd")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audit torch.compile graph breaks')
    parser.add_argument('--config', default='production/config_strix_halo.yaml')
    parser.add_argument('--quick', action='store_true', help='Skip warm pass')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    run_audit(args.config, quick=args.quick, device=args.device)
