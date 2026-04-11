#!/usr/bin/env python3
"""Profile a single training step on Strix Halo (or any GPU).

Breaks down wall-clock time into: data loading, forward pass, backward pass,
optimizer step, and other overhead. Also reports memory usage breakdown.

Usage:
    python scripts/profile_strix_halo.py                    # Default config
    python scripts/profile_strix_halo.py --config production/config.yaml
    python scripts/profile_strix_halo.py --steps 5          # Average over 5 steps
    python scripts/profile_strix_halo.py --torch-profile    # Full torch.profiler trace
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def profile_step(config_path: str, device: str, num_steps: int, use_torch_profile: bool):
    import torch
    from production.config_loader import load_config
    from production.model import NanoDiT
    from production.train import flow_matching_loss

    config = load_config(config_path)
    model_cfg = config.model
    training_cfg = config.training

    print("=" * 60)
    print("Strix Halo Training Step Profiler")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    hip = getattr(torch.version, 'hip', None)
    if hip:
        print(f"ROCm/HIP: {hip}")
        arch = torch.cuda.get_device_properties(device).gcnArchName if torch.cuda.is_available() else 'N/A'
        print(f"GCN Arch: {arch}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"Memory: {mem_gb:.1f} GB")
    print(f"Steps to profile: {num_steps}")
    print(f"Batch size: {training_cfg.batch_size}")
    print(f"Precision: {training_cfg.precision}")
    print(f"Grad checkpointing: {training_cfg.gradient_checkpointing}")
    print(f"torch.compile: {training_cfg.compile}")
    print()

    # Build model
    model = NanoDiT(
        in_channels=model_cfg.in_channels,
        hidden_size=model_cfg.hidden_size,
        depth=model_cfg.depth,
        num_heads=model_cfg.num_heads,
        patch_size=model_cfg.patch_size,
        mlp_ratio=model_cfg.mlp_ratio,
        use_gradient_checkpointing=training_cfg.gradient_checkpointing,
        bottleneck_size=model_cfg.bottleneck_size,
        num_pose_joints=model_cfg.num_pose_joints,
        pose_confidence_threshold=model_cfg.pose_confidence_threshold,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count / 1e6:.1f}M params")

    if training_cfg.compile:
        fullgraph = training_cfg.compile_fullgraph
        mode = 'max-autotune' if training_cfg.inductor_max_autotune else 'default'
        print(f"Compiling (fullgraph={fullgraph}, mode='{mode}')...")
        model = torch.compile(model, dynamic=True, fullgraph=fullgraph, mode=mode)

    # Optimizer (simplified — just AdamW for profiling)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    cfg_probs = training_cfg.cfg_dropout.to_dict()
    tread_cfg = training_cfg.tread if training_cfg.tread.enabled else None
    dtype = torch.bfloat16 if training_cfg.precision == 'bfloat16' else torch.float16

    # Synthetic data at target resolution
    H, W = 256, 256
    patch_size = model_cfg.patch_size
    B = training_cfg.batch_size
    hidden = model_cfg.hidden_size
    h_p, w_p = H // patch_size, W // patch_size
    n_patches = h_p * w_p
    # Conditioning dims match NanoDiT defaults (not hidden_size)
    dino_dim = 1024
    dino_patch_dim = 1024
    text_dim = 1024

    def make_batch():
        return {
            'image_data': torch.randn(B, 3, H, W, device=device),
            'dino_embedding': torch.randn(B, dino_dim, device=device),
            'dinov3_patches': torch.randn(B, n_patches, dino_patch_dim, device=device),
            'dinov3_patches_mask': torch.ones(B, n_patches, dtype=torch.long, device=device),
            't5_hidden': torch.randn(B, 512, text_dim, device=device),
            't5_mask': torch.ones(B, 512, dtype=torch.long, device=device),
            'pose_keypoints': torch.randn(B, 133, 3, device=device),
        }

    # Warmup (1 step to trigger JIT compilation)
    print("\nWarmup step (includes JIT compilation)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    batch = make_batch()
    t0 = time.time()
    with torch.amp.autocast('cuda', dtype=dtype):
        loss = flow_matching_loss(
            model, batch['image_data'], batch['dino_embedding'],
            batch['dinov3_patches'], batch['t5_hidden'], batch['t5_mask'],
            cfg_probs, dino_patches_mask=batch['dinov3_patches_mask'],
            pose_kpts=batch['pose_keypoints'],
            prediction_type=model_cfg.prediction_type, tread_config=tread_cfg,
        )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    warmup_time = time.time() - t0
    print(f"  Warmup: {warmup_time:.2f}s")

    # Profile multiple steps
    print(f"\nProfiling {num_steps} steps...")
    timings = {'data': [], 'forward': [], 'backward': [], 'optimizer': [], 'total': []}

    for step in range(num_steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        # Data creation (simulates loading)
        t_data = time.time()
        batch = make_batch()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        timings['data'].append(time.time() - t_data)

        # Forward
        t_fwd = time.time()
        with torch.amp.autocast('cuda', dtype=dtype):
            loss = flow_matching_loss(
                model, batch['image_data'], batch['dino_embedding'],
                batch['dinov3_patches'], batch['t5_hidden'], batch['t5_mask'],
                cfg_probs, dino_patches_mask=batch['dinov3_patches_mask'],
                pose_kpts=batch['pose_keypoints'],
                prediction_type=model_cfg.prediction_type, tread_config=tread_cfg,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        timings['forward'].append(time.time() - t_fwd)

        # Backward
        t_bwd = time.time()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        timings['backward'].append(time.time() - t_bwd)

        # Optimizer
        t_opt = time.time()
        optimizer.step()
        optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        timings['optimizer'].append(time.time() - t_opt)

        timings['total'].append(
            timings['data'][-1] + timings['forward'][-1] +
            timings['backward'][-1] + timings['optimizer'][-1]
        )

    # Memory report
    print(f"\n{'='*60}")
    print("MEMORY REPORT")
    print(f"{'='*60}")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        current = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        print(f"  Peak allocated:    {peak:.2f} GB")
        print(f"  Current allocated: {current:.2f} GB")
        print(f"  Reserved:          {reserved:.2f} GB")
    else:
        print("  (CUDA not available — memory stats unavailable)")

    # Weight memory
    weight_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"  Model weights:     {weight_mem:.2f} GB")

    # Timing report
    print(f"\n{'='*60}")
    print(f"TIMING REPORT (averaged over {num_steps} steps)")
    print(f"{'='*60}")
    import statistics
    for phase in ['data', 'forward', 'backward', 'optimizer', 'total']:
        vals = timings[phase]
        avg = statistics.mean(vals) * 1000  # ms
        std = statistics.stdev(vals) * 1000 if len(vals) > 1 else 0
        print(f"  {phase:12s}: {avg:8.2f} ms ± {std:6.2f} ms")

    total_avg = statistics.mean(timings['total'])
    print(f"\n  Steps/second: {1.0/total_avg:.2f}")
    print(f"  Images/second: {B/total_avg:.2f}")

    # Breakdown percentages
    print(f"\n  Breakdown:")
    for phase in ['data', 'forward', 'backward', 'optimizer']:
        avg = statistics.mean(timings[phase])
        pct = avg / total_avg * 100
        print(f"    {phase:12s}: {pct:5.1f}%")

    # Torch profiler trace (optional)
    if use_torch_profile:
        print(f"\n{'='*60}")
        print("TORCH PROFILER TRACE")
        print(f"{'='*60}")
        trace_path = 'profile_trace.json'
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            batch = make_batch()
            with torch.amp.autocast('cuda', dtype=dtype):
                loss = flow_matching_loss(
                    model, batch['image_data'], batch['dino_embedding'],
                    batch['dinov3_patches'], batch['t5_hidden'], batch['t5_mask'],
                    cfg_probs, dino_patches_mask=batch['dinov3_patches_mask'],
                    pose_kpts=batch['pose_keypoints'],
                    prediction_type=model_cfg.prediction_type, tread_config=tread_cfg,
                )
            loss.backward()

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        prof.export_chrome_trace(trace_path)
        print(f"\n  Chrome trace saved to: {trace_path}")
        print(f"  Open in chrome://tracing or https://ui.perfetto.dev/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Profile training step')
    parser.add_argument('--config', default='production/config_strix_halo.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--torch-profile', action='store_true')
    args = parser.parse_args()

    profile_step(args.config, args.device, args.steps, args.torch_profile)
