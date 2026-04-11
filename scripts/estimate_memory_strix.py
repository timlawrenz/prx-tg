"""Estimate memory requirements for Strix Halo training.

Calculates the total memory needed for:
  - Model weights (bf16)
  - Optimizer states (Muon + AdamW)
  - Activations per batch (with/without gradient checkpointing)
  - Batch data buffers

Usage:
    python -m scripts.estimate_memory_strix
    python -m scripts.estimate_memory_strix --batch-size 8 --resolution 1024
"""

import argparse
import sys
sys.path.insert(0, '.')


def estimate_model_memory(hidden=768, depth=18, heads=12, mlp_ratio=4.0,
                          patch_size=16, in_channels=3, num_pose=133,
                          dtype_bytes=2):
    """Estimate model weight memory in bytes."""
    mlp_dim = int(hidden * mlp_ratio)
    
    # Patch embedding: Conv2d(3, hidden, patch_size, patch_size)
    patch_embed = in_channels * hidden * patch_size * patch_size
    
    # Per DiTBlock:
    #   Self-attn: QKV (3 * hidden * hidden) + out (hidden * hidden) = 4 * hidden^2
    #   Cross-attn: QKV (3 * hidden * hidden) + out (hidden * hidden) = 4 * hidden^2
    #   MLP: up (hidden * mlp_dim) + down (mlp_dim * hidden) = 2 * hidden * mlp_dim
    #   AdaLN: hidden * 6*hidden (modulation params)
    #   LayerNorms: 3 * 2 * hidden (weight + bias each)
    per_block = (
        4 * hidden * hidden +          # self-attn
        4 * hidden * hidden +          # cross-attn
        2 * hidden * mlp_dim +         # MLP
        hidden * 6 * hidden +          # adaLN modulation
        3 * 2 * hidden                 # layer norms
    )
    
    # REPA projection: hidden -> 1024
    repa_proj = hidden * 1024
    
    # Timestep MLP: hidden -> hidden -> hidden
    time_mlp = 2 * hidden * hidden
    
    # DINO projections: 1024 -> hidden (CLS + patches)
    dino_proj = 2 * 1024 * hidden
    
    # T5 projection: 1024 -> hidden
    t5_proj = 1024 * hidden
    
    # Pose MLP: 3 -> hidden -> hidden + embeddings
    pose_proj = 3 * hidden + hidden * hidden + num_pose * hidden
    
    # Final layer: norm + linear(hidden, patch_size^2 * in_channels)
    final = 2 * hidden + hidden * (patch_size * patch_size * in_channels)
    
    total_params = patch_embed + depth * per_block + repa_proj + time_mlp + dino_proj + t5_proj + pose_proj + final
    
    return total_params, total_params * dtype_bytes


def estimate_optimizer_memory(total_params, optimizer='Muon', dtype_bytes=4):
    """Estimate optimizer state memory.
    
    Muon (2D weights): momentum buffer (1 copy)
    AdamW (non-2D weights): exp_avg + exp_avg_sq (2 copies)
    EMA: full model copy
    """
    # Rough split: ~99.5% of params are 2D (Muon), ~0.5% non-2D (AdamW)
    muon_params = int(total_params * 0.995)
    adamw_params = total_params - muon_params
    
    if optimizer == 'Muon':
        # Muon: 1 momentum buffer per param (float32)
        muon_mem = muon_params * dtype_bytes
        # AdamW: 2 states per param (float32)
        adamw_mem = adamw_params * 2 * dtype_bytes
        opt_mem = muon_mem + adamw_mem
    else:
        # Pure AdamW: 2 states per param
        opt_mem = total_params * 2 * dtype_bytes
    
    # EMA: full model copy in same dtype as model
    ema_mem = total_params * 2  # bf16
    
    return opt_mem, ema_mem


def estimate_activation_memory(batch_size, resolution, hidden=768, depth=18,
                               patch_size=16, mlp_ratio=4.0, heads=12,
                               gradient_checkpointing=False,
                               dtype_bytes=2):
    """Estimate activation memory for forward + backward pass."""
    seq_len = (resolution // patch_size) ** 2
    mlp_dim = int(hidden * mlp_ratio)
    head_dim = hidden // heads
    
    # Per block activations (stored for backward pass):
    #   Input to block: (B, seq_len, hidden)
    #   QKV: (B, 3, heads, seq_len, head_dim) for self-attn
    #   Attention output: (B, seq_len, hidden)
    #   Cross-attn QKV + output
    #   MLP intermediate: (B, seq_len, mlp_dim)
    #   AdaLN intermediates
    per_block_act = batch_size * (
        seq_len * hidden * 4 +          # input + self-attn output + cross-attn output + mlp output
        seq_len * mlp_dim +             # MLP intermediate
        3 * heads * seq_len * head_dim  # QKV (self-attn)
    ) * dtype_bytes
    
    if gradient_checkpointing:
        # Only store input per block (recompute the rest)
        total_act = depth * batch_size * seq_len * hidden * dtype_bytes
    else:
        total_act = depth * per_block_act
    
    # Conditioning data (kept throughout):
    # DINOv3 patches: ~4096 tokens × 1024 dim × B
    dino_patches = batch_size * 4096 * 1024 * dtype_bytes
    # T5: 512 × 1024 × B
    t5 = batch_size * 512 * 1024 * dtype_bytes
    
    return total_act + dino_patches + t5


def estimate_batch_data_memory(batch_size, resolution, dtype_bytes=2):
    """Estimate memory for one batch of input data."""
    # Image: (B, 3, H, W)
    image = batch_size * 3 * resolution * resolution * dtype_bytes
    # DINOv3 patches: (B, ~4096, 1024)
    dino_patches = batch_size * 4096 * 1024 * dtype_bytes
    # DINOv3 CLS: (B, 1024)
    dino_cls = batch_size * 1024 * dtype_bytes
    # T5: (B, 512, 1024)
    t5 = batch_size * 512 * 1024 * dtype_bytes
    # Pose: (B, 133, 3)
    pose = batch_size * 133 * 3 * dtype_bytes
    
    return image + dino_patches + dino_cls + t5 + pose


def main():
    parser = argparse.ArgumentParser(description='Estimate Strix Halo training memory')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--hidden', type=int, default=768)
    parser.add_argument('--depth', type=int, default=18)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--optimizer', type=str, default='Muon', choices=['Muon', 'AdamW'])
    parser.add_argument('--no-gradient-checkpointing', action='store_true', default=True)
    parser.add_argument('--gradient-checkpointing', action='store_true', default=False)
    parser.add_argument('--total-memory-gb', type=float, default=128.0)
    args = parser.parse_args()
    
    use_gc = args.gradient_checkpointing and not args.no_gradient_checkpointing
    
    print("=" * 60)
    print("STRIX HALO MEMORY ESTIMATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {args.hidden}h × {args.depth}d × {args.heads} heads, patch={args.patch_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Resolution: {args.resolution}×{args.resolution}")
    print(f"  Sequence length: {(args.resolution // args.patch_size) ** 2} tokens")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Gradient checkpointing: {'YES' if use_gc else 'NO'}")
    print(f"  Total system memory: {args.total_memory_gb:.0f} GB")
    
    # Model weights
    total_params, model_mem = estimate_model_memory(
        args.hidden, args.depth, args.heads, patch_size=args.patch_size
    )
    
    # Optimizer states
    opt_mem, ema_mem = estimate_optimizer_memory(total_params, args.optimizer)
    
    # Activations
    act_mem = estimate_activation_memory(
        args.batch_size, args.resolution, args.hidden, args.depth,
        args.patch_size, gradient_checkpointing=use_gc
    )
    
    # Batch data
    data_mem = estimate_batch_data_memory(args.batch_size, args.resolution)
    
    # Gradients (same size as model weights, in float32 for accumulation)
    grad_mem = total_params * 4  # float32 gradients
    
    total = model_mem + opt_mem + ema_mem + act_mem + data_mem + grad_mem
    
    print(f"\nMemory Breakdown:")
    print(f"  Model weights (bf16):       {model_mem / 1024**3:8.2f} GB  ({total_params/1e6:.1f}M params)")
    print(f"  Gradients (fp32):           {grad_mem / 1024**3:8.2f} GB")
    print(f"  Optimizer states:           {opt_mem / 1024**3:8.2f} GB  ({args.optimizer})")
    print(f"  EMA weights (bf16):         {ema_mem / 1024**3:8.2f} GB")
    print(f"  Activations:                {act_mem / 1024**3:8.2f} GB  ({'checkpointed' if use_gc else 'full cache'})")
    print(f"  Batch data:                 {data_mem / 1024**3:8.2f} GB")
    print(f"  {'─' * 42}")
    print(f"  TOTAL:                      {total / 1024**3:8.2f} GB")
    
    available = args.total_memory_gb * 0.9  # 90% usable (OS + system overhead)
    headroom = available - total / 1024**3
    
    print(f"\n  Available (~90% of {args.total_memory_gb:.0f}GB): {available:.1f} GB")
    print(f"  Headroom:                   {headroom:8.2f} GB")
    
    if headroom < 0:
        print(f"\n  ⚠️  WILL NOT FIT — reduce batch_size or enable gradient checkpointing")
    elif headroom < 10:
        print(f"\n  ⚠️  TIGHT FIT — limited room for PyTorch overhead")
    else:
        print(f"\n  ✅ FITS with {headroom:.1f} GB to spare")
    
    # Sweep batch sizes
    print(f"\n{'─' * 60}")
    print(f"Batch Size Sweep (resolution={args.resolution}, gc={'ON' if use_gc else 'OFF'}):")
    print(f"  {'BS':>4}  {'Activations':>12}  {'Batch Data':>12}  {'TOTAL':>12}  {'Status':>10}")
    for bs in [1, 2, 4, 8, 16, 32]:
        act = estimate_activation_memory(
            bs, args.resolution, args.hidden, args.depth,
            args.patch_size, gradient_checkpointing=use_gc
        )
        data = estimate_batch_data_memory(bs, args.resolution)
        t = model_mem + opt_mem + ema_mem + act + data + grad_mem
        t_gb = t / 1024**3
        status = "✅" if t_gb < available else "❌ OOM"
        print(f"  {bs:>4}  {act/1024**3:>10.2f}GB  {data/1024**3:>10.2f}GB  {t_gb:>10.2f}GB  {status:>10}")


if __name__ == '__main__':
    main()
