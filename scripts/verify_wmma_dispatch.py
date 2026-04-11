#!/usr/bin/env python3
"""Verify WMMA (Wave Matrix Multiply Accumulate) dispatch on RDNA 3.5.

RDNA 3.5 supports WMMA instructions for bfloat16 matrix math. If PyTorch
doesn't dispatch to WMMA, matmuls fall back to scalar ALUs — a 2-5× slowdown.

This script runs a small bfloat16 matmul and checks via torch.profiler
whether the correct kernels are being used.

Usage:
    python scripts/verify_wmma_dispatch.py
    python scripts/verify_wmma_dispatch.py --size 2048   # Larger matrix
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def verify_wmma(device: str = 'cuda', size: int = 1024):
    import torch

    print("=== WMMA Dispatch Verification ===")
    print(f"PyTorch: {torch.__version__}")
    hip = getattr(torch.version, 'hip', None)
    print(f"ROCm/HIP: {hip or 'Not available (CUDA)'}")

    if not torch.cuda.is_available():
        print("\nERROR: No GPU available. Cannot verify WMMA dispatch.")
        print("This script requires a GPU (CUDA or ROCm).")
        return False

    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}")
    gcn = getattr(props, 'gcnArchName', 'N/A')
    print(f"GCN Arch: {gcn}")
    print(f"Total Memory: {props.total_mem / 1e9:.1f} GB")
    print()

    # Check if this is a Strix Halo / RDNA 3.5
    is_rdna35 = 'gfx1151' in gcn or 'gfx115' in gcn
    is_rdna3 = 'gfx11' in gcn
    hsa_override = os.environ.get('HSA_OVERRIDE_GFX_VERSION', '')

    if is_rdna35:
        print(f"✓ Detected RDNA 3.5 (Strix Halo) — gfx1151")
    elif is_rdna3:
        print(f"✓ Detected RDNA 3.x — {gcn}")
    elif hsa_override:
        print(f"⚠ GCN arch {gcn} with HSA_OVERRIDE_GFX_VERSION={hsa_override}")
    else:
        print(f"ℹ Non-AMD GPU detected ({gcn}). WMMA check may not apply.")

    # Run bfloat16 matmul and profile
    print(f"\nRunning bfloat16 matmul ({size}×{size})...")
    A = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    B = torch.randn(size, size, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(3):
        _ = torch.mm(A, B)
    torch.cuda.synchronize(device)

    # Profiled run
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(10):
            C = torch.mm(A, B)
        torch.cuda.synchronize(device)

    # Analyze kernels
    print("\nTop GPU kernels:")
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
    print(table)

    # Check for WMMA/MFMA indicators
    events = prof.key_averages()
    kernel_names = [e.key for e in events if e.device_time_total > 0]

    wmma_indicators = ['wmma', 'mfma', 'dot', 'gemm']
    scalar_indicators = ['valu', 'naive']

    found_wmma = any(
        any(ind in name.lower() for ind in wmma_indicators)
        for name in kernel_names
    )
    found_scalar = any(
        any(ind in name.lower() for ind in scalar_indicators)
        for name in kernel_names
    )

    print("\n=== Verdict ===")
    if found_wmma and not found_scalar:
        print("✓ WMMA/MFMA kernels detected — bfloat16 matmuls are hardware-accelerated")
        return True
    elif found_wmma and found_scalar:
        print("⚠ Mixed dispatch — some operations use WMMA, others use scalar ALUs")
        print("  This may be expected for small matrices or non-matmul ops")
        return True
    else:
        print("✗ No WMMA/MFMA kernels detected — matmuls may be using scalar ALUs")
        print("\nPossible fixes:")
        print("  1. Set HSA_OVERRIDE_GFX_VERSION=11.0.0")
        print("  2. Install gfx1151 nightlies: pip install --pre torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/")
        print("  3. Enable AOTriton: export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1")
        print(f"\nNote: Kernel names seen: {kernel_names[:5]}")
        print("  Look for 'wmma', 'mfma', or 'dot' in kernel names for confirmation")
        return False

    # Also test with torch.compile to check Triton dispatch
    print(f"\nTesting with torch.compile (Triton)...")

    @torch.compile(mode='default')
    def compiled_matmul(a, b):
        return torch.mm(a, b)

    # Warmup compile
    _ = compiled_matmul(A, B)
    torch.cuda.synchronize(device)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof2:
        for _ in range(10):
            _ = compiled_matmul(A, B)
        torch.cuda.synchronize(device)

    print("\nCompiled matmul kernels:")
    print(prof2.key_averages().table(sort_by="cuda_time_total", row_limit=5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify WMMA dispatch')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--size', type=int, default=1024, help='Matrix size')
    args = parser.parse_args()

    verify_wmma(args.device, args.size)
