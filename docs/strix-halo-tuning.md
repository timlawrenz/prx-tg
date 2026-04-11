# Strix Halo Training Tuning Guide

System-level optimizations for running the prx-tg NanoDiT training pipeline on AMD Strix Halo APU (RDNA 3.5, gfx1151) with 128GB unified LPDDR5X memory.

## Table of Contents

- [PyTorch Installation (AMD gfx1151 Nightlies)](#pytorch-installation-amd-gfx1151-nightlies)
- [AOTriton — Critical for FlashAttention Performance](#aotriton--critical-for-flashattention-performance)
- [Known bf16 Bugs on gfx1151](#known-bf16-bugs-on-gfx1151)
- [GTT/Shared Memory Configuration](#gttshared-memory-configuration)
- [Power Budget Rebalancing](#power-budget-rebalancing)
- [WMMA Dispatch Verification](#wmma-dispatch-verification)
- [Environment Variables](#environment-variables)
- [Profiling](#profiling)

---

## PyTorch Installation (AMD gfx1151 Nightlies)

AMD publishes nightly PyTorch builds compiled specifically for Strix Halo (gfx1151)
with native AOTriton kernels. These ship ahead of the stable ROCm releases and
include critical performance fixes not yet available in mainline wheels.

As of April 2026, the nightlies provide:
- **PyTorch 2.10+** with ROCm 7.12 backend
- **Triton 3.4+** with gfx1151-specific kernel compilation
- **AOTriton** — ahead-of-time compiled FlashAttention/SDPA kernels
- **Native gfx1151 support** — `HSA_OVERRIDE_GFX_VERSION` is no longer needed

### Installation

```bash
# Clean install (recommended)
pip uninstall torch torchvision torchaudio -y
pip install -U --pre torch torchvision torchaudio \
    --index-url https://rocm.nightlies.amd.com/v2/gfx1151/

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'HIP: {torch.version.hip}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

> **Note**: These nightlies are built against a newer ROCm than stable releases.
> The `PYTORCH_HIP_ALLOC_CONF=backend:malloc` variable set by some ROCm shell
> profiles will crash PyTorch — `unset` it or let `train_production.py` handle it.

---

## AOTriton — Critical for FlashAttention Performance

AOTriton (Ahead-of-Time Triton) pre-compiles GPU kernels for gfx1151, eliminating
JIT compilation overhead. **This is the single most impactful optimization** — it
gives a **19× speedup on SDPA** (44ms → 2.3ms per call).

### Enable AOTriton

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

This is set automatically by `train_production.py` when Strix Halo is detected.
Without this flag, FlashAttention falls back to a naive implementation that makes
training unviably slow.

**Benchmark** (from [ROCm/ROCm#6034](https://github.com/ROCm/ROCm/issues/6034)):
| SDPA Mode         | Time per call | Speedup |
|-------------------|--------------|---------|
| Without AOTriton  | 44.0 ms      | 1×      |
| With AOTriton     | 2.3 ms       | **19×** |

### Verify AOTriton is Active

```python
import torch, os
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
x = torch.randn(2, 12, 256, 64, device='cuda', dtype=torch.bfloat16)
# This should complete in < 5ms, not 40ms+
with torch.no_grad():
    out = torch.nn.functional.scaled_dot_product_attention(x, x, x)
print("SDPA completed — AOTriton active")
```

---

## Known bf16 Bugs on gfx1151

Community testing ([ROCm/ROCm#6034](https://github.com/ROCm/ROCm/issues/6034),
93 experiments) has identified several bf16 precision boundaries on Strix Halo.
These affect training stability and must be worked around:

| Bug | Symptom | Workaround |
|-----|---------|------------|
| Small batch accumulation | NaN within 15 steps at effective batch < 2^15 | Use effective batch ≥ 32768 tokens |
| Small head dim (32) | NaN crash | Use `HEAD_DIM ≥ 64` (NanoDiT uses 64 ✓) |
| Deep networks (depth > 12) | Non-deterministic NaN/timeout | Monitor loss early; our depth=18 needs testing |
| Wide aspect ratios (>64) | Timeout/crash | Keep aspect ratio ≤ 64 (our buckets are ≤ 2:1 ✓) |
| High Muon matrix LR (>0.15) | Sharp NaN cliff | Keep Muon LR ≤ 0.15 |

**Our NanoDiT (hidden=768, depth=18, head_dim=64) hits the "deep network" boundary.**
This means you should:
1. Monitor for NaN in early training steps
2. Consider starting at depth=10-12 and scaling up
3. Use gradient clipping (already configured at 1.0)

---

## GTT/Shared Memory Configuration

The Strix Halo APU shares its 128GB LPDDR5X between CPU and GPU. Out of the box, Linux may limit the GPU's virtual address space (GPUVM/GTT).

### Check Current Allocation

```bash
# Check GPU-visible memory
rocm-smi --showmeminfo all

# Check TTM (Translation Table Manager) limits
cat /sys/class/drm/card0/device/mem_info_vram_total
cat /sys/class/drm/card0/device/mem_info_gtt_total
```

### Maximize GPU-Visible Memory

**BIOS Settings:**
- Set "UMA Frame Buffer Size" or "VRAM Size" to maximum (often 96GB or "Auto")
- Some BIOS versions: Advanced → AMD CBS → NBIO → GFX Configuration

**Kernel Parameters:**
```bash
# Add to GRUB_CMDLINE_LINUX in /etc/default/grub:
# amdgpu.vm_size=1024 amdgpu.gttsize=98304
# Then: sudo update-grub && reboot
```

**Runtime (amd-ttm utility):**
```bash
# If available, adjust GTT size dynamically
# Target: 96GB+ for GPU-visible memory
```

### Verification

```bash
# After configuration, verify with:
python -c "
import torch
props = torch.cuda.get_device_properties(0)
print(f'GPU Memory: {props.total_mem / 1e9:.1f} GB')
print(f'GCN Arch: {props.gcnArchName}')
"
```

---

## Power Budget Rebalancing

Strix Halo operates under a shared TDP. During training, CPU handles data loading while GPU handles matrix math. If CPU draws too much power, GPU clocks throttle.

### Limit CPU Power During Training

```bash
# Check current CPU governor
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set CPU to powersave during training (frees thermal budget for GPU)
sudo cpupower frequency-set -g powersave

# Or cap maximum CPU frequency (e.g., 2.5 GHz, leaving headroom for GPU)
sudo cpupower frequency-set --max 2500MHz

# Restore after training
sudo cpupower frequency-set -g performance
```

### Monitor Thermal/Power State

```bash
# Watch GPU clocks and temperature during training
watch -n 1 rocm-smi

# Check if GPU is throttling
rocm-smi --showclocks --showtemp --showpower
```

### Automated Power Script

```bash
#!/bin/bash
# save as: scripts/power_training_mode.sh

echo "Switching to training power mode..."
echo "  - CPU: powersave governor, max 2.5 GHz"
echo "  - GPU: performance mode"

sudo cpupower frequency-set -g powersave
sudo cpupower frequency-set --max 2500MHz

# Optional: set GPU to highest performance level
# echo "high" | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level

echo "Power mode set. Run training, then restore with:"
echo "  sudo cpupower frequency-set -g performance"
```

---

## WMMA Dispatch Verification

RDNA 3.5 supports WMMA (Wave Matrix Multiply Accumulate) for bfloat16 matrix math. If PyTorch doesn't recognize the hardware correctly, it may fall back to scalar ALUs — severely degrading performance.

### Quick Check

```bash
# Run the verify script (creates a small model, runs one step, checks profiler)
python scripts/verify_wmma_dispatch.py
```

### Manual Verification with rocprof

```bash
# Profile a single training step
rocprof --hip-trace --hsa-trace python scripts/profile_strix_halo.py --steps 1

# Look for WMMA kernels in the trace:
grep -i "wmma\|mfma\|matrix" results.csv

# If you see only VALU (vector ALU) instructions for matmuls, WMMA is NOT being used
```

### Forcing Correct Dispatch

If WMMA is not engaged and you're on an older ROCm build without native gfx1151 kernels:

```bash
# Force GFX version mapping (treat gfx1151 as gfx1100/Navi 31)
# NOTE: Not needed with AMD gfx1151 nightlies (rocm.nightlies.amd.com/v2/gfx1151/)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

---

## Environment Variables

Key environment variables for Strix Halo training:

```bash
# AOTriton experimental kernels — CRITICAL for FlashAttention/SDPA performance
# (set automatically by train_production.py on Strix Halo detection)
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Clear crash-inducing HIP allocator config (set by some ROCm shell profiles)
unset PYTORCH_HIP_ALLOC_CONF

# Triton cache (persist compiled kernels across runs)
export TORCHINDUCTOR_CACHE_DIR=~/.cache/torch_inductor

# Inductor settings (set automatically when inductor_max_autotune=true)
export TORCH_INDUCTOR_MAX_AUTOTUNE=1

# Disable NUMA balancing (can cause latency spikes on APU)
echo 0 | sudo tee /proc/sys/kernel/numa_balancing

# Set HIP visible devices (if multiple GPUs somehow detected)
export HIP_VISIBLE_DEVICES=0

# Increase file descriptor limit (for mmap data loading with many .npy files)
ulimit -n 65536

# NOTE: HSA_OVERRIDE_GFX_VERSION is NOT needed with gfx1151 nightlies.
# Only set it if you're on an older ROCm build without native gfx1151 support.
```

---

## Profiling

### Step-Level Profiling

```bash
# Profile 5 training steps, report per-phase timing breakdown
python scripts/profile_strix_halo.py --steps 5

# With torch.profiler Chrome trace
python scripts/profile_strix_halo.py --torch-profile
# Then open profile_trace.json in chrome://tracing
```

### Graph Break Audit

```bash
# Check for torch.compile graph breaks (each break = lost optimization)
TORCH_LOGS="graph_breaks" python scripts/audit_graph_breaks.py

# Quick mode (just check if fullgraph compiles)
python scripts/audit_graph_breaks.py --quick
```

### Memory Estimation

```bash
# Estimate memory usage for different batch sizes (no GPU needed)
python scripts/estimate_memory_strix.py
```

### Kernel-Level with rocprof

```bash
# Full kernel trace
rocprof --hip-trace python scripts/profile_strix_halo.py --steps 1

# Specific counters (memory bandwidth)
rocprof --timestamp on -i counters.txt python scripts/profile_strix_halo.py --steps 1
```
