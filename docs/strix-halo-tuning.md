# Strix Halo Training Tuning Guide

System-level optimizations for running the prx-tg NanoDiT training pipeline on AMD Strix Halo APU (RDNA 3.5, gfx1151) with 128GB unified LPDDR5X memory.

## Table of Contents

- [TheROCk Stack Installation](#therock-stack-installation)
- [GTT/Shared Memory Configuration](#gttshared-memory-configuration)
- [Power Budget Rebalancing](#power-budget-rebalancing)
- [WMMA Dispatch Verification](#wmma-dispatch-verification)
- [Environment Variables](#environment-variables)
- [Profiling](#profiling)

---

## TheROCk Stack Installation

Standard ROCm 7.2 has known performance issues and broken FlashAttention for gfx1151. The "TheROCk" preview stack (ROCm 7.9+/8.0) is recommended.

### Why TheROCk?

- **FlashAttention**: Restored for Strix Halo — critical for the quad-conditioned NanoDiT's dense sequence lengths
- **Faster Inductor/Dynamo**: Significantly reduced cold-start compilation times for torch.compile
- **gfx1151 support**: Better hardware detection and dispatch for RDNA 3.5

### Installation

```bash
# TheROCk nightlies (check https://github.com/ROCm/TheROCk for latest)
# These instructions will evolve as TheROCk stabilizes

# 1. Remove existing ROCm (if installed)
sudo apt purge rocm-* hip-* 2>/dev/null

# 2. Add TheROCk repository
# (Follow latest instructions at https://github.com/ROCm/TheROCk/releases)

# 3. Install PyTorch with TheROCk support
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.3
# OR build from source against TheROCk headers

# 4. Verify
python -c "import torch; print(torch.version.hip); print(torch.cuda.is_available())"
```

### PyTorch "TheRock" Builds

AMD maintains custom PyTorch builds optimized for TheROCk:

```bash
# Check https://github.com/ROCm/pytorch for latest compatible builds
# These may include patches for FlashAttention on gfx1151
```

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

If WMMA is not engaged:

```bash
# Force GFX version mapping (treat gfx1151 as gfx1100/Navi 31)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# This is set automatically by production/train_production.py when it
# detects a Strix Halo (gfx1151) architecture, but you can set it
# manually for testing.
```

---

## Environment Variables

Key environment variables for Strix Halo training:

```bash
# ROCm device targeting (set automatically by train_production.py)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Triton cache (persist compiled kernels across runs)
export TORCHINDUCTOR_CACHE_DIR=~/.cache/torch_inductor

# Inductor settings (set automatically when inductor_max_autotune=true)
# These maximize kernel autotuning for the specific hardware
export TORCH_INDUCTOR_MAX_AUTOTUNE=1

# Disable NUMA balancing (can cause latency spikes on APU)
echo 0 | sudo tee /proc/sys/kernel/numa_balancing

# Set HIP visible devices (if multiple GPUs somehow detected)
export HIP_VISIBLE_DEVICES=0

# Increase file descriptor limit (for mmap data loading with many .npy files)
ulimit -n 65536
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
