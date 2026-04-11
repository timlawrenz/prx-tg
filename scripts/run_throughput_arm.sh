#!/usr/bin/env bash
# Runner script for throughput experiment arms.
# Invoked by: bash scripts/run_throughput_arm.sh <config_yaml>
# Outputs METRICS:{json} on the last line for fleet executor extraction.
set -euo pipefail

CONFIG="${1:?Usage: $0 <config_yaml>}"

echo "=== Throughput Arm Runner ==="
echo "Config: $CONFIG"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'unknown')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"
echo ""

# Install g++ if missing (required for torch.compile/inductor)
if ! command -v g++ &>/dev/null; then
    echo "Installing g++ for torch.compile..."
    apt-get update -qq && apt-get install -y -qq g++ > /dev/null 2>&1 || true
fi

# Run training, capture output
TRAIN_LOG=$(mktemp /tmp/train_log.XXXXXX)
set +e
python -m production.train_production --config "$CONFIG" 2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=$?
set -e

# Extract metrics from training log
python3 -c "
import json, re, sys

log_path = '$TRAIN_LOG'
exit_code = $TRAIN_EXIT

with open(log_path) as f:
    lines = f.readlines()

# Parse iter_per_sec and loss from log lines
# The trainer logs JSON-lines to training.log, but also prints to stdout
iter_speeds = []
losses = []
peak_vram = 0.0

for line in lines:
    # Look for iter_per_sec in log output
    if 'iter_per_sec' in line:
        m = re.search(r'\"iter_per_sec\":\s*([\d.]+)', line)
        if m:
            iter_speeds.append(float(m.group(1)))
    if 'peak_vram_gb' in line:
        m = re.search(r'\"peak_vram_gb\":\s*([\d.]+)', line)
        if m:
            peak_vram = max(peak_vram, float(m.group(1)))
    if '\"loss\"' in line:
        m = re.search(r'\"loss\":\s*([\d.]+)', line)
        if m:
            losses.append(float(m.group(1)))

# Also try parsing the JSON log file
import os
log_file = None
for candidate in ['training.log', 'experiments/training.log']:
    # Search in likely experiment directories
    for root, dirs, files in os.walk('.'):
        if 'training.log' in files:
            log_file = os.path.join(root, 'training.log')
            break
    if log_file:
        break

if log_file and os.path.exists(log_file):
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                step = entry.get('step', 0)
                # Skip warmup steps (first 10)
                if step > 10:
                    if 'iter_per_sec' in entry:
                        iter_speeds.append(entry['iter_per_sec'])
                    if 'loss' in entry:
                        losses.append(entry['loss'])
                if 'peak_vram_gb' in entry:
                    peak_vram = max(peak_vram, entry['peak_vram_gb'])
            except (json.JSONDecodeError, KeyError):
                pass

# Compute averages (skip first 10 entries for warmup)
warmup_skip = 10
post_warmup_speeds = iter_speeds[warmup_skip:] if len(iter_speeds) > warmup_skip else iter_speeds

avg_iter_per_sec = sum(post_warmup_speeds) / len(post_warmup_speeds) if post_warmup_speeds else 0.0
final_loss = losses[-1] if losses else float('inf')

# GPU memory from nvidia-smi as fallback
if peak_vram == 0:
    import subprocess
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            text=True
        )
        peak_vram = float(out.strip()) / 1024  # MiB to GiB
    except Exception:
        pass

result = {
    'avg_iter_per_sec': round(avg_iter_per_sec, 3),
    'peak_vram_gb': round(peak_vram, 2),
    'final_loss': round(final_loss, 6) if final_loss != float('inf') else 'Infinity',
    'total_steps': len(iter_speeds),
    'exit_code': exit_code,
}

print(f'METRICS:{json.dumps(result)}')
sys.exit(exit_code)
"

rm -f "$TRAIN_LOG"
