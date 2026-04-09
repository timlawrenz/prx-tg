#!/usr/bin/env bash
# Run a single throughput experiment arm.
# Called by the fleet orchestrator — one instance per arm.
#
# Usage:
#   bash scripts/run_throughput_arm.sh <config_yaml>
#
# Example:
#   bash scripts/run_throughput_arm.sh experiments/throughput/config_0_baseline.yaml
#
# Environment variables (optional):
#   GPU=0          GPU index (default: 0)
#   SEED=42        Random seed for reproducibility

set -euo pipefail

CONFIG="${1:?Usage: $0 <config_yaml>}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: Config not found: $CONFIG"
  exit 1
fi

ARM_NAME=$(basename "$CONFIG" .yaml)

echo "============================================================"
echo "THROUGHPUT EXPERIMENT: $ARM_NAME"
echo "Config: $CONFIG"
echo "GPU: $GPU"
echo "Seed: $SEED"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Print GPU info
nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current \
  --format=csv,noheader -i "$GPU" 2>/dev/null || echo "nvidia-smi not available"

# Print host info for hardware matching verification
echo "Host RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "CPU: $(lscpu | grep 'Model name' | sed 's/Model name: *//')"
echo ""

# Set seed for reproducibility
export PYTHONHASHSEED="$SEED"

# Run training
CUDA_VISIBLE_DEVICES="$GPU" python -m production.train_production \
  --config "$CONFIG" \
  --seed "$SEED"

echo ""
echo "$ARM_NAME complete: $(date '+%Y-%m-%d %H:%M:%S')"
