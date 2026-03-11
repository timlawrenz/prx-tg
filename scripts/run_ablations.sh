#!/usr/bin/env bash
# Run the full ablation grid (configs A-D) sequentially.
# Each run creates a timestamped experiment directory under experiments/.
#
# Usage:
#   bash scripts/run_ablations.sh              # Run all 4
#   bash scripts/run_ablations.sh B D          # Run only B and D
#
# GPU selection (default: 0):
#   GPU=1 bash scripts/run_ablations.sh

set -euo pipefail

GPU="${GPU:-0}"
CONFIGS_DIR="experiments/ablations"

declare -A CONFIGS=(
  [A]="config_A_baseline.yaml"
  [B]="config_B_tread_adamw.yaml"
  [C]="config_C_tread_muon.yaml"
  [D]="config_D_full_stack.yaml"
)

DESCRIPTIONS=(
  [A]="Baseline (No TREAD, AdamW, No REPA)"
  [B]="+TREAD (TREAD, AdamW, No REPA)"
  [C]="+Muon (TREAD, Muon, No REPA)"
  [D]="Full Stack (TREAD, Muon, REPA)"
)

# Default: run all configs in order
if [[ $# -gt 0 ]]; then
  RUNS=("$@")
else
  RUNS=(A B C D)
fi

echo "============================================================"
echo "ABLATION STUDY"
echo "============================================================"
echo "GPU: $GPU"
echo "Runs: ${RUNS[*]}"
echo "============================================================"
echo ""

for RUN in "${RUNS[@]}"; do
  CONFIG_FILE="${CONFIGS_DIR}/${CONFIGS[$RUN]}"
  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config not found: $CONFIG_FILE"
    exit 1
  fi

  echo "============================================================"
  echo "RUN $RUN: ${DESCRIPTIONS[$RUN]}"
  echo "Config: $CONFIG_FILE"
  echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  python -m production.train_production \
    --config "$CONFIG_FILE" \
    --gpu "$GPU"

  echo ""
  echo "RUN $RUN complete: $(date '+%Y-%m-%d %H:%M:%S')"
  echo ""
done

echo "============================================================"
echo "ALL ABLATION RUNS COMPLETE"
echo "============================================================"
