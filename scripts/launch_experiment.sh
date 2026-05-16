#!/usr/bin/env bash
# scripts/launch_experiment.sh — Governed experiment launcher
# Enforces pre-flight checks, freezes config, creates metadata, then trains.
#
# Usage:
#   ./scripts/launch_experiment.sh <config_yaml> [--gpu 0] [--resume <ckpt>] [--experiment-name <name>]
#
# Example:
#   ./scripts/launch_experiment.sh scripts/arm_e_config.yaml --experiment-name arm_e_seg_weight

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Parse arguments ──────────────────────────────────────────────────────────
CONFIG=""
GPU="0"
RESUME=""
EXP_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --experiment-name) EXP_NAME="$2"; shift 2 ;;
    -*) echo "Unknown flag: $1"; exit 1 ;;
    *) CONFIG="$1"; shift ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "ERROR: No config file specified."
  echo "Usage: $0 <config.yaml> [--gpu N] [--resume <ckpt>] [--experiment-name <name>]"
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: Config file not found: $CONFIG"
  exit 1
fi

# ── Pre-flight checks ───────────────────────────────────────────────────────

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              PRX-TG EXPERIMENT LAUNCHER                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 1. Git cleanliness
GIT_COMMIT=$(git rev-parse HEAD)
GIT_SHORT=$(git rev-parse --short HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [[ -n "$(git status --porcelain)" ]]; then
  echo "⚠  WARNING: Uncommitted changes detected!"
  echo "   This run will NOT be reproducible."
  echo ""
  git status --short
  echo ""
  read -p "Continue anyway? [y/N] " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. Commit your changes first."
    exit 1
  fi
  GIT_DIRTY=true
else
  echo "✓  Git is clean: $GIT_SHORT ($GIT_BRANCH)"
  GIT_DIRTY=false
fi

# 2. Data availability
STRATUM_DIR="${STRATUM_DIR:-}"
SHARD_DIR=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
print(c.get('data',{}).get('shard_base_dir','data/shards/faces7k'))
" 2>/dev/null || echo "data/shards/faces7k")

if [[ -n "$STRATUM_DIR" && -d "$STRATUM_DIR" ]]; then
  STRATUM_COUNT=$(ls "$STRATUM_DIR" 2>/dev/null | wc -l)
  echo "✓  Stratum dir: $STRATUM_DIR ($STRATUM_COUNT items)"
elif [[ -d "$SHARD_DIR" ]]; then
  SHARD_COUNT=$(find "$SHARD_DIR" -name "*.tar" 2>/dev/null | wc -l)
  echo "✓  Shard dir: $SHARD_DIR ($SHARD_COUNT shards)"
else
  echo "⚠  WARNING: No data directory found (STRATUM_DIR unset, $SHARD_DIR missing)"
fi

# 3. GPU check
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU" 2>/dev/null || echo "unknown")
  GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader -i "$GPU" 2>/dev/null || echo "unknown")
  echo "✓  GPU $GPU: $GPU_NAME ($GPU_MEM)"
else
  echo "⚠  nvidia-smi not found (running on CPU?)"
fi

# ── Create experiment directory ──────────────────────────────────────────────
TIMESTAMP=$(date +%Y-%m-%d_%H%M)
EXP_DIR="experiments/${TIMESTAMP}"
if [[ -n "$EXP_NAME" ]]; then
  EXP_DIR="experiments/${EXP_NAME}/${TIMESTAMP}"
fi
mkdir -p "$EXP_DIR/checkpoints" "$EXP_DIR/tensorboard" "$EXP_DIR/validation" "$EXP_DIR/visual_debug"

# ── Freeze config ────────────────────────────────────────────────────────────
cp "$CONFIG" "$EXP_DIR/config.yaml"
echo "✓  Config frozen to: $EXP_DIR/config.yaml"

# ── Write metadata ───────────────────────────────────────────────────────────
COMMAND="python -m production.train_production --config $CONFIG --gpu $GPU"
if [[ -n "$RESUME" ]]; then
  COMMAND="$COMMAND --resume $RESUME"
fi

cat > "$EXP_DIR/metadata.json" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "config_source": "$CONFIG",
  "config_frozen": "$EXP_DIR/config.yaml",
  "command": "$COMMAND",
  "git_commit": "$GIT_COMMIT",
  "git_branch": "$GIT_BRANCH",
  "git_dirty": $GIT_DIRTY,
  "gpu_index": $GPU,
  "gpu_name": "${GPU_NAME:-unknown}",
  "experiment_name": "${EXP_NAME:-}",
  "resume_from": "${RESUME:-null}",
  "hostname": "$(hostname)",
  "pytorch_version": "$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
}
EOF

echo "✓  Metadata written to: $EXP_DIR/metadata.json"
echo ""

# ── Print summary ────────────────────────────────────────────────────────────
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│  Config:     $CONFIG"
echo "│  Experiment: $EXP_DIR"
echo "│  Commit:     $GIT_SHORT ($GIT_BRANCH)"
echo "│  GPU:        $GPU (${GPU_NAME:-unknown})"
if [[ -n "$RESUME" ]]; then
echo "│  Resume:     $RESUME"
fi
echo "│  Dirty:      $GIT_DIRTY"
echo "└─────────────────────────────────────────────────────────────┘"
echo ""

# ── Launch training ──────────────────────────────────────────────────────────
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Launching training..."
echo ""

TRAIN_ARGS=(
  --config "$EXP_DIR/config.yaml"
  --gpu "$GPU"
)
if [[ -n "$RESUME" ]]; then
  TRAIN_ARGS+=(--resume "$RESUME")
fi
if [[ -n "$EXP_NAME" ]]; then
  TRAIN_ARGS+=(--experiment-name "$EXP_NAME")
fi

python -m production.train_production "${TRAIN_ARGS[@]}" 2>&1 | tee "$EXP_DIR/training.log"

EXIT_CODE=${PIPESTATUS[0]}

# ── Post-run ─────────────────────────────────────────────────────────────────
echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "✓  Training completed successfully."
  echo "   Run validation with:"
  echo "   python scripts/run_checkpoint_validation.py \\"
  echo "     --config $EXP_DIR/config.yaml \\"
  echo "     --checkpoint $EXP_DIR/checkpoints/checkpoint_final.pt"
else
  echo "✗  Training failed with exit code $EXIT_CODE"
fi

# Append completion status to metadata
python3 -c "
import json
with open('$EXP_DIR/metadata.json') as f:
    m = json.load(f)
m['exit_code'] = $EXIT_CODE
m['completed_at'] = '$(date -Iseconds)'
with open('$EXP_DIR/metadata.json', 'w') as f:
    json.dump(m, f, indent=2)
"

exit $EXIT_CODE
