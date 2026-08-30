#!/bin/bash
# Chain: wait for BT-Net → launch Arm O → SIGTERM at 6am
# Run with: nohup bash /home/tim/source/activity/prx-tg/scripts/chain_arm_o.sh > /tmp/chain_arm_o.log 2>&1 &

set -euo pipefail

LOGFILE=/tmp/chain_arm_o.log
exec >> "$LOGFILE" 2>&1

echo "=== Chain started at $(date) ==="

# ── Phase 1: Wait for BreastTissue-Net training to finish ──
echo "[chain] Phase 1: Waiting for BT-Net (pgrep 'train.py.*config.json')..."
while pgrep -f "[p]ython.*train\\.py.*config\\.json" > /dev/null 2>&1; do
    echo "[chain]   BT-Net still running at $(date +%H:%M:%S)..."
    sleep 30
done
echo "[chain] Phase 1 complete: BT-Net finished at $(date)"

# Short cooldown to let GPU memory settle
sleep 10

# ── Phase 2: Launch Arm O ──
echo "[chain] Phase 2: Launching Arm O..."
cd /home/tim/source/activity/prx-tg

PYTORCH_ALLOC_CONF=expandable_segments:True \
    nohup .venv/bin/python3 -m production.train_production \
    --config experiments/zg-token-basis-cfg-guard/config.yaml \
    > /tmp/arm_o_train.log 2>&1 &

ARM_O_PID=$!
echo "[chain]   Arm O PID: $ARM_O_PID"
echo "[chain]   Training log: /tmp/arm_o_train.log"

# ── Phase 3: Calculate sleep until 06:00 ──
TARGET=$(date -d "06:00" +%s)
NOW=$(date +%s)
if [ "$TARGET" -le "$NOW" ]; then
    TARGET=$(date -d "tomorrow 06:00" +%s)
fi
SLEEP_SEC=$((TARGET - NOW))
SLEEP_MIN=$((SLEEP_SEC / 60))
echo "[chain] Phase 3: Sleeping ${SLEEP_SEC}s (~${SLEEP_MIN}m) until $(date -d @$TARGET)"

sleep "$SLEEP_SEC"

# ── Phase 4: SIGTERM Arm O at 6am ──
echo "[chain] Phase 4: $(date) — Sending SIGTERM to Arm O (PID=$ARM_O_PID)..."
kill -TERM "$ARM_O_PID" 2>/dev/null || echo "[chain]   Process already gone"

# Wait for interrupt checkpoint save
sleep 15
if kill -0 "$ARM_O_PID" 2>/dev/null; then
    echo "[chain]   Process still alive after 15s — sending SIGKILL..."
    kill -KILL "$ARM_O_PID" 2>/dev/null || true
fi

echo "=== Chain finished at $(date) ==="
