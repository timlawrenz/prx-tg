#!/bin/bash
# Queue the EIR probe pair with the GPU scheduler and run them when granted.
#
# Requests a 4090 slot, polls until the scheduler grants it (the current holder
# releases at its block boundary and the queued job goes first), then runs:
#   1. the z_g yaw sweep on step 5000  (Arm O's emergence step, for comparison)
#   2. the z_g yaw sweep on the newest checkpoint (trajectory)
#   3. the seed-variation identity probe on the newest checkpoint
# and releases. Deliberately does NOT set -e: an untested probe must not strand
# the claim, so every step runs and the release always happens.
set -u

REPO=/home/tim/source/activity/prx-tg
PY=$REPO/.venv/bin/python
SCHED=/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py
JOB=prx-tg-eir-probes-v1
RUN=/mnt/nas-ai-models/training-data/prx-tg/eidolon-identity-renderer/runs/2026-09-25_0734
CK=$RUN/checkpoints
BASE=/home/tim/.hermes/profiles/prx-tg/cache/scratch
LOG=$BASE/probes_runner.log
OUT=$BASE/eir_probes
CFG=$REPO/experiment-configs/eidolon-identity-renderer/config.yaml

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
cd "$REPO" || exit 1

mkdir -p "$OUT/step5000" "$OUT/latest" "$OUT/seeds"
log "=== probe runner start (pid $$) ==="

$PY "$SCHED" request --gpu 4090 --project prx-tg --vram 8 --duration 45m --job-id "$JOB" >>"$LOG" 2>&1
log "requested $JOB"

R=""
for i in $(seq 1 720); do                      # up to 12 h
  R=$($PY "$SCHED" poll --gpu 4090 --job-id "$JOB" 2>&1 | tail -1)
  log "poll -> $R"
  echo "$R" | grep -qi "claimed" && break
  sleep 60
done

if ! echo "$R" | grep -qi "claimed"; then
  log "NEVER GRANTED after 12h — exiting without running (nothing claimed, nothing held)"
  exit 1
fi

$PY "$SCHED" activate --gpu 4090 --job-id "$JOB" >>"$LOG" 2>&1
log "granted + activated; running probes"

# 1. step 5000 (the step where Arm O's binding appeared)
if [ -f "$CK/checkpoint_step0005000.pt" ]; then
  log "probe 1: yaw sweep @ step 5000"
  $PY scripts/diag_eidolon_zg_dilution.py --config "$CFG" \
      --checkpoint "$CK/checkpoint_step0005000.pt" --out-dir "$OUT/step5000" >>"$LOG" 2>&1
  log "probe 1 exit=$?"
fi

LATEST=$(ls -t "$CK"/checkpoint_step*.pt 2>/dev/null | head -1)
log "newest checkpoint = $LATEST"

# 2. newest checkpoint — same sweep
log "probe 2: yaw sweep @ $LATEST"
$PY scripts/diag_eidolon_zg_dilution.py --config "$CFG" \
    --checkpoint "$LATEST" --out-dir "$OUT/latest" >>"$LOG" 2>&1
log "probe 2 exit=$?"

# 3. seed-variation identity probe on the newest checkpoint
log "probe 3: seed-variation identity"
$PY scripts/diag_identity_seed_consistency.py --config "$CFG" \
    --checkpoint "$LATEST" --seeds 6 --out-dir "$OUT/seeds" >>"$LOG" 2>&1
log "probe 3 exit=$?"

$PY "$SCHED" release --gpu 4090 --job-id "$JOB" --status completed >>"$LOG" 2>&1
log "released; runner done"
