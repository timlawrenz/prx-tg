#!/bin/bash
# Second probe pass for the eidolon-identity-renderer arm:
#   A. z_g yaw sweep on the NEWEST checkpoint (extends the 5000 -> 7000 emergence curve)
#   B. seed-variation identity probe (same identity conditioning, different noise seeds)
# The step-5000 sweep is already measured and committed, so it is skipped here.
# Requests the GPU through the scheduler and waits its turn; never sets -e so an
# untested probe cannot strand the claim.
set -u

REPO=/home/tim/source/activity/prx-tg
PY=$REPO/.venv/bin/python
SCHED=/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py
JOB=prx-tg-eir-probes-v2
RUN=/mnt/nas-ai-models/training-data/prx-tg/eidolon-identity-renderer/runs/2026-09-25_0734
CK=$RUN/checkpoints
BASE=/home/tim/.hermes/profiles/prx-tg/cache/scratch
LOG=$BASE/probes_runner2.log
OUT=$BASE/eir_probes2
CFG=$REPO/experiment-configs/eidolon-identity-renderer/config.yaml

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
cd "$REPO" || exit 1
mkdir -p "$OUT/latest" "$OUT/seeds"
log "=== probe runner v2 start (pid $$) ==="

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
  log "NEVER GRANTED after 12h - exiting, nothing claimed, nothing held"
  exit 1
fi

$PY "$SCHED" activate --gpu 4090 --job-id "$JOB" >>"$LOG" 2>&1
log "granted + activated"

LATEST=$(ls -t "$CK"/checkpoint_step*.pt 2>/dev/null | head -1)
log "newest checkpoint = $LATEST"

log "probe A: yaw sweep @ $LATEST"
$PY scripts/diag_eidolon_zg_dilution.py --config "$CFG" \
    --checkpoint "$LATEST" --out-dir "$OUT/latest" >>"$LOG" 2>&1
log "probe A exit=$?"

log "probe B: seed-variation identity @ $LATEST"
$PY scripts/diag_identity_seed_consistency.py --config "$CFG" \
    --checkpoint "$LATEST" --seeds 6 --out-dir "$OUT/seeds" >>"$LOG" 2>&1
log "probe B exit=$?"

$PY "$SCHED" release --gpu 4090 --job-id "$JOB" --status completed >>"$LOG" 2>&1
log "released; runner v2 done"
