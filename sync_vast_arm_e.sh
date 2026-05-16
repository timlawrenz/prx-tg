#!/bin/bash
TIMESTAMP=$(date +%Y-%m-%d_%H%M)
DEST="/mnt/nas-ai-models/training-data/prx-tg/vast_sync_arm_e_${TIMESTAMP}"
FINAL_DEST="/mnt/nas-ai-models/training-data/prx-tg/ablation_arm_e"
SSH_OPTS="-o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -i /home/tim/.ssh/id_rsa -p 13840"
HOST="root@ssh3.vast.ai"

mkdir -p "$DEST"
mkdir -p "$FINAL_DEST"

rsync -avz --size-only -e "ssh $SSH_OPTS" "$HOST:/workspace/armE.log" "$DEST/" || true
rsync -avz --size-only -e "ssh $SSH_OPTS" "$HOST:/workspace/bootstrap.log" "$DEST/" || true
rsync -avz --size-only --link-dest="$FINAL_DEST" -e "ssh $SSH_OPTS" "$HOST:/workspace/experiments/" "$DEST/experiments/" || true
rsync -avz --size-only "$DEST/experiments/" "$FINAL_DEST/" || true
ls -dt /mnt/nas-ai-models/training-data/prx-tg/vast_sync_arm_e_* | tail -n +8 | xargs rm -rf 2>/dev/null || true
echo "[$(date)] Sync complete: $DEST and $FINAL_DEST"
