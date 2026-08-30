#!/usr/bin/env python3
import json
import os
import subprocess
import sys

JOB_ID = "fdcb5b74df70"
GPU = "4090"
SCHEDULER_PY = "/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py"
LOG_JSONL = "/home/tim/source/activity/prx-tg/experiments/zg-token-basis-cfg-guard/runs/2026-07-20_0603/training_log.jsonl"

def is_training_running():
    res = subprocess.run(["pgrep", "-f", "production.train_production"], capture_output=True, text=True)
    return res.returncode == 0

def get_latest_metrics():
    if not os.path.exists(LOG_JSONL):
        return 0, 0.0
    try:
        with open(LOG_JSONL, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            if not lines:
                return 0, 0.0
            last_line = lines[-1]
            data = json.loads(last_line)
            step = data.get("step", 0)
            vram_gb = round(data.get("vram_reserved_gb", 0.0), 1)
            return step, vram_gb
    except Exception as e:
        return 0, 0.0

def main():
    if is_training_running():
        step, vram_gb = get_latest_metrics()
        cmd = [
            sys.executable, SCHEDULER_PY, "heartbeat",
            "--gpu", GPU,
            "--job-id", JOB_ID,
            "--progress", str(step),
            "--vram-used", str(vram_gb)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Heartbeat failed: {res.stderr or res.stdout}")
            sys.exit(1)
        # Success: silent (no stdout)
    else:
        # Released slot and alert
        cmd = [sys.executable, SCHEDULER_PY, "release", "--gpu", GPU, "--job-id", JOB_ID, "--status", "failed"]
        subprocess.run(cmd, capture_output=True, text=True)
        print(f"ALERT: Training process for job {JOB_ID} is no longer running! Released GPU slot.")
        sys.exit(0)

if __name__ == "__main__":
    main()
