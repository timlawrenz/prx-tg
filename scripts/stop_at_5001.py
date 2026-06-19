import time
import json
import subprocess
import os
import sys

log_file = "experiments/2026-06-04_2116/training_log.jsonl"
target_step = 5001

print(f"Waiting for step >= {target_step} in {log_file}...")

with open(log_file, 'r') as f:
    # Seek to end so we don't re-process old events if we restart the script
    # Wait, actually let's read from the beginning to catch if it already reached it.
    pass

with open(log_file, 'r') as f:
    while True:
        line = f.readline()
        if not line:
            # Check if training process is still alive
            res = subprocess.run(["pgrep", "-f", "production.train_production"], capture_output=True)
            if res.returncode != 0:
                print("Training process exited before reaching target step.")
                sys.exit(0)
            time.sleep(5)
            continue
        
        try:
            data = json.loads(line.strip())
            step = data.get('step', 0)
            if step >= target_step:
                print(f"Reached step {step}! Sending SIGINT to gracefully stop training...")
                subprocess.run(["pkill", "-SIGINT", "-f", "production.train_production"])
                print("Signal sent. Exiting watcher.")
                sys.exit(0)
        except Exception:
            pass
