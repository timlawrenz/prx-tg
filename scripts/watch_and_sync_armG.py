import time
import subprocess
import sys

vast_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15", "-i", "/home/tim/.ssh/id_rsa", "-p", "13840", "root@ssh3.vast.ai"]

def is_running():
    try:
        res = subprocess.run(vast_cmd + ["pgrep", "-f", "production.train_production"], capture_output=True, text=True, timeout=30)
        if res.returncode == 0 and res.stdout.strip():
            return True
        if res.returncode == 1: # pgrep found nothing
            return False
        # If return code is 255 (SSH error) or something else, assume network issue
        return True
    except Exception as e:
        print(f"Network check error: {e}")
        return True # Assume running to avoid false positive exit

print("Starting watchdog for Arm G training on ssh3.vast.ai...")

# Wait loop
failures = 0
while True:
    if not is_running():
        failures += 1
        print(f"Process not found. Verification {failures}/3...")
        if failures >= 3:
            print("Confirmed process is no longer running. Proceeding to sync.")
            break
        time.sleep(60)
    else:
        failures = 0
        time.sleep(300) # Check every 5 minutes

print("Initiating final rsync...")

rsync_cmd_dir = [
    "rsync", "-avz", "-e", "ssh -o StrictHostKeyChecking=no -i /home/tim/.ssh/id_rsa -p 13840",
    "root@ssh3.vast.ai:/workspace/prx-tg/experiments/2026-05-21_1315/",
    "experiments/asym-flow-ablation/runs/2026-05-21_1315/"
]
subprocess.run(rsync_cmd_dir, check=False)

rsync_cmd_log = [
    "rsync", "-avz", "-e", "ssh -o StrictHostKeyChecking=no -i /home/tim/.ssh/id_rsa -p 13840",
    "root@ssh3.vast.ai:/workspace/armG.log",
    "experiments/asym-flow-ablation/runs/2026-05-21_1315/"
]
subprocess.run(rsync_cmd_log, check=False)

print("DONE! Final assets and logs synced successfully.")
