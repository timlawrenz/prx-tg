#!/usr/bin/env bash
# setup_vastai.sh — bootstrap stratum dataset on a fresh Vast.ai instance
#
# Sources:
#   HuggingFace  timlawrenz/stratum-ffhq  → seg, depth, pose, t5, dinov3, caption, metadata
#   Cloudflare R2 ffhq-pixels             → pixel.npy  (first 7000 samples only)
#
# Usage:
#   export HF_TOKEN=...
#   export R2_ACCESS_KEY_ID=...
#   export R2_SECRET_ACCESS_KEY=...
#   bash scripts/setup_vastai.sh
#
# After completion, STRATUM_DIR will contain 00000/–06999/ each with:
#   caption.txt  metadata.json  seg.npy  depth.npy  pose.npy
#   t5_hidden.npy  t5_mask.npy  dinov3_cls.npy  dinov3_patches.npy  pixel.npy

set -euo pipefail

STRATUM_DIR="${STRATUM_DIR:-/workspace/stratum}"
N_SAMPLES=7000        # first 7k for Arm E ablation
HF_REPO="timlawrenz/stratum-ffhq"
R2_ENDPOINT="https://78be4b701cdf857a09e55203cec6d2d4.r2.cloudflarestorage.com"
R2_BUCKET="s3://ffhq-pixels"
JOBS=8                # parallel R2 downloads

: "${HF_TOKEN:?HF_TOKEN not set}"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID not set}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY not set}"

mkdir -p "$STRATUM_DIR"
echo "[setup] STRATUM_DIR = $STRATUM_DIR"
echo "[setup] N_SAMPLES   = $N_SAMPLES"

# ---------------------------------------------------------------------------
# 1. Install dependencies
# ---------------------------------------------------------------------------
echo "[setup] Installing huggingface_hub, pyarrow, awscli..."
pip install -q huggingface_hub pyarrow awscli

# ---------------------------------------------------------------------------
# 2. Caption + metadata from parquet (first N_SAMPLES rows)
# ---------------------------------------------------------------------------
echo "[setup] Writing caption.txt and metadata.json..."
python3 - <<'PYEOF'
import os, json, math
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

stratum_dir = os.environ["STRATUM_DIR"]
n_samples   = int(os.environ["N_SAMPLES"])
hf_repo     = os.environ["HF_REPO"]
token       = os.environ["HF_TOKEN"]

# parquet shards are 10k rows each; we only need shard 0 (00000-09999)
needed_shards = math.ceil(n_samples / 10000)  # =1 for 7k
shard_files = [f"data/{10000*i:05d}-{10000*(i+1)-1:05d}.parquet" for i in range(needed_shards)]

rows = []
for sf in shard_files:
    path = hf_hub_download(hf_repo, sf, repo_type="dataset", token=token)
    tbl  = pq.read_table(path)
    rows.extend(tbl.to_pylist())

rows = rows[:n_samples]
print(f"[caption] Writing {len(rows)} samples...")

for row in rows:
    iid = row["image_id"]   # e.g. "00000"
    d = os.path.join(stratum_dir, iid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "caption.txt"), "w") as f:
        f.write(row["caption"])
    meta = {
        "image_id":      iid,
        "width":         row["width"],
        "height":        row["height"],
        "aspect_bucket": row["aspect_bucket"],
        "source_path":   "",
    }
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump(meta, f)

print("[caption] Done.")
PYEOF

# ---------------------------------------------------------------------------
# 3. npy_tar layers: seg, depth, pose, t5, dinov3
#    Tar internal structure: NNNNN/<file>.npy  → extract directly into STRATUM_DIR
# ---------------------------------------------------------------------------
echo "[setup] Downloading and extracting npy_tar layers..."

python3 - <<'PYEOF'
import os, math, tarfile
from huggingface_hub import hf_hub_download

stratum_dir = os.environ["STRATUM_DIR"]
n_samples   = int(os.environ["N_SAMPLES"])
hf_repo     = os.environ["HF_REPO"]
token       = os.environ["HF_TOKEN"]

# Each tar shard covers 1000 samples: 00000-00999, 01000-01999, ...
needed_shards = math.ceil(n_samples / 1000)  # =7 for 7k

LAYERS = {
    "seg":    ["seg.npy"],
    "depth":  ["depth.npy"],
    "pose":   ["pose.npy"],
    "t5":     ["t5_hidden.npy", "t5_mask.npy"],
    "dinov3": ["dinov3_cls.npy", "dinov3_patches.npy"],
}

for layer, files in LAYERS.items():
    print(f"[{layer}] Downloading {needed_shards} shards...")
    for s in range(needed_shards):
        lo = s * 1000
        hi = lo + 999
        
        if layer == "dinov3":
            # DINOv3 is sharded by 7 samples instead of 1000
            # For Arm E, skip DINOv3 in this general download loop and handle specially
            print("[dinov3] Skipping general 1000-shard download; must be handled per 7-samples")
            break
            
        tar_name = f"{layer}/{lo:05d}-{hi:05d}.tar"
        local_tar = hf_hub_download(hf_repo, tar_name, repo_type="dataset", token=token)
        with tarfile.open(local_tar) as tf:
            # Only extract members within our sample range
            members = [m for m in tf.getmembers()
                       if any(m.name.endswith(fn) for fn in files)]
            tf.extractall(path=stratum_dir, members=members)

    if layer == "dinov3":
        import math
        needed_7_shards = math.ceil(n_samples / 7)
        print(f"[dinov3] Downloading {needed_7_shards} 7-sample shards...")
        for s in range(needed_7_shards):
            lo = s * 7
            hi = lo + 6
            if lo >= n_samples:
                break
            tar_name = f"dinov3/{lo:05d}-{hi:05d}.tar"
            try:
                local_tar = hf_hub_download(hf_repo, tar_name, repo_type="dataset", token=token)
                with tarfile.open(local_tar) as tf:
                    tf.extractall(path=stratum_dir)
            except Exception as e:
                print(f"Failed to fetch {tar_name}: {e}")
    print(f"[{layer}] Done.")

print("[npy_tar] All layers extracted.")
PYEOF

# ---------------------------------------------------------------------------
# 4. pixel.npy from R2 (parallel with xargs)
# ---------------------------------------------------------------------------
echo "[setup] Downloading pixel.npy from R2 ($N_SAMPLES samples, $JOBS workers)..."

export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export STRATUM_DIR R2_ENDPOINT R2_BUCKET

R2_LOG="/tmp/setup_r2_pixels.log"
: > "$R2_LOG"

download_pixel() {
    local idx=$1
    local padded
    padded=$(printf "%05d" "$idx")
    local dst="$STRATUM_DIR/$padded/pixel.npy"
    if [[ -f "$dst" ]]; then
        echo "SKIP $idx" >> "$R2_LOG"
        return
    fi
    aws s3 cp "$R2_BUCKET/$padded/pixel.npy" "$dst" \
        --endpoint-url "$R2_ENDPOINT" \
        --no-progress 2>>"$R2_LOG" \
        && echo "OK $idx" >> "$R2_LOG" \
        || echo "FAIL $idx" >> "$R2_LOG"
}

export -f download_pixel
export STRATUM_DIR R2_BUCKET R2_ENDPOINT R2_LOG

seq 0 $((N_SAMPLES - 1)) | \
    xargs -P "$JOBS" -I{} bash -c 'download_pixel "$@"' _ {}

OK_COUNT=$(grep -c "^OK\|^SKIP" "$R2_LOG" || true)
FAIL_COUNT=$(grep -c "^FAIL" "$R2_LOG" || true)
echo "[setup] R2 pixel.npy: OK/SKIP=$OK_COUNT  FAIL=$FAIL_COUNT / $N_SAMPLES"
if [[ "$FAIL_COUNT" -gt 0 ]]; then
    echo "[setup] WARNING: $FAIL_COUNT pixel downloads failed. Check $R2_LOG"
fi

# ---------------------------------------------------------------------------
# 5. Sanity check
# ---------------------------------------------------------------------------
echo "[setup] Sanity check on first 3 samples..."
python3 - <<'PYEOF'
import os, json
import numpy as np

stratum_dir = os.environ["STRATUM_DIR"]
expected = ["caption.txt", "metadata.json", "seg.npy", "depth.npy",
            "pose.npy", "t5_hidden.npy", "t5_mask.npy",
            "dinov3_cls.npy", "dinov3_patches.npy", "pixel.npy"]

for i in range(3):
    d = os.path.join(stratum_dir, f"{i:05d}")
    missing = [f for f in expected if not os.path.exists(os.path.join(d, f))]
    if missing:
        print(f"  FAIL {i:05d}: missing {missing}")
    else:
        pixel = np.load(os.path.join(d, "pixel.npy"))
        seg   = np.load(os.path.join(d, "seg.npy"))
        print(f"  OK {i:05d}: pixel={pixel.shape} seg={seg.shape}")

print("[setup] Complete.")
PYEOF

echo ""
echo "================================================================"
echo " Setup complete. Train with:"
echo "   export STRATUM_DIR=$STRATUM_DIR"
echo "   python -m production.train_production --config scripts/arm_e_config.yaml"
echo "================================================================"
