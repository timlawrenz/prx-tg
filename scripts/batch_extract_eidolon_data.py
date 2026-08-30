#!/usr/bin/env python3
"""One-shot batch: copy z_g.npy + compute auraface_lda.npy for all stratum items."""
import sys, os, time
import numpy as np
from pathlib import Path

STRATUM = Path("/mnt/nas-ai-models/training-data/ffhq/stratum")
ZG_SRC  = Path("/mnt/nas-ai-models/training-data/ffhq/zg")
AF_SRC  = Path("/mnt/nas-ai-models/training-data/ffhq/auraface")
ARTIFACT_DIR = Path("/home/tim/source/activity/eidolon/experiments/geometry_pca/output")

# --- Load reference artifacts -------------------------------------------------
prep = np.load(ARTIFACT_DIR / "auraface_preprocess.npz")
mu    = prep["pooled_mean"]       # (512,)
pc1   = prep["pc1_direction"]     # (512,)
yaw   = prep["yaw_direction"]     # (512,)

lda = np.load(ARTIFACT_DIR / "auraface_lda.npz")
W      = lda["lda_basis"]         # (512, 64)
mu_lda = lda["pooled_mean"]       # (512,)

def clean_auraface(v):
    """Remove domain (PC1) and pose (yaw) nuisances, L2 renormalize."""
    v = np.asarray(v, dtype=np.float64)
    was_1d = v.ndim == 1
    v = np.atleast_2d(v)
    vc = v - mu
    vc = vc - np.outer(vc @ pc1, pc1)
    vc = vc - np.outer(vc @ yaw, yaw)
    norms = np.linalg.norm(vc, axis=1, keepdims=True)
    vc = vc / (norms + 1e-12)
    return vc[0] if was_1d else vc

def project_to_lda(v_clean):
    """Project cleaned AuraFace vector onto LDA identity basis → (64,) float64."""
    v = np.atleast_2d(np.asarray(v_clean, dtype=np.float64))
    vc = v - mu_lda
    return (vc @ W).squeeze()

# --- Main processing ----------------------------------------------------------
stratum_ids = sorted(d.name for d in STRATUM.iterdir() if d.is_dir())
total = len(stratum_ids)
print(f"Found {total} stratum directories")

t0 = time.time()
zg_copied = 0
af_processed = 0
af_skipped_exists = 0
zg_missing = 0
af_missing = 0

for i, sid in enumerate(stratum_ids):
    # --- z_g: copy ------------------------------------------------------------
    zg_src = ZG_SRC / sid / "zg.npy"
    zg_dst = STRATUM / sid / "z_g.npy"
    if zg_src.exists():
        if not zg_dst.exists():
            import shutil
            shutil.copy2(zg_src, zg_dst)
            zg_copied += 1
    else:
        zg_missing += 1

    # --- auraface_lda: clean + project ----------------------------------------
    af_src = AF_SRC / f"{sid}.npy"
    af_dst = STRATUM / sid / "auraface_lda.npy"
    if af_src.exists():
        if not af_dst.exists():
            raw = np.load(af_src)               # (512,) float32
            cleaned = clean_auraface(raw)         # (512,) float64
            lda_vec = project_to_lda(cleaned)    # (64,) float64
            np.save(af_dst, lda_vec)
            af_processed += 1
        else:
            af_skipped_exists += 1
    else:
        af_missing += 1

    if (i + 1) % 5000 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (total - i - 1) / rate
        print(f"  [{i+1}/{total}] {rate:.0f}/s, ETA {eta:.0f}s  "
              f"| zg_copied={zg_copied} af_processed={af_processed}")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")
print(f"  z_g:     copied={zg_copied}  missing_src={zg_missing}")
print(f"  auraface: processed={af_processed}  skipped_existing={af_skipped_exists}  missing_src={af_missing}")

# --- Verify a few samples -----------------------------------------------------
print("\n=== Verification (first 3 samples) ===")
for sid in stratum_ids[:3]:
    dirpath = STRATUM / sid
    zg = np.load(dirpath / "z_g.npy")
    af = np.load(dirpath / "auraface_lda.npy")
    print(f"  {sid}: z_g={zg.shape} {zg.dtype}  auraface_lda={af.shape} {af.dtype}")
