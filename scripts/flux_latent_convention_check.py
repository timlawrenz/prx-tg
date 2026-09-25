#!/usr/bin/env python3
"""Pre-flight for flux-latent generation: prove the encoder path is the SAME
convention that produced the existing on-disk latents.

Why this exists
---------------
FFHQ's flux_latent.npy files were produced by the comfy AutoencodingEngine path
(raw 16-ch latents, no scaling). prx-tg's own production/flux_ae.py is a
decode-only diffusers wrapper. If a different encoder (e.g. one that applies a
scaling factor) generated hegre's latents, the two datasets would sit on
different latent scales in the same training slot — a silent confound of exactly
the class that already burned this project once (mixed identity bases).

So: before encoding 31,711 new samples, re-encode samples whose latents ALREADY
exist and require the reconstruction to match. This is the control for
Measurement Provenance Equivalence — numbers are only comparable within an
identical execution context, and this makes that context explicit.

Checks
------
  A. FFHQ round-trip: encode pixel.npy -> compare against on-disk flux_latent.npy
     (same weights, same code path). Reports max/mean/p95 abs diff.
  B. Hegre encode sanity: encode N hegre samples, assert shape (16,128,128),
     floating dtype, all-finite, plausible magnitude. Does NOT write anything.

Exit 0 and print "PRECHECK PASS" only if BOTH pass, so the launcher can use this
as a fail-closed gate before taking a multi-hour GPU window.

Usage:
    /mnt/fscache/essdee/ComfyUI/.venv/bin/python3 scripts/flux_latent_convention_check.py \
        --n 4 --n-hegre 3 --device cuda
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

FFHQ = "/mnt/nas-ai-models/training-data/ffhq/stratum"
HEGRE = "/mnt/nas-ai-models/training-data/eidolon/hegre_corpus"
AE_PATH = "/mnt/models/vae/ae.safetensors"
sys.path.insert(0, "/mnt/fscache/essdee/ComfyUI")

import comfy.sd as csd          # noqa: E402
import comfy.utils as cu        # noqa: E402

# Gate thresholds. A scale/mapping error (the failure this guards) shows up as
# O(0.1-1) differences; fp16 storage rounding of a correct path is O(1e-3).
MAX_ABS_PASS = 0.02
MAX_ABS_FAIL = 0.10


def encode_one(vae, pixel_path, device):
    """Identical call path to scripts/generate_flux_latents.py."""
    pix = np.load(pixel_path, mmap_mode="r").astype(np.float32)
    if pix.shape != (3, 1024, 1024):
        raise ValueError(f"unexpected pixel shape {pix.shape} in {pixel_path}")
    t = torch.from_numpy(pix).unsqueeze(0).to(device)
    with torch.no_grad():
        z = vae.encode(t.permute(0, 2, 3, 1).contiguous())  # comfy wants BHWC
        if isinstance(z, tuple):
            z = z[0]
    return z.to(torch.float16)


def pick(root, n, prefer_latent_exists=False):
    """Deterministic spread across the dir listing (not the first N)."""
    dirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if prefer_latent_exists:
        dirs = [d for d in dirs
                if os.path.exists(os.path.join(root, d, "pixel.npy"))
                and os.path.exists(os.path.join(root, d, "flux_latent.npy"))]
    else:
        dirs = [d for d in dirs if os.path.exists(os.path.join(root, d, "pixel.npy"))]
    if not dirs:
        return []
    step = max(1, len(dirs) // max(n, 1))
    return dirs[::step][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="FFHQ control samples")
    ap.add_argument("--n-hegre", type=int, default=3, help="hegre sanity samples")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--ffhq-root", type=str, default=FFHQ)
    ap.add_argument("--hegre-root", type=str, default=HEGRE)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[init] device={args.device} torch={torch.__version__}")
    vae = csd.VAE(sd=cu.load_torch_file(AE_PATH))
    try:
        vae = vae.to(args.device)
    except Exception:
        pass
    print(f"[vae] FLUX AE loaded ({time.time()-t0:.1f}s)")

    ok = True

    # ---- A. FFHQ round-trip control -------------------------------------
    names = pick(args.ffhq_root, args.n, prefer_latent_exists=True)
    if len(names) < args.n:
        print(f"[A] FATAL: only {len(names)} FFHQ dirs have both pixel and flux_latent")
        ok = False
    diffs = []
    for d in names:
        dp = os.path.join(args.ffhq_root, d)
        on_disk = np.load(os.path.join(dp, "flux_latent.npy")).astype(np.float32)
        z = encode_one(vae, os.path.join(dp, "pixel.npy"), args.device)
        zf = z[0].cpu().numpy().astype(np.float32)
        if zf.shape != on_disk.shape:
            print(f"[A] {d}: SHAPE MISMATCH fresh{zf.shape} vs disk{on_disk.shape}")
            ok = False
            continue
        ad = np.abs(zf - on_disk)
        mad = float(ad.mean())
        mx = float(ad.max())
        p95 = float(np.percentile(ad, 95))
        amp = float(on_disk.std())
        diffs.append((d, mx, mad, p95, amp))
        print(f"[A] {d}: max={mx:.5f} mean={mad:.6f} p95={p95:.5f} "
              f"disk_std={amp:.4f} fresh_std={float(zf.std()):.4f}")

    if diffs:
        worst = max(x[1] for x in diffs)
        print(f"[A] worst max|diff| = {worst:.5f}  (pass<{MAX_ABS_PASS}, fail>{MAX_ABS_FAIL})")
        if worst > MAX_ABS_FAIL:
            print("[A] FAIL: encoder path NOT convention-identical to on-disk latents")
            ok = False
        elif worst > MAX_ABS_PASS:
            print("[A] WARN: diff above fp16-rounding band -- inspect before mass-encoding")
        else:
            print("[A] PASS: convention identical")

    # ---- B. Hegre encode sanity (writes nothing) ------------------------
    hnames = pick(args.hegre_root, args.n_hegre)
    for d in hnames:
        dp = os.path.join(args.hegre_root, d)
        try:
            z = encode_one(vae, os.path.join(dp, "pixel.npy"), args.device)
            zf = z[0].cpu().numpy().astype(np.float32)
            finite = bool(np.isfinite(zf).all())
            print(f"[B] {d}: shape={tuple(zf.shape)} dtype={z.dtype} "
                  f"finite={finite} std={zf.std():.4f} absmax={np.abs(zf).max():.3f}")
            if tuple(zf.shape) != (16, 128, 128) or not finite:
                print(f"[B] FAIL on {d}")
                ok = False
            if zf.std() < 0.05 or zf.std() > 50:
                print(f"[B] FAIL on {d}: implausible latent scale (std={zf.std():.4f})")
                ok = False
        except Exception as e:
            print(f"[B] FAIL on {d}: {type(e).__name__}: {e}")
            ok = False

    print(f"[done] {time.time()-t0:.1f}s")
    print("PRECHECK PASS" if ok else "PRECHECK FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
