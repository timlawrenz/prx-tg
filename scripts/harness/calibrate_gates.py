#!/usr/bin/env python3
"""M2 — freeze G0 photorealism gates calibration from the real-FFHQ reference set.

Deterministic: seed-driven stratified sampling over sorted stratum dirs.
All gates are computed on 512-downsampled luminance (identical preprocessing is
applied later to MODEL outputs — bands are a RELATIVE instrument, so the exact
downsampling choice is part of the frozen contract, recorded in the output).

Stdlib + numpy only. No GPU. USAGE:
    .venv/bin/python3 scripts/harness/calibrate_gates.py [--n 1000] [--out research/avenues/gates_calibration.json]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

STRATUM = Path("/mnt/nas-ai-models/training-data/ffhq/stratum")
FACE_SKIN_V1 = 2  # v1 28-class taxonomy: Face_Neck = 2 (v2 shifts +1)
REF_SIZE = 512
MIN_SKIN_FRAC = 0.005


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    """Boolean erosion via shifts; borders set False (no wraparound)."""
    m = mask.copy()
    for dy in (-r, 0, r):
        for dx in (-r, 0, r):
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(mask, dx, axis=1), dy, axis=0)
            if dx < 0:
                shifted[:, dx:] = False
            elif dx > 0:
                shifted[:, :dx] = False
            if dy < 0:
                shifted[dy:, :] = False
            elif dy > 0:
                shifted[:dy, :] = False
            m &= shifted
    return m


def downsample(a: np.ndarray, size: int) -> np.ndarray:
    h, w = a.shape[:2]
    s = max(1, min(h, w) // size)
    hc, wc = (h // s) * s, (w // s) * s
    a = a[:hc:s, :wc:s]
    # crop/pad to exactly size
    if a.shape[0] > size:
        a = a[:size]
    if a.shape[1] > size:
        a = a[:, :size]
    if a.shape < (size, size):
        out = np.zeros((size, size), dtype=a.dtype)
        out[:a.shape[0], :a.shape[1]] = a
        a = out
    return a


def lumin(pix: np.ndarray) -> np.ndarray:
    f = pix.astype(np.float32)
    if f.ndim == 3 and f.shape[0] == 3:  # channels-first (verified on-disk: (3, H, W))
        f = np.moveaxis(f, 0, -1)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def fft_gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    h, w = a.shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.rfftfreq(w)
    fy2 = fy[:, None] ** 2 + fx[None, :] ** 2
    g = np.exp(-2 * (np.pi * sigma) ** 2 * fy2).astype(np.float32)
    return np.fft.irfft2(np.fft.rfft2(a) * g, s=(h, w))[:h, :w]


def radial_spectrum(a: np.ndarray, fmin: float = 0.1, fmax: float = 0.9, bins: int = 12):
    h, w = a.shape
    mag = np.abs(np.fft.rfft2(a - a.mean()))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    freq = np.sqrt(fy ** 2 + fx ** 2) / 0.5  # normalized nyquist
    lo, hi = np.log10(fmin), np.log10(fmax)
    edges = np.logspace(lo, hi, bins + 1)
    power, centers = [], []
    for i in range(bins):
        m = (freq >= edges[i]) & (freq < edges[i + 1]) & (freq > 0)
        if m.sum() < 8:
            continue
        power.append(np.log10(mag[m].mean() + 1e-12))
        centers.append(np.log10(0.5 * (edges[i] + edges[i + 1])))
    if len(power) < 4:
        return None
    return float(np.polyfit(centers, power, 1)[0])


def block_stats(a: np.ndarray, mask: np.ndarray, block: int = 8):
    h, w = a.shape
    hb, wb = h // block * block, w // block * block
    a_b = a[:hb, :wb].reshape(hb // block, block, wb // block, block)
    m_b = mask[:hb, :wb].reshape(hb // block, block, wb // block, block).mean(axis=(1, 3))
    std_blocks = a_b.reshape(hb // block, wb // block, -1).std(axis=-1)
    keep = (m_b > 0.7) & np.isfinite(std_blocks)
    if keep.sum() < 10:
        return None
    return std_blocks[keep].astype(np.float64)


def sample_dirs(pool: list[str], n: int, seed: int) -> list[str]:
    dirs = sorted(pool)
    rng = np.random.default_rng(seed)
    if n >= len(dirs):
        return dirs
    stride = max(1, len(dirs) // n)
    idx = np.arange(0, len(dirs), stride)
    return [dirs[int(i)] for i in rng.choice(idx, size=min(n, len(idx)), replace=False)]


def process_dir(d: Path):
    """Return dict of gate values or None if the image can't be measured."""
    try:
        pixel = np.load(d / "pixel.npy", mmap_mode="r")
        seg = np.load(d / "seg.npy", mmap_mode="r")
    except FileNotFoundError:
        return None
    if pixel.ndim != 3 or 3 not in pixel.shape:  # channels-first (3,H,W) or channels-last (H,W,3)
        return None
    lum = downsample(lumin(pixel), REF_SIZE)
    mask = downsample((seg == FACE_SKIN_V1), REF_SIZE)
    mask = erode(mask, r=1)
    if mask.sum() < MIN_SKIN_FRAC * REF_SIZE * REF_SIZE:
        return None

    hp = lum - fft_gaussian_blur(lum, 2.5)
    spec = radial_spectrum(lum)

    skin = mask
    g0a = float(np.log10(np.std(hp[skin]) + 1e-12))
    g0c = float(np.mean(hp[skin] ** 2) / (np.mean(lum[skin] ** 2) + 1e-12))
    bs = block_stats(lum, mask)
    out = {
        "g0a_sensor_noise_floor": g0a,
        "g0c_skin_texture_energy": g0c,
    }
    if spec is not None:
        out["g0b_spectral_slope"] = spec
    if bs is not None:
        out["g0d_local_contrast"] = {
            "p5": float(np.percentile(bs, 5)),
            "p50": float(np.percentile(bs, 50)),
            "p95": float(np.percentile(bs, 95)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="research/avenues/gates_calibration.json")
    ap.add_argument("--stratum", default=str(STRATUM))
    args = ap.parse_args()

    stratum = Path(args.stratum)
    if not stratum.is_dir():
        print(f"ERROR stratum dir missing: {stratum}", file=sys.stderr)
        return 1

    # Readdir-only pool listing: per-dir stat over 70k NFS entries is minutes;
    # readdir is ~4s. pixel.npy presence is verified only for sampled dirs.
    pool = sorted(os.listdir(stratum))
    pool = [p for p in pool if not p.startswith(".")]
    sel = sample_dirs(pool, args.n, args.seed)
    print(f"pool={len(pool)} sampled={len(sel)} seed={args.seed}", file=sys.stderr)

    acc: dict[str, list] = {}
    skipped = 0
    misses = 0
    t0 = time.time()
    for i, name in enumerate(sel):
        g = process_dir(stratum / name)
        if g is None:
            misses += 1
            continue
        for k, v in g.items():
            if isinstance(v, dict):
                acc.setdefault(k + "_p5", []).append(v["p5"])
                acc.setdefault(k + "_p50", []).append(v["p50"])
                acc.setdefault(k + "_p95", []).append(v["p95"])
            else:
                acc.setdefault(k, []).append(v)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sel)} ({time.time()-t0:.0f}s)", file=sys.stderr)

    cal = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:8],
        "reference": {
            "path": str(stratum),
            "pool_size": len(pool),
            "n_sampled": len(sel),
            "seed": args.seed,
        },
        "preprocessing": {
            "resample": REF_SIZE,
            "mode": "class-2 (Face_Neck v1 28-class) skin mask, eroded r=1, min skin frac 0.005",
            "luminance": "0.299/0.587/0.114 float32",
            "note": "Bands are RELATIVE: model outputs must go through the identical downsample path.",
        },
        "gates": {},
    }
    n_per = {}
    for k, vals in sorted(acc.items()):
        a = np.asarray(vals, dtype=np.float64)
        cal["gates"][k] = {
            "n": int(len(a)),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "p5": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)),
        }
        n_per["_".join(k.split("_")[:2])] = len(a)
    cal["gates"]["__meta__"] = {
        "unmeasurable_images": misses,
        "per_gate_n": {k: v for k, v in n_per.items() if not k.startswith("__")},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cal, indent=2))
    tmp.replace(out)
    print(f"frozen: {out} (n_sampled={len(sel)}, unmeasurable={misses})")
    return 0


if __name__ == "__main__":
    sys.exit(main())