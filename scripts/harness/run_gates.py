#!/usr/bin/env python3
"""Phase 1 producer — compute G0 photorealism gates on a directory of images.

Consumes generated/real PNGs (e.g. quality_metrics prompt images from a
checkpoint, or the real-FFHQ reference set) and emits tick-consumable JSONL:
one line per gate with the AGGREGATE value (mean over images) + per-image
detail lines to a sidecar. Identical preprocessing to calibrate_gates.py is
imported from that module — the frozen contract, single source of truth.

Generated images have no seg masks, so v1 computes the MASK-FREE gates
(g0a_full, g0b, g0c_full, g0d_full) — these have matching calibration bands.
Masked gates (g0a/g0c/g0d) are emitted only when a seg mask dir is supplied
(--masks <dir> with same basenames).

USAGE:
    .venv/bin/python3 scripts/harness/run_gates.py \
        --images <dir-with-pngs> \
        [--calibration research/avenues/gates_calibration.json] \
        --out research/avenues/gates/<arm>/<step>/gates.jsonl
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_gates import downsample, lumin, fft_gaussian_blur, radial_spectrum, block_stats  # noqa: E402

REF_SIZE = 512


def measure_image(path: Path, mask: np.ndarray | None = None):
    """Return dict of gate values for one image; mask-free unless a mask is given."""
    img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    lum = downsample(lumin(img), REF_SIZE)
    hp = lum - fft_gaussian_blur(lum, 2.5)
    spec = radial_spectrum(lum)

    out: dict = {
        "g0a_sensor_noise_floor_full": float(np.log10(np.std(hp) + 1e-12)),
        "g0c_skin_texture_energy_full": float(np.mean(hp ** 2) / (np.mean(lum ** 2) + 1e-12)),
    }
    if spec is not None:
        out["g0b_spectral_slope"] = spec
    bs = block_stats(lum, np.ones_like(lum, dtype=bool))
    if bs is not None:
        out["g0d_local_contrast_full_p5"] = float(np.percentile(bs, 5))
        out["g0d_local_contrast_full_p50"] = float(np.percentile(bs, 50))
        out["g0d_local_contrast_full_p95"] = float(np.percentile(bs, 95))

    if mask is not None:
        m = downsample(mask.astype(bool), REF_SIZE)
        g0a = float(np.log10(np.std(hp[m]) + 1e-12))
        g0c = float(np.mean(hp[m] ** 2) / (np.mean(lum[m] ** 2) + 1e-12))
        out["g0a_sensor_noise_floor"] = g0a
        out["g0c_skin_texture_energy"] = g0c
        bsm = block_stats(lum, m)
        if bsm is not None:
            out["g0d_local_contrast_p5"] = float(np.percentile(bsm, 5))
            out["g0d_local_contrast_p50"] = float(np.percentile(bsm, 50))
            out["g0d_local_contrast_p95"] = float(np.percentile(bsm, 95))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--masks", type=Path, default=None,
                    help="optional dir with .npy binary masks, same basenames as images")
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    images = sorted(glob.glob(str(args.images / "*.png")) +
                    glob.glob(str(args.images / "*.jpg")))
    if not images:
        print(f"ERROR no PNG/JPG found in {args.images}", file=sys.stderr)
        return 2

    calib = None
    if args.calibration is not None:
        calib = json.loads(args.calibration.read_text())["gates"]

    detail: list[dict] = []
    agg: dict[str, list[float]] = {}
    for p in images:
        stem = Path(p).stem
        mask = None
        if args.masks is not None:
            mask_path = args.masks / f"{stem}.npy"
            if mask_path.is_file():
                mask = np.load(mask_path)
        vals = measure_image(Path(p), mask)
        detail.append({"image": Path(p).name, **vals})
        for k, v in vals.items():
            agg.setdefault(k, []).append(v)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for gate, values in sorted(agg.items()):
        a = np.asarray(values, dtype=np.float64)
        rec = {"gate_id": gate, "value": float(a.mean()), "n": int(len(a)),
               "std": float(a.std())}
        if calib is not None and gate in calib:
            band = calib[gate]
            rec["in_band"] = band["p5"] <= rec["value"] <= band["p95"]
            rec["band"] = [band["p5"], band["p95"]]
        lines.append(json.dumps(rec))
    tmp = out.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(out)
    (out.parent / (out.stem + "_detail.jsonl")).write_text(
        "\n".join(json.dumps(d) for d in detail) + "\n")
    print(f"{len(images)} images -> {out} ({len(lines)} gate lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())