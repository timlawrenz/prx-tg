#!/usr/bin/env python3
"""Blind-review pool builder (Phase 1).

Builds a deterministic review pool from model-output images and hidden real
photos. Pairs are rendered as unlabeled side-by-side PNGs; ground truth (which
side is real / which model) lives ONLY in pool.json and is never shown to
raters. Two question blocks are generated as separate pools — realism and
quality are never mixed in one session (attractive fakes corrupt realism
judgment; the two questions must be asked separately).

USAGE:
    .venv/bin/python3 scripts/harness/review_pool.py \
        --a-dir <champion-images> --b-dir <arm-images> --real-dir <real-photos> \
        --block realism --n-pairs 30 --seed 42 --out-dir research/avenues/review/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

VALID_BLOCKS = ("realism", "quality")


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _side_by_side(left: np.ndarray, right: np.ndarray, height=768) -> np.ndarray:
    def resize(im):
        h, w = im.shape[:2]
        nh = height
        nw = int(w * height / h)
        return np.asarray(Image.fromarray(im).resize((nw, nh)))

    l, r = resize(left), resize(right)
    h = max(l.shape[0], r.shape[0])
    out = np.full((h, l.shape[1] + 24 + r.shape[1], 3), 255, dtype=np.uint8)
    out[:l.shape[0], :l.shape[1]] = l
    out[:r.shape[0], l.shape[1] + 24:] = r
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", required=True, type=Path, help="model A images (e.g. champion)")
    ap.add_argument("--b-dir", required=True, type=Path, help="model B images (e.g. arm candidate)")
    ap.add_argument("--real-dir", required=True, type=Path, help="hidden real photos (calibration ladder)")
    ap.add_argument("--block", required=True, choices=VALID_BLOCKS)
    ap.add_argument("--n-pairs", type=int, default=30)
    ap.add_argument("--real-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    a_imgs = sorted(args.a_dir.glob("*.png")) + sorted(args.a_dir.glob("*.jpg"))
    b_imgs = sorted(args.b_dir.glob("*.png")) + sorted(args.b_dir.glob("*.jpg"))
    real_imgs = sorted(args.real_dir.glob("*.png")) + sorted(args.real_dir.glob("*.jpg"))
    if not (a_imgs and b_imgs and real_imgs):
        print("ERROR: all three image dirs must contain images", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    n_real = max(1, int(args.n_pairs * args.real_fraction))
    n_ab = args.n_pairs - n_real

    img_out = args.out_dir / args.block / "pairs"
    img_out.mkdir(parents=True, exist_ok=True)

    pairs = []
    for i in range(n_ab):
        a = a_imgs[i % len(a_imgs)]
        b = b_imgs[i % len(b_imgs)]
        swap = bool(rng.integers(0, 2))
        left, right = (b, a) if swap else (a, b)
        pid = f"{args.block}_ab_{i:04d}"
        out_png = img_out / f"{pid}.png"
        Image.fromarray(_side_by_side(_load(left), _load(right))).save(out_png)
        pairs.append({
            "pair_id": pid, "block": args.block, "kind": "ab",
            "image": str(out_png),
            "left": str(left), "right": str(right),
            "left_model": "b" if swap else "a", "right_model": "a" if swap else "b",
            "left_is_real": False, "right_is_real": False,
        })

    for i in range(n_real):
        real = real_imgs[i % len(real_imgs)]
        model = (a_imgs if i % 2 == 0 else b_imgs)[i % len(a_imgs)]
        swap = bool(rng.integers(0, 2))
        left, right = (model, real) if swap else (real, model)
        pid = f"{args.block}_cal_{i:04d}"
        out_png = img_out / f"{pid}.png"
        Image.fromarray(_side_by_side(_load(left), _load(right))).save(out_png)
        pairs.append({
            "pair_id": pid, "block": args.block, "kind": "calibration",
            "image": str(out_png),
            "left": str(left), "right": str(right),
            "left_model": "model" if swap else None, "right_model": None if swap else "model",
            "left_is_real": not swap, "right_is_real": swap,
        })

    pool_path = args.out_dir / args.block / "pool.json"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool = {
        "block": args.block,
        "seed": args.seed,
        "n_pairs": len(pairs),
        "n_ab": n_ab,
        "n_calibration": n_real,
        "a_dir": str(args.a_dir),
        "b_dir": str(args.b_dir),
        "real_dir": str(args.real_dir),
        "pairs": pairs,
    }
    pool_path.write_text(json.dumps(pool, indent=2))
    print(f"{len(pairs)} pairs -> {pool_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())