#!/usr/bin/env python3
"""Is the training data complete and evenly distributed across yaw (and pitch)?

Measures yaw, pitch and roll proxies from the pose.npy landmarks over N training images,
then asks the question that matters for the sweep: the sweep commands z_g dim0 in
[-3, +3] standard deviations. Mapping dim0 -> yaw via the data's own regression, how many
real training images actually sit at the yaw the sweep asks for? If the commanded yaw is
outside the data's support, the model is being asked to extrapolate.

CPU only. Usage: yaw_pitch_coverage.py [N]
"""
import os
import random
import sys

import numpy as np

FFHQ = "/mnt/nas-ai-models/training-data/ffhq/stratum"
HEGRE_CANDIDATES = [
    "/mnt/nas-ai-models/training-data/hegre/stratum",
    "/mnt/nas-ai-models/training-data/hegre-faces/stratum",
    "/mnt/nas-ai-models/training-data/hegre",
]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000


def proxies(p):
    face = p[23:91]
    if face.shape[0] < 68 or face[:, 2].mean() < 0.3:
        return None
    pts = np.stack([(face[:, 0] + 1) * 512.0, (face[:, 1] + 1) * 512.0], 1)
    el, er, nose = pts[36:42].mean(0), pts[42:48].mean(0), pts[30]
    inter = float(np.linalg.norm(el - er))
    if inter < 20:
        return None
    ecx, ecy = 0.5 * (el[0] + er[0]), 0.5 * (el[1] + er[1])
    yaw = (nose[0] - ecx) / inter
    pitch = (nose[1] - ecy) / inter
    roll = float(np.degrees(np.arctan2(er[1] - el[1], er[0] - el[0])))
    return float(yaw), float(pitch), roll, inter


def collect(root, n, need_zg=False, seed=0):
    dirs = [d.path for d in os.scandir(root) if d.is_dir()]
    random.seed(seed)
    random.shuffle(dirs)
    yaw, pit, rol, zg = [], [], [], []
    seen = 0
    for d in dirs:
        pp = os.path.join(d, "pose.npy")
        if not os.path.exists(pp):
            continue
        if need_zg and not os.path.exists(os.path.join(d, "z_g.npy")):
            continue
        try:
            r = proxies(np.load(pp).astype(np.float64))
            if r is None:
                continue
            yaw.append(r[0]); pit.append(r[1]); rol.append(r[2])
            if need_zg:
                zg.append(float(np.load(os.path.join(d, "z_g.npy")).astype(np.float64).reshape(-1)[0]))
            seen += 1
        except Exception:
            continue
        if seen >= n:
            break
    return (np.array(yaw), np.array(pit), np.array(rol), np.array(zg) if need_zg else None)


def report(tag, yaw, pit, rol):
    print("=== %s: n=%d ===" % (tag, len(yaw)))
    for name, v in (("yaw", yaw), ("pitch", pit), ("roll(deg)", rol)):
        q = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
        print("  %-9s std %6.3f  p1 %+6.3f p5 %+6.3f p25 %+6.3f med %+6.3f p75 %+6.3f p95 %+6.3f p99 %+6.3f  min %+6.3f max %+6.3f"
              % ((name, v.std()) + tuple(q) + (v.min(), v.max())))
    for t in (0.10, 0.25, 0.40, 0.60):
        print("  |yaw| > %.2f : %5.2f%%   |pitch| > %.2f : %5.2f%%"
              % (t, 100 * np.mean(np.abs(yaw) > t), t, 100 * np.mean(np.abs(pit) > t)))
    print("  sign: yaw<0 %5.2f%%  yaw>0 %5.2f%%   (asymmetry %.3f)"
          % (100 * np.mean(yaw < 0), 100 * np.mean(yaw > 0),
             float(np.mean(yaw > 0) - np.mean(yaw < 0))))


def main():
    yaw, pit, rol, zg = collect(FFHQ, N, need_zg=True)
    report("FFHQ", yaw, pit, rol)

    if zg is not None and len(zg) == len(yaw):
        b, a = np.polyfit(zg, yaw, 1)
        r = float(np.corrcoef(zg, yaw)[0, 1])
        print("\n=== dim0 -> yaw regression (FFHQ) ===")
        print("  yaw = %.4f + %.4f * dim0      r = %+.3f" % (a, b, r))
        print("  the sweep commands these dim0 values; is the implied yaw inside the data's support?")
        print("  dim0    -> implied yaw   n(within 0.05)  n(within 0.10)   inside p1..p99?")
        lo, hi = np.percentile(yaw, 1), np.percentile(yaw, 99)
        for dv in (-3.0, -1.5, 0.0, 1.5, 3.0):
            ty = a + b * dv
            n5 = int(np.sum(np.abs(yaw - ty) <= 0.05))
            n10 = int(np.sum(np.abs(yaw - ty) <= 0.10))
            print("  %+4.1f    %+8.4f      %5d / %d       %5d / %d      %s"
                  % (dv, ty, n5, len(yaw), n10, len(yaw), "yes" if lo <= ty <= hi else "NO - EXTRAPOLATION"))

    for c in HEGRE_CANDIDATES:
        if os.path.isdir(c) and any(os.path.exists(os.path.join(d.path, "pose.npy")) for d in os.scandir(c) if d.is_dir()):
            print()
            hy, hp, hr, _ = collect(c, min(N, 4000), seed=1)
            if len(hy):
                report("hegre (%s)" % c, hy, hp, hr)
            break


if __name__ == "__main__":
    main()
