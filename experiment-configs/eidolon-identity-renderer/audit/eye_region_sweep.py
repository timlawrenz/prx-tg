#!/usr/bin/env python3
"""Quantify the operator's observation on the yaw sweeps:

  "5 different images of roughly the same person... not five different poses...
   a grey patch on the person's left eye that monotonically shrinks from left to right."

For every sweep row at every checkpoint: split the collage into its 5 panels, locate the
face, crop the LEFT-EYE region, and measure how that region changes across the sweep.

Metrics per panel (i = 0..4 across dim0 = -3.0 .. +3.0):
  lum   mean luminance of the eye crop
  grey  fraction of desaturated mid-luminance pixels ("grey patch" proxy)
  d0    L2 distance of the crop vs panel 0
A "grey patch that monotone shrinks" predicts: grey decreasing monotonically to the right.

Also reports the same three metrics over the WHOLE panel, as a control: if the head were
rotating, whole-panel distance would grow while the eye patch (a local feature) would not
be the dominant term.

CPU only. Usage: eye_region_sweep.py <run_dir> [<run_dir> ...]
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/tim/source/activity/prx-tg/scripts")
from dwpose_onnx import DWPoseDetector  # noqa: E402

VALUES = [-3.0, -1.5, 0.0, 1.5, 3.0]


def panels(path):
    a = cv2.imread(path)
    h, w = a.shape[:2]
    pw = w // 5
    return [a[:, i * pw:(i + 1) * pw] for i in range(5)]


def grey_metrics(crop):
    """(luminance, grey fraction) — grey = desaturated (low chroma) and mid-luminance."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    lum = float(val.mean())
    chromatic = (sat > 40) & (val > 40)
    greyish = (sat <= 40) & (val > 40) & (val < 230)
    frac = float(greyish.mean())
    return lum, frac, chromatic


def main(det, run_dirs):
    print("value order (left to right):", VALUES)
    for rd in run_dirs:
        step = rd.rstrip("/").split("/")[-1]
        for f in sorted(glob.glob(os.path.join(rd, "eidolon_geometry_sweep", "*_sweep.png"))):
            name = os.path.basename(f).replace("_sweep.png", "")
            ps = panels(f)
            rows = []
            for i, p in enumerate(ps):
                try:
                    kp, sc, bb = det(p, single_person=True)
                except Exception as e:
                    rows.append((i, None, None, None, "err %s" % type(e).__name__))
                    continue
                if kp is None or len(kp) == 0:
                    rows.append((i, None, None, None, "no_face"))
                    continue
                kp = np.asarray(kp[0] if kp.ndim == 3 else kp)
                nose, le, re = kp[0][:2], kp[1][:2], kp[2][:2]
                inter = float(np.linalg.norm(le - re))
                if inter < 1e-6:
                    rows.append((i, None, None, None, "degenerate"))
                    continue
                r = inter * 0.35
                x0, y0 = int(max(0, le[0] - r)), int(max(0, le[1] - r))
                x1, y1 = int(min(p.shape[1], le[0] + r)), int(min(p.shape[0], le[1] + r))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    rows.append((i, None, None, None, "crop_too_small"))
                    continue
                crop = p[y0:y1, x0:x1]
                lum, grey, chromatic = grey_metrics(crop)
                rows.append((i, lum, grey, float(chromatic.mean()), "ok"))
                if i == 0:
                    ref = cv2.resize(crop, (64, 64)).astype(np.float32)
                else:
                    cur = cv2.resize(crop, (64, 64)).astype(np.float32)
                    rows[-1] = rows[-1] + (float(np.sqrt(((cur - ref) ** 2).mean())),)

            ok = [r for r in rows if r[4] == "ok"]
            print("\n=== %s / %s  (%d/5 panels with a face) ===" % (step, name, len(ok)))
            if not ok:
                continue
            print("  panel   dim0     lum    greyFrac   eyeL2vs0")
            for r in rows:
                if r[4] != "ok":
                    print("   %d      %+.1f      -         -          -     [%s]" % (r[0], VALUES[r[0]], r[4]))
                else:
                    d0 = r[5] if len(r) > 5 else 0.0
                    print("   %d      %+.1f   %6.1f    %7.4f    %8.2f" % (r[0], VALUES[r[0]], r[1], r[2], d0))
            g = [r[2] for r in ok]
            d = [r[5] for r in ok if len(r) > 5]
            def mono(seq):
                if len(seq) < 3:
                    return "n/a"
                inc = all(b >= a - 1e-9 for a, b in zip(seq, seq[1:]))
                dec = all(b <= a + 1e-9 for a, b in zip(seq, seq[1:]))
                return "MONOTONE-UP" if inc else ("MONOTONE-DOWN" if dec else "non-monotone")
            print("  greyFrac: %s -> %s" % (mono(g), " ".join("%.4f" % v for v in g)))
            if d:
                print("  eyeL2   : %s -> %s" % (mono(d), " ".join("%.1f" % v for v in d)))


if __name__ == "__main__":
    det = DWPoseDetector(device="cpu")
    main(det, sys.argv[1:])
