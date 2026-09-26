#!/usr/bin/env python3
"""Is the rendered yaw a real 3D projection, or a 2D shortening?

Operator's read of the step-7000 geometry-dominant grid:
  "The geometry shift is partial deformation (the sides are 'shortened'), partial real
   3d occlusion (the nose shows the person's nose left side, hides the nose's right side
   and vice versa on the other end of the spectrum."

Two mechanisms make different predictions in the landmarks:
  REAL ROTATION   projected interocular shrinks as cos(yaw); the far cheek is HIDDEN, so the
                  nose-to-far-edge distance collapses while the near side holds.
  SQUASH/SKEW     both sides shorten together: the ratios stay symmetric, and interocular
                  shrinks more than cos(yaw) allows.

Reference: n real FFHQ faces, same 68 iBUG landmarks from pose.npy, same metric definitions.
CPU only.
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/tim/source/activity/prx-tg/scripts")
from dwpose_onnx import DWPoseDetector  # noqa: E402

PANELS = "/home/tim/.hermes/profiles/prx-tg/cache/scratch/eir_probes/latest"
STRAT = "/mnt/nas-ai-models/training-data/ffhq/stratum"


def face68_from_133(kp):
    """COCO-WholeBody 133 -> the 68 iBUG face points are rows 23:91."""
    return np.asarray(kp)[23:91]


def metrics(pts):
    """pts: (68,2) in pixels. Returns the comparison metrics."""
    eye_l, eye_r = pts[36:42].mean(0), pts[42:48].mean(0)
    nose = pts[30]
    jaw = pts[0:17]
    inter = float(np.linalg.norm(eye_l - eye_r))
    ecx = 0.5 * (eye_l[0] + eye_r[0])
    yaw = float((nose[0] - ecx) / max(inter, 1e-6))
    xmin, xmax = float(jaw[:, 0].min()), float(jaw[:, 0].max())
    width = xmax - xmin
    return dict(inter=inter, yaw=yaw, width=width,
                nose_to_near=None, nose_to_far=None,
                left=nose[0] - xmin, right=xmax - nose[0],
                nose_x=float(nose[0]), cx=ecx)


def main():
    det = DWPoseDetector(device="cpu")

    # ---- reference: real FFHQ faces, landmarks from pose.npy ------------------------
    import random
    dirs = [d for d in os.scandir(STRAT) if d.is_dir()]
    random.seed(0)
    random.shuffle(dirs)
    real = []
    for d in dirs:
        p = os.path.join(d.path, "pose.npy")
        if not os.path.exists(p):
            continue
        try:
            f = np.load(p).astype(np.float64)[23:91]
            if f[:, 2].mean() < 0.3:
                continue
            pts = np.stack([(f[:, 0] + 1) * 512.0, (f[:, 1] + 1) * 512.0], 1)
            m = metrics(pts)
            if m["inter"] < 20:
                continue
            real.append(m)
        except Exception:
            continue
        if len(real) >= 500:
            break

    inter_med = float(np.median([m["inter"] for m in real]))
    print("REAL FFHQ reference: n=%d  median interocular=%.1f px" % (len(real), inter_med))
    print("  real: inter/median vs yaw, and side asymmetry (left/right jaw distance from nose tip)")
    bins = [(-9, -0.25), (-0.25, -0.10), (-0.10, 0.0), (0.0, 0.10), (0.10, 0.25), (0.25, 9)]
    for lo, hi in bins:
        sel = [m for m in real if lo <= m["yaw"] < hi]
        if len(sel) < 10:
            continue
        r_inter = np.mean([m["inter"] / inter_med for m in sel])
        aL = np.mean([m["left"] / m["inter"] for m in sel])
        aR = np.mean([m["right"] / m["inter"] for m in sel])
        pred_cos = np.cos(np.arcsin(np.clip(np.mean([m["yaw"] for m in sel]), -0.99, 0.99))) if False else None
        print("  yaw %+0.2f..%+0.2f n=%3d  inter/med %.3f   left %.2f right %.2f  (L/R %.2f)"
              % (lo, hi, len(sel), r_inter, aL, aR, aL / max(aR, 1e-9)))

    # ---- the render: step-7000 panels, geometry-dominant ladder --------------------
    print()
    print("RENDER step 7000, geoonly_id0_geo10 (same metrics, DWPose landmarks):")
    for s in (10, 30, 50):
        files = sorted(glob.glob(os.path.join(PANELS, "s%d_geoonly_id0_geo10_dim0_*.png" % s)))
        files = sorted(files, key=lambda f: float(f.split("dim0_")[-1].replace("+", "").replace(".png", "")))
        print("  --- sample %d ---" % s)
        base = None
        for f in files:
            v = os.path.basename(f).split("dim0_")[-1].replace(".png", "")
            img = cv2.imread(f)
            kp, sc, bb = det(img, single_person=True)
            if kp is None or len(kp) == 0:
                print("   dim0 %-5s  no_face" % v)
                continue
            pts = face68_from_133(np.asarray(kp)[0] if np.asarray(kp).ndim == 3 else np.asarray(kp))
            m = metrics(pts)
            if base is None:
                base = m["inter"]
            print("   dim0 %-5s  inter %6.1f (%.3f of dim0=-3.0)  yaw %+0.3f  width %6.1f  left %5.1f right %5.1f (L/R %.2f)"
                  % (v, m["inter"], m["inter"] / base, m["yaw"], m["width"], m["left"], m["right"],
                     m["left"] / max(m["right"], 1e-9)))


if __name__ == "__main__":
    main()
