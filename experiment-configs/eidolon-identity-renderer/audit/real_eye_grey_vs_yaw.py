#!/usr/bin/env python3
"""Does the training data contain the eye-region signal the renderer is emitting?

The renderer's sweep shows the LEFT-EYE grey mass falling monotonically with z_g dim0
(0.316 -> 0.081 on step0006000/sample30). Two competing explanations:

  H_data  the data contains a monotone eye-region appearance correlate of yaw, and the
          model faithfully reproduced that cheap correlate instead of rotating the head.
  H_model there is no such signal in the data; the model is manufacturing an ordered
          response because it is still learning how to abide by the conditioning.

This measures the SAME metric (desaturated mid-luminance fraction of the eye crop, same
crop geometry) on REAL FFHQ training images against their true yaw from pose.npy.
H_data predicts a monotone grey/yaw relationship; H_model predicts none.

CPU only. Usage: real_eye_grey_vs_yaw.py [n_images]
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np

STRAT = Path("/mnt/nas-ai-models/training-data/ffhq/stratum")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def grey_frac(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    return float(((sat <= 40) & (val > 40) & (val < 230)).mean()), float(val.mean())


def as_image(a):
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[0] == 3:            # CHW
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        m = float(np.nanmax(a))
        a = (a * 255.0).clip(0, 255).astype(np.uint8) if m <= 1.5 else a.clip(0, 255).astype(np.uint8)
    return a


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    def rank(v):
        o = np.argsort(np.argsort(v))
        return o.astype(float)
    return pearson(rank(np.asarray(x, float)), rank(np.asarray(y, float)))


def main():
    dirs = [d for d in STRAT.iterdir() if d.is_dir()]
    random.seed(0)
    random.shuffle(dirs)
    pick = []
    for d in dirs:
        if (d / "z_g.npy").exists() and (d / "pose.npy").exists() and (d / "pixel.npy").exists():
            pick.append(d)
        if len(pick) >= N:
            break
    print("sampled %d real FFHQ images with z_g + pose + pixel sidecars" % len(pick))

    yaw, zg0, gA, gB, lA, lB, seen = [], [], [], [], [], [], 0
    for d in pick:
        try:
            p = np.load(d / "pose.npy").astype(np.float64)
            face = p[23:91, :]
            if face.shape[0] < 68 or face[:, 2].mean() < 0.3:
                continue
            img = as_image(np.load(d / "pixel.npy"))
            H, W = img.shape[:2]
            # pose.npy landmarks are normalized to [-1, 1]; convert to pixels
            lx = (face[:, 0] + 1.0) * 0.5 * W
            ly = (face[:, 1] + 1.0) * 0.5 * H
            pts = np.stack([lx, ly], axis=1)
            eye_left = pts[36:42].mean(0)      # image-left eye
            eye_right = pts[42:48].mean(0)     # image-right eye
            nose = pts[30]
            inter = float(np.linalg.norm(eye_left - eye_right))
            if inter < 12:
                continue
            y = float((nose[0] - 0.5 * (eye_left[0] + eye_right[0])) / inter)
            r = inter * 0.35
            out = {}
            for tag, c in (("A", eye_left), ("B", eye_right)):
                x0, y0 = int(max(0, c[0] - r)), int(max(0, c[1] - r))
                x1, y1 = int(min(img.shape[1], c[0] + r)), int(min(img.shape[0], c[1] + r))
                if x1 - x0 < 6 or y1 - y0 < 6:
                    out[tag] = None
                    continue
                out[tag] = grey_frac(img[y0:y1, x0:x1])
            if out.get("A") is None or out.get("B") is None:
                continue
            z = np.load(d / "z_g.npy").astype(np.float64).reshape(-1)
            yaw.append(y); zg0.append(float(z[0]))
            gA.append(out["A"][0]); lA.append(out["A"][1])
            gB.append(out["B"][0]); lB.append(out["B"][1])
            seen += 1
        except Exception:
            continue

    yaw = np.array(yaw); zg0 = np.array(zg0)
    gA = np.array(gA); gB = np.array(gB)
    lA = np.array(lA); lB = np.array(lB)
    print("usable: %d" % seen)
    print("yaw proxy: std=%.4f  range=[%.3f, %.3f]" % (yaw.std(), yaw.min(), yaw.max()))
    print("z_g dim0 : std=%.3f  range=[%.2f, %.2f]" % (zg0.std(), zg0.min(), zg0.max()))
    print()
    print("grey fraction, image-LEFT eye  (A): mean=%.4f std=%.4f  p5=%.4f p95=%.4f  min=%.4f max=%.4f"
          % (gA.mean(), gA.std(), np.percentile(gA, 5), np.percentile(gA, 95), gA.min(), gA.max()))
    print("grey fraction, image-RIGHT eye (B): mean=%.4f std=%.4f  p5=%.4f p95=%.4f"
          % (gB.mean(), gB.std(), np.percentile(gB, 5), np.percentile(gB, 95)))
    print()
    print("== the decisive test: does real grey mass track yaw? ==")
    for tag, g in (("image-LEFT (A)", gA), ("image-RIGHT (B)", gB)):
        print("  %-16s r(grey, yaw)     pearson %+0.4f  spearman %+0.4f" % (tag, pearson(g, yaw), spearman(g, yaw)))
        print("  %-16s r(grey, |yaw|)   pearson %+0.4f  spearman %+0.4f" % ("", pearson(g, np.abs(yaw)), spearman(g, np.abs(yaw))))
        print("  %-16s r(grey, z_g dim0) pearson %+0.4f  spearman %+0.4f" % ("", pearson(g, zg0), spearman(g, zg0)))
    print()
    print("== grey mass by yaw quintile (image-LEFT eye) ==")
    q = np.quantile(yaw, [0, .2, .4, .6, .8, 1.0])
    for i in range(5):
        m = (yaw >= q[i]) & (yaw <= q[i + 1])
        if m.sum() == 0:
            continue
        print("  yaw %+.3f..%+.3f  n=%3d  grey mean=%.4f  (lum %.1f)" % (q[i], q[i + 1], m.sum(), gA[m].mean(), lA[m].mean()))
    print()
    print("== reference: the RENDERED range on step0006000/sample30 ==")
    print("  grey 0.3158 -> 0.0805 across z_g dim0 -3.0 -> +3.0 (monotone-down), lum flat (171.6 -> 169.7)")
    print("  verdict: H_data supported iff real grey shows a monotone yaw trend of comparable magnitude;")


if __name__ == "__main__":
    main()
