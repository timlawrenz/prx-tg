#!/usr/bin/env python3
"""Crop the LEFT-EYE region from each of the 5 sweep panels and stack the crops into
one strip per row, so the operator's "grey patch that monotonically shrinks" can be
verified by eye. CPU only."""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/tim/source/activity/prx-tg/scripts")
from dwpose_onnx import DWPoseDetector  # noqa: E402

OUT = "/home/tim/.hermes/profiles/prx-tg/cache/scratch/eye_strips"


def panels(path):
    a = cv2.imread(path)
    w = a.shape[1] // 5
    return [a[:, i * w:(i + 1) * w] for i in range(5)]


def main(det, run_dirs):
    os.makedirs(OUT, exist_ok=True)
    for rd in run_dirs:
        step = rd.rstrip("/").split("/")[-1]
        for f in sorted(glob.glob(os.path.join(rd, "eidolon_geometry_sweep", "*_sweep.png"))):
            name = os.path.basename(f).replace("_sweep.png", "")
            crops = []
            for p in panels(f):
                kp, sc, bb = det(p, single_person=True)
                if kp is None or len(kp) == 0:
                    crops.append(None)
                    continue
                kp = np.asarray(kp[0] if kp.ndim == 3 else kp)
                le, re = kp[1][:2], kp[2][:2]
                inter = float(np.linalg.norm(le - re))
                if inter < 1e-6:
                    crops.append(None)
                    continue
                r = inter * 0.40
                x0, y0 = int(max(0, le[0] - r)), int(max(0, le[1] - r))
                x1, y1 = int(min(p.shape[1], le[0] + r)), int(min(p.shape[0], le[1] + r))
                c = p[y0:y1, x0:x1]
                if c.size == 0:
                    crops.append(None)
                    continue
                crops.append(cv2.resize(c, (220, 220), interpolation=cv2.INTER_NEAREST))

            if all(c is None for c in crops):
                print("%s/%s: no crops" % (step, name))
                continue
            blank = np.zeros((220, 220, 3), np.uint8)
            strip = np.hstack([c if c is not None else blank for c in crops])
            labels = np.zeros((90, strip.shape[1], 3), np.uint8)
            for i, v in enumerate([-3.0, -1.5, 0.0, 1.5, 3.0]):
                cv2.putText(labels, "%+.1f" % v, (i * 220 + 80, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
            out = np.vstack([labels, strip])
            path = os.path.join(OUT, "%s_%s_left_eye.png" % (step, name))
            cv2.imwrite(path, out)
            print("wrote", path)


if __name__ == "__main__":
    main(DWPoseDetector(device="cpu"), sys.argv[1:])
