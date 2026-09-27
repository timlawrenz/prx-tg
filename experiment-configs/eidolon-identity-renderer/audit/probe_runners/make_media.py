#!/usr/bin/env python3
"""Downscale the contact sheet + the step-8000 sweep grids to postable JPEGs,
and print the step-8000 yaw spreads."""
import json
import os

import cv2

B = "/home/tim/.hermes/profiles/prx-tg/cache/scratch/eir_probes2"
M = "/home/tim/.hermes/profiles/prx-tg/cache/scratch/probe_media"


def shrink(src, name, width=2400, q=88):
    if not os.path.exists(src):
        print("MISSING", src)
        return None
    im = cv2.imread(src)
    h, w = im.shape[:2]
    sc = min(1.0, float(width) / w)
    out = os.path.join(M, name)
    cv2.imwrite(out, cv2.resize(im, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, q])
    print("POST %-34s %dx%d  %5d KB" % (name, w, h, os.path.getsize(out) // 1024))
    return out


shrink(os.path.join(M, "seed_contact_sheet.png"), "sheet_post.jpg")
shrink(os.path.join(M, "yaw8000_fullcfg3-2.png"), "yaw8000_fullcfg_post.jpg")
shrink(os.path.join(M, "yaw8000_geoonly10.png"), "yaw8000_geo10_post.jpg")

print()
print("=== YAW SWEEP @ step 8000 (z_g dim0 -3.0 -> +3.0) ===")
a = json.load(open(B + "/latest/results.json"))
print("checkpoint:", a["checkpoint"].split("/")[-1],
      "| no_face panels:", a["no_face_panels_total"], "/", a["panels_total"],
      "| seed:", a["seed"], "| steps:", a["num_steps"])
for c in a["configs"]:
    print("  %s (id %.1f / geo %.1f)" % (c["name"], c["identity_scale"], c["geometry_scale"]))
    for s in c["samples"]:
        if "yaw_spread" in s:
            print("     s%-3d spread %.4f (%.0f px)  %s" % (s["sample_idx"], s["yaw_spread"],
                                                            s["yaw_spread"] / 0.0041, s["monotone"]))
