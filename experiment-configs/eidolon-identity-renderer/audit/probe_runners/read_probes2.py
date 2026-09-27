#!/usr/bin/env python3
"""Copy probe outputs to parser-safe filenames and print the probe results."""
import glob
import json
import os
import shutil

B = "/home/tim/.hermes/profiles/prx-tg/cache/scratch/eir_probes2"
OUT = "/home/tim/.hermes/profiles/prx-tg/cache/scratch/probe_media"
os.makedirs(OUT, exist_ok=True)


def clean(src, name):
    if not os.path.exists(src):
        print("MISSING", src)
        return
    dst = os.path.join(OUT, name)
    shutil.copy(src, dst)
    print("COPIED %-28s %6d KB" % (name, os.path.getsize(dst) // 1024))


for f in sorted(glob.glob(B + "/seeds/contact*.png")):
    clean(f, "seed_contact_sheet.png")
for f in sorted(glob.glob(B + "/seeds/*.png")):
    base = os.path.basename(f)
    if base != "seed_contact_sheet.png" and "row" not in base:
        clean(f, "seed_" + base[-30:].replace("[", "_").replace("]", "_").replace("=", "-").replace(",", "_"))
for f in sorted(glob.glob(B + "/latest/grid_geoonly_id0_geo10*.png")):
    clean(f, "yaw8000_geoonly10.png")
for f in sorted(glob.glob(B + "/latest/grid_fullcfg_id3_geo2*.png")):
    clean(f, "yaw8000_fullcfg3-2.png")

print()
print("=== SEED-IDENTITY PROBE (results.json) ===")
d = json.load(open(B + "/seeds/results.json"))
print("keys:", list(d.keys()))
print(json.dumps(d, indent=1, default=str)[:5000])
print()
print("=== YAW SWEEP @ step 8000 ===")
a = json.load(open(B + "/latest/results.json"))
print("checkpoint:", a["checkpoint"].split("/")[-1],
      "| no_face panels:", a["no_face_panels_total"], "/", a["panels_total"])
for c in a["configs"]:
    for s in c["samples"]:
        if "yaw_spread" in s:
            print("  %-18s s%-3d spread %.4f (%.0f px)  %s"
                  % (c["name"], s["sample_idx"], s["yaw_spread"], s["yaw_spread"] / 0.0041, s["monotone"]))
