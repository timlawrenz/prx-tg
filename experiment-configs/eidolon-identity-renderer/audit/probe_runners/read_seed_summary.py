#!/usr/bin/env python3
"""Print the seed-identity probe summary and make a postable-size contact sheet."""
import json
import os

import cv2

B = "/home/tim/.hermes/profiles/prx-tg/cache/scratch/eir_probes2"
M = "/home/tim/.hermes/profiles/prx-tg/cache/scratch/probe_media"

d = json.load(open(B + "/seeds/results.json"))
print("=== SEED-IDENTITY PROBE @ %s ===" % d["checkpoint"].split("/")[-1])
print("question :", d["question"])
print("pool_size:", d.get("pool_size"), "| chance_R@1:", d.get("chance_R@1"))
print("seeds    :", d.get("seeds"))
print("conventions:", json.dumps(d.get("conventions", {}))[:300])
print("notes    :", json.dumps(d.get("notes", []))[:400])
print()
print("--- summary ---")
print(json.dumps(d.get("summary", {}), indent=1)[:2000])
print()
print("--- per_sample (condensed) ---")
for s in d.get("per_sample", []):
    keep = {k: v for k, v in s.items()
            if k in ("sample_idx", "image_id", "r_at_1", "R@1", "correct", "rank",
                     "pairwise_cosine", "mean_pairwise_cosine", "min_pairwise_cosine",
                     "top1_id", "top1_cosine", "no_face", "skipped")}
    print("  ", json.dumps(keep, default=str)[:400])

src = os.path.join(M, "seed_contact_sheet.png")
if os.path.exists(src):
    im = cv2.imread(src)
    h, w = im.shape[:2]
    sc = min(1.0, 2400.0 / w)
    out = os.path.join(M, "seed_contact_sheet_post.jpg")
    cv2.imwrite(out, cv2.resize(im, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("\ncontact sheet %dx%d -> %s (%d KB)" % (w, h, out, os.path.getsize(out) // 1024))
