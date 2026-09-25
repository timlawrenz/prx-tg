#!/usr/bin/env python3
"""Does the detection/crop convention actually move an AuraFace-LDA identity vector?

Background: two extraction conventions exist in this ecosystem.
  FFHQ   : tools/../scripts/pipeline/extract_ffhq_auraface.py  -> full image,
           det_size=(512,512), providers CUDA-then-CPU
  hegre  : tools/hegre_dataset/enrichment.py                   -> 512px MTCNN face
           crop, app.prepare(ctx_id=0) i.e. default 640, CPU only, and the padded
           retry (PAD_FRACTION 0.20) is a ROUTINE path, not an exception.

Both land in the same LDA basis (120e1c5a1dc4f423), so the vectors are comparable
in principle. The question is whether they are comparable in fact, against an
identity margin of only ~0.0015 cosine.

Harness: FFHQ ships 70k RAW originals (00000.png ...) and 69,960 stored vectors
mapping 1:1 by name, so the det_size=(512,512) convention can be reproduced from
the true source image and checked against what is on disk.

Configs (ONE det_size PER PROCESS -- the extractor caches per det_size and its
docstring forbids mixing them in one process):
  full_d512   raw image, det_size=(512,512)   == the FFHQ convention (reproduction check)
  full_d640   raw image, det_size=None (640)  == the corpus convention
  full_d320   raw image, det_size=(320,320)   == sensitivity trend
  crop_d512   center 512x512 crop, det_size=(512,512)
  crop_d640   center 512x512 crop, det_size=None  == the hegre-like framing regime

Reads: full_d512 vs stored = convention chain reproduces at all; full_d512 vs
full_d640 = the det_size effect; full_d512 vs crop_d512 = the FRAMING effect
(Tim's hypothesis: the crop should not matter much, because insightface aligns the
face internally from landmarks regardless of the detected box).

Honest limitation: the stored FFHQ vectors were produced with a CUDA provider and
this probe is CPU-only (the 4090 is claimed by the hegre latent encode), so a
reproduction mismatch would be provider-or-resolution and cannot be attributed to
either alone. The provider axis is therefore NOT measured here.

    <full-stack python> scripts/eidolon_det_size_probe.py            # run all + aggregate
    <full-stack python> scripts/eidolon_det_size_probe.py --one CONFIG  # single config (internal)
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/tim/source/activity/eidolon")
from tools.auraface import describe_instrument, extract_auraface  # noqa: E402

RAW = Path("/mnt/nas-ai-models/training-data/ffhq/raw")
STORE = Path("/mnt/nas-ai-models/training-data/ffhq/auraface")
OUT = Path("/home/tim/source/activity/prx-tg/research/results/eidolon-identity-instrument/det_size_probe.json")
TMP = Path("/tmp")

N = 24
CONFIGS = {
    "full_d512": {"det_size": (512, 512), "crop": False},
    "full_d640": {"det_size": None, "crop": False},
    "full_d320": {"det_size": (320, 320), "crop": False},
    "crop_d512": {"det_size": (512, 512), "crop": True},
    "crop_d640": {"det_size": None, "crop": True},
}


def pick():
    """Deterministic sample of images that have a stored vector."""
    names = sorted(p.name for p in RAW.glob("*.png"))
    names = [n for n in names[:2000] if (STORE / (Path(n).stem + ".npy")).exists()]
    rng = random.Random(1234)
    return sorted(rng.sample(names, min(N, len(names))))


def l2n(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def do_one(cfg_name):
    import cv2
    cfg = CONFIGS[cfg_name]
    names = pick()
    print(f"[{cfg_name}] det_size={cfg['det_size']} crop={cfg['crop']} n={len(names)}", flush=True)
    print(f"[{cfg_name}] instrument: {json.dumps({k: describe_instrument(det_size=cfg['det_size'])[k] for k in ('det_size', 'model_dir', 'pad_fraction')})}", flush=True)
    rec = {}
    t0 = time.time()
    for i, name in enumerate(names):
        stem = Path(name).stem
        img = cv2.imread(str(RAW / name))
        if img is None:
            rec[stem] = {"error": "unreadable"}
            continue
        if cfg["crop"]:
            h, w = img.shape[:2]
            cy, cx = h // 2, w // 2
            img = img[max(0, cy - 256):cy + 256, max(0, cx - 256):cx + 256]
        res = extract_auraface(img, source_path=name, det_size=cfg["det_size"],
                              keep_embedding=True, keep_lda=True)
        entry = {
            "ok": bool(res.ok), "outcome": res.outcome, "n_faces": int(res.n_faces),
            "ambiguous": bool(getattr(res, "ambiguous", False)),
        }
        if res.ok:
            entry["lda"] = [float(x) for x in np.asarray(res.lda_coords)]
            entry["emb"] = [float(x) for x in np.asarray(res.normed_embedding)]
        rec[stem] = entry
        if (i + 1) % 6 == 0:
            print(f"[{cfg_name}] {i+1}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)
    out = TMP / f"detsize_probe_{cfg_name}.json"
    out.write_text(json.dumps(rec))
    print(f"[{cfg_name}] wrote {out} in {time.time()-t0:.0f}s", flush=True)


def load(cfg_name):
    return json.loads((TMP / f"detsize_probe_{cfg_name}.json").read_text())


def cos_stats(a_list, b_list):
    """cosine over L2-normalized vectors, elementwise across the sample list."""
    if not a_list:
        return None
    A = np.stack([l2n(a) for a in a_list])
    B = np.stack([l2n(b) for b in b_list])
    c = (A * B).sum(axis=1)
    return {"n": int(len(c)), "mean": float(c.mean()), "median": float(np.median(c)),
            "min": float(c.min()), "max": float(c.max()),
            "min_abs_delta": float(1.0 - c.min())}


def aggregate():
    data = {k: load(k) for k in CONFIGS}
    stems = [s for s in data["full_d512"]
             if data["full_d512"][s].get("ok")]
    print(f"\n[aggregate] {len(stems)} samples usable in the reference config\n")

    # stored FFHQ vectors
    stored = {}
    for s in stems:
        p = STORE / f"{s}.npy"
        if p.exists():
            stored[s] = np.load(p)
    s_keys = sorted(stored)

    def get(cfg, s, field):
        e = data[cfg].get(s, {})
        return e.get(field)

    results = {}
    space = {  # label -> (config_a, config_b, field, note)
        "reproduce_full_d512_vs_stored": None,  # handled separately (stored is 512-d)
        "det_size_full_d512_vs_full_d640": ("full_d512", "full_d640", "lda", "det_size effect, full-frame framing"),
        "det_size_full_d512_vs_full_d320": ("full_d512", "full_d320", "lda", "sensitivity trend"),
        "framing_full_d512_vs_crop_d512": ("full_d512", "crop_d512", "lda", "FRAMING effect (Tim's hypothesis)"),
        "det_size_under_crop_crop_d512_vs_crop_d640": ("crop_d512", "crop_d640", "lda", "det_size effect, face-filling framing"),
        "combined_corpus_vs_ffhq_full_d512_vs_crop_d640": ("full_d512", "crop_d640", "lda", "FFHQ convention vs the hegre-like regime"),
    }
    for label, spec in space.items():
        if spec is None:
            continue
        ca, cb, field, note = spec
        keys = [s for s in s_keys if get(ca, s, field) and get(cb, s, field)]
        st = cos_stats([get(ca, s, field) for s in keys], [get(cb, s, field) for s in keys])
        results[label] = {"note": note, "config_a": ca, "config_b": cb, "cos": st}

    # reproduction: recomputed full_d512 (512-d embedding) vs the stored vector
    keys = [s for s in s_keys if get("full_d512", s, "emb")]
    st = cos_stats([get("full_d512", s, "emb") for s in keys], [stored[s] for s in keys])
    results["reproduce_full_d512_vs_stored_512d"] = {
        "note": "recomputed raw->embed at the FFHQ convention vs the stored vector "
                "(CPU here, stored was CUDA: a mismatch is provider-or-resolution)",
        "cos": st}

    # outcomes per config (the padded-retry path is a routine hegre path)
    outcomes = {c: {} for c in CONFIGS}
    for c in CONFIGS:
        for s in s_keys:
            o = get(c, s, "outcome")
            if o:
                outcomes[c][o] = outcomes[c].get(o, 0) + 1

    print(f"{'comparison':52s} {'n':>3s} {'mean':>8s} {'median':>8s} {'min':>8s} {'1-min':>9s}")
    for label, r in results.items():
        st = r["cos"]
        if not st:
            print(f"{label:52s}  (no data)")
            continue
        print(f"{label:52s} {st['n']:3d} {st['mean']:8.5f} {st['median']:8.5f} "
              f"{st['min']:8.5f} {st['min_abs_delta']:9.5f}")
    print("\noutcomes per config (detected / detected_after_padding / no_face):")
    for c, o in outcomes.items():
        print(f"  {c:14s} {o}")

    payload = {
        "measurement": "detection_convention_sensitivity (det_size + framing)",
        "question": "do the FFHQ (det_size 512, full image) and hegre (default 640, "
                    "face-filling crop) conventions produce comparable identity vectors?",
        "reference_margin_cosine": 0.0015,
        "reference_vae_floor": 0.0002,
        "n_samples": len(s_keys),
        "sample_selection": "deterministic sample (seed 1234) of the first 2000 FFHQ raws",
        "configs": {k: {"det_size": v["det_size"], "crop": v["crop"]} for k, v in CONFIGS.items()},
        "instrument_provenance": describe_instrument(),
        "results": results,
        "outcomes": outcomes,
        "untested_axis": "provider (FFHQ used CUDA-then-CPU; this probe is CPU-only "
                         "because the 4090 is claimed by the hegre latent encode)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.uname().nodename,
        "interpreter": sys.executable,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\n[out] {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", choices=list(CONFIGS), help="run a single config (internal)")
    args = ap.parse_args()
    if args.one:
        do_one(args.one)
        return
    print(f"[driver] interpreter: {sys.executable}")
    print(f"[driver] samples: {pick()}")
    for cfg in CONFIGS:
        subprocess.run([sys.executable, os.path.abspath(__file__), "--one", cfg], check=True)
    aggregate()


if __name__ == "__main__":
    main()
