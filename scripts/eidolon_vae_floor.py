#!/usr/bin/env python3
"""Step (a) of the Eidolon identity instrument: the VAE round-trip identity floor.

Question: how much identity does putting the FLUX VAE in the measurement path
cost, before any renderer exists?

Three measurements per sample, all in LDA coordinates:
  a1  embed(pixel) vs embed(decode(latent))   -- pure VAE identity cost
  a2  embed(pixel) vs the corpus's stored auraface_lda.npy
      -- does the identity vector we CONDITION ON actually describe the image we
         TRAIN ON? The corpus vectors were extracted from the source image, while
         pixel.npy is a separate downscaled artifact; if these disagree, the model
         is being asked to condition on a description of a different image. Nobody
         has checked this, and it is cheap to check.
  a3  MSE(pixel, decode(latent))              -- reconstruction sanity

Control: cross-identity cosines (embed(dec_i) vs embed(orig_j), i != j) give the
floor that a1 must beat to be meaningful. A VAE that shifted every identity by
more than the identity separation would make a1 indistinguishable from the
control, and no renderer measurement downstream could mean anything.

CONVENTIONS (getting any of these wrong silently poisons the number):
  - pixel.npy is (3,H,W) float16 in [0,1], RGB.
  - decode_latents() returns (B,3,8H,8W) in [-1,1] -> (x+1)/2 for [0,1].
  - extract_auraface() wants (H,W,3) BGR uint8, i.e. cv2.imread order.
  - LDA coords on disk and from the helper are RAW scale (norm ~153); the corpus's
    stored vectors and the DiT identity slot are L2-normalized (norm 1.0). Every
    cosine here is taken after L2-normalizing both sides.

    .venv/bin/python scripts/eidolon_vae_floor.py --n 24 [--device cpu]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/tim/source/activity/eidolon")
from tools.auraface import describe_instrument, extract_auraface  # noqa: E402

sys.path.insert(0, "/home/tim/source/activity/prx-tg")
from production.flux_ae import decode_latents, load_flux_ae_decoder  # noqa: E402

CORPUS = "/mnt/nas-ai-models/training-data/eidolon/hegre_corpus"
OUT = "/home/tim/source/activity/prx-tg/research/results/eidolon-identity-instrument/vae_floor.json"


def l2n(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def to_bgr_u8(arr_chw):
    """(3,H,W) float [0,1] RGB -> (H,W,3) uint8 BGR, as cv2.imread would return."""
    a = np.asarray(arr_chw, dtype=np.float32)
    a = np.clip(a, 0.0, 1.0)
    a = (a * 255.0).round().astype(np.uint8)
    hwc = np.transpose(a, (1, 2, 0))
    return np.ascontiguousarray(hwc[..., ::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="samples (distinct personas)")
    ap.add_argument("--device", default="cpu", help="cpu (no GPU claim) or cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[instrument] {json.dumps(describe_instrument(), default=str)}", flush=True)

    # samples from DISTINCT personas so the cross-identity control is meaningful
    entries = []
    seen = set()
    for d in sorted(os.listdir(CORPUS)):
        p = os.path.join(CORPUS, d)
        if not os.path.isdir(p) or "--" not in d:
            continue
        persona = d.split("--", 1)[0]
        if persona in seen:
            continue
        if not (os.path.exists(os.path.join(p, "flux_latent.npy"))
                and os.path.exists(os.path.join(p, "pixel.npy"))):
            continue
        seen.add(persona)
        entries.append(p)
        if len(entries) >= args.n:
            break
    if len(entries) < 4:
        sys.exit(f"[samples] only {len(entries)} usable; hegre latents may still be encoding")

    print(f"[samples] {len(entries)} dirs, {len(seen)} distinct personas", flush=True)
    print(f"[vae] loading FLUX AE decoder on {args.device}", flush=True)
    vae = load_flux_ae_decoder(device=args.device)

    rows = []
    dec_lda = {}          # dir -> decoded LDA vector, for the cross-identity control
    t0 = time.time()
    for i, p in enumerate(entries):
        d = os.path.basename(p)
        pix = np.load(os.path.join(p, "pixel.npy"))
        lat = np.load(os.path.join(p, "flux_latent.npy"))
        z = torch.from_numpy(lat).unsqueeze(0).to(vae.device, dtype=vae.dtype)

        dec = decode_latents(vae, z)
        dec = (dec[0].float().cpu().numpy() + 1.0) / 2.0        # [-1,1] -> [0,1]

        r_orig = extract_auraface(to_bgr_u8(pix), source_path=p)
        r_dec = extract_auraface(to_bgr_u8(dec), source_path=p)

        mse = float(((np.clip(dec, 0, 1).astype(np.float32) - pix.astype(np.float32)) ** 2).mean())
        row = {
            "dir": d, "persona": d.split("--", 1)[0],
            "orig_ok": bool(r_orig.ok), "orig_outcome": r_orig.outcome,
            "dec_ok": bool(r_dec.ok), "dec_outcome": r_dec.outcome,
            "dec_mse": mse,
            "dec_lda_norm_raw": float(np.linalg.norm(r_dec.lda_coords)) if r_dec.lda_coords is not None else None,
            "n_faces_orig": int(r_orig.n_faces), "n_faces_dec": int(r_dec.n_faces),
        }
        if r_orig.ok and r_dec.ok:
            lo, ld = l2n(r_orig.lda_coords), l2n(r_dec.lda_coords)
            row["a1_lda_cos"] = float(lo @ ld)
            row["a1_auraface_cos"] = float(l2n(r_orig.normed_embedding) @ l2n(r_dec.normed_embedding))
        if r_dec.ok:
            dec_lda[d] = np.asarray(r_dec.lda_coords)
        # NB: the corpus ships the PERSONA CENTROID in every sample dir (bit-identical
        # across a persona's samples), not a per-image vector. So a2 asks "does the
        # image we train on sit near the centroid we condition on" -- directly
        # comparable to the cross-shoot ceiling's own-centroid cosine.
        stored = os.path.join(p, "auraface_lda.npy")
        if r_orig.ok and os.path.exists(stored):
            row["a2_cos_to_persona_centroid"] = float(l2n(r_orig.lda_coords) @ l2n(np.load(stored)))
        rows.append(row)
        print(f"  [{i+1}/{len(entries)}] {d[:44]:46s} a1_lda={row.get('a1_lda_cos', float('nan')):.4f} "
              f"a2={row.get('a2_cos_to_persona_centroid', float('nan')):.4f} mse={mse:.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    a1 = np.array([r["a1_lda_cos"] for r in rows if "a1_lda_cos" in r])
    a2 = np.array([r["a2_cos_to_persona_centroid"] for r in rows if "a2_cos_to_persona_centroid" in r])
    # cross-identity control over decoded vectors vs OTHER samples' originals
    dec_v = {d: l2n(v) for d, v in dec_lda.items()}
    ctrl = []
    vecs = [(r["dir"], r["persona"]) for r in rows if "a1_lda_cos" in r]
    for di, pi in vecs:
        if di not in dec_v:
            continue
        for dj, pj in vecs:
            if pi != pj:
                ctrl.append(float(dec_v[di] @ l2n(
                    np.load(os.path.join(CORPUS, dj, "auraface_lda.npy")))))
    ctrl = np.array(ctrl)

    print()
    print(f"[a1 VAE identity cost] cos(embed(pixel), embed(decode(latent))) over n={len(a1)}")
    print(f"    mean {a1.mean():.4f}  median {np.median(a1):.4f}  min {a1.min():.4f}  "
          f"max {a1.max():.4f}  frac>0.99 {float((a1 > 0.99).mean()):.2f}")
    if len(a2):
        print(f"[a2 corpus vector describes the trained image?] cos(embed(pixel), stored) n={len(a2)}")
        print(f"    mean {a2.mean():.4f}  median {np.median(a2):.4f}  min {a2.min():.4f}  max {a2.max():.4f}")
    if len(ctrl):
        print(f"[control cross-identity] cos over {len(ctrl)} wrong pairs")
        print(f"    mean {ctrl.mean():.4f}  p95 {np.percentile(ctrl,95):.4f}  max {ctrl.max():.4f}")
    mses = [r["dec_mse"] for r in rows]
    print(f"[a3 reconstruction] MSE mean {np.mean(mses):.5f} max {np.max(mses):.5f}")

    payload = {
        "measurement": "vae_roundtrip_identity_floor",
        "definition": {
            "a1": "cos(embed(pixel.npy), embed(decode(flux_latent.npy))), LDA, both L2-normalized",
            "a2": "cos(embed(pixel.npy), the corpus persona centroid we condition on) — within-persona consistency, comparable to the ceiling's own-centroid cosine",
            "a3": "MSE(pixel.npy, decode(latent))",
            "control": "cos(embed(decoded_i), stored_j) for personas i != j",
        },
        "instrument_provenance": describe_instrument(),
        "conventions": {
            "pixel_npy": "(3,H,W) float16 [0,1] RGB",
            "decode_latents": "(B,3,8H,8W) float [-1,1] -> (x+1)/2",
            "extract_auraface_input": "(H,W,3) BGR uint8",
            "lda_on_disk_helper": "raw scale (~153)",
            "corpus_stored_and_dit_slot": "L2-normalized (norm 1.0)",
        },
        "config": vars(args),
        "samples": rows,
        "a1_summary": {"n": int(len(a1)), "mean": float(a1.mean()) if len(a1) else None,
                       "median": float(np.median(a1)) if len(a1) else None,
                       "min": float(a1.min()) if len(a1) else None,
                       "max": float(a1.max()) if len(a1) else None},
        "a2_summary": {"n": int(len(a2)), "mean": float(a2.mean()) if len(a2) else None,
                       "median": float(np.median(a2)) if len(a2) else None},
        "control_summary": {"n": int(len(ctrl)), "mean": float(ctrl.mean()) if len(ctrl) else None,
                            "p95": float(np.percentile(ctrl, 95)) if len(ctrl) else None,
                            "max": float(ctrl.max()) if len(ctrl) else None},
        "a3_mse_mean": float(np.mean(mses)), "a3_mse_max": float(np.max(mses)),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": os.uname().nodename,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[out] {OUT}")


if __name__ == "__main__":
    main()
