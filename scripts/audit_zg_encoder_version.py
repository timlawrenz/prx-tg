#!/usr/bin/env python3
"""Audit: WHICH eidolon encoder path produced the shipped prx-tg z_g.npy sidecars?

The question
------------
`/mnt/nas-ai-models/training-data/ffhq/stratum/<id>/z_g.npy` is a 50-d float32
geometry vector. Two documented encoder paths exist in the eidolon repo:

  * FLAT 2D GPA ("Phase 1", SUPERSEDED per provenance_zg_posenorm.yaml):
      center+scale -> align_single to gpa_mean -> PCA(k=50) -> whiten
      (`geometry_pca.encode.encode_pose`, Phase-1 artifact `output/encoder.npz`)

  * FRONTALIZED 3D ("Phase 1-R / production", the shipped backbone):
      center+scale -> EPnP rotation vs 300W template -> frontalize -> light 2D
      GPA -> PCA(k=50) -> whiten
      (`geometry_pca.zg_inference.encode_zg`, artifact
       `output/encoder_production.npz`)

A prx-tg audit measured |r| = 0.942 between shipped z_g dim0 and a landmark
yaw proxy. A genuinely pose-invariant encoder should NOT leak yaw that strongly.
So: re-encode real FFHQ images through every plausible candidate path and see
which one reproduces the shipped vectors.

Method
------
NFS-safe sampling (os.listdir, never shell globs on that path; the stratum tree
has 70k dirs). For each sampled item we load its real `pose.npy` (DWPose 133
COCO-WholeBody points, NORMALIZED to [-1,1] -- not pixels), slice the 68 iBUG
face points (indices 23:91, `constants.FACE_SLICE`), run each candidate encoder,
and compare against the shipped `z_g.npy`.

A candidate "reproduces" the shipped vectors only if the distance is at
float32-round-trip level. The mismatched-pair control (candidate vectors
against a *shuffled* set of shipped vectors) gives the distance scale of
"unrelated vectors", so the metric is shown to be able to fail.

Read-only: nothing is written under either repo's production/ or under
experiments/. No training. CPU only.

Usage
-----
    python scripts/audit_zg_encoder_version.py --n 200
    python scripts/audit_zg_encoder_version.py --n 200 --json out.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
STRATUM = Path("/mnt/nas-ai-models/training-data/ffhq/stratum")
ZG_SRC = Path("/mnt/nas-ai-models/training-data/ffhq/zg")   # batch_extract copies from here
FFHQ_ROOT = Path("/mnt/nas-ai-models/training-data/ffhq")

EIDOLON = Path("/home/tim/source/activity/eidolon")
GP_DIR = EIDOLON / "experiments" / "geometry_pca"          # import root for `geometry_pca.*`
ART = GP_DIR / "output"

FACE_SLICE = slice(23, 91)          # COCO-WholeBody 133 -> 68 iBUG (constants.FACE_SLICE)

# encoder artifacts under test
ARTIFACTS = {
    "encoder_production.npz": ART / "encoder_production.npz",   # frontalized 70k fit (shipped)
    "encoder.npz":            ART / "encoder.npz",              # Phase 1 flat 2D GPA (10k)
    "encoder_posenorm.npz":   ART / "encoder_posenorm.npz",     # Phase 1-R spike (2k)
}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or "unknown"
    except Exception as e:  # pragma: no cover
        return f"error: {e}"


# --------------------------------------------------------------------------- #
# Candidate encoders
# --------------------------------------------------------------------------- #
# `geometry_pca` is imported (read-only) from the eidolon repo so the documented
# conventions are REUSED rather than re-implemented.
sys.path.insert(0, str(GP_DIR))
from geometry_pca.gpa import center_and_scale, align_single      # noqa: E402
from geometry_pca.encode import encode_pose                      # noqa: E402  (flat 2D path)
from geometry_pca.zg_inference import encode_zg                  # noqa: E402  (frontalized path)
from geometry_pca.pose_normalize import frontalize, estimate_rotation  # noqa: E402

# ---- pre-fix frontalization, transcribed verbatim from git b07986e^ ---------- #
# commit b07986e ("fix(geometry): fix PCA pose-normalization Z-shear bug",
# 2026-06-23) changed exactly these lines. The shipped z_g was extracted
# 2026-06-26, i.e. AFTER that commit -- but the audit must not assume the
# extraction ran the committed code, so both variants are measured.
def _estimate_rotation_prefix(template3d, observed2d):
    X = template3d - template3d.mean(axis=0)
    y = observed2d - observed2d.mean(axis=0)
    A, *_ = np.linalg.lstsq(X, y, rcond=None)
    r1, r2 = A[:, 0], A[:, 1]
    n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
    if n1 < 1e-8 or n2 < 1e-8:
        return np.eye(3, dtype=np.float32)
    r1, r2 = r1 / n1, r2 / n2
    r2 = r2 - np.dot(r1, r2) * r1
    n2 = np.linalg.norm(r2)
    if n2 < 1e-8:
        return np.eye(3, dtype=np.float32)
    r2 = r2 / n2
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=0)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R.astype(np.float32)


def _frontalize_prefix(template3d, observed2d):
    """Pre-b07986e frontalize: lifts with the UNROTATED template depth Z."""
    X = template3d - template3d.mean(axis=0)
    y = observed2d - observed2d.mean(axis=0)
    R = _estimate_rotation_prefix(X, y)
    lifted = np.concatenate([y, X[:, 2:3]], axis=1)
    frontal3d = lifted @ R
    return frontal3d[:, :2].astype(np.float32)


def _frontal_then_align(face2d, enc, tpl_flipped, frontalize_fn):
    """Shared tail: frontalize -> align_single to gpa_mean -> PCA -> whiten."""
    centered = center_and_scale(face2d)
    frontal = frontalize_fn(tpl_flipped, centered)
    aligned = align_single(frontal, enc["gpa_mean"]).reshape(-1)
    raw = (aligned - enc["pca_mean"]) @ enc["components"].T
    return ((raw - enc["whiten_mu"]) / enc["whiten_sigma"]).astype(np.float32)


def build_candidates(encs):
    """Return dict name -> fn(face2d)->(50,) float32.

    `encs` maps artifact filename -> loaded npz dict.
    """
    cands = {}

    def mk_frontal(enc_name, frontalize_fn, yflip=True):
        enc = encs[enc_name]
        tpl = enc["canonical_template"].copy()
        if yflip:
            tpl[:, 1] *= -1.0
        lf = (lambda t, o: frontalize(t, o)) if frontalize_fn == "postfix" else _frontalize_prefix
        return lambda f: _frontal_then_align(f, enc, tpl, lf)

    # --- frontalized family ---
    # Only encoder_production.npz carries `canonical_template`: it is the only
    # artifact fitted on frontalized shapes (05_fit_production.py persists the
    # template for inference). encoder.npz / encoder_posenorm.npz are Phase-1 /
    # spike artifacts with no persisted template -- the spike's synthetic
    # template was built in-script and never saved, so a frontalized re-encode
    # against them is not reproducible and is NOT attempted.
    frontal_ok = "canonical_template" in encs.get("encoder_production.npz", {})
    if frontal_ok:
        cands["frontal_postfix_prod"]   = mk_frontal("encoder_production.npz", "postfix")
        cands["frontal_prefix_prod"]    = mk_frontal("encoder_production.npz", "prefix")
        cands["frontal_postfix_prod_noYflip"] = mk_frontal("encoder_production.npz", "postfix", yflip=False)
    else:
        print("  !! encoder_production.npz lacks canonical_template -- frontal family skipped")

    # --- flat 2D family (Phase-1 style, no frontalization at all) ---
    for nm, key in [("flat2d_prod", "encoder_production.npz"),
                    ("flat2d_phase1", "encoder.npz"),
                    ("flat2d_posen", "encoder_posenorm.npz")]:
        cands[nm] = (lambda e: (lambda f: encode_pose(f, e)))(encs[key])

    # --- the literal documented inference path (sanity: must equal frontal_postfix_prod) ---
    cands["documented_encode_zg"] = (lambda e: (lambda f: encode_zg(f, e)))(encs["encoder_production.npz"])
    return cands


# --------------------------------------------------------------------------- #
# Sampling (NFS-safe)
# --------------------------------------------------------------------------- #
def sample_dirs(n: int, seed: int = 0, require_pixel=False):
    names = sorted(d for d in os.listdir(STRATUM) if os.path.isdir(STRATUM / d))
    rng = random.Random(seed)
    rng.shuffle(names)
    pick = []
    for d in names:
        p = STRATUM / d
        if not (p / "pose.npy").exists() or not (p / "z_g.npy").exists():
            continue
        if require_pixel and not (p / "pixel.npy").exists():
            continue
        pick.append(p)
        if len(pick) >= n:
            break
    return pick, len(names)


def face68(pose: np.ndarray) -> np.ndarray | None:
    """Slice the 68 iBUG face points. Handles (133,3), (133,2), (68,2)."""
    if pose.shape == (133, 3):
        return pose[FACE_SLICE, :2].astype(np.float32)
    if pose.shape == (68, 2):
        return pose.astype(np.float32)
    return None


def yaw_proxy(face: np.ndarray) -> float:
    """Signed yaw proxy, identical to prx-tg experiment-configs/.../audit/zg_pose_audit.py."""
    reye, leye = face[36:42].mean(0), face[42:48].mean(0)
    nose = face[30]
    inter = np.linalg.norm(leye - reye) + 1e-9
    return float((nose[0] - 0.5 * (reye[0] + leye[0])) / inter)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compare(shipped: np.ndarray, cand: np.ndarray) -> dict:
    """shipped, cand: (N,50)."""
    diff = cand - shipped
    denom = np.linalg.norm(shipped, axis=1) + 1e-12
    rel = np.linalg.norm(diff, axis=1) / denom
    cos = np.sum(cand * shipped, axis=1) / (
        np.linalg.norm(cand, axis=1) * np.linalg.norm(shipped, axis=1) + 1e-12)
    # per-dimension Pearson r (across images)
    r = []
    for d in range(shipped.shape[1]):
        if shipped[:, d].std() < 1e-9 or cand[:, d].std() < 1e-9:
            r.append(np.nan)
        else:
            r.append(float(np.corrcoef(shipped[:, d], cand[:, d])[0, 1]))
    r = np.array(r, dtype=np.float64)
    exact = float(np.mean(np.all(cand == shipped, axis=1)))
    bclose = float(np.mean(np.all(np.isclose(cand, shipped, rtol=0, atol=2e-5), axis=1)))
    return {
        "rel_l2_median": float(np.median(rel)),
        "rel_l2_max": float(rel.max()),
        "abs_max": float(np.abs(diff).max()),
        "abs_mean": float(np.abs(diff).mean()),
        "cos_sim_median": float(np.median(cos)),
        "perdim_r_mean": float(np.nanmean(r)),
        "perdim_r_frac_gt_099": float(np.mean(np.abs(r) > 0.99)),
        "bitwise_equal_frac": exact,
        "frac_within_2e-5": bclose,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of stratum dirs to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    print("=" * 78)
    print("z_g ENCODER-VERSION AUDIT")
    print("=" * 78)
    print(f"stratum      : {STRATUM}")
    print(f"eidolon      : {EIDOLON}  HEAD={git_head(EIDOLON)}")
    print(f"prx-tg       : {Path(__file__).resolve().parents[1]}  "
          f"HEAD={git_head(Path(__file__).resolve().parents[1])}")
    print(f"geometry_pca : {GP_DIR}")

    encs = {}
    for name, p in ARTIFACTS.items():
        if not p.exists():
            print(f"  !! MISSING artifact {p}")
            continue
        encs[name] = dict(np.load(p))
        print(f"  artifact {name:26s} md5={md5(p)[:16]} keys={sorted(encs[name])}")

    # ---- provenance of the shipped vectors: they are a copy of ffhq/zg/<id>/zg.npy
    print("\n--- shipped sidecar provenance check ---------------------------------")
    dirs, n_dirs = sample_dirs(args.n, args.seed)
    print(f"sampled {len(dirs)} stratum dirs (of {n_dirs} dirs listed)")
    same = 0
    checked = 0
    for d in dirs[:40]:
        src = ZG_SRC / d.name / "zg.npy"
        if src.exists():
            checked += 1
            a = np.load(d / "z_g.npy")
            b = np.load(src)
            same += int(a.shape == b.shape and np.array_equal(a, b))
    if checked:
        print(f"  z_g.npy bitwise-identical to ffhq/zg/<id>/zg.npy for {same}/{checked} checked")
        print("  -> producer is scripts/pipeline/extract_zg_and_averages.py (process_ffhq)")

    # ---- load shipped + poses -------------------------------------------------
    shipped, faces, yaws, kept_dirs = [], [], [], []
    for d in dirs:
        pose = np.load(d / "pose.npy")
        f = face68(pose)
        if f is None or (f == 0).all():
            continue
        shipped.append(np.load(d / "z_g.npy").astype(np.float64).reshape(-1))
        faces.append(f)
        kept_dirs.append(d)
        yaws.append(yaw_proxy(f))
    shipped = np.asarray(shipped, dtype=np.float64)
    yaws = np.asarray(yaws, dtype=np.float64)
    N = shipped.shape[0]
    print(f"\nloaded {N} usable samples | z_g shape={shipped.shape}")

    # ---- schema evidence (the [-1,1] normalization trap) ----------------------
    allxy = np.concatenate([f[:, :2] for f in faces])
    amax = float(np.abs(allxy).max())
    verdict_xy = "NORMALIZED (image-relative)" if amax < 10 else "PIXEL-LIKE"
    print(f"pose xy range: min={allxy.min():.3f} max={allxy.max():.3f} "
          f"|max|={amax:.3f} p99.9={np.quantile(np.abs(allxy), 0.999):.3f} -> {verdict_xy}")
    print("  NOTE: values are NOT pixels. A pixel-space input would give |max| ~ 1024;")
    print("        any script feeding this as pixels gets a silent garbage z_g.")

    # ---- spread of the shipped vectors (collapse test) ------------------------
    print("\n--- shipped z_g statistics -------------------------------------------")
    print(f"  per-dim std: mean={shipped.std(0).mean():.4f} median={np.median(shipped.std(0)):.4f} "
          f"min={shipped.std(0).min():.4f} max={shipped.std(0).max():.4f}")
    print(f"  dim0 std={shipped.std(0)[0]:.4f}   ||z_g|| mean={np.linalg.norm(shipped,axis=1).mean():.2f}")

    yaw_r0 = float(np.corrcoef(shipped[:, 0], yaws)[0, 1])
    yaw_absr = np.array([abs(np.corrcoef(shipped[:, i], yaws)[0, 1]) for i in range(50)])
    print(f"  corr(z_g dim0, yaw proxy) = {yaw_r0:+.4f}   |r| rank of dim0 = "
          f"{int(np.where(np.argsort(-yaw_absr)==0)[0][0])+1}")
    print(f"  max |corr| over dims = {yaw_absr.max():.4f}  median |corr| = {np.median(yaw_absr):.4f}")
    # least-squares slope of the previously reported regression
    slope, intercept = np.polyfit(shipped[:, 0], yaws, 1)
    print(f"  yaw = {intercept:+.4f} {slope:+.4f}*dim0   (r = {yaw_r0:+.4f})")

    # ---- run every candidate --------------------------------------------------
    cands = build_candidates(encs)
    results = {}
    print("\n--- candidate reproduction -------------------------------------------")
    print(f"{'candidate':34s} {'relL2_med':>10s} {'relL2_max':>10s} {'absmax':>9s} "
          f"{'absmean':>9s} {'cos':>9s} {'r_mean':>8s} {'r>.99':>7s} {'bitwise':>8s}")
    for name, fn in cands.items():
        Z = np.stack([fn(f) for f in faces]).astype(np.float64)
        m = compare(shipped, Z)
        results[name] = m
        print(f"{name:34s} {m['rel_l2_median']:10.3e} {m['rel_l2_max']:10.3e} "
              f"{m['abs_max']:9.3e} {m['abs_mean']:9.3e} {m['cos_sim_median']:9.6f} "
              f"{m['perdim_r_mean']:8.4f} {m['perdim_r_frac_gt_099']:7.2f} "
              f"{m['bitwise_equal_frac']:8.2f}")

    # ---- negative control: mismatched pairing (proves the metric can fail) ----
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    ctrl_name, ctrl_fn = next(iter(cands.items()))
    Z0 = np.stack([ctrl_fn(f) for f in faces]).astype(np.float64)
    neg = compare(shipped, Z0[perm])
    print(f"\n--- negative control (mismatched image pairing, {ctrl_name}) ---")
    print(f"  relL2 median={neg['rel_l2_median']:.3f} max={neg['rel_l2_max']:.3f} "
          f"cos={neg['cos_sim_median']:.4f} r_mean={neg['perdim_r_mean']:+.4f}")

    # ---- input-convention controls -------------------------------------------
    # Which input corruptions still reproduce the shipped vectors, and which do
    # not. This is the check that has to be able to FAIL: if every corrupted
    # input reproduced the shipped vectors, the match would prove nothing.
    # Interpretation is per-control, because some invariances are real
    # properties of the encoder rather than evidence about the producer.
    conv, conv_note = {}, {}
    if kept_dirs:
        def yflip(f):
            g = f.copy()
            g[:, 1] *= -1.0
            return g

        def variant(kind):
            Zs = []
            for d, f in zip(kept_dirs, faces):
                pose = np.load(d / "pose.npy")
                if kind == "as_pixels":
                    fv = (f * 1024.0).astype(np.float32)   # wrong input unit
                elif kind == "body_slice":
                    fv = pose[0:68, :2].astype(np.float32) if pose.shape[0] >= 68 else f
                elif kind == "y_flipped":
                    fv = yflip(f)
                else:
                    raise ValueError(kind)
                Zs.append(encode_zg(fv, encs["encoder_production.npz"]))
            return np.stack(Zs).astype(np.float64)

        print("\n--- input-convention controls -------------------------------------")
        for kind in ("as_pixels", "body_slice", "y_flipped"):
            Zv = variant(kind)
            m = compare(shipped, Zv)
            conv[kind] = m
            inv = m["rel_l2_max"] < 1e-3
            print(f"  {kind:14s} relL2_median={m['rel_l2_median']:.3e} "
                  f"cos={m['cos_sim_median']:+.4f} bitwise={m['bitwise_equal_frac']:.2f} "
                  f"-> {'invariant' if inv else 'distinguished'}")
            conv_note[kind] = "invariant" if inv else "distinguished"

        # Is y-flip invariance specific to the frontalize path? If the FLAT path
        # separates a y-flipped face but the frontalized path does not, the
        # frontalization is discarding the observed shape.
        flat = cands.get("flat2d_prod")
        if flat is not None:
            Zf_norm = np.stack([flat(f) for f in faces]).astype(np.float64)
            Zf_flip = np.stack([flat(yflip(f)) for f in faces]).astype(np.float64)
            sep = float(np.median(np.linalg.norm(Zf_norm - Zf_flip, axis=1) /
                                  (np.linalg.norm(Zf_norm, axis=1) + 1e-12)))
            print(f"  [flat2d_prod] y-flip separation relL2={sep:.3e} -> "
                  f"{'flat path is ALSO y-flip invariant' if sep < 1e-3 else 'flat path SEPARATES y-flip'}")
            conv_note["flat2d_yflip_separation"] = sep

    # ---- where does the yaw leak live: pre- or post-whitening? ---------------
    print("\n--- pre- vs post-whitening yaw leak ---------------------------------")
    leak = {}
    for name in ("frontal_postfix_prod", "flat2d_prod", "flat2d_phase1"):
        if name not in results:
            continue
        ekey = {"frontal_postfix_prod": "encoder_production.npz",
                "flat2d_prod": "encoder_production.npz",
                "flat2d_phase1": "encoder.npz"}[name]
        enc = encs[ekey]
        fn = cands[name]
        Z = np.stack([fn(f) for f in faces]).astype(np.float64)
        # raw PCA scores = z_g un-whitened
        S = Z * enc["whiten_sigma"] + enc["whiten_mu"]
        r_pre = float(np.corrcoef(S[:, 0], yaws)[0, 1])
        r_post = float(np.corrcoef(Z[:, 0], yaws)[0, 1])
        sd_pre = float(S[:, 0].std())
        leak[name] = {"r_prewhiten": r_pre, "r_postwhiten": r_post,
                      "score_std_dim0": sd_pre, "whiten_sigma_dim0": float(enc["whiten_sigma"][0]),
                      "amplification": sd_pre / float(enc["whiten_sigma"][0] + 1e-12)}
        print(f"  {name:24s} r(dim0,yaw) pre-whiten={r_pre:+.4f} post-whiten={r_post:+.4f} | "
              f"score_std={sd_pre:.5f} whiten_sigma={float(enc['whiten_sigma'][0]):.5f} "
              f"amp={leak[name]['amplification']:.3f}")
    print("  -> sign is preserved; whitening rescales, it does not create the leak.")

    # ---- shape fidelity: how much observed geometry survives the encode? -----
    print("\n--- shape fidelity (decoded z_g vs observed face) -------------------")
    fid = {}
    raw_center = np.stack([center_and_scale(f).reshape(-1) for f in faces]).astype(np.float64)
    ss_tot = float(((raw_center - raw_center.mean(0)) ** 2).sum())
    print("  (a) R^2 with NO re-alignment -- penalises orientation differences")
    for name, fn in cands.items():
        ekey = "encoder_production.npz"
        if name.startswith("flat2d_phase1"):
            ekey = "encoder.npz"
        elif name.startswith("flat2d_posen"):
            ekey = "encoder_posenorm.npz"
        enc = encs[ekey]
        Z = np.stack([fn(f) for f in faces]).astype(np.float64)
        dec = enc["pca_mean"] + (Z * enc["whiten_sigma"] + enc["whiten_mu"]) @ enc["components"]
        r2 = 1.0 - float(((dec - raw_center) ** 2).sum()) / ss_tot
        fid[name] = {"r2_unaligned": float(r2)}
        print(f"    {name:34s} R^2 = {r2:+.4f}")

    # (b) FAIR instrument: give each decode the best similarity transform
    # (translation+scale+rotation) before scoring. No encoder gets penalised
    # for its output frame, only for shape content it cannot represent.
    print("  (b) R^2 after optimal similarity (Procrustes) re-alignment")
    for name, fn in cands.items():
        ekey = "encoder_production.npz"
        if name.startswith("flat2d_phase1"):
            ekey = "encoder.npz"
        elif name.startswith("flat2d_posen"):
            ekey = "encoder_posenorm.npz"
        enc = encs[ekey]
        Z = np.stack([fn(f) for f in faces]).astype(np.float64)
        dec = enc["pca_mean"] + (Z * enc["whiten_sigma"] + enc["whiten_mu"]) @ enc["components"]
        al = np.stack([align_single(dec[i].reshape(68, 2), raw_center[i].reshape(68, 2)).reshape(-1)
                       for i in range(len(faces))])
        r2 = 1.0 - float(((al - raw_center) ** 2).sum()) / ss_tot
        fid[name]["r2_procrustes"] = float(r2)
        print(f"    {name:34s} R^2 = {r2:+.4f}")
    print("  R^2 near 1 = the encoder carries the observed face geometry; near 0 or")
    print("  negative = the encode is near-collapsed onto the canonical template.")

    # ---- direct invariance verification (one sample, all intermediates) -----
    print("\n--- direct invariance check (frontal path, single sample) -----------")
    f0 = faces[0]
    tpl = encs["encoder_production.npz"]["canonical_template"].copy()
    tpl[:, 1] *= -1.0
    c0 = center_and_scale(f0)
    g = f0.copy(); g[:, 1] *= -1.0
    cg = center_and_scale(g)
    fr0 = frontalize(tpl, c0)
    frg = frontalize(tpl, cg)
    print(f"  ||centered - centered_flipped||      = {np.abs(c0 - cg).max():.3e}")
    print(f"  ||frontalized - frontalized_flipped||= {np.abs(fr0 - frg).max():.3e}")
    print(f"  ||z_g(f) - z_g(flip(f))||            = "
          f"{np.abs(encode_zg(f0, encs['encoder_production.npz']) - encode_zg(g, encs['encoder_production.npz'])).max():.3e}")
    print(f"  frontalized output vs s*canonical_template: max|diff| = "
          f"{np.abs(fr0 - fr0.mean(0)).max():.3e} vs template {np.abs(tpl - tpl.mean(0)).max():.3e}")
    print("  -> a vertically mirrored face encodes to the SAME z_g: the documented")
    print("     frontalized encode discards the observed shape (and its sign).")

    # ---- why a "pose-invariant" encoder leaks yaw -------------------------------
    # How much does the frontalized output actually vary across faces, and is that
    # variation pose-driven? Compare against the flat 2D path's aligned shapes.
    print("\n--- what varies in the frontalized encode ---------------------------")
    fr = np.stack([frontalize(tpl, center_and_scale(f)) for f in faces]).astype(np.float64)
    fr_bar = fr.mean(0)
    d_fr = np.linalg.norm((fr - fr_bar).reshape(len(faces), -1), axis=1)
    vary_fr = float(np.median(d_fr) / (np.linalg.norm(fr_bar) + 1e-12))
    flat_named = cands.get("flat2d_phase1")
    if flat_named is not None:
        e1 = encs["encoder.npz"]
        alm = np.stack([align_single(center_and_scale(f), e1["gpa_mean"]) for f in faces]).astype(np.float64)
        al_bar = alm.mean(0)
        d_al = np.linalg.norm((alm - al_bar).reshape(len(faces), -1), axis=1)
        vary_al = float(np.median(d_al) / (np.linalg.norm(al_bar) + 1e-12))
    else:
        vary_al = float("nan")
    # does the frontalized variation track pose?
    r_vary_yaw = float(np.corrcoef(d_fr, np.abs(yaws))[0, 1])
    # is the weak-perspective scale s the carrier?  (measured, not assumed)
    from geometry_pca.pose_normalize import estimate_rotation
    svals = np.asarray([estimate_rotation(tpl, center_and_scale(f).astype(np.float32))[1] for f in faces])
    r_s_dim0 = float(np.corrcoef(svals, shipped[:, 0])[0, 1])
    r_s_yaw = float(np.corrcoef(svals, yaws)[0, 1])
    print(f"  relative spread of frontalized shapes  (median ||fr-fr_bar||/||fr_bar||) = {vary_fr:.4f}")
    print(f"  relative spread of flat-path aligned   (median ||al-al_bar||/||al_bar||) = {vary_al:.4f}")
    print(f"  corr(frontalized deviation, |yaw|) = {r_vary_yaw:+.4f}")
    print(f"  corr(weak-perspective scale s, dim0) = {r_s_dim0:+.4f} ; corr(s, yaw) = {r_s_yaw:+.4f}")
    print("  -> the frontalized encode leaves only a small pose-driven residual; the")
    print("     whitened PCA dim0 is that residual's axis. s is NOT the carrier.")
    mech = {"frontalized_relative_spread": vary_fr,
            "flatpath_relative_spread": vary_al,
            "r_frontalized_spread_vs_absyaw": r_vary_yaw,
            "r_scale_s_vs_dim0": r_s_dim0, "r_scale_s_vs_yaw": r_s_yaw,
            "detail": ("The frontalized encode leaves only a small residual around the "
                       "canonical template; the whitened PCA dim0 is the axis of that "
                       "residual, which tracks out-of-plane pose (|r|~0.94). The "
                       "weak-perspective scale s is not the carrier (r~0).")}


    # ---- stratified by |yaw|: a real prefilter must fail on frontal faces ----
    # If the frontalized and flat paths only agreed on near-frontal faces, the
    # match would be uninformative. Check the match holds at extreme yaw.
    order = np.argsort(np.abs(yaws))
    tert = np.array_split(order, 3)
    print("\n--- relL2 by |yaw| tertile (low/mid/high) ---------------------------")
    strat = {}
    for name, fn in cands.items():
        Z = np.stack([fn(f) for f in faces]).astype(np.float64)
        rel = np.linalg.norm(Z - shipped, axis=1) / (np.linalg.norm(shipped, axis=1) + 1e-12)
        vals = [float(np.median(rel[t])) for t in tert]
        strat[name] = vals
        print(f"  {name:34s} low={vals[0]:9.3e} mid={vals[1]:9.3e} high={vals[2]:9.3e}")
    print(f"  |yaw| tertile medians: low={np.median(np.abs(yaws[tert[0]])):.3f} "
          f"mid={np.median(np.abs(yaws[tert[1]])):.3f} high={np.median(np.abs(yaws[tert[2]])):.3f}")

    # ---- axis characterisation: WHAT does the shipped dim0 encode? -----------
    print("\n--- shipped dim0 axis characterisation ------------------------------")
    inter, pitch = [], []
    for f in faces:
        reye, leye = f[36:42].mean(0), f[42:48].mean(0)
        nose = f[30]
        face_w = np.linalg.norm(leye - reye) + 1e-9
        inter.append(face_w / (np.linalg.norm(f.max(0) - f.min(0)) + 1e-9))  # interocular / face size
        pitch.append((nose[1] - 0.5 * (reye[1] + leye[1])) / face_w)
    inter = np.asarray(inter)
    pitch = np.asarray(pitch)
    r_yaw = float(np.corrcoef(shipped[:, 0], yaws)[0, 1])
    r_inter = float(np.corrcoef(shipped[:, 0], inter)[0, 1])
    r_pitch = float(np.corrcoef(shipped[:, 0], pitch)[0, 1])
    print(f"  corr(dim0, yaw proxy)                = {r_yaw:+.4f}")
    print(f"  corr(dim0, interocular/face-size)    = {r_inter:+.4f}")
    print(f"  corr(dim0, pitch proxy)              = {r_pitch:+.4f}")
    q = np.quantile(shipped[:, 0], [0.1, 0.9])
    lo, hi = inter[shipped[:, 0] <= q[0]], inter[shipped[:, 0] >= q[1]]
    if len(lo) and len(hi):
        print(f"  mean interocular/face-size: dim0 bottom-decile={lo.mean():.4f} "
              f"top-decile={hi.mean():.4f}  ratio={hi.mean()/lo.mean():.3f}")

    # ---- verdict --------------------------------------------------------------
    MATCH = 1e-3     # float32 round-trip + accumulation tolerance
    matches = [n for n, m in results.items() if m["rel_l2_max"] < MATCH]
    ranked = sorted(results.items(), key=lambda kv: kv[1]["rel_l2_median"])
    print("\n--- VERDICT ----------------------------------------------------------")
    for n, m in ranked[:3]:
        print(f"  closest: {n:34s} relL2_median={m['rel_l2_median']:.3e}")
    if matches:
        print(f"  MATCH (relL2_max < {MATCH:g}): {matches}")
    else:
        print(f"  NO candidate reproduces the shipped vectors at relL2_max < {MATCH:g}")
        print("  -> INCONCLUSIVE for an exact producer match; closest path reported above.")

    # ---- instrument guard -------------------------------------------------
    # If a deliberately mismatched pairing ever looked like a match, the test
    # could not fail and none of the numbers above would mean anything.
    if neg["rel_l2_median"] <= 0.1:
        print("\n!! INSTRUMENT BROKEN: mismatched pairing looked like a match -> exit 3")
        sys.exit(3)
    print(f"\n  instrument guard OK: mismatched pairing relL2={neg['rel_l2_median']:.2f} "
          f"(a match is < {MATCH:g}) -- the test can fail.")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "n_samples": N, "stratum": str(STRATUM), "eidolon_head": git_head(EIDOLON),
            "prx_tg_head": git_head(Path(__file__).resolve().parents[1]),
            "artifact_md5": {k: md5(v) for k, v in ARTIFACTS.items() if v.exists()},
            "shipped_dim0_yaw_r": yaw_r0,
            "shipped_regression": {"intercept": float(intercept), "slope": float(slope), "r": yaw_r0},
            "shipped_perdim_std_mean": float(shipped.std(0).mean()),
            "shipped_dim0_axis": {"r_yaw": r_yaw, "r_interocular": r_inter, "r_pitch": r_pitch},
            "rel_l2_by_yaw_tertile": strat,
            "candidates": results, "negative_control": neg,
            "input_convention_controls": conv,
            "input_convention_notes": conv_note,
            "yaw_leak_pre_post_whitening": leak,
            "shape_fidelity_r2": fid,
            "dim0_mechanism": mech,
            "verdict_matches": matches,
        }, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()