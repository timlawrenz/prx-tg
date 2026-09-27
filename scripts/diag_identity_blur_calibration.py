#!/usr/bin/env python3
"""DIAGNOSTIC: what R@1 does the AuraFace -> 64-d LDA identity instrument give on
REAL faces as image quality drops -- i.e. the QUALITY-MATCHED BASELINE for a
low-quality render?

Why this exists
---------------
`experiments/{eidolon-identity-renderer}` renders faces whose identity R@1 against
a real-identity pool is 0.0588, against a chance rate of 0.0196. That number
cannot be interpreted on its own: it is not known whether 0.0588 is "the identity
stream is broken" or "the instrument cannot recognise a face this degraded".
The recogniser itself is validated at R@1 0.891 on clean real faces
(`scripts/eidolon_identity_replication.py`), so the missing piece is the
DEGRADATION CURVE between those two ends: feed the instrument REAL FFHQ faces with
known identity vectors, degrade them in a controlled, monotone way, and measure
how R@1 falls. The render's own quality then reads off that curve as: at the
render's degradation level, a real face would score X -- and the render scores
0.0588.

What is measured
----------------
* Pool: FFHQ identity holdout (`experiment-configs/eidolon-identity-renderer/
  holdout.json`, 6,996 identity dirs that never enter training; one identity ==
  one directory, one image per identity). Sampled with a fixed seed to
  `--identities` (default 150). Each sampled identity must have BOTH
  `pixel.npy` (3,1024,1024) float16 in [0,1] AND the sidecar
  `auraface_lda.npy` (the ground-truth identity vector the validation loader
  hands the model).
* Degradation families (each level applied to the SAME sampled identities):
    blur   -- Gaussian blur at sigma in {0, 0.5, 1, 1.5, 2, 3, 4, 6, 8} px at
              1024px (sigma 0 == the identity/control level);
    downup -- downscale to 256px (INTER_AREA) then back up (INTER_LINEAR);
    noise  -- additive white Gaussian noise, sigma given as a fraction of the
              [0,1] range (secondary family; see below).
* Per level: (a) a sharpness proxy, the variance of the Laplacian of the uint8
  grey image, averaged over the sampled identities; (b) 1-shot rank-based
  identity R@1 (with R@5, median rank, mean margin) of the degraded image's LDA
  against the index of the other identities' ground-truth LDA, with a bootstrap
  CI over identities (>=2000 reps); (c) the per-level `no_face` count. A level
  with a missing measurement is reported as missing -- no value is ever
  substituted for a `no_face`.
* The arm's existing renders are measured with the same sharpness proxy, so the
  render's degradation level can be read off the curve by interpolation.

Two indexes, and why both are reported
--------------------------------------
  index A (`stored sidecar`) -- the pool index is the stored `auraface_lda.npy`
    of the other identities. This is the convention the render's 0.0588 was
    produced against.
  index B (`clean re-embed`) -- the pool index is the sigma=0 re-embedding of
    the SAME pool by this instrument. This removes the query/index convention
    mismatch (the FFHQ sidecars were built by the FFHQ path with
    det_size=(512,512) -- `eidolon/scripts/pipeline/extract_ffhq_auraface.py`;
    this script uses CORPUS behaviour, det_size=None == 640x640, as instructed),
    so index B isolates the DEGRADATION effect and index A additionally carries
    any convention mismatch. Both are reported; the sigma=0 row of index A is the
    like-for-like anchor for the render's 0.0588.

The sharpness proxy turned out NOT to bracket the render (a finding, not a bug)
------------------------------------------------------------------------------
Variance of the Laplacian is a HIGH-FREQUENCY ENERGY measure: it is inflated by
noise as much as by detail. The arm's renders are noise-dominated (visibly
speckled / grainy), so their proxy sits ABOVE the clean-real endpoint -- no blur
level matches them, and a blur-only calibration can therefore only say "the
render is not simply a blurred real face". For that reason a `noise` family is
implemented alongside the requested blur sweep: it is the family that actually
brackets the render, so `R@1 at the render's matched sharpness` is reported from
whichever family brackets it, with the matching family named in every output. No
interpolation is ever extrapolated past the measured range: when the render's
proxy lies outside a family's range that family reports `in_range=false` and no
interpolated number.

Conventions (copied, not invented)
---------------------------------
* AuraFace instrument (`model_dir` /mnt/nas-ai-models/models/auraface, det_size
  None == insightface default 640x640 == CORPUS behaviour, pad_fraction 0.2,
  `normed_embedding` 512-d, per-image outcome detected / detected_after_padding /
  no_face):  /home/tim/source/activity/eidolon/tools/auraface/extract.py
  (`extract_auraface`, `describe_instrument`, `CORPUS_DET_SIZE`, `PAD_FRACTION`).
* clean -> project recipe (`auraface_preprocess.npz` pooled_mean / pc1_direction /
  yaw_direction, `auraface_lda.npz` lda_basis 512x64 / pooled_mean, and the
  L2-normalize-the-query rule = trap 6):
      scripts/batch_extract_eidolon_data.py          (the sidecar producer)
      scripts/eidolon_identity_retrieval.py          (trap 6, ranking)
      scripts/diag_identity_seed_consistency.py      (AuraFaceLDA, rank_of_own)
* Image convention: `pixel.npy` is (3,H,W) in the same channel order as the VAE
  decode (`production/flux_ae.py` validates the AE by decoding a latent and
  comparing it to `pixel.npy`), i.e. RGB; the instrument takes BGR uint8 in
  cv2.imread order, so the channel order is reversed explicitly. This is exactly
  what the render path does (tensor -> RGB PIL -> PNG -> cv2.imread -> BGR).

Interpreter / cost
------------------
CPU-only by design (`insightface` is pinned to CPUExecutionProvider by the
instrument). No CUDA, no training, no writes under production/, no modification
of any checkpoint. Measured on this host: ~0.21 s per AuraFace extraction after a
~6 s warm-up, so `--identities 150` x (9 blur + 1 downup) levels is ~6 minutes.
Run with:

    .venv/bin/python scripts/diag_identity_blur_calibration.py

`--check-paths` resolves the pool / basis / render paths and exits without
loading the insightface app or embedding anything.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# The AuraFace identity instrument lives in the sibling eidolon repo. APPEND it,
# not insert(0): the prx-tg repo root must keep winning name resolution for
# `production` and for the `scripts`/`experiments` namespaces, which also exist
# under the eidolon root (same rule as scripts/diag_identity_seed_consistency.py).
EIDOLON_ROOT = Path("/home/tim/source/activity/eidolon")
if str(EIDOLON_ROOT) not in sys.path:
    sys.path.append(str(EIDOLON_ROOT))

from tools.auraface import (  # noqa: E402
    CORPUS_DET_SIZE,
    PAD_FRACTION,
    describe_instrument,
    extract_auraface,
)

# ---------------------------------------------------------------------------
# Fixed locations
# ---------------------------------------------------------------------------
# LDA basis artifacts: the directory scripts/batch_extract_eidolon_data.py reads
# (ARTIFACT_DIR) and tools/auraface/extract.py pins hashes for (BASIS_DIR).
BASIS_DIR = EIDOLON_ROOT / "experiments" / "geometry_pca" / "output"
PREPROCESS_NPZ = BASIS_DIR / "auraface_preprocess.npz"
LDA_NPZ = BASIS_DIR / "auraface_lda.npz"

AURAFACE_DIR = Path("/mnt/nas-ai-models/models/auraface")
STRATUM = Path("/mnt/nas-ai-models/training-data/ffhq/stratum")
HOLDOUT_MANIFEST = (
    REPO / "experiment-configs" / "eidolon-identity-renderer" / "holdout.json"
)

# The arm's existing renders (sharpness only; no embedding of rendered faces here
# -- their R@1 is already recorded by the arm's own diagnostics).
RENDER_GLOBS = [
    Path("/home/tim/.hermes/profiles/prx-tg/cache/scratch/eir_probes2/latest/"
         "s*_geoonly_id0_geo10_dim0_*.png"),
    Path("/home/tim/.hermes/profiles/prx-tg/cache/scratch/eir_probes2/latest/"
         "seeds/s*_seed0*.png"),
]
RECON_DIR = Path(
    "/mnt/nas-ai-models/training-data/prx-tg/eidolon-identity-renderer/"
    "runs/2026-09-25_0734/validation/step0008000/reconstruction"
)

DEFAULT_OUT_DIR = REPO / "_diag_identity_blur_calibration"

# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------
BLUR_SIGMAS_DEFAULT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
DOWNUP_SIZES_DEFAULT = [256]
# Additive white Gaussian noise, sigma in [0,1] image units. Chosen to bracket the
# renders' high-frequency energy (measured: clean real ~96-115 lap-var, the
# geoonly renders ~453): 0.02 -> ~341, 0.03 -> ~600.
NOISE_SIGMAS_DEFAULT = [0.01, 0.02, 0.03, 0.04, 0.06]

IDENTITIES_DEFAULT = 150
BOOTSTRAP_REPS_DEFAULT = 2000


# ---------------------------------------------------------------------------
# AuraFace -> 64-d LDA identity instrument
# ---------------------------------------------------------------------------
# Recipe copied faithfully from scripts/batch_extract_eidolon_data.py via
# scripts/diag_identity_seed_consistency.py. The `l2n` step is mandatory:
# `project_to_lda` returns an UNNORMALIZED vector (~norm 150) while every identity
# vector stored in this tree is unit-norm, so a cosine without it is meaningless
# (~150x scale mismatch) -- trap 6 in scripts/eidolon_identity_retrieval.py.
class AuraFaceLDA:
    def __init__(self):
        prep = np.load(PREPROCESS_NPZ)
        self.mu = prep["pooled_mean"]        # (512,)
        self.pc1 = prep["pc1_direction"]     # (512,)
        self.yaw = prep["yaw_direction"]     # (512,)

        lda = np.load(LDA_NPZ)
        self.W = lda["lda_basis"]            # (512, 64)
        self.mu_lda = lda["pooled_mean"]     # (512,)

    def clean_auraface(self, v):
        """Remove domain (PC1) and pose (yaw) nuisances, L2 renormalize."""
        v = np.asarray(v, dtype=np.float64)
        was_1d = v.ndim == 1
        v = np.atleast_2d(v)
        vc = v - self.mu
        vc = vc - np.outer(vc @ self.pc1, self.pc1)
        vc = vc - np.outer(vc @ self.yaw, self.yaw)
        norms = np.linalg.norm(vc, axis=1, keepdims=True)
        vc = vc / (norms + 1e-12)
        return vc[0] if was_1d else vc

    def project_to_lda(self, v_clean):
        """Project cleaned AuraFace vector onto LDA identity basis -> (64,)."""
        v = np.atleast_2d(np.asarray(v_clean, dtype=np.float64))
        vc = v - self.mu_lda
        return (vc @ self.W).squeeze()

    def embed(self, image):
        """BGR uint8 (H,W,3) -> per-image outcome + L2-normalized 64-d LDA.

        NEVER guesses: a no-face image yields ok=False and lda_unit=None.
        """
        res = extract_auraface(
            image,
            det_size=CORPUS_DET_SIZE,   # None == CORPUS behaviour (640x640)
            pad_fraction=PAD_FRACTION,  # 0.2, per side
            keep_embedding=True,
            keep_lda=True,
        )
        out = {
            "ok": bool(res.ok),
            "outcome": str(res.outcome),
            "n_faces": int(res.n_faces),
            "n_faces_first_pass": int(res.n_faces_first_pass),
            "used_padding": bool(res.used_padding),
            "ambiguous": bool(res.ambiguous),
        }
        if not res.ok:
            out["lda_unit"] = None
            out["lda_norm_raw"] = None
            out["cos_vs_helper_lda"] = None
            return out
        lda_raw = self.project_to_lda(self.clean_auraface(res.normed_embedding))
        out["lda_norm_raw"] = float(np.linalg.norm(lda_raw))
        out["lda_unit"] = l2n(lda_raw)
        out["cos_vs_helper_lda"] = (
            float(np.asarray(out["lda_unit"]) @ l2n(res.lda_coords))
            if res.lda_coords is not None else None
        )
        return out


def l2n(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n else v


# ---------------------------------------------------------------------------
# Image I/O and degradation
# ---------------------------------------------------------------------------
def load_pixel_rgb(dirpath: Path) -> np.ndarray:
    """(3,H,W) float16 [0,1] RGB-ish tensor -> (H,W,3) float32 [0,255] RGB.

    Channel order: `pixel.npy` matches the VAE decode output (production/flux_ae.py
    validates the AE by decoding a latent and comparing against pixel.npy), which is
    written to PNG as RGB by production.sample.tensor_to_pil. So the array is the
    same layout the render path round-trips through cv2.imread.
    """
    a = np.load(dirpath / "pixel.npy").astype(np.float32)
    return np.ascontiguousarray(np.transpose(a, (1, 2, 0)) * 255.0)


def to_bgr_u8(rgb_f: np.ndarray) -> np.ndarray:
    """(H,W,3) float [0,255] RGB -> BGR uint8, the instrument's input convention."""
    u8 = np.clip(np.rint(rgb_f), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(u8[:, :, ::-1])


def blur_rgb(rgb_f: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur at 1024px. sigma 0 is an exact no-op (the control level)."""
    if sigma <= 0:
        return rgb_f
    return cv2.GaussianBlur(rgb_f, (0, 0), sigma)


def downup_rgb(rgb_f: np.ndarray, size: int) -> np.ndarray:
    """Downscale to `size` (INTER_AREA) then back to the original size (INTER_LINEAR)."""
    h, w = rgb_f.shape[:2]
    small = cv2.resize(rgb_f, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def noise_rgb(rgb_f: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive white Gaussian noise; `sigma` is a fraction of the [0,1] range."""
    if sigma <= 0:
        return rgb_f
    return rgb_f + rng.normal(0.0, sigma * 255.0, rgb_f.shape).astype(np.float32)


def sharpness(bgr: np.ndarray) -> float:
    """Variance of the Laplacian of the uint8 grey image (the requested proxy).

    OpenCV's Laplacian does not accept float32 in this build, so the input is
    uint8 -- exactly the quantisation every image here (real and rendered) shares.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# Pool selection
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_pool(n: int, seed: int, manifest: Path, stratum_root: Path):
    """Sample `n` eligible FFHQ holdout identities, deterministically.

    Eligible == has both pixel.npy and auraface_lda.npy. The holdout manifest's
    identity list is shuffled with a fixed seed and walked until `n` eligible
    identities are found, so only a handful of NFS stat calls are needed instead
    of stat-ing all 6,996 directories.
    """
    info = {"manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
            "stratum_root": str(stratum_root)}
    ids = None
    source = None
    if manifest.exists():
        d = json.loads(manifest.read_text())
        ids = [str(x) for x in d["ffhq"]["holdout"]]
        info["holdout_declared_count"] = int(d["ffhq"]["holdout_count"])
        info["holdout_declared_sha256"] = d["ffhq"]["holdout_sha256"]
        info["holdout_purpose"] = d.get("purpose")
        source = "holdout_manifest"
    else:
        # Fallback (NOT the documented pool): every stratum dir.
        ids = sorted(x.name for x in stratum_root.iterdir() if x.is_dir())
        source = "stratum_scan_fallback"
        info["note"] = ("holdout manifest absent -- fell back to scanning the whole "
                        "stratum tree; this is NOT the locked validation pool.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    chosen, scanned, rejected = [], 0, {"missing_pixel": 0, "missing_lda": 0}
    for j in order:
        if len(chosen) >= n:
            break
        sid = ids[int(j)]
        scanned += 1
        d = stratum_root / sid
        if not (d / "pixel.npy").exists():
            rejected["missing_pixel"] += 1
            continue
        if not (d / "auraface_lda.npy").exists():
            rejected["missing_lda"] += 1
            continue
        chosen.append(sid)
    info.update({"source": source, "pool_size": len(chosen), "dirs_scanned": scanned,
                 "rejected": rejected, "seed": seed, "requested": n})
    return chosen, info


# ---------------------------------------------------------------------------
# Ranking + bootstrap
# ---------------------------------------------------------------------------
def ranks_against(query_units, pool_unit, own_idx):
    """Rank (1-based) of own identity for each query, plus margin.

    `own_idx[i]` is the query's own row in `pool_unit`, or -1 when the identity is
    not present in that index at all (index B) -- such a query is SKIPPED for that
    index, never scored against a wrong row. Queries that are missing (None) are
    skipped, not substituted.
    """
    ranks, own, best, margin = [], [], [], []
    for qi, q in zip(own_idx, query_units):
        if q is None or qi is None or qi < 0:
            continue
        sims = pool_unit @ q
        order = np.argsort(-sims)
        r = int(np.where(order == qi)[0][0]) + 1
        o = float(sims[qi])
        others = np.delete(sims, qi)
        b = float(others.max()) if others.size else float("nan")
        ranks.append(r)
        own.append(o)
        best.append(b)
        margin.append(o - b)
    return {
        "rank": np.asarray(ranks, dtype=np.int64),
        "own_cos": np.asarray(own, dtype=np.float64),
        "best_other_cos": np.asarray(best, dtype=np.float64),
        "margin": np.asarray(margin, dtype=np.float64),
    }


def bootstrap_r1(ranks: np.ndarray, reps: int, seed: int):
    """CI over IDENTITIES (resample the per-identity hit indicator with replacement)."""
    if ranks.size == 0:
        return {"n": 0, "R@1": None, "ci_lo": None, "ci_hi": None, "reps": reps}
    hits = (ranks == 1).astype(np.float64)
    rng = np.random.default_rng(seed)
    n = hits.size
    draws = rng.integers(0, n, size=(reps, n))
    means = hits[draws].mean(axis=1)
    return {
        "n": int(n),
        "R@1": float(hits.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "reps": int(reps),
    }


def summarise(res: dict, pool_size: int, reps: int, seed: int) -> dict:
    r = res["rank"]
    out = {"chance_R@1": 1.0 / pool_size if pool_size else None}
    out.update(bootstrap_r1(r, reps, seed))
    if r.size:
        out.update({
            "R@5": float((r <= 5).mean()),
            "median_rank": float(np.median(r)),
            "mean_rank": float(r.mean()),
            "mean_own_cos": float(res["own_cos"].mean()),
            "mean_best_other_cos": float(res["best_other_cos"].mean()),
            "mean_margin": float(res["margin"].mean()),
            "frac_own_ge_best_other": float((res["margin"] >= 0).mean()),
        })
    else:
        out.update({"R@5": None, "median_rank": None, "mean_rank": None,
                    "mean_own_cos": None, "mean_best_other_cos": None,
                    "mean_margin": None, "frac_own_ge_best_other": None})
    return out


# ---------------------------------------------------------------------------
# Interpolation (never extrapolates)
# ---------------------------------------------------------------------------
def interp_sigma_at_sharpness(target: float, levels: list):
    """Interpolate degradation strength at a target sharpness. No extrapolation.

    `levels` is a list of dicts with keys `x` (degradation strength) and `sharp`.
    Returns (x_hat, in_range, note). `in_range` False means the target lies outside
    the measured sharpness span, so NO interpolated value is reported.
    """
    pts = [(lv.get("sharp_mean"), lv["x"]) for lv in levels
           if lv.get("sharp_mean") is not None]
    if len(pts) < 2:
        return None, False, "fewer than 2 measured sharpness levels"
    pts.sort(key=lambda p: p[0])              # ascending in sharpness for np.interp
    sharp_sorted = np.array([p[0] for p in pts], dtype=np.float64)
    x_sorted = np.array([p[1] for p in pts], dtype=np.float64)
    if target < sharp_sorted.min() or target > sharp_sorted.max():
        return None, False, (
            f"target sharpness {target:.2f} outside measured span "
            f"[{sharp_sorted.min():.2f}, {sharp_sorted.max():.2f}]")
    # np.interp needs a monotone increasing xp; use log-space for the sharpness
    # axis because it spans orders of magnitude.
    x_hat = float(np.interp(np.log(max(target, 1e-9)), np.log(np.maximum(sharp_sorted, 1e-9)), x_sorted))
    return x_hat, True, "log-sharpness linear interpolation"


def interp_value_at_x(x_target: float, levels: list, key: str):
    """Interpolate a per-level float value at x_target along the degradation axis."""
    pts = [(lv["x"], lv[key]) for lv in levels if lv.get(key) is not None]
    if len(pts) < 2:
        return None, False, "fewer than 2 measured levels"
    pts.sort(key=lambda p: p[0])
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    if x_target < xs.min() or x_target > xs.max():
        return None, False, f"{key} target {x_target:.3f} outside measured span [{xs.min():.3f}, {xs.max():.3f}]"
    return float(np.interp(x_target, xs, ys)), True, "linear interpolation"


def interp_r1_at_x(x_target: float, levels: list, summary_key: str):
    """Interpolate R@1 (from a per-level summary dict) at x_target. No extrapolation."""
    pts = [(lv["x"], lv[summary_key]["R@1"]) for lv in levels
           if lv.get(summary_key) and lv[summary_key].get("R@1") is not None]
    if len(pts) < 2:
        return None, False, f"fewer than 2 measured {summary_key} R@1 levels"
    pts.sort(key=lambda p: p[0])
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    if x_target < xs.min() or x_target > xs.max():
        return None, False, (
            f"{summary_key} R@1 target x={x_target:.3f} outside measured span "
            f"[{xs.min():.3f}, {xs.max():.3f}]")
    return float(np.interp(x_target, xs, ys)), True, "linear interpolation"


# ---------------------------------------------------------------------------
# Render sharpness
# ---------------------------------------------------------------------------
def render_sharpness(extra_dirs=()):
    out = {"groups": [], "files": []}
    for pattern in list(RENDER_GLOBS) + [p for p in extra_dirs]:
        files = sorted(Path(pattern).parent.glob(Path(pattern).name)) if isinstance(pattern, Path) else []
        if not files:
            out["groups"].append({"pattern": str(pattern), "n": 0, "note": "no files"})
            continue
        vals = []
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                out["files"].append({"path": str(f), "sharpness": None, "note": "unreadable"})
                continue
            s = sharpness(img)
            vals.append(s)
            out["files"].append({"path": str(f), "shape": list(img.shape), "sharpness": s})
        a = np.asarray(vals, dtype=np.float64)
        out["groups"].append({
            "pattern": str(pattern), "n": int(a.size),
            "sharpness_mean": float(a.mean()) if a.size else None,
            "sharpness_median": float(np.median(a)) if a.size else None,
            "sharpness_min": float(a.min()) if a.size else None,
            "sharpness_max": float(a.max()) if a.size else None,
        })
    return out


def reconstruction_sharpness():
    if not RECON_DIR.is_dir():
        return {"dir": str(RECON_DIR), "n": 0, "note": "absent"}
    files = sorted(RECON_DIR.glob("*.png"))
    vals = []
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            vals.append(sharpness(img))
    a = np.asarray(vals, dtype=np.float64)
    return {"dir": str(RECON_DIR), "n": int(a.size),
            "sharpness_mean": float(a.mean()) if a.size else None,
            "sharpness_median": float(np.median(a)) if a.size else None,
            "sharpness_min": float(a.min()) if a.size else None,
            "sharpness_max": float(a.max()) if a.size else None}


# ---------------------------------------------------------------------------
# --check-paths
# ---------------------------------------------------------------------------
def check_paths(args):
    print("=" * 72)
    print("PATH CHECK (no insightface app, no embedding, no writes)")
    print("=" * 72)
    print(f"  interpreter : {sys.executable}")
    print(f"  repo        : {REPO}")
    problems = []

    def report(label, path: Path, must_be_file=None):
        exists = path.exists()
        kind = ("file" if path.is_file() else "dir") if exists else "-"
        ok = exists and (must_be_file is None or path.is_file() == must_be_file)
        print(f"  {'OK ' if ok else 'MISSING'} {label:32s} [{kind:4s}] {path}")
        if not ok:
            problems.append(f"{label}: {path}")
        return ok

    report("preprocess npz", PREPROCESS_NPZ, must_be_file=True)
    report("lda npz", LDA_NPZ, must_be_file=True)
    report("auraface model dir", AURAFACE_DIR, must_be_file=False)
    for f in ("glintr100.onnx", "scrfd_10g_bnkps.onnx"):
        report(f"auraface/{f}", AURAFACE_DIR / f, must_be_file=True)
    report("stratum root", STRATUM, must_be_file=False)
    report("holdout manifest", HOLDOUT_MANIFEST, must_be_file=True)
    report("eidolon instrument root", EIDOLON_ROOT / "tools" / "auraface",
           must_be_file=False)

    for pattern in RENDER_GLOBS:
        n = len(list(Path(pattern).parent.glob(Path(pattern).name)))
        print(f"  {'OK ' if n else 'MISSING'} render glob {pattern.name[:34]:34s} n={n}")
    report("reconstruction dir", RECON_DIR, must_be_file=False)

    try:
        prov = describe_instrument(det_size=CORPUS_DET_SIZE)
        print(f"  OK  instrument: det_size={prov['det_size']} "
              f"pad_fraction={prov['pad_fraction']} model_dir={prov['model_dir']}")
        print(f"      basis_fingerprint={prov['basis_fingerprint']} "
              f"(expected {prov['expected_basis_fingerprint']})")
    except Exception as e:
        print(f"  MISSING instrument: {type(e).__name__}: {e}")
        problems.append(f"instrument: {e}")

    if HOLDOUT_MANIFEST.exists():
        ids, info = select_pool(3, 0, HOLDOUT_MANIFEST, STRATUM)
        print(f"  pool probe: source={info['source']} scanned={info['dirs_scanned']} "
              f"eligible={info['pool_size']} sample={ids}")

    print("-" * 72)
    print(f"PROBLEMS: {len(problems)}")
    for p in problems:
        print(f"  - {p}")
    return 1 if problems else 0


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(fig_path: Path, blur_levels, noise_levels, matched, render_sharp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # --- panel 1: R@1 vs blur sigma ---
    ax = axes[0]
    xs = [lv["x"] for lv in blur_levels]
    for key, label, colour in (("summary_indexA", "index A: stored sidecar GT", "#c0392b"),
                               ("summary_indexB", "index B: clean re-embed", "#2471a3")):
        ys = [(lv[key]["R@1"] if lv[key]["R@1"] is not None else np.nan) for lv in blur_levels]
        lo = [(lv[key]["ci_lo"] if lv[key]["ci_lo"] is not None else np.nan) for lv in blur_levels]
        hi = [(lv[key]["ci_hi"] if lv[key]["ci_hi"] is not None else np.nan) for lv in blur_levels]
        ax.plot(xs, ys, "o-", color=colour, label=label, lw=2)
        ax.fill_between(xs, lo, hi, color=colour, alpha=0.15)
    if blur_levels and blur_levels[0]["summary_indexA"]["chance_R@1"]:
        ax.axhline(blur_levels[0]["summary_indexA"]["chance_R@1"], color="grey",
                   ls=":", label="chance (1/pool)")
    m = matched.get("blur_matched") or {}
    if m.get("in_range") and m.get("sigma") is not None and m.get("R@1_indexA") is not None:
        ax.plot([m["sigma"]], [m["R@1_indexA"]], "k*", ms=18,
                label=f"render-matched (sigma={m['sigma']:.2f})")
        ax.annotate(f"R@1={m['R@1_indexA']:.3f}", (m["sigma"], m["R@1_indexA"]),
                    textcoords="offset points", xytext=(8, 10), fontsize=9)
    for lv in blur_levels:
        if lv["x"] == 0.0:
            ax.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("Gaussian blur sigma (px @ 1024)")
    ax.set_ylabel("identity R@1 (1-shot)")
    ax.set_title("R@1 vs blur (bootstrap 95% CI over identities)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

    # --- panel 2: sharpness vs degradation, with the render marked ---
    ax2 = axes[1]
    ys = [lv["sharp_mean"] for lv in blur_levels]
    ax2.semilogy(xs, ys, "o-", color="#c0392b", label="blur family")
    if noise_levels:
        nxs = [lv["x"] for lv in noise_levels]
        nys = [lv["sharp_mean"] for lv in noise_levels]
        ax2.semilogy(nxs, nys, "s--", color="#7d3c98", label="noise family (sigma/255)")
    if render_sharp is not None:
        ax2.axhline(render_sharp, color="k", ls="--", lw=1.2,
                    label=f"arm render ({render_sharp:.0f} lap-var)")
    ax2.set_xlabel("degradation strength (blur sigma px | noise sigma/255)")
    ax2.set_ylabel("sharpness = var(Laplacian)  [log]")
    ax2.set_title("Sharpness proxy vs degradation")
    ax2.grid(alpha=0.25, which="both")
    ax2.legend(fontsize=8, loc="upper right")

    fig.suptitle("AuraFace->LDA identity R@1 vs image degradation (real FFHQ holdout faces)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--identities", type=int, default=IDENTITIES_DEFAULT,
                    help=f"real FFHQ holdout identities to sample (default {IDENTITIES_DEFAULT})")
    ap.add_argument("--seed", type=int, default=20260926,
                    help="sampling seed for pool selection and for the noise draws")
    ap.add_argument("--families", type=str, default="blur,downup",
                    help="comma list from {blur,downup,noise} (default blur,downup)")
    ap.add_argument("--blur-sigmas", type=float, nargs="+", default=BLUR_SIGMAS_DEFAULT)
    ap.add_argument("--downup-sizes", type=int, nargs="+", default=DOWNUP_SIZES_DEFAULT)
    ap.add_argument("--noise-sigmas", type=float, nargs="+", default=NOISE_SIGMAS_DEFAULT)
    ap.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS_DEFAULT)
    ap.add_argument("--pool-start", type=int, default=0,
                    help="skip the first N sampled identities (for a second, disjoint run)")
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--check-paths", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if args.check_paths:
        return check_paths(args)
    out.mkdir(parents=True, exist_ok=True)

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    for f in families:
        if f not in ("blur", "downup", "noise"):
            raise SystemExit(f"unknown family {f!r}; choose from blur,downup,noise")

    # ---- build the level list -------------------------------------------------
    levels = []
    if "blur" in families:
        for s in args.blur_sigmas:
            levels.append({"family": "blur", "label": f"blur_sigma={s:g}", "x": float(s)})
    if "downup" in families:
        for z in args.downup_sizes:
            levels.append({"family": "downup", "label": f"downup_{z}", "x": float(z)})
    if "noise" in families:
        for s in args.noise_sigmas:
            levels.append({"family": "noise", "label": f"noise_sigma={s:g}", "x": float(s)})

    print("=" * 72)
    print("IDENTITY BLUR / QUALITY CALIBRATION (real FFHQ holdout faces)")
    print("=" * 72)
    print(f"  interpreter : {sys.executable}")
    print(f"  out-dir     : {out}")
    print(f"  families    : {families}  ({len(levels)} levels)")

    pool_ids, pool_info = select_pool(
        args.identities + args.pool_start, args.seed, HOLDOUT_MANIFEST, STRATUM
    )
    pool_ids = pool_ids[args.pool_start:]
    print(f"  pool        : {len(pool_ids)} identities from {pool_info['source']} "
          f"(scanned {pool_info['dirs_scanned']} dirs, rejected {pool_info['rejected']})")
    if len(pool_ids) < 4:
        raise SystemExit(f"pool of {len(pool_ids)} is too small to measure R@1")

    print("  loading AuraFace instrument provenance...", flush=True)
    instrument = describe_instrument(det_size=CORPUS_DET_SIZE)
    lda_inst = AuraFaceLDA()
    print(f"  instrument  : det_size={instrument['det_size']} "
          f"(None == 640x640 CORPUS) pad={instrument['pad_fraction']} "
          f"basis_fingerprint={instrument['basis_fingerprint']}")

    # ground-truth sidecars -> index A
    gt_unit = []
    keep = []
    for sid in pool_ids:
        v = np.load(STRATUM / sid / "auraface_lda.npy").astype(np.float64)
        gt_unit.append(l2n(v))
        keep.append(sid)
    gt_unit = np.stack(gt_unit)                     # (N, 64) unit-norm
    N = len(keep)
    print(f"  index A     : {N} stored ground-truth sidecar LDA vectors "
          f"(chance R@1 = {1.0 / N:.4f})")

    # ---- sweep ---------------------------------------------------------------
    # query_units[level_index][identity_index]
    query_units = [[None] * N for _ in levels]
    sharp_per_level = [[] for _ in levels]
    outcomes_per_level = [[] for _ in levels]
    per_identity = []
    t0 = time.time()

    for i, sid in enumerate(keep):
        d = STRATUM / sid
        rgb0 = load_pixel_rgb(d)
        rec = {"identity": sid, "levels": []}
        for li, lv in enumerate(levels):
            fam, x = lv["family"], lv["x"]
            if fam == "blur":
                rgb = blur_rgb(rgb0, x)
            elif fam == "downup":
                rgb = downup_rgb(rgb0, int(x))
            else:
                rng = np.random.default_rng([args.seed, li, i])
                rgb = noise_rgb(rgb0, x, rng)
            bgr = to_bgr_u8(rgb)
            sh = sharpness(bgr)
            sharp_per_level[li].append(sh)
            m = lda_inst.embed(bgr)
            outcomes_per_level[li].append(m["outcome"])
            if m["lda_unit"] is None:
                query_units[li][i] = None
            else:
                query_units[li][i] = m["lda_unit"]
            rec["levels"].append({
                "label": lv["label"],
                "sharpness": sh,
                "outcome": m["outcome"],
                "ok": m["ok"],
                "lda_norm_raw": m["lda_norm_raw"],
                "cos_vs_helper_lda": m["cos_vs_helper_lda"],
                "lda_unit": None if m["lda_unit"] is None
                else [float(v) for v in m["lda_unit"]],
            })
        per_identity.append(rec)
        if (i + 1) % 10 == 0 or i == 0:
            el = time.time() - t0
            print(f"    {i + 1}/{N} identities  ({el:.0f}s, {el / (i + 1):.2f}s/identity)",
                  flush=True)

    elapsed = time.time() - t0
    print(f"  sweep done in {elapsed:.0f}s")

    # ---- index B: clean re-embed (sigma=0 blur level is the reference) ---------
    zero_li = next((li for li, lv in enumerate(levels)
                    if lv["family"] == "blur" and lv["x"] == 0.0), None)
    if zero_li is None:
        indexB = None
        indexB_note = ("no blur sigma=0 level in this run (index B unavailable; "
                       "add blur 0.0 to --blur-sigmas)")
        print(f"  index B     : UNAVAILABLE -- {indexB_note}")
    else:
        clean = [q for q in query_units[zero_li]]
        indexB = np.stack([q for q in clean if q is not None])
        indexB_note = (
            "index B is the clean re-embedding of the SAME pool by this instrument "
            "(det_size None). Because queries that returned no_face at sigma=0 are "
            "absent, index B is smaller than index A and its own-index positions are "
            "remapped; ranks are computed against the remapped pool."
        )
        indexB_ids = [keep[i] for i, q in enumerate(clean) if q is not None]
        print(f"  index B     : {indexB.shape[0]} clean re-embeddings "
              f"(chance R@1 = {1.0 / indexB.shape[0]:.4f})")

    # ---- per-level metrics ---------------------------------------------------
    for li, lv in enumerate(levels):
        lv["sharp_mean"] = float(np.mean(sharp_per_level[li])) if sharp_per_level[li] else None
        lv["sharp_median"] = float(np.median(sharp_per_level[li])) if sharp_per_level[li] else None
        lv["sharp_min"] = float(np.min(sharp_per_level[li])) if sharp_per_level[li] else None
        lv["sharp_max"] = float(np.max(sharp_per_level[li])) if sharp_per_level[li] else None
        n_no_face = sum(1 for o in outcomes_per_level[li] if o == "no_face")
        lv["n_identities"] = len(outcomes_per_level[li])
        lv["no_face"] = n_no_face
        lv["no_face_rate"] = n_no_face / len(outcomes_per_level[li]) if outcomes_per_level[li] else None
        lv["outcome_counts"] = {
            o: outcomes_per_level[li].count(o) for o in sorted(set(outcomes_per_level[li]))
        }
        lv["n_used_padding"] = sum(1 for o in outcomes_per_level[li]
                                   if o == "detected_after_padding")
        lv["n_measured"] = sum(1 for q in query_units[li] if q is not None)
        lv["query_units"] = query_units[li]

        lv["summary_indexA"] = summarise(
            ranks_against(query_units[li], gt_unit, list(range(N))),
            N, args.bootstrap_reps, args.seed + 1000 + li)
        lv["chance_R@1"] = 1.0 / N

        if indexB is not None:
            # remap own index: position of this identity in the indexB pool
            b_pos = {sid: k for k, sid in enumerate(indexB_ids)}
            own = [b_pos.get(sid, -1) for sid in keep]
            lv["summary_indexB"] = summarise(
                ranks_against(query_units[li], indexB, own),
                indexB.shape[0], args.bootstrap_reps, args.seed + 2000 + li)
        else:
            lv["summary_indexB"] = None

    # ---- render sharpness + matching ----------------------------------------
    renders = render_sharpness()
    recon = reconstruction_sharpness()
    geoonly = next((g for g in renders["groups"]
                    if "geoonly_id0_geo10" in g["pattern"] and g["n"]), None)
    seeds_g = next((g for g in renders["groups"]
                    if "/seeds/" in g["pattern"] and g["n"]), None)
    primary = geoonly or seeds_g
    if primary:
        render_sharp = primary["sharpness_median"]
        render_sharp_src = {"group": primary["pattern"], "stat": "median",
                            "value": render_sharp}
    elif recon.get("n"):
        render_sharp = recon["sharpness_median"]
        render_sharp_src = {"group": recon["dir"], "stat": "median", "value": render_sharp}
    else:
        render_sharp, render_sharp_src = None, None
    print(f"  render sharpness: {render_sharp}  ({render_sharp_src})")

    matched = {}
    if render_sharp is not None:
        for fam in ("blur", "downup", "noise"):
            sub = [lv for lv in levels if lv["family"] == fam]
            if len(sub) < 2:
                continue
            x_hat, in_range, note = interp_sigma_at_sharpness(render_sharp, sub)
            entry = {"family": fam, "sigma": x_hat, "in_range": in_range, "note": note}
            if in_range and x_hat is not None:
                a1, ok1, n1 = interp_r1_at_x(x_hat, sub, "summary_indexA")
                entry["R@1_indexA"] = a1 if ok1 else None
                entry["note_indexA"] = n1
                if all(lv["summary_indexB"] for lv in sub):
                    b1, ok2, n2 = interp_r1_at_x(x_hat, sub, "summary_indexB")
                    entry["R@1_indexB"] = b1 if ok2 else None
                    entry["note_indexB"] = n2
                else:
                    entry["R@1_indexB"] = None
            matched[f"{fam}_matched"] = entry

    # ---- console tables ------------------------------------------------------
    print("\n" + "=" * 72)
    print("PER-LEVEL RESULTS")
    print("=" * 72)
    hdr = (f"{'level':>18s} {'sharp':>9s} {'no_face':>8s} {'n':>4s} "
           f"{'R@1(A)':>7s} {'95% CI':>15s} {'R@5':>6s} {'medR':>5s} {'R@1(B)':>7s}")
    print(hdr)
    for lv in levels:
        a = lv["summary_indexA"]
        b = lv["summary_indexB"]
        r1 = "n/a" if a["R@1"] is None else f"{a['R@1']:.4f}"
        ci = ("n/a" if a["ci_lo"] is None
              else f"[{a['ci_lo']:.3f},{a['ci_hi']:.3f}]")
        r5 = "n/a" if a["R@5"] is None else f"{a['R@5']:.3f}"
        mr = "n/a" if a["median_rank"] is None else f"{a['median_rank']:.0f}"
        b1 = "n/a" if (b is None or b["R@1"] is None) else f"{b['R@1']:.4f}"
        print(f"{lv['label']:>18s} {lv['sharp_mean']:9.2f} "
              f"{lv['no_face']:4d}/{lv['n_identities']:<3d} {lv['n_measured']:4d} "
              f"{r1:>7s} {ci:>15s} {r5:>6s} {mr:>5s} {b1:>7s}")
    print(f"\nchance R@1 (pool {N}) = {1.0 / N:.4f}")

    print("\n" + "=" * 72)
    print("HEADLINE")
    print("=" * 72)
    if zero_li is not None:
        z = levels[zero_li]
        zA = z["summary_indexA"]["R@1"]
        zB = z["summary_indexB"]["R@1"] if z["summary_indexB"] else None
        print(f"  R@1 at sigma=0 (clean real faces): "
              f"{'n/a' if zA is None else f'{zA:.4f}'} (index A, stored-sidecar GT), "
              f"{'n/a' if zB is None else f'{zB:.4f}'} (index B, clean re-embed)  "
              f"[pool {N}, chance {1.0 / N:.4f}]")
        print(f"  no_face at sigma=0: {z['no_face']}/{z['n_identities']}")
    print(f"  render sharpness (var Laplacian) = "
          f"{'n/a' if render_sharp is None else f'{render_sharp:.1f}'}  {render_sharp_src}")
    for fam in ("blur", "downup", "noise"):
        e = matched.get(f"{fam}_matched")
        if not e:
            continue
        if e["in_range"]:
            a1 = e["R@1_indexA"]
            b1 = e.get("R@1_indexB")
            print(f"  [{fam}] render matches {e['sigma']:.3f} -> "
                  f"R@1(indexA) = {'n/a' if a1 is None else f'{a1:.4f}'}, "
                  f"R@1(indexB) = {'n/a' if b1 is None else f'{b1:.4f}'}")
        else:
            print(f"  [{fam}] NO MATCH: {e['note']}")
    nf_max = max(lv["no_face_rate"] for lv in levels)
    print(f"  worst no_face rate across levels = {nf_max:.4f}")

    # ---- write results -------------------------------------------------------
    results = {
        "diagnostic": "identity_blur_calibration",
        "question": ("how does the AuraFace->LDA identity instrument's R@1 degrade "
                     "on REAL faces as image quality drops, and what is the "
                     "quality-matched baseline for the arm's low-quality renders?"),
        "pool": pool_info,
        "pool_ids": keep,
        "pool_size": N,
        "chance_R@1": 1.0 / N,
        "families": families,
        "blur_sigmas": args.blur_sigmas,
        "downup_sizes": args.downup_sizes,
        "noise_sigmas": args.noise_sigmas,
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "instrument": instrument,
        "index_B": {"available": indexB is not None,
                    "n": (None if indexB is None else int(indexB.shape[0])),
                    "ids": (None if indexB is None else indexB_ids),
                    "note": indexB_note},
        "levels": [{k: v for k, v in lv.items() if k != "query_units"} for lv in levels],
        "sharpness_curve": [
            {"family": lv["family"], "x": lv["x"], "sharp_mean": lv["sharp_mean"],
             "sharp_median": lv["sharp_median"]} for lv in levels
        ],
        "r1_curve": [
            {"family": lv["family"], "x": lv["x"],
             "R@1_indexA": lv["summary_indexA"]["R@1"],
             "ci_lo_indexA": lv["summary_indexA"]["ci_lo"],
             "ci_hi_indexA": lv["summary_indexA"]["ci_hi"],
             "R@1_indexB": (lv["summary_indexB"]["R@1"] if lv["summary_indexB"] else None)}
            for lv in levels
        ],
        "render_sharpness": renders,
        "render_sharpness_primary": render_sharp_src,
        "reconstruction_sharpness": recon,
        "render_matched": matched,
        "per_identity": per_identity,
        "notes": [],
        "provenance": {
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": os.uname().nodename,
            "interpreter": sys.executable,
            "elapsed_sweep_seconds": elapsed,
            "device": "cpu (insightface pinned to CPUExecutionProvider)",
            "git_commit": _git_commit(),
            "script": str(Path(__file__).resolve()),
            "conventions": {
                "auraface_input": "(H,W,3) BGR uint8 -- pixel.npy is RGB (matches the "
                                  "VAE decode) and is reversed explicitly",
                "det_size": CORPUS_DET_SIZE,
                "det_size_note": ("None == insightface default 640x640, the CORPUS "
                                  "convention. NOTE: the FFHQ sidecars were produced by "
                                  "the FFHQ path with det_size=(512,512) "
                                  "(eidolon/scripts/pipeline/extract_ffhq_auraface.py), so "
                                  "index A carries a query/index convention mismatch that "
                                  "index B removes."),
                "pad_fraction": PAD_FRACTION,
                "embedding": "normed_embedding (512-d)",
                "lda": "clean_auraface -> project_to_lda -> L2-normalize the query (trap 6)",
                "index_A": "other identities' stored auraface_lda.npy, L2-normalized",
                "index_B": "clean (sigma=0) re-embeddings of the same pool",
                "sharpness": "variance of the Laplacian of the uint8 grey image",
                "no_face": "outcome no_face -> lda_unit=null, excluded from R@1; never "
                           "substituted",
                "interpolation": "log-sharpness linear interpolation, never extrapolated",
            },
        },
    }

    # findings / caveats
    if render_sharp is not None:
        blur_sharp0 = next((lv["sharp_mean"] for lv in levels
                            if lv["family"] == "blur" and lv["x"] == 0.0), None)
        if blur_sharp0 is not None:
            if render_sharp > blur_sharp0:
                results["notes"].append(
                    f"FINDING: the render's sharpness proxy ({render_sharp:.1f}) is ABOVE "
                    f"the clean-real mean ({blur_sharp0:.1f}), so NO blur level matches it. "
                    "Variance of the Laplacian measures high-frequency ENERGY, which the "
                    "renders' speckle/grain inflates; the renders are not merely blurred "
                    "real faces, they are noise-dominated. The blur sweep therefore cannot "
                    "supply the quality-matched baseline on its own; read the noise-family "
                    "match for the render and the blur curve for the degradation trend."
                )
    if any(lv["no_face"] for lv in levels):
        results["notes"].append(
            "some levels returned no_face; those identities are excluded from R@1 at that "
            "level (n_measured per level is reported) and no value was substituted."
        )
    if indexB is not None and indexB.shape[0] < N:
        results["notes"].append(
            f"index B has {indexB.shape[0]} of {N} identities (sigma=0 no_face removes "
            "identities from it), so index A and index B are not computed over identical "
            "pools; index A uses all "
            f"{N} stored sidecars as queries whenever the level's image is detectable."
        )
    costs = [r["cos_vs_helper_lda"] for r in [l for pi in per_identity for l in pi["levels"]]
             if r["cos_vs_helper_lda"] is not None]
    if costs and min(costs) < 0.99999:
        results["notes"].append(
            f"the inline clean->project recipe and the instrument's own lda_coords "
            f"disagree (min cosine {min(costs):.6f}); the basis artifacts may have moved."
        )

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults.json -> {out / 'results.json'}")

    blur_levels = [lv for lv in levels if lv["family"] == "blur"]
    noise_levels = [lv for lv in levels if lv["family"] == "noise"]
    fig_path = out / "identity_blur_calibration.png"
    if blur_levels:
        make_figure(fig_path, blur_levels, noise_levels, matched, render_sharp)
        print(f"figure       -> {fig_path}")

    for note in results["notes"]:
        print(f"  note: {note}")
    print("\nDONE.")
    return 0


def _git_commit():
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
