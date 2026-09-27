#!/usr/bin/env python3
"""Is the shipped z_g dim0 axis a real 3D head rotation, or a 2D in-plane shear?

WHY THIS EXISTS
---------------
The arm `eidolon-identity-renderer` conditions on z_g dim0. Two of our own
measurements pin dim0 to head yaw:

    corr(z_g dim0, DWPose yaw proxy) = -0.94   (|r| rank 1 / 50, n=300)
    yaw = -0.0037 - 0.1831 * dim0              (r = -0.937, n=10000)

but a correlation with yaw CANNOT distinguish the two mechanisms that both
produce a yaw-like signal:

  ROTATION    a genuine 3D head rotation. Under true yaw the far-side jaw
              collapses behind the face while the PROJECted interocular ratio
              holds ~1.00 (it shrinks by cos(yaw), a few % at most at the yaws
              the sweep reaches, and DWPose's eye-corner fit absorbs part of it).
              Signature: inter/median FLAT about 1.00, L/R swings hard.

  SHEAR       a 2D in-plane shear / registration artifact passed through from
              the geometry encoder. The eye line STRETCHES as the axis moves.
              Signature: inter/median WIDENS away from the centre bin.

Published result on real faces binned by DWPose yaw (yaw_mechanism_test.py):
inter/median 0.916 0.992 1.004 1.004 0.991 0.927 (flat about 1.00) while L/R
swings 0.19 -> 4.25 (22x) -- the ROTATION signature.

This script asks the same question of the DATA along dim0 instead of along yaw:
bin the same real FFHQ images by the SHIPPED z_g dim0 value, compute the same
two statistics with the SAME helper functions, and run the yaw binning in the
same run as an internal control (it must reproduce the published flat curve).

The arm's RENDERS widen interocular by 7-12% with dim0. If the real data along
dim0 does NOT widen, the renders' widening is a model/projection artifact, not
a property of the conditioning axis.

CONVENTIONS (copied verbatim from the reference + repo audit scripts; do not
re-invent):
  * pose.npy is DWPose COCO-WholeBody 133 points, 68 iBUG face points = rows
    23:91, values NORMALIZED to [-1, 1] -- NOT pixels. Convert with
    (v + 1) * 512.0 exactly as yaw_mechanism_test.py does.
  * acceptance: mean face confidence >= 0.3 and interocular >= 20 px.
  * statistics: copied verbatim from yaw_mechanism_test.py `metrics()`.
  * yaw / pitch / roll proxies: copied verbatim from yaw_pitch_coverage.py.

CPU only, read-only on the NAS. Nothing under production/ is touched.

Usage
-----
    python scripts/audit_zg_dim0_projection_signature.py --n 500
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

STRATUM = Path("/mnt/nas-ai-models/training-data/ffhq/stratum")
DEFAULT_OUT = Path(__file__).resolve().parents[1] / \
    "experiment-configs/eidolon-identity-renderer/audit/zg_dim0_projection_signature"
N_PIXEL_CHECK = 5          # how many pixel.npy we actually open (existence is required for all)


# --------------------------------------------------------------------------- #
# helpers copied VERBATIM from
# experiment-configs/eidolon-identity-renderer/audit/yaw_mechanism_test.py
# --------------------------------------------------------------------------- #
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


def pixels_from_normalized(f):
    """f: (68,3) normalized [-1,1] face rows -> (68,2) pixels, reference convention."""
    return np.stack([(f[:, 0] + 1) * 512.0, (f[:, 1] + 1) * 512.0], 1)


# --------------------------------------------------------------------------- #
# helpers copied VERBATIM from
# experiment-configs/eidolon-identity-renderer/audit/yaw_pitch_coverage.py
# --------------------------------------------------------------------------- #
def proxies(pts):
    """pts: (68,2) in pixels -> (yaw, pitch, roll_deg)."""
    el, er, nose = pts[36:42].mean(0), pts[42:48].mean(0), pts[30]
    inter = float(np.linalg.norm(el - er))
    ecx, ecy = 0.5 * (el[0] + er[0]), 0.5 * (el[1] + er[1])
    yaw = (nose[0] - ecx) / inter
    pitch = (nose[1] - ecy) / inter
    roll = float(np.degrees(np.arctan2(er[1] - el[1], er[0] - el[0])))
    return float(yaw), float(pitch), roll


# --------------------------------------------------------------------------- #
# sampling (NFS-safe: os.listdir + os.path.exists, never globs on this path)
# --------------------------------------------------------------------------- #
def sample_dirs(n, seed=0):
    names = sorted(d for d in os.listdir(STRATUM) if os.path.isdir(os.path.join(STRATUM, d)))
    rng = random.Random(seed)
    rng.shuffle(names)
    pick, miss = [], {"z_g.npy": 0, "pose.npy": 0, "pixel.npy": 0}
    for d in names:
        p = os.path.join(STRATUM, d)
        have = {f: os.path.exists(os.path.join(p, f)) for f in miss}
        if not all(have.values()):
            for f, h in have.items():
                if not h:
                    miss[f] += 1
            continue
        pick.append((d, p))
        if len(pick) >= n:
            break
    return pick, len(names), miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("z_g dim0 PROJECTION-SIGNATURE AUDIT  (rotation vs 2D shear)")
    print("=" * 78)
    print(f"stratum: {STRATUM}")
    print(f"output : {out}\n")

    pick, n_listed, miss = sample_dirs(args.n, args.seed)
    print(f"sampled {len(pick)} dirs with z_g.npy + pose.npy + pixel.npy "
          f"(of {n_listed} dirs listed)")
    print(f"  skipped for missing sidecar: z_g.npy {miss['z_g.npy']}, "
          f"pose.npy {miss['pose.npy']}, pixel.npy {miss['pixel.npy']}")

    # ---- optional sanity: prove pixel.npy is readable+shaped as documented ------
    pchk = []
    for d, p in pick[:N_PIXEL_CHECK]:
        try:
            a = np.load(os.path.join(p, "pixel.npy"), mmap_mode="r")
            pchk.append({"id": d, "shape": list(a.shape), "dtype": str(a.dtype)})
        except Exception as e:                                   # pragma: no cover
            pchk.append({"id": d, "error": str(e)})
    print(f"  pixel.npy readability spot-check (mmap, first {len(pchk)}): {pchk}")

    # ---- load -------------------------------------------------------------------
    excl = {"missing_pose": 0, "face_conf_lt_0.3": 0, "short_face": 0,
            "inter_lt_20": 0, "load_error": 0, "missing_zg": 0}
    dim0, yaw, pitch, roll, inter, width, left, right, inter_med_agn = [], [], [], [], [], [], [], [], []
    XY_MIN, XY_MAX = np.inf, -np.inf
    for d, p in pick:
        try:
            zg = np.load(os.path.join(p, "z_g.npy"))
        except Exception:
            excl["missing_zg"] += 1
            continue
        try:
            f = np.load(os.path.join(p, "pose.npy")).astype(np.float64)
        except Exception:
            excl["missing_pose"] += 1
            continue
        if f.ndim != 2 or f.shape[0] < 91:
            excl["short_face"] += 1
            continue
        face = f[23:91]
        XY_MIN = min(XY_MIN, float(face[:, :2].min()))
        XY_MAX = max(XY_MAX, float(face[:, :2].max()))
        if face[:, 2].mean() < 0.3:
            excl["face_conf_lt_0.3"] += 1
            continue
        pts = pixels_from_normalized(face)
        m = metrics(pts)                       # reference metrics (also yaw proxy)
        if m["inter"] < 20:
            excl["inter_lt_20"] += 1
            continue
        y, pi, ro = proxies(pts)               # reference yaw/pitch/roll proxies
        dim0.append(float(np.asarray(zg).astype(np.float64).reshape(-1)[0]))
        yaw.append(m["yaw"])
        pitch.append(pi)
        roll.append(ro)
        inter.append(m["inter"])
        width.append(m["width"])
        left.append(m["left"])
        right.append(m["right"])

    dim0 = np.array(dim0); yaw = np.array(yaw); pitch = np.array(pitch); roll = np.array(roll)
    inter = np.array(inter); width = np.array(width)
    left = np.array(left); right = np.array(right)
    n = len(dim0)
    print(f"\nusable samples: {n}")
    print("excluded: " + ", ".join(f"{k}={v}" for k, v in excl.items()))
    print(f"pose xy range over usable set: [{XY_MIN:.3f}, {XY_MAX:.3f}]  -> "
          f"{'NORMALIZED [-1,1] (converted to px)' if XY_MAX <= 1.05 else 'PIXEL-LIKE'}")
    if n < 100:
        sys.exit("refusing to report on <100 usable samples")

    inter_med = float(np.median(inter))
    print(f"median interocular (px): {inter_med:.1f}   median face width: {np.median(width):.1f}")

    # ---- the two statistics, exactly as the reference computes them -------------
    def bin_stats(idx):
        sel = np.asarray(idx)
        if sel.size == 0:
            return None
        return {
            "n": int(sel.size),
            "inter_over_median": float(np.mean(inter[sel] / inter_med)),
            "width_over_inter": float(np.mean(width[sel] / inter[sel])),
            "left_over_inter": float(np.mean(left[sel] / inter[sel])),
            "right_over_inter": float(np.mean(right[sel] / inter[sel])),
            "LR_ratio": float(np.mean(left[sel] / inter[sel]) /
                              max(np.mean(right[sel] / inter[sel]), 1e-9)),
            "dim0_mean": float(dim0[sel].mean()),
            "yaw_mean": float(yaw[sel].mean()),
            "pitch_mean": float(pitch[sel].mean()),
            "roll_mean_deg": float(roll[sel].mean()),
            "dim0_median": float(np.median(dim0[sel])),
        }

    # ---- binning A: SHIPPED z_g dim0, quintiles ---------------------------------
    qs = np.percentile(dim0, [20, 40, 60, 80])
    edges = np.array([dim0.min(), *qs, dim0.max()])
    dim0_bins = []
    print("\n--- A. binned by SHIPPED z_g dim0 (quintiles) ---")
    print("  bin            dim0_cut      n   inter/med   width/inter   L/inter  R/inter   L/R")
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        sel = np.where((dim0 >= lo) & (dim0 < hi))[0] if i < 4 else \
            np.where((dim0 >= lo) & (dim0 <= hi))[0]
        b = bin_stats(sel)
        if b is None or b["n"] < 10:
            print(f"  q{i+1}  [{lo:+.3f},{hi:+.3f}]   n={0 if b is None else b['n']}  (too few)")
            if b is not None:
                b["lo"], b["hi"] = float(lo), float(hi)
                dim0_bins.append(b)
            continue
        b["lo"], b["hi"] = float(lo), float(hi)
        dim0_bins.append(b)
        print("  q%d  [%+.3f,%+.3f]  %4d   %8.3f   %10.3f   %7.2f  %7.2f  %6.2f"
              % (i + 1, lo, hi, b["n"], b["inter_over_median"], b["width_over_inter"],
                 b["left_over_inter"], b["right_over_inter"], b["LR_ratio"]))

    # ---- binning B: DWPose yaw, the reference's exact fixed bins (control) ------
    ybins = [(-9, -0.25), (-0.25, -0.10), (-0.10, 0.0), (0.0, 0.10), (0.10, 0.25), (0.25, 9)]
    yaw_bins = []
    print("\n--- B. CONTROL: binned by DWPose yaw proxy (reference's exact bins) ---")
    print("  yaw            n   inter/med   width/inter   L/inter  R/inter   L/R   dim0_mean")
    for lo, hi in ybins:
        sel = np.where((yaw >= lo) & (yaw < hi))[0]
        b = bin_stats(sel)
        if b is None or b["n"] < 10:
            continue
        b["lo"], b["hi"] = float(lo), float(hi)
        yaw_bins.append(b)
        print("  [%+0.2f,%+0.2f) %4d   %8.3f   %10.3f   %7.2f  %7.2f  %6.2f   %+.3f"
              % (lo, hi, b["n"], b["inter_over_median"], b["width_over_inter"],
                 b["left_over_inter"], b["right_over_inter"], b["LR_ratio"], b["dim0_mean"]))

    # ---- correlations: is dim0 yaw-only? ---------------------------------------
    def corr(a, b_):
        return float(np.corrcoef(a, b_)[0, 1])

    corrs = {"yaw": corr(dim0, yaw), "pitch": corr(dim0, pitch), "roll_deg": corr(dim0, roll)}
    slope, intercept = np.polyfit(dim0, yaw, 1)
    print("\n--- C. what is dim0? Pearson r against the DWPose proxies (n=%d) ---" % n)
    for k, v in corrs.items():
        print(f"  r(dim0, {k:9s}) = {v:+.4f}")
    print(f"  yaw = {intercept:+.4f} {slope:+.4f}*dim0   (r = {corrs['yaw']:+.4f})")
    print(f"  R^2 shared with yaw alone: {corrs['yaw']**2:.3f}  "
          f"(residual yaw variance {1 - corrs['yaw']**2:.3f})")

    # ---- signature magnitudes --------------------------------------------------
    d_inter = np.array([b["inter_over_median"] for b in dim0_bins if b["n"] >= 10])
    d_lr = np.array([b["LR_ratio"] for b in dim0_bins if b["n"] >= 10])
    d_width = np.array([b["width_over_inter"] for b in dim0_bins if b["n"] >= 10])
    y_inter = np.array([b["inter_over_median"] for b in yaw_bins])
    y_lr = np.array([b["LR_ratio"] for b in yaw_bins])
    summary = {
        "dim0_inter_over_median_range": [float(d_inter.min()), float(d_inter.max())],
        "dim0_inter_over_median_extreme_span_pct": float(100 * (d_inter.max() - d_inter.min())),
        "dim0_inter_vs_centre_dev_pct": [float(100 * (v - d_inter[len(d_inter) // 2]))
                                         for v in d_inter],
        "dim0_LR_ratio_range": [float(d_lr.min()), float(d_lr.max())],
        "dim0_LR_swing_factor": float(d_lr.max() / max(d_lr.min(), 1e-9)),
        "dim0_width_over_inter_range": [float(d_width.min()), float(d_width.max())],
        "yaw_inter_over_median_range": [float(y_inter.min()), float(y_inter.max())],
        "yaw_inter_over_median_span_pct": float(100 * (y_inter.max() - y_inter.min())),
        "yaw_LR_ratio_range": [float(y_lr.min()), float(y_lr.max())],
        "yaw_LR_swing_factor": float(y_lr.max() / max(y_lr.min(), 1e-9)),
    }
    print("\n--- D. signature magnitudes ---")
    print("  dim0 bins : inter/med %.3f..%.3f (span %.1f%%)   L/R %.2f..%.2f (%.1fx swing)"
          % (d_inter.min(), d_inter.max(), summary["dim0_inter_over_median_extreme_span_pct"],
             d_lr.min(), d_lr.max(), summary["dim0_LR_swing_factor"]))
    print("  yaw  bins : inter/med %.3f..%.3f (span %.1f%%)   L/R %.2f..%.2f (%.1fx swing)"
          % (y_inter.min(), y_inter.max(), summary["yaw_inter_over_median_span_pct"],
             y_lr.min(), y_lr.max(), summary["yaw_LR_swing_factor"]))

    # ---- E. SHAPE test: symmetric dip (rotation) vs monotone rise (shear) -------
    # The span alone cannot decide. What decides is the SHAPE:
    #   ROTATION  inter/median dips at BOTH |dim0| extremes (a cos-like curve in
    #             true yaw) -- the two ends fall on the SAME side of the centre bin.
    #   SHEAR     inter/median rises monotonically with dim0 -- the two ends fall on
    #             OPPOSITE sides, and the rise across the range matches the render's
    #             7-12% over the +/-3 sweep.
    ratio = inter / inter_med

    def ols(Xcols, yv, names_):
        X = np.column_stack([np.ones(n)] + list(Xcols))
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        dof = n - X.shape[1]
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        out = {"r2": float(1 - (resid @ resid) / ss_tot)}
        for i, nm in enumerate(["const"] + names_):
            out[nm] = {"coef": float(beta[i]), "se": float(se[i]),
                       "t": float(beta[i] / se[i]) if se[i] > 0 else float("nan")}
        return out

    ctr = float(np.median(dim0))
    m_lin = ols([dim0], ratio, ["dim0"])
    m_quad = ols([dim0, (dim0 - ctr) ** 2], ratio, ["dim0", "dim0_sq"])
    rot_pred = np.sqrt(np.clip(1.0 - yaw ** 2, 1e-6, 1.0))          # cos(arcsin(yaw)) as a ratio
    rot_pred = rot_pred / float(np.median(rot_pred))
    m_rot = ols([rot_pred], ratio, ["cos_arcsin_yaw"])
    m_ay = ols([np.abs(yaw)], ratio, ["abs_yaw"])

    shear_pct_per_unit = [100 * 7.0 / 6.0, 100 * 12.0 / 6.0]         # 7-12% over a +/-3 sweep
    lin_pct_per_unit = 100 * m_lin["dim0"]["coef"]
    span_lo = float(dim0_bins[0]["dim0_median"] if dim0_bins else ctr)
    span_hi = float(dim0_bins[-1]["dim0_median"] if dim0_bins else ctr)
    lin_pct_across_range = lin_pct_per_unit * (span_hi - span_lo)
    dip = float((d_inter[0] + d_inter[-1]) / 2.0 - d_inter[len(d_inter) // 2])
    asym = float(d_inter[-1] - d_inter[0])

    print("\n--- E. SHAPE test: symmetric dip (rotation) vs monotone rise (shear) ---")
    print("  per-sample OLS on inter/median (n=%d, ratio scale):" % n)
    print("    inter_ratio = %+.4f %+.4f*dim0        r2=%.4f  t(dim0)=%+.2f"
          % (m_lin["const"]["coef"], m_lin["dim0"]["coef"], m_lin["r2"], m_lin["dim0"]["t"]))
    print("    + (dim0-med)^2 : coef = %+.5f (t=%+.2f)   r2=%.4f   "
          "[rotation wants this NEGATIVE = dip]"
          % (m_quad["dim0_sq"]["coef"], m_quad["dim0_sq"]["t"], m_quad["r2"]))
    print("    vs the yaw-derived rotation predictor cos(arcsin(yaw)): coef=%+.4f (t=%+.2f) r2=%.4f"
          % (m_rot["cos_arcsin_yaw"]["coef"], m_rot["cos_arcsin_yaw"]["t"], m_rot["r2"]))
    print("  linear dim0 slope = %+.3f %%/unit dim0  -> %+.2f%% across the observed bin range "
          "[%+.2f,%+.2f]" % (lin_pct_per_unit, lin_pct_across_range, span_lo, span_hi))
    print("  the render's shear would put this slope at %+.2f..%+.2f %%/unit dim0"
          % (shear_pct_per_unit[0], shear_pct_per_unit[1]))
    print("  symmetry: dip = mean(ends)-centre = %+.4f ; asymmetry = q5-q1 = %+.4f  "
          "-> %s" % (dip, asym, "SYMMETRIC dip (rotation-like)" if abs(dip) > abs(asym)
                     else "ASYMMETRIC / monotone (shear-like)"))

    # per-bin rotation prediction from the SAME images' true yaw
    print("\n  per-dim0-bin check that true yaw alone explains the inter/median variation:")
    print("    bin        n   observed  pred_by_yaw  resid   |  yaw_mean")
    obs_v, pred_v = [], []
    for b in dim0_bins:
        if b["n"] < 10:
            continue
        sel = np.where((dim0 >= b["lo"]) & (dim0 <= b["hi"]))[0]
        p = float(np.mean(np.sqrt(np.clip(1.0 - yaw[sel] ** 2, 1e-6, 1.0))) /
                  np.median(rot_pred))
        b["pred_by_yaw_inter_over_median"] = p
        b["resid_vs_yaw_pred"] = float(b["inter_over_median"] - p)
        obs_v.append(b["inter_over_median"]); pred_v.append(p)
        print("    [%+.2f,%+.2f] %4d   %.4f     %.4f   %+.4f  |  %+.3f"
              % (b["lo"], b["hi"], b["n"], b["inter_over_median"], p,
                 b["resid_vs_yaw_pred"], b["yaw_mean"]))
    obs_v = np.array(obs_v); pred_v = np.array(pred_v)
    rms_rot = float(np.sqrt(np.mean((obs_v - pred_v) ** 2)))
    fit_shear = 1.0 + (np.mean(shear_pct_per_unit) / 100.0) * \
        np.array([b["dim0_median"] - ctr for b in dim0_bins if b["n"] >= 10])
    rms_shear = float(np.sqrt(np.mean((obs_v - fit_shear) ** 2)))
    rms_flat = float(np.sqrt(np.mean((obs_v - 1.0) ** 2)))
    print("    RMS(observed - yaw_rotation_prediction) = %.4f" % rms_rot)
    print("    RMS(observed - render_shear_prediction) = %.4f  (7-12%% over +/-3 dim0)" % rms_shear)
    print("    RMS(observed - flat 1.000)              = %.4f" % rms_flat)

    # ---- explicit verdict -------------------------------------------------------
    widen_pct = summary["dim0_inter_over_median_extreme_span_pct"]
    lr_swing = summary["dim0_LR_swing_factor"]
    shear_slope_lo, shear_slope_hi = shear_pct_per_unit
    if lin_pct_per_unit >= shear_slope_lo and asym > 0 and dip < 0.002:
        verdict = "SHEAR"
        why = ("inter/median rises monotonically with dim0 at %.2f %%/unit, inside the "
               "render-shear band %.2f..%.2f %%/unit, and the two extremes sit on "
               "opposite sides of the centre bin." % (lin_pct_per_unit, shear_slope_lo, shear_slope_hi))
    elif dip < -0.005 and lin_pct_per_unit < 0.5 * shear_slope_lo and lr_swing >= 2.0:
        verdict = "ROTATION"
        why = ("inter/median dips at BOTH |dim0| extremes (dip %+.4f, ends %+.4f/%+.4f around "
               "centre %+.4f), the linear dim0 slope is %+.2f %%/unit -- below half the "
               "render-shear band %.2f..%.2f -- and L/R swings %.1fx."
               % (dip, d_inter[0], d_inter[-1], d_inter[len(d_inter) // 2],
                  lin_pct_per_unit, shear_slope_lo, shear_slope_hi, lr_swing))
    else:
        verdict = "AMBIGUOUS"
        why = ("the shape matches neither signature cleanly (dip %+.4f, asymmetry %+.4f, "
               "linear slope %+.2f %%/unit, L/R swing %.1fx)." % (dip, asym, lin_pct_per_unit, lr_swing))
    print("\n--- VERDICT ---------------------------------------------------------")
    print("  dim0 bins span %.1f%% of inter/median end-to-end; the yaw-binned control spans "
          "%.1f%%." % (widen_pct, summary["yaw_inter_over_median_span_pct"]))
    print("  %s" % why)
    print("  => Z_G DIM0 IN THE REAL DATA: %s" % verdict)
    if verdict == "ROTATION":
        print("     The data's inter/median along dim0 is a cos-like dip about the centre, not a")
        print("     widening: the axis turns the head in 3D. The renders' 7-12% WIDENING is therefore")
        print("     NOT inherited from the conditioning data -- it is a model/projection artifact.")
    elif verdict == "SHEAR":
        print("     The data's inter/median widens along dim0 the way the renders do: the axis")
        print("     carries a 2D in-plane stretch, and the renders pass it through.")

    # ---- outputs ----------------------------------------------------------------
    results = {
        "script": str(Path(__file__).resolve()),
        "stratum": str(STRATUM),
        "seed": args.seed,
        "requested_n": args.n,
        "n_dirs_listed": n_listed,
        "n_sampled_dirs": len(pick),
        "skipped_missing_sidecar": miss,
        "n_usable": int(n),
        "excluded": excl,
        "pose_xy_range": [float(XY_MIN), float(XY_MAX)],
        "pose_convention": "normalized [-1,1], converted to px as (v+1)*512 (reference convention)",
        "acceptance": "face_conf_mean >= 0.3 and interocular >= 20 px (reference convention)",
        "pixel_npy_spotcheck": pchk,
        "median_interocular_px": inter_med,
        "median_face_width_px": float(np.median(width)),
        "dim0_quintile_edges": [float(e) for e in edges],
        "dim0_bins": dim0_bins,
        "yaw_bins_control": yaw_bins,
        "yaw_bin_edges_control": [[float(a), float(b)] for a, b in ybins],
        "correlations_dim0": corrs,
        "dim0_yaw_regression": {"intercept": float(intercept), "slope": float(slope),
                                "r": corrs["yaw"], "r2": float(corrs["yaw"] ** 2)},
        "shape_test": {
            "inter_ratio_ols_linear_dim0": m_lin,
            "inter_ratio_ols_dim0_and_dim0sq": m_quad,
            "inter_ratio_ols_cos_arcsin_yaw": m_rot,
            "inter_ratio_ols_abs_yaw": m_ay,
            "linear_slope_pct_per_unit_dim0": float(lin_pct_per_unit),
            "linear_slope_pct_across_bin_range": float(lin_pct_across_range),
            "bin_range_dim0": [span_lo, span_hi],
            "render_shear_prediction_pct_per_unit_dim0": [float(x) for x in shear_pct_per_unit],
            "symmetric_dip_mean_ends_minus_centre": dip,
            "asymmetry_q5_minus_q1": asym,
            "rms_obs_vs_yaw_rotation_prediction": rms_rot,
            "rms_obs_vs_render_shear_prediction": rms_shear,
            "rms_obs_vs_flat": rms_flat,
        },
        "signature_summary": summary,
        "verdict": verdict,
        "verdict_reason": why,
        "verdict_rule": ("SHEAR if the linear dim0 slope of inter/median reaches the render's "
                         "7-12%-over-+/-3 band (>=1.17 %%/unit) with the ends on opposite sides; "
                         "ROTATION if inter/median dips at BOTH |dim0| extremes (dip < -0.005) with "
                         "linear slope < 0.58 %%/unit and L/R swing >= 2x; else AMBIGUOUS"),
    }
    rj = out / "results.json"
    rj.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {rj}")

    # ---- figure -----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    ax = axes[0, 0]
    xs = np.arange(len(d_inter))
    ax.plot(xs, d_inter, "o-", color="#c0392b", label="observed, binned by z_g dim0")
    ax.plot(xs, [b["pred_by_yaw_inter_over_median"] for b in dim0_bins if b["n"] >= 10],
            "^--", color="#2471a3", label="predicted from the SAME faces' true yaw\n(cos(arcsin yaw))")
    ax.plot(xs, fit_shear, ":", color="#27ae60",
            label="render-shear model (7-12% over +/-3 dim0)")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(["[%+.2f,%+.2f]" % (b["lo"], b["hi"]) for b in dim0_bins if b["n"] >= 10],
                       fontsize=6, rotation=20)
    ax.set_xlabel("z_g dim0 quintile"); ax.set_ylabel("mean inter / median inter")
    ax.set_title("(a) inter/median vs SHIPPED dim0\ndip at both ends => rotation; monotone rise => 2D shear",
                 fontsize=9)
    ax.legend(fontsize=6, loc="lower center")

    ax = axes[0, 1]
    ax.plot(xs, d_lr, "s-", color="#c0392b", label="binned by z_g dim0")
    ax.set_xticks(xs)
    ax.set_xticklabels(["[%+.2f,%+.2f]" % (b["lo"], b["hi"]) for b in dim0_bins if b["n"] >= 10],
                       fontsize=6, rotation=20)
    ax.set_xlabel("z_g dim0 quintile"); ax.set_ylabel("L/R jaw distance from nose tip")
    ax.set_title("(b) L/R asymmetry vs SHIPPED dim0\nlarge swing => the axis turns the head", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ys = np.arange(len(y_inter))
    ax.plot(ys, y_inter, "o-", color="#2471a3", label="binned by DWPose yaw (control)")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(ys)
    ax.set_xticklabels(["[%+.2f,%+.2f)" % (b["lo"], b["hi"]) for b in yaw_bins], fontsize=6, rotation=20)
    ax.set_xlabel("DWPose yaw proxy"); ax.set_ylabel("mean inter / median inter")
    ax.set_title("(c) CONTROL: inter/median vs yaw\nmust reproduce the published flat curve", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(ys, y_lr, "s-", color="#2471a3", label="binned by DWPose yaw (control)")
    ax.set_xticks(ys)
    ax.set_xticklabels(["[%+.2f,%+.2f)" % (b["lo"], b["hi"]) for b in yaw_bins], fontsize=6, rotation=20)
    ax.set_xlabel("DWPose yaw proxy"); ax.set_ylabel("L/R jaw distance from nose tip")
    ax.set_title("(d) CONTROL: L/R vs yaw (the 22x swing)", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    ax.scatter(dim0, yaw, s=4, alpha=0.35, color="#7f8c8d")
    xx = np.linspace(dim0.min(), dim0.max(), 50)
    ax.plot(xx, intercept + slope * xx, color="#c0392b", lw=1.2,
            label="yaw = %+.4f %+.4f*dim0\nr = %+.3f" % (intercept, slope, corrs["yaw"]))
    ax.set_xlabel("z_g dim0"); ax.set_ylabel("DWPose yaw proxy")
    ax.set_title("(e) dim0 really is yaw\n(but a yaw correlation cannot pick the mechanism)", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1, 2]
    ax.axis("off")
    txt = ("VERDICT: %s\n\n"
           "z_g dim0 bins (n=%d)\n"
           "  inter/median  %.3f %.3f %.3f %.3f %.3f\n"
           "  L/R jaw       %.2f %.2f %.2f %.2f %.2f\n"
           "  dip(ends-centre) %+.4f   asym(q5-q1) %+.4f\n\n"
           "per-sample OLS: inter_ratio = %+.4f %+.4f*dim0\n"
           "  linear slope %+.2f %%/unit dim0\n"
           "  render-shear predicts 1.17..2.00 %%/unit\n\n"
           "DWPose yaw bins (control, n=%d)\n"
           "  inter/median  %.3f .. %.3f\n"
           "  L/R jaw       %.2f .. %.2f  (%.1fx swing)\n\n"
           "RMS vs yaw-rotation model %.4f\n"
           "RMS vs render-shear model %.4f\n\n"
           "dim0 is yaw-only: r(yaw)=%+.3f r(pitch)=%+.3f\nr(roll)=%+.3f"
           % (verdict, n, *d_inter, *d_lr, dip, asym,
              m_lin["const"]["coef"], m_lin["dim0"]["coef"], lin_pct_per_unit,
              n, y_inter.min(), y_inter.max(), y_lr.min(), y_lr.max(),
              summary["yaw_LR_swing_factor"], rms_rot, rms_shear,
              corrs["yaw"], corrs["pitch"], corrs["roll_deg"]))
    ax.text(0.0, 0.98, txt, va="top", ha="left", fontsize=8, family="monospace")

    fig.suptitle("z_g dim0 projection signature on %d real FFHQ faces — rotation or 2D shear?" % n,
                 fontsize=12)
    fig.tight_layout()
    png = out / "zg_dim0_projection_signature.png"
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()