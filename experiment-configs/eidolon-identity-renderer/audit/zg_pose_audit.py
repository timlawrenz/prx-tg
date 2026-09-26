#!/usr/bin/env python3
"""Audit: does the z_g we actually condition on encode yaw? Plots what the pose data looks like.

Decisive test = per-dim Pearson correlation between z_g and a landmark-derived yaw
proxy on real training images. Also decodes z_g -> 68 landmarks through the frozen
production encoder and draws mean +/- 3sigma along a dim.
"""
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = Path("/home/tim/source/activity/eidolon/experiments/geometry_pca/output")
STRAT = Path("/mnt/nas-ai-models/training-data/ffhq/stratum")
OUT = Path("/home/tim/.hermes/profiles/prx-tg/cache/scratch/zg_audit")
OUT.mkdir(parents=True, exist_ok=True)

enc = np.load(ART / "encoder_production.npz")
C, pm = enc["components"], enc["pca_mean"]            # (50,136), (136,)
wmu, wsig = enc["whiten_mu"], enc["whiten_sigma"]     # (50,), (50,)
gpa = enc["gpa_mean"]                                 # (68,2)
print("encoder_production.npz: C%s pm%s wmu%s wsig%s gpa%s" % (C.shape, pm.shape, wmu.shape, wsig.shape, gpa.shape))
print("retained variance: %.6f" % float(enc["explained_variance_ratio"].sum()))

# ---- sample real training images that have both z_g and pose sidecars -------------
dirs = sorted(d for d in STRAT.iterdir() if d.is_dir())
random.seed(0)
random.shuffle(dirs)
pick = []
for d in dirs:
    if (d / "z_g.npy").exists() and (d / "pose.npy").exists():
        pick.append(d)
    if len(pick) >= 300:
        break
print("sampled %d dirs with both z_g.npy and pose.npy" % len(pick))

ZG, YAW, SCHEMA = [], [], None
for d in pick:
    zg = np.load(d / "z_g.npy").astype(np.float64).reshape(-1)
    p = np.load(d / "pose.npy").astype(np.float64)     # (133,3) COCO-WholeBody
    face = p[23:91, :2]                                # 68 iBUG landmarks
    if SCHEMA is None:
        SCHEMA = (p.shape, p[:, :2].min(), p[:, :2].max(), p[23:91, 2].mean() if p.shape[1] > 2 else np.nan)
    reye, leye = face[36:42].mean(0), face[42:48].mean(0)
    nose = face[30]                                    # nose tip
    inter = np.linalg.norm(leye - reye) + 1e-9
    yaw = (nose[0] - 0.5 * (leye[0] + reye[0])) / inter   # signed yaw proxy, monotone in yaw
    ZG.append(zg)
    YAW.append(yaw)

ZG = np.array(ZG)
YAW = np.array(YAW)
print("pose.npy schema: shape=%s xy_range=(%.1f, %.1f) face_conf_mean=%.3f" % SCHEMA)
print("z_g: n=%d  dims=%d  per-dim std mean=%.3f  dim0 std=%.3f" % (ZG.shape[0], ZG.shape[1], ZG.std(0).mean(), ZG.std(0)[0]))
print("yaw proxy: mean=%.4f std=%.4f  min=%.3f max=%.3f" % (YAW.mean(), YAW.std(), YAW.min(), YAW.max()))

# ---- the decisive number: per-dim correlation with yaw ---------------------------
corr = np.array([np.corrcoef(ZG[:, i], YAW)[0, 1] for i in range(ZG.shape[1])])
order = np.argsort(-np.abs(corr))
print("\n=== |corr(z_g dim, yaw proxy)| top 10 ===")
for i in order[:10]:
    print("   dim %-3d |corr| = %.3f  (r = %+.3f)" % (i, abs(corr[i]), corr[i]))
rank0 = int(np.where(order == 0)[0][0]) + 1
print("   >>> dim0: |corr| = %.3f  (r = %+.3f)  -- rank %d of 50" % (abs(corr[0]), corr[0], rank0))
print("   max |corr| over all dims = %.3f ; median |corr| = %.3f" % (np.abs(corr).max(), np.median(np.abs(corr))))

# ---- decode z_g -> 68 landmarks through the frozen encoder -----------------------
mean_land = (pm + (wmu @ C)).reshape(68, 2)            # landmarks at z_g = 0
def shape_at(vec):
    return (pm + ((vec * wsig + wmu) @ C)).reshape(68, 2)
# variation per dim (whitened): moving dim d by v shifts landmarks by v*wsig[d]*C[d]
def dim_delta(d):
    return (wsig[d] * C[d]).reshape(68, 2)

def draw(ax, dims, tag):
    ax.scatter(mean_land[:, 0], mean_land[:, 1], s=6, c="k", zorder=5)
    cols = {-3.0: "#c0392b", 0.0: "#2c3e50", 3.0: "#2471a3"}
    for d in dims:
        delta = dim_delta(d)
        for v, c in cols.items():
            lm = mean_land + v * delta
            ax.plot(lm[:, 0], lm[:, 1], ".", ms=4, color=c, alpha=0.85)
    for v, c in cols.items():
        lm = mean_land + v * dim_delta(dims[0])
        ax.plot(lm[:, 0], lm[:, 1], "-", lw=0.5, color=c, alpha=0.6, label="dim%d=%+.1f" % (dims[0], v))
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_title(tag, fontsize=9)
    ax.legend(fontsize=6, loc="upper right"); ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
draw(axes[0], [0], "dim0 (the sweep's 'yaw' axis)\nz_g = -3 / 0 / +3")
draw(axes[1], [int(order[0])], "dim %d  (strongest yaw corr, |r|=%.2f)" % (order[0], abs(corr[order[0]])))
axes[2].hist(ZG[:, 0], bins=40, color="#8e44ad", alpha=0.8)
for v in (-3.0, -1.5, 0.0, 1.5, 3.0):
    axes[2].axvline(v, color="#c0392b", ls="--", lw=1)
    axes[2].annotate("%+.1f" % v, (v, 0), fontsize=6, color="#c0392b", rotation=90, va="bottom")
axes[2].set_title("training z_g dim0 distribution\n(dashed = the 5 sweep values)", fontsize=9)
axes[2].set_ylabel("count"); axes[2].set_xlabel("z_g dim0")
fig.suptitle("What the pose conditioning actually contains — FFHQ training z_g (n=%d)" % ZG.shape[0], fontsize=11)
fig.tight_layout()
p1 = OUT / "zg_pose_audit.png"
fig.savefig(p1, dpi=130)
print("\nwrote %s" % p1)

# ---- figure 2: all 50 dims' yaw correlation -------------------------------------
fig2, ax = plt.subplots(figsize=(11, 3.2))
ax.bar(range(50), corr, color=["#c0392b" if i == 0 else "#7f8c8d" for i in range(50)])
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("z_g dimension"); ax.set_ylabel("r with yaw proxy")
ax.set_title("Per-dimension correlation of z_g with measured yaw (dim0 in red — the axis the sweep varies)", fontsize=10)
fig2.tight_layout()
p2 = OUT / "zg_yaw_corr_perdim.png"
fig2.savefig(p2, dpi=130)
print("wrote %s" % p2)

# ---- the TEST side: what the sweep vectors look like in landmark space -----------
base = ZG[np.argmin(np.abs(YAW))]      # a near-frontal real sample
fig3, ax = plt.subplots(figsize=(5.2, 5.2))
ax.scatter(mean_land[:, 0], mean_land[:, 1], s=6, c="k", zorder=5)
for v, c in ((-3.0, "#c0392b"), (-1.5, "#e67e22"), (0.0, "#2c3e50"), (1.5, "#16a085"), (3.0, "#2471a3")):
    bv = base.copy(); bv[0] = v                     # exactly what the sweep feeds: base z_g with dim0 replaced
    lm = shape_at(bv)
    ax.plot(lm[:, 0], lm[:, 1], "o-", ms=4, lw=0.7, color=c, label="dim0 = %+.1f" % v)
ax.invert_yaxis(); ax.set_aspect("equal")
ax.set_title("TEST conditioning: landmarks encoded by the 5 sweep vectors\n(decoded through the frozen production encoder)", fontsize=9)
ax.legend(fontsize=7); ax.set_xticks([]); ax.set_yticks([])
fig3.tight_layout()
p3 = OUT / "zg_sweep_landmarks.png"
fig3.savefig(p3, dpi=130)
print("wrote %s" % p3)
