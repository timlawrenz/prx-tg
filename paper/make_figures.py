#!/usr/bin/env python3
"""Generate publication figures for the prx-tg ablation paper from verified metrics + training logs."""
import json, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (paper/ -> ..)
os.chdir(ROOT)
os.makedirs("paper/figures", exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True,
                     "font.family": "DejaVu Sans"})

m = json.load(open("paper/data/verified_metrics.json"))
L = m["lpips_step5000"]; T = m["throughput"]
arms = ["A", "B", "C", "D", "G", "H", "I"]
labels = {"A": "A\nBaseline", "B": "B\nTREAD\n+AdamW", "C": "C\nTREAD\n+Muon",
          "D": "D\nFull Stack", "G": "G\nAsymFlow", "H": "H\nShared\nadaLN", "I": "I\nFP8"}

TEAL = "#2a6f7f"; OCHRE = "#c98a3b"; GREEN = "#5a8f4f"; RED = "#b0413e"

# ---- Figure 1: LPIPS grouped bars ----
recon = [L[a]["recon"] for a in arms]
texto = [L[a]["text_only"] for a in arms]
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(arms)); w = 0.38
b1 = ax.bar(x - w / 2, recon, w, label="Reconstruction LPIPS", color=TEAL)
b2 = ax.bar(x + w / 2, texto, w, label="Text-only LPIPS", color=OCHRE)
ax.set_xticks(x); ax.set_xticklabels([labels[a] for a in arms], fontsize=8.5)
ax.set_ylabel("LPIPS (lower is better)"); ax.set_ylim(0.85, 1.05)
ax.axhline(L["D"]["recon"], ls="--", color=TEAL, alpha=0.5, lw=1)
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Final-checkpoint perceptual quality (step 5000)")
for b in list(b1) + list(b2):
    ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=6.5, rotation=90, xytext=(0, 2),
                textcoords="offset points")
plt.tight_layout()
plt.savefig("paper/figures/fig_lpips_bars.pdf"); plt.savefig("paper/figures/fig_lpips_bars.png")
plt.close()
print("fig_lpips_bars done")

# ---- Figure 2: throughput vs VRAM ----
eff = ["A", "B", "C", "D", "G", "I"]
its = [T[a]["median_iter_per_sec"] for a in eff]
vram = [T[a]["peak_vram_gb"] for a in eff]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax2 = ax.twinx()
xx = np.arange(len(eff))
bar = ax.bar(xx, its, 0.5, color=GREEN, label="Throughput (it/s)")
ln, = ax2.plot(xx, vram, "o-", color=RED, label="Peak VRAM (GB)", lw=2)
ax.set_xticks(xx); ax.set_xticklabels(eff)
ax.set_ylabel("Optimizer-step throughput (it/s)"); ax2.set_ylabel("Peak VRAM (GB)")
ax.set_title("Throughput and memory across the optimization stack")
for i, v in enumerate(its):
    ax.annotate(f"{v:.4f}", (i, v), ha="center", va="bottom", fontsize=8, xytext=(0, 2),
                textcoords="offset points")
for i, v in enumerate(vram):
    ax2.annotate(f"{v:.1f}", (i, v), ha="center", va="bottom", fontsize=8, color=RED,
                 xytext=(0, 4), textcoords="offset points")
ax.legend([bar, ln], [bar.get_label(), ln.get_label()], loc="upper left", fontsize=9)
ax.set_ylim(0, 0.06); ax2.set_ylim(0, 12); ax2.grid(False)
plt.tight_layout()
plt.savefig("paper/figures/fig_throughput.pdf"); plt.savefig("paper/figures/fig_throughput.png")
plt.close()
print("fig_throughput done")

# ---- Figure 3: training loss curves (A-D) ----
runs = {
    "A Baseline": ("experiments/vast_final_2026-05-12_0902/ablation_baseline", "#888888"),
    "B TREAD+AdamW": ("experiments/vast_final_2026-05-12_0902/ablation_tread_adamw", OCHRE),
    "C TREAD+Muon": ("experiments/vast_final_2026-05-12_0902/ablation_tread_muon", GREEN),
    "D Full Stack": ("experiments/vast_final_2026-05-12_0902/ablation_full_stack", TEAL),
}


def load_loss(rd):
    cands = glob.glob(os.path.join(rd, "**", "training_log.jsonl"), recursive=True)
    if not cands:
        return None, None
    f = max(cands, key=os.path.getsize)
    steps, losses = [], []
    for line in open(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "loss" in d and "step" in d:
            steps.append(d["step"]); losses.append(d["loss"])
    return np.array(steps), np.array(losses)


def smooth(y, k=25):
    if y is None or len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


fig, ax = plt.subplots(figsize=(8, 4.5))
for name, (rd, c) in runs.items():
    s, l = load_loss(rd)
    if s is None:
        print("no loss for", name); continue
    ls = smooth(l); ss = s[:len(ls)]
    ax.plot(ss, ls, color=c, lw=1.5, label=name)
ax.set_xlabel("Optimizer step"); ax.set_ylabel("Training loss (smoothed, k=25)")
ax.set_yscale("log"); ax.set_title("Training loss: A-D ablation arms")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("paper/figures/fig_loss_curves.pdf"); plt.savefig("paper/figures/fig_loss_curves.png")
plt.close()
print("fig_loss_curves done")
