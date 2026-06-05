#!/usr/bin/env python3
"""Issue #5a quantification spike (READ-ONLY, no production changes).

Loads one real training batch via the production dataloader and reports the
distribution stats of the flow-matching endpoints to confirm/quantify the
[0,1] vs [-1,1] pixel normalization bias.

Run: ./.venv/bin/python paper/spike_pixel_norm.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from production.config_loader import load_config
from production.data import get_production_dataloader


def stats(name, t):
    print(f"  {name:16} shape={tuple(t.shape)} min={t.min():.4f} max={t.max():.4f} "
          f"mean={t.mean():.4f} std={t.std():.4f}")


def main():
    CFG = "production/config.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(CFG)

    # Force single-process loading for this one-batch probe.
    try:
        cfg.data.num_workers = 0
    except Exception:
        pass

    dl = get_production_dataloader(cfg, device)
    batch = next(iter(dl))
    x0 = batch["image_data"].to(device).float()

    print("=== Issue #5a: flow-matching endpoint statistics ===")
    print("\n[x0] data endpoint (dataloader output / used at train.py:478):")
    stats("x0", x0)

    torch.manual_seed(0)
    z1 = torch.randn_like(x0)
    print("\n[z1] noise endpoint (train.py:243, randn_like):")
    stats("z1", z1)

    print("\n[zt] interpolation zt=(1-t)*x0 + t*z1 across t (train.py:249):")
    for t in [0.1, 0.25, 0.5, 0.75, 0.9]:
        zt = (1 - t) * x0 + t * z1
        stats(f"zt(t={t})", zt)

    v_target = z1 - x0
    print("\n[v_target] = z1 - x0 (train.py:263):")
    stats("v_target", v_target)

    x0c = x0 * 2 - 1
    print("\n=== If data were centered to [-1,1] (proposed fix) ===")
    stats("x0_centered", x0c)
    print("  v_target_centered = z1 - x0_centered:")
    stats("v_target_c", z1 - x0c)

    print("\n=== Bias summary ===")
    print(f"  Current data mean (want ~0 for clean flow matching): {x0.mean():.4f}")
    print(f"  Current data std  (want ~1 to match N(0,1) noise):   {x0.std():.4f}")
    print(f"  -> data endpoint offset ~{x0.mean():.3f}; "
          f"~{1/max(float(x0.std()),1e-6):.2f}x narrower than the noise.")


if __name__ == "__main__":
    main()
