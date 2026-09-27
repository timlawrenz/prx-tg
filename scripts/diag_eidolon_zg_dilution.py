#!/usr/bin/env python3
"""DIAGNOSTIC: is the eval-time dual-CFG ladder DILUTING z_g (geometry)?

Context
-------
`eidolon-identity-renderer` (Arm EIR) is conditioned on identity (64-d, adaLN)
plus geometry z_g (50-d, cross-attention). z_g dim0 was verified to be a
near-pure head-yaw axis (|r| = 0.94 against yaw measured from real DWPose
landmarks). Yet the model's renders do NOT turn the head when z_g dim0 is
swept, and its sensitivity to z_g appears to be shrinking as training proceeds.

Hypothesis under test
---------------------
The eval-time dual-CFG ladder dilutes geometry. sample.py builds

    v = v_uncond
        + identity_scale * (v_identity - v_uncond)
        + geometry_scale * (v_geometry - v_uncond)

with the configuration default identity_scale = 3.0 vs geometry_scale = 2.0
(experiment-configs/eidolon-identity-renderer/config.yaml -> sampling). With the
identity stream dominated, the geometry branch's contribution is attenuated and
the yaw sweep may not move the rendered head at all.

Test
----
Sweep z_g dim0 over [-3.0, -1.5, 0.0, 1.5, 3.0] (the default; `--sweep-values`
overrides the list, e.g. `--sweep-values -1.5 0.0 1.5` runs a +/-1.5 sigma
in-distribution row without editing this file) for sample indices [10, 30, 50]
under FOUR CFG settings, including a GEOMETRY-ONLY ladder (identity_scale = 0.0)
and saturated geometry scales:

    fullcfg_id3_geo2   identity 3.0   geometry 2.0    <- the config default
    geoonly_id0_geo4   identity 0.0   geometry 4.0    <- geometry only
    geoonly_id0_geo10  identity 0.0   geometry 10.0   <- geometry only, saturated
    id1_geo8           identity 1.0   geometry 8.0    <- geometry-dominant

Noise-matched: the SAME fixed seed is reset immediately before EVERY single
generation, so any difference between panels is attributable to conditioning
(z_g value / CFG scales), never to a different initial noise draw.

Measurement
-----------
Every rendered panel is measured with the project's own CPU DWPose ONNX
detector (scripts/dwpose_onnx.py). Yaw proxy = horizontal nose offset from the
eye midpoint, normalised by inter-ocular distance (same proxy used by
/home/tim/.hermes/profiles/prx-tg/cache/scratch/pose_on_sweep.py). The headline
per-config number is the yaw SPREAD across the 5 sweep values (max - min) for
each sample index, plus the count of no-face panels. A config where geometry
actually binds shows a clear monotone progression across the sweep values; the
per-value numbers are printed in sweep order so monotonicity is visible.

Outputs (under --out-dir)
-------------------------
    results.json                       machine-readable
    s{idx:02d}_{cfg}_dim0_{val:+.1f}.png   individual panels
    grid_{cfg}_rows=sample...png       5 panels wide, one row per sample index

Provenance of the copied API calls
----------------------------------
* Model construction + checkpoint load + validation-dataloader/conditioning
  extraction + EulerSampler invocation style:
      scripts/diag_pose_cfg_sweep.py  (source of truth for model/sampler calls)
* Latent-vs-pixel shape selection, VAE decode and the identity/basis config
  guards (this arm is FLUX-AE LATENT space, which diag_pose_cfg_sweep.py never
  handled — it only covers pixel-space arms):
      production/validate.py (ValidationSampler.generate, run_validation,
      create_validation_fn) and production/sample.py (EulerSampler.sample,
      decode_latents, tensor_to_pil)
* DWPose detector setup + yaw proxy:
      /home/tim/.hermes/profiles/prx-tg/cache/scratch/pose_on_sweep.py
* Sweep indices / values / dim, and the "one seed for the whole sweep" rule:
      production/validate.py (EIDOLON_GEOMETRY_SWEEP_*)

This script does NOT run any training, does not touch the checkpoint on disk,
and does not modify anything under production/ or any existing script.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Repo root on sys.path, exactly as scripts/diag_pose_cfg_sweep.py does.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from production.config_loader import (  # noqa: E402
    load_config,
    resolve_latent_space,
    resolve_sampling_gamma,
)
from production.model import NanoDiT  # noqa: E402
from production.sample import (  # noqa: E402
    EulerSampler,
    decode_latents,
    load_vae_decoder,
    tensor_to_pil,
)
from production.data import get_deterministic_validation_dataloader  # noqa: E402
from scripts.dwpose_onnx import DWPoseDetector  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed experiment parameters (mirroring validate.py EIDOLON_GEOMETRY_SWEEP_*)
# ---------------------------------------------------------------------------
CONFIG_PATH = REPO / "experiment-configs" / "eidolon-identity-renderer" / "config.yaml"
DEFAULT_CKPT_DIR = Path(
    "/mnt/nas-ai-models/training-data/prx-tg/eidolon-identity-renderer/"
    "runs/2026-09-25_0734/checkpoints"
)
DEFAULT_OUT_DIR = REPO / "_diag_zg_dilution"

SWEEP_DIM = 0
SWEEP_VALUES = [-3.0, -1.5, 0.0, 1.5, 3.0]       # ~+/-42 deg yaw at 14 deg/sigma
SAMPLE_INDICES = [10, 30, 50]

# (name, identity_scale, geometry_scale) -- passed to sampler.sample() as
# text_scale / dino_scale respectively. For the eidolon adapter the sampler's
# "text" branch IS identity-only and its "dino" branch IS geometry-only
# (production/sample.py, sample.py dual-CFG block; documented in the arm config
# under `sampling:`), so these two numbers are identity and geometry guidance.
CFG_CONFIGS = [
    ("fullcfg_id3_geo2", 3.0, 2.0),
    ("geoonly_id0_geo4", 0.0, 4.0),
    ("geoonly_id0_geo10", 0.0, 10.0),
    ("id1_geo8", 1.0, 8.0),
]

NUM_STEPS = 50
SEED = 1234          # reset before EVERY generation (noise-matched sweep)


# ---------------------------------------------------------------------------
# Checkpoint selection
# ---------------------------------------------------------------------------
def find_default_checkpoint(ckpt_dir: Path) -> Path:
    """Most recently modified checkpoint_step*.pt or checkpoint_interrupt.pt."""
    candidates = sorted(ckpt_dir.glob("checkpoint_step*.pt"))
    interrupt = ckpt_dir / "checkpoint_interrupt.pt"
    if interrupt.exists():
        candidates.append(interrupt)
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint_step*.pt or checkpoint_interrupt.pt under {ckpt_dir}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Model construction -- copied faithfully from scripts/diag_pose_cfg_sweep.py
# ---------------------------------------------------------------------------
def build_model(config, ckpt_path: Path, device: torch.device) -> NanoDiT:
    kw = dict(
        input_size=config.model.input_size,
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        use_gradient_checkpointing=False,
    )
    ad = getattr(config, "adapter", None)
    if ad:
        kw["adapter_kwargs"] = {
            "name": ad.name,
            "identity_dim": getattr(ad, "identity_dim", 64),
            "z_g_dim": getattr(ad, "z_g_dim", 50),
            "geometry_token_basis": getattr(ad, "geometry_token_basis", False),
        }
    model = NanoDiT(**kw)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Training weights, NOT ema (matches diag_pose_cfg_sweep.py / visual_debug).
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Validation samples -- copied faithfully from scripts/diag_pose_cfg_sweep.py
# ---------------------------------------------------------------------------
def load_samples(config, device, latent_space: bool):
    """Grab SAMPLE_INDICES from the deterministic validation dataloader.

    diag_pose_cfg_sweep.py passes target_latent_size=config.model.input_size and
    does not forward the identity-basis guard. For a LATENT-space arm,
    production/validate.py uses input_size * 8 (a 1024px image -> 128x128 FLUX
    latents), so this script follows validate.py for that one argument only. The
    value is ignored for conditioning extraction (we only use identity_emb /
    geometry_emb), and production/data_stratum.py documents that
    target_latent_size is IGNORED in latent mode anyway.
    """
    target_latent_size = (
        config.model.input_size * 8 if latent_space else config.model.input_size
    )
    loader = get_deterministic_validation_dataloader(
        shard_dir=None,
        batch_size=1,
        target_latent_size=target_latent_size,
        source="stratum",
        stratum_dir=config.data.stratum_dir,
        adapter_name="eidolon",
        # Forward the guard values the arm ships (production/train_production.py
        # L503-507 does this for the validation loader) so the basis guard does
        # not reject the root and silently yield zero samples.
        expected_basis_fingerprint=getattr(config.data, "basis_fingerprint", None),
        allow_unstamped_identity=getattr(config.data, "allow_unstamped_identity", False),
    )

    maxi = max(SAMPLE_INDICES)
    samples = []
    it = iter(loader)
    cnt = 0
    while cnt <= maxi:
        b = next(it)
        n = (
            b["image_data"].shape[0]
            if "image_data" in b
            else b["identity_emb"].shape[0]
        )
        for i in range(n):
            if cnt <= maxi:
                samples.append(
                    {
                        "identity_emb": b["identity_emb"][i],
                        "geometry_emb": b["geometry_emb"][i],
                        "image_id": b["image_ids"][i],
                    }
                )
                cnt += 1
    if len(samples) <= maxi:
        raise RuntimeError(
            f"Validation loader yielded only {len(samples)} samples; need index {maxi}"
        )
    return samples


# ---------------------------------------------------------------------------
# Grid builder -- copied from scripts/diag_pose_cfg_sweep.py (grid()).
# rows: list of list of (3,H,W) tensors in [-1,1]. One COLUMN per sweep value,
# one ROW per sample index.
# ---------------------------------------------------------------------------
def grid(rows, spacing=8):
    pil_rows = []
    for r in rows:
        imgs = []
        for img in r:
            a = (img.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
            imgs.append(Image.fromarray((a.transpose(1, 2, 0) * 255).astype(np.uint8)))
        pil_rows.append(imgs)
    h = pil_rows[0][0].height
    w = pil_rows[0][0].width
    ncol = max(len(r) for r in pil_rows)
    nrow = len(pil_rows)
    W = ncol * w + (ncol - 1) * spacing
    H = nrow * h + (nrow - 1) * spacing
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    for ri, r in enumerate(pil_rows):
        for ci, im in enumerate(r):
            canvas.paste(im, (ci * (w + spacing), ri * (h + spacing)))
    return canvas


# ---------------------------------------------------------------------------
# Yaw measurement -- detector setup + yaw proxy copied from the scratch
# pose_on_sweep.py script (which already runs correctly, CPU-only).
# ---------------------------------------------------------------------------
def build_detector() -> DWPoseDetector:
    print("loading DWPose (ONNX, CPU)...", flush=True)
    return DWPoseDetector(device="cpu")


def measure_panel_yaw(det: DWPoseDetector, panel_path: Path):
    """Return (yaw, conf, n_kp) or None if no face/keypoints were detected.

    Exactly the proxy from pose_on_sweep.py:
      yaw = (nose_x - (le_x + re_x)/2) / |le - re|
    COCO-133 face points (nose=0, left-eye=1, right-eye=2). A panel with no
    detection, an empty keypoint array, or a degenerate inter-ocular distance
    returns None -- never a substituted guess.
    """
    img = cv2.imread(str(panel_path))
    if img is None:
        return None
    try:
        kp, sc, bb = det(img, single_person=True)
    except Exception as e:  # detector failure must not kill the run
        print(f"    [warn] detector error on {panel_path.name}: "
              f"{type(e).__name__}: {e}")
        return None
    if kp is None or len(kp) == 0:
        return None
    kp = np.asarray(kp[0] if kp.ndim == 3 else kp)
    sc = (
        np.asarray(sc[0] if sc is not None and np.asarray(sc).ndim == 2 else sc)
        if sc is not None
        else None
    )
    n_kp = len(kp)
    conf = float(np.mean(sc)) if sc is not None and len(sc) else float("nan")
    nose, le, re = kp[0], kp[1], kp[2]
    inter = float(np.linalg.norm(le - re))
    if inter <= 1e-6:
        return None
    yaw = float((nose[0] - (le[0] + re[0]) / 2.0) / inter)
    if not math.isfinite(yaw):
        return None
    return yaw, conf, n_kp


def spread(values):
    """max - min over the non-None yaws, or None if fewer than 2 measured."""
    v = [x for x in values if x is not None]
    if len(v) < 2:
        return None
    return max(v) - min(v)


def monotonicity(values):
    """'increasing' / 'decreasing' / 'non-monotone' / None from sweep-ordered yaws."""
    v = [x for x in values if x is not None]
    if len(v) < 2:
        return None
    if all(b > a for a, b in zip(v, v[1:])):
        return "increasing"
    if all(b < a for a, b in zip(v, v[1:])):
        return "decreasing"
    return "non-monotone"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # `global` must precede the parser build, which reads SWEEP_VALUES for the
    # --sweep-values default (that read happens before the reassignment below).
    global SWEEP_VALUES

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=str, default=str(CONFIG_PATH),
                    help=f"arm config (default: {CONFIG_PATH})")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="checkpoint .pt (default: most recently modified "
                         "checkpoint_step*.pt / checkpoint_interrupt.pt under "
                         f"{DEFAULT_CKPT_DIR})")
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                    help=f"output directory (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--sweep-values", type=float, nargs="+",
                    default=list(SWEEP_VALUES),
                    help=f"z_g dim{SWEEP_DIM} values to sweep, in sweep order "
                         f"(default {SWEEP_VALUES}) -- e.g. pass "
                         f"'-1.5 0.0 1.5' for a +/-1.5 sigma in-distribution row "
                         f"without editing this file")
    args = ap.parse_args()

    # CLI-overridable copy of the module default. When --sweep-values is omitted,
    # args.sweep_values == list(SWEEP_VALUES), so behaviour is byte-identical to
    # before this flag existed (main() is the only reader of SWEEP_VALUES).
    SWEEP_VALUES = list(args.sweep_values)
    if not SWEEP_VALUES:
        raise SystemExit("--sweep-values must contain at least one value")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else find_default_checkpoint(DEFAULT_CKPT_DIR)
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    print("=" * 70)
    print("EIDOLON z_g DILUTION DIAGNOSTIC (identity-CFG vs geometry-CFG)")
    print("=" * 70)
    print(f"  config     : {args.config}")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  out-dir    : {out}")
    print(f"  sweep      : dim {SWEEP_DIM} over {SWEEP_VALUES} "
          f"(sweep order), samples {SAMPLE_INDICES}, seed {SEED}")
    for name, ids, gs in CFG_CONFIGS:
        print(f"  cfg        : {name:18s} identity_scale={ids}  geometry_scale={gs}")

    config = load_config(str(args.config))
    latent_space = resolve_latent_space(config)
    gamma = resolve_sampling_gamma(config)
    device = torch.device(args.device)
    print(f"  latent_space={latent_space}  gamma={gamma}  device={device}")

    model = build_model(config, ckpt_path, device)
    print(f"  model: {sum(p.numel() for p in model.parameters()):,} params")

    vae = load_vae_decoder(device=device) if latent_space else None

    samples = load_samples(config, device, latent_space)
    print(f"  loaded {len(samples)} validation samples")

    det = build_detector()

    # Generation shape: pixel-space arms sample RGB at full res; this arm is
    # latent-space, so we sample 16ch latents at the model grid and VAE-decode
    # (production/sample.py ValidationSampler.generate, production/validate.py).
    if latent_space:
        in_channels, spatial = 16, config.model.input_size
    else:
        in_channels, spatial = 3, 1024
    shape = (1, in_channels, spatial, spatial)
    sampler = EulerSampler(num_steps=NUM_STEPS)

    results = {
        "config": str(args.config),
        "checkpoint": str(ckpt_path),
        "sweep_dim": SWEEP_DIM,
        "sweep_values": SWEEP_VALUES,
        "sample_indices": SAMPLE_INDICES,
        "seed": SEED,
        "num_steps": NUM_STEPS,
        "gamma": gamma,
        "latent_space": latent_space,
        "cfg_configs": [
            {"name": n, "identity_scale": ids, "geometry_scale": gs}
            for n, ids, gs in CFG_CONFIGS
        ],
        "notes": [],
        "configs": [],
    }

    panels_total = 0
    no_face_total = 0

    for cname, id_scale, geo_scale in CFG_CONFIGS:
        print(f"\n######## CFG {cname}  (identity_scale={id_scale}, "
              f"geometry_scale={geo_scale}) ########")
        cfg_entry = {
            "name": cname,
            "identity_scale": id_scale,
            "geometry_scale": geo_scale,
            "samples": [],
        }
        grid_rows = []

        for si in SAMPLE_INDICES:
            s = samples[si]
            idv = s["identity_emb"].unsqueeze(0).to(device)
            baseg = s["geometry_emb"].unsqueeze(0).to(device)
            print(f"  -- sample {si} (image_id={s['image_id']}) "
                  f"base z_g[{SWEEP_DIM}]={float(baseg[0, SWEEP_DIM]):+.2f}")

            row_tensors = []
            yaws, confs, nkps = [], [], []
            panel_paths = []

            for val in SWEEP_VALUES:
                g = baseg.clone()
                g[:, SWEEP_DIM] = val
                # Noise-matched: reset the same fixed seed before EVERY single
                # generation (diag_pose_cfg_sweep.py does exactly this), so a
                # panel difference is attributable to conditioning only.
                torch.manual_seed(SEED)
                with torch.no_grad():
                    o = sampler.sample(
                        model=model,
                        shape=shape,
                        identity_emb=idv,
                        geometry_emb=g,
                        device=device,
                        text_scale=id_scale,
                        dino_scale=geo_scale,
                        prediction_type=config.model.prediction_type,
                    )
                if latent_space:
                    img_t = decode_latents(vae, o)[0]          # (3, H*8, W*8) [-1,1]
                else:
                    img_t = o[0].clamp(0, 1) * 2 - 1           # (3, H, W) [-1,1]
                row_tensors.append(img_t)

                panel_path = out / f"s{si:02d}_{cname}_dim0_{val:+.1f}.png"
                tensor_to_pil(img_t).save(panel_path)
                panel_paths.append(panel_path)

            # Measure yaw on every rendered panel (sweep order preserved).
            for val, panel_path in zip(SWEEP_VALUES, panel_paths):
                m = measure_panel_yaw(det, panel_path)
                panels_total += 1
                if m is None:
                    yaws.append(None)
                    confs.append(None)
                    nkps.append(None)
                    no_face_total += 1
                    print(f"     z_g={val:+.1f}: no face detected "
                          f"(panel={panel_path.name})")
                else:
                    yaw, conf, n_kp = m
                    yaws.append(yaw)
                    confs.append(conf)
                    nkps.append(n_kp)
                    print(f"     z_g={val:+.1f}: yaw_proxy={yaw:+.4f} "
                          f"conf={conf:.3f} kp={n_kp}")

            sp = spread(yaws)
            mono = monotonicity(yaws)
            print(f"     {cname} sample {si}: yaw_spread={sp} "
                  f"monotone={mono} "
                  f"no_face={sum(1 for y in yaws if y is None)}/{len(SWEEP_VALUES)}")

            cfg_entry["samples"].append(
                {
                    "sample_idx": si,
                    "image_id": str(s["image_id"]),
                    "base_zg_dim0": float(baseg[0, SWEEP_DIM].item()),
                    "sweep_values": SWEEP_VALUES,
                    "yaw_proxy": yaws,          # sweep order, None = no face
                    "det_conf": confs,
                    "n_keypoints": nkps,
                    "no_face_panels": sum(1 for y in yaws if y is None),
                    "yaw_spread": sp,
                    "monotone": mono,
                }
            )
            grid_rows.append(row_tensors)

        g_img = grid(grid_rows)
        g_path = (
            out
            / f"grid_{cname}_rows=samples[{','.join(str(i) for i in SAMPLE_INDICES)}]"
              f"_cols=dim0[{','.join(f'{v:+.1f}' for v in SWEEP_VALUES)}].png"
        )
        g_img.save(g_path)
        print(f"  GRID -> {g_path}")

        results["configs"].append(cfg_entry)

    results["panels_total"] = panels_total
    results["no_face_panels_total"] = no_face_total
    if no_face_total:
        results["notes"].append(
            f"{no_face_total}/{panels_total} panels had no detectable face; "
            "their yaw is recorded as null and excluded from spread/monotonicity. "
            "No value was substituted."
        )
    if latent_space:
        results["notes"].append(
            "latent-space arm: latents sampled at "
            f"{in_channels}ch x {spatial}x{spatial} and VAE-decoded to RGB."
        )

    # ---------------- summary table ----------------
    print("\n" + "=" * 70)
    print("SUMMARY -- yaw proxy in SWEEP ORDER "
          f"dim{SWEEP_DIM} {SWEEP_VALUES}")
    print("=" * 70)
    hdr = f"{'cfg':18s} {'sidx':>4s} " + " ".join(f"{v:+6.1f}" for v in SWEEP_VALUES) \
          + f"  {'spread':>8s}  {'mono':>12s}  noface"
    print(hdr)
    for cfg_entry in results["configs"]:
        for sm in cfg_entry["samples"]:
            cells = " ".join(
                "  n/a " if y is None else f"{y:+6.3f}" for y in sm["yaw_proxy"]
            )
            sp = "n/a" if sm["yaw_spread"] is None else f"{sm['yaw_spread']:.3f}"
            print(f"{cfg_entry['name']:18s} {sm['sample_idx']:>4d} {cells}  "
                  f"{sp:>8s}  {str(sm['monotone']):>12s}  "
                  f"{sm['no_face_panels']}/{len(SWEEP_VALUES)}")

    results_path = out / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults.json -> {results_path}")
    print(f"panels={panels_total}  no_face={no_face_total}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
