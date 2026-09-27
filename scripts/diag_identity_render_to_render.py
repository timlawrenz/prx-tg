#!/usr/bin/env python3
"""DIAGNOSTIC: does `eidolon-identity-renderer` (Arm EIR) retrieve the RIGHT
identity among OTHER RENDERS, rather than among REAL faces?

Why this exists
---------------
The arm's identity gate scores retrieval with a rank-based R@1 whose index is
REAL images (the validation pool's stored `auraface_lda.npy`). At the current
render quality that number is CONFOUNDED:

  * the AuraFace recogniser is validated on clean real faces (R@1 0.891 on the
    published index);
  * on low-quality renders every embedding collapses toward the mean, so the
    measured R@1 against the real index was 0.0588 vs a chance of 0.0196 -- and
    the cosine to the best OTHER identity often EXCEEDS the cosine to the
    render's OWN identity.

The fix is to retrieve against an index of OTHER RENDERS, so every candidate
shares the render's quality (the recogniser's quality gap cancels). If the arm
learned identity, a render's nearest neighbour among the other renders is
overwhelmingly its own source identity.

Test
----
Load the same deterministic validation sample pool the seed probe uses. Render
each of the first N samples ONCE under the FULL CFG ladder (identity_scale 3.0 /
geometry_scale 2.0, i.e. sampling.text_scale / sampling.dino_scale), with a seed
that is FIXED PER SAMPLE and reset via `torch.manual_seed(seed)` immediately
before EVERY generation (the noise-matched convention shared with
`scripts/diag_eidolon_zg_dilution.py` and
`scripts/diag_identity_seed_consistency.py`).

Measurement
-----------
Every render is embedded by the project's SHARED AuraFace instrument and
projected to the 64-d LDA identity space with the SAME 'clean_auraface' recipe
the seed probe uses (see "instrument" below). TWO retrievals are computed and
BOTH reported:

  (i)  RENDER-TO-RENDER -- index = the LDA embeddings of the OTHER RENDERS.
       Each candidate shares the query's render quality, so the recogniser's
       domain gap cancels. chance = 1 / (n_measured - 1).

  (ii) RENDER-TO-REAL -- index = the real ground-truth LDA of the same pool
       (the sidecar `auraface_lda.npy` that the validation loader hands the
       model, L2-normalized). This is the confounded number the gate currently
       reports, kept here for direct comparison. chance = 1 / (pool - 1).

For each retrieval the script reports R@1, R@5, median rank, chance, the
own-minus-best-other margin (mean / median), the fraction of queries whose own
cosine is >= the best other, and a BOOTSTRAP CI on R@1 that resamples IDENTITIES
(>= 2000 reps). Explicit `no_face` counts and rates are reported: a face-filling
crop can legitimately return `no_face` from the detector, and an excluded render
is ALWAYS reported, never silently dropped and never substituted with a guess.

Outputs (under --out-dir)
-------------------------
    results.json              machine-readable, incl. instrument + provenance
    s{idx:02d}.png            the individual renders
    contact_sheet.png         one small labelled tile per render (sample id)

Instrument (which recogniser, and why)
--------------------------------------
`/mnt/nas-ai-models/models/auraface/` holds the `auraface` insightface pack.
`glintr100.onnx` is the recognition net that emits the 512-d `normed_embedding`
this project's LDA basis was fit on; `scrfd_10g_bnkps.onnx` is the detector
(used for detection *and* the 5-point alignment crop). The instrument is
addressed through the shared pack loader (`tools.auraface.extract_auraface`),
and `describe_instrument()` records `det_size`, `pad_fraction`, the basis
artifact sha256 AND the sha256 of both ONNX files, so "which AuraFace
convention produced these numbers?" is answerable without rereading the code.

Detection convention: `det_size = CORPUS_DET_SIZE = None`, i.e. insightface's
default 640x640 -- the CORPUS behaviour every stored `auraface_lda.npy` was
produced with. `pad_fraction 0.2` is the instrument's own retry contract.

Provenance of the copied API calls
----------------------------------
* Model construction, checkpoint load (`ckpt["model"]`, NOT ema), validation
  dataloader call WITH the identity-basis guard values, latent-space shape
  selection, VAE decode, sampler invocation and the noise-matched
  `torch.manual_seed` reset:
      scripts/diag_eidolon_zg_dilution.py
* The validation sample pool loader, the AuraFace->64-d LDA recipe
  (`clean_auraface` -> `project_to_lda`, L2-normalize the query before any
  cosine -- "trap 6"), `rank_of_own`, `stat`, `_jsonable`, `_git_commit` and the
  `_ensure_webdataset` fail-closed stub:
      scripts/diag_identity_seed_consistency.py
* The AuraFace instrument itself (`extract_auraface`, `describe_instrument`,
  `CORPUS_DET_SIZE`, `pad_fraction 0.2`, per-image outcome):
      /home/tim/source/activity/eidolon/tools/auraface/extract.py

This script does NOT train, does NOT modify the checkpoint on disk, does NOT
touch anything under production/, and does NOT modify any existing script.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# Repo root on sys.path, exactly as scripts/diag_identity_seed_consistency.py.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# The AuraFace identity instrument lives in the sibling eidolon repo. APPEND it,
# not insert(0): the prx-tg repo root must keep winning name resolution for
# `production` (a regular package here) and the `scripts`/`experiments`
# namespaces, which also exist under the eidolon root.
EIDOLON_ROOT = Path("/home/tim/source/activity/eidolon")
if str(EIDOLON_ROOT) not in sys.path:
    sys.path.append(str(EIDOLON_ROOT))

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


def _ensure_webdataset():
    """Make `import webdataset` inside production/data.py succeed when the
    package is absent, WITHOUT allowing any of its API to be used.

    `production/data.py` imports webdataset at module scope but only uses it in
    the shard-based ValidationDataset branch; the `source="stratum"` branch that
    this diagnostic (and every eidolon arm) actually uses never touches it. The
    placeholder is FAIL-CLOSED: fetching any attribute raises, so a future code
    path that DOES need webdataset cannot be silently mis-served.

    Returns "present" | "stubbed".
    """
    import importlib
    import types

    try:
        importlib.import_module("webdataset")
        return "present"
    except ModuleNotFoundError:
        pass

    stub = types.ModuleType("webdataset")
    msg = (
        "webdataset is not installed in this interpreter, and this diagnostic "
        "only stubbed it because the source='stratum' validation branch never "
        "uses it. Something tried to use webdataset for real. Install it "
        "(`pip install webdataset`) or run under an interpreter that has it."
    )

    def _boom(*_a, **_k):
        raise RuntimeError(msg)

    stub.__getattr__ = _boom          # PEP 562 module-level fallback
    for _name in ("WebDataset", "warn_and_continue", "DataLoader", "ShardList",
                  "split_by_node", "split_by_worker"):
        setattr(stub, _name, _boom)
    sys.modules["webdataset"] = stub
    return "stubbed"


WEBDATASET_STATUS = _ensure_webdataset()

from production.data import get_deterministic_validation_dataloader  # noqa: E402

from tools.auraface import (  # noqa: E402
    CORPUS_DET_SIZE,
    describe_instrument,
    extract_auraface,
)

# ---------------------------------------------------------------------------
# Fixed experiment parameters
# ---------------------------------------------------------------------------
CONFIG_PATH = REPO / "experiment-configs" / "eidolon-identity-renderer" / "config.yaml"
DEFAULT_CKPT_DIR = Path(
    "/mnt/nas-ai-models/training-data/prx-tg/eidolon-identity-renderer/"
    "runs/2026-09-25_0734/checkpoints"
)

SAMPLES_DEFAULT = 40            # first N samples rendered (max ~51 == pool size)
POOL_SIZE_DEFAULT = 51          # the validation identity pool the seed probe loads
NUM_STEPS_DEFAULT = 50
SEED_BASE_DEFAULT = 0           # render seed = seed_base + sample_idx (fixed per sample)
BOOTSTRAP_REPS_DEFAULT = 2000
BOOTSTRAP_SEED = 20260926       # fixed, so the CI is reproducible
THUMB_DEFAULT = 160             # contact-sheet tile size (px, square)
COLS_DEFAULT = 8                # tiles per row on the contact sheet

# LDA basis artifacts. The same directory scripts/batch_extract_eidolon_data.py
# reads (`ARTIFACT_DIR`) and the same one tools/auraface/extract.py pins hashes
# for (`BASIS_DIR`).
BASIS_DIR = Path("/home/tim/source/activity/eidolon/experiments/geometry_pca/output")
PREPROCESS_NPZ = BASIS_DIR / "auraface_preprocess.npz"
LDA_NPZ = BASIS_DIR / "auraface_lda.npz"

# The AuraFace model pack (self-resolved by the instrument, recorded here too).
AURAFACE_DIR = Path("/mnt/nas-ai-models/models/auraface")


# ---------------------------------------------------------------------------
# Checkpoint selection -- copied from scripts/diag_eidolon_zg_dilution.py
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
# Model construction -- copied faithfully from scripts/diag_eidolon_zg_dilution.py
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
    # Training weights, NOT ema (matches diag_eidolon_zg_dilution.py).
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Validation sample pool -- copied faithfully from
# scripts/diag_identity_seed_consistency.py's load_samples(), with the pool
# loaded to a fixed COUNT (not to a max index) and graceful exhaustion.
# ---------------------------------------------------------------------------
def load_pool(config, latent_space: bool, pool_size: int):
    """Grab up to `pool_size` samples from the deterministic validation loader.

    diag_pose_cfg_sweep.py passes target_latent_size=config.model.input_size and
    does not forward the identity-basis guard. For a LATENT-space arm,
    production/validate.py uses input_size * 8 (a 1024px image -> 128x128 FLUX
    latents), so this follows validate.py for that one argument only. The value
    is ignored for conditioning extraction (only identity_emb / geometry_emb are
    used) and production/data_stratum.py documents that target_latent_size is
    IGNORED in latent mode anyway.
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

    samples = []
    it = iter(loader)
    while len(samples) < pool_size:
        try:
            b = next(it)
        except StopIteration:
            break
        n = (
            b["image_data"].shape[0]
            if "image_data" in b
            else b["identity_emb"].shape[0]
        )
        for i in range(n):
            if len(samples) < pool_size:
                samples.append(
                    {
                        "identity_emb": b["identity_emb"][i],
                        "geometry_emb": b["geometry_emb"][i],
                        "image_id": b["image_ids"][i],
                    }
                )
    return samples


# ---------------------------------------------------------------------------
# AuraFace -> 64-d LDA identity instrument
# ---------------------------------------------------------------------------
# Recipe copied faithfully from scripts/diag_identity_seed_consistency.py /
# scripts/batch_extract_eidolon_data.py: load reference artifacts -> clean ->
# project. The ONLY addition is l2n(): project_to_lda returns an UNNORMALIZED
# vector (~norm 150) while the identity vectors on disk are unit-norm, so every
# cosine is taken after L2-normalizing BOTH sides (documented trap 6).
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
        """PNG path or (H,W,3) BGR uint8 -> dict with the instrument's outcome
        and the L2-normalized 64-d LDA vector (or None). NEVER guesses: a
        no-face render yields ok=False and lda_unit=None.
        """
        res = extract_auraface(
            image,
            det_size=CORPUS_DET_SIZE,   # None == insightface default 640x640
            pad_fraction=0.20,          # the instrument's own retry contract
            keep_embedding=True,
            keep_lda=True,
        )
        out = {
            "ok": bool(res.ok),
            "outcome": str(res.outcome),          # detected | detected_after_padding | no_face
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
        # Cross-check against the instrument's own projection (same artifacts,
        # same math; a disagreement means the basis moved under us).
        if res.lda_coords is not None:
            helper = l2n(res.lda_coords)
            out["cos_vs_helper_lda"] = float(np.asarray(out["lda_unit"]) @ helper)
        else:
            out["cos_vs_helper_lda"] = None
        return out


def l2n(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n else v


# ---------------------------------------------------------------------------
# Contact sheet -- small labelled tiles, one per render (sample id drawn in).
# ---------------------------------------------------------------------------
def contact_sheet(tiles_in, spacing=6, thumb=THUMB_DEFAULT, cols=COLS_DEFAULT):
    """tiles_in: list of (sample_idx, (3,H,W) tensor in [-1,1]) -> labelled PNG."""
    tiles = []
    for si, img in tiles_in:
        a = (img.detach().cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        pil = Image.fromarray((a.transpose(1, 2, 0) * 255).astype(np.uint8))
        pil = pil.resize((thumb, thumb), Image.LANCZOS)
        tiles.append((si, pil))

    n = len(tiles)
    ncol = min(cols, max(1, n))
    nrow = (n + ncol - 1) // ncol
    label_h = 14
    cell_w = thumb
    cell_h = thumb + label_h
    W = ncol * cell_w + (ncol - 1) * spacing
    H = nrow * cell_h + (nrow - 1) * spacing
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for k, (si, im) in enumerate(tiles):
        r, c = divmod(k, ncol)
        x = c * (cell_w + spacing)
        y = r * (cell_h + spacing)
        canvas.paste(im, (x, y + label_h))
        draw.text((x + 2, y + 1), f"s{si:02d}", fill=(0, 0, 0))
    return canvas


# ---------------------------------------------------------------------------
# Ranking + statistics helpers
# ---------------------------------------------------------------------------
def rank_of_own(q_unit, pool_unit, own_idx):
    """Rank (1-based) of own entry among the pool, plus (own_cos, best_other_cos)."""
    sims = pool_unit @ q_unit                      # pool rows are unit-norm
    order = np.argsort(-sims)
    rank = int(np.where(order == own_idx)[0][0]) + 1
    own = float(sims[own_idx])
    others = np.delete(sims, own_idx)
    best_other = float(others.max()) if others.size else float("nan")
    return rank, own, best_other


def stat(values):
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "mean": None, "min": None, "max": None,
                "median": None, "p95": None}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "min": float(v.min()),
        "max": float(v.max()),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
    }


def bootstrap_r1_ci(hits, reps=BOOTSTRAP_REPS_DEFAULT, seed=BOOTSTRAP_SEED):
    """Percentile bootstrap on R@1 resampling IDENTITIES (== queries).

    Each query is one render of one identity, so resampling queries with
    replacement IS resampling identities. Returns the point estimate plus the
    2.5/97.5 percentiles. Deterministic: the RNG is a private numpy generator
    seeded with `seed`, so it never touches the torch RNG used for generation.
    """
    h = np.asarray(hits, dtype=np.float64)
    if h.size == 0:
        return {"R@1": None, "ci_low": None, "ci_high": None,
                "reps": int(reps), "resample_unit": "identity"}
    rng = np.random.default_rng(seed)
    n = h.size
    means = np.empty(int(reps), dtype=np.float64)
    for b in range(int(reps)):
        means[b] = h[rng.integers(0, n, n)].mean()
    return {
        "R@1": float(h.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "reps": int(reps),
        "resample_unit": "identity",
    }


def summarize_retrieval(ranks, margins, hits, chance, pool_size, index_kind):
    """One retrieval's headline block: R@1/R@5/median rank/chance/margin/CI."""
    rk = np.asarray(ranks, dtype=np.int64) if len(ranks) else np.zeros(0, np.int64)
    mg = np.asarray(margins, dtype=np.float64) if len(margins) else np.zeros(0)
    return {
        "index": index_kind,
        "pool_size": int(pool_size),
        "n_queries": int(rk.size),
        "chance_R@1": float(chance),
        "R@1": (float((rk == 1).mean()) if rk.size else None),
        "R@5": (float((rk <= 5).mean()) if rk.size else None),
        "median_rank": (float(np.median(rk)) if rk.size else None),
        "margin_own_minus_best_other": stat(margins),
        "frac_own_ge_best_other": (float((mg >= 0).mean()) if mg.size else None),
        "bootstrap_R@1": bootstrap_r1_ci(hits),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=str, default=str(CONFIG_PATH),
                    help=f"arm config (default: {CONFIG_PATH})")
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="checkpoint .pt to render from (REQUIRED), e.g. "
                         "experiments/eidolon-identity-renderer/runs/"
                         "2026-09-25_0734/checkpoints/checkpoint_step0009000.pt")
    ap.add_argument("--samples", type=int, default=SAMPLES_DEFAULT,
                    help=f"number of the FIRST validation samples to render "
                         f"(default {SAMPLES_DEFAULT}, max == pool size ~51)")
    ap.add_argument("--num-steps", type=int, default=NUM_STEPS_DEFAULT,
                    help=f"Euler steps (default {NUM_STEPS_DEFAULT}, == arm config)")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="output directory (REQUIRED)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--pool-size", type=int, default=POOL_SIZE_DEFAULT,
                    help=f"how many validation identities to load into the pool "
                         f"(default {POOL_SIZE_DEFAULT})")
    ap.add_argument("--seed-base", type=int, default=SEED_BASE_DEFAULT,
                    help="render seed = seed_base + sample_idx (fixed per sample; "
                         f"default {SEED_BASE_DEFAULT})")
    ap.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS_DEFAULT,
                    help=f"bootstrap resamples for the R@1 CI "
                         f"(default {BOOTSTRAP_REPS_DEFAULT}; identities resampled)")
    ap.add_argument("--thumb", type=int, default=THUMB_DEFAULT,
                    help=f"contact-sheet tile size in px (default {THUMB_DEFAULT})")
    ap.add_argument("--cols", type=int, default=COLS_DEFAULT,
                    help=f"contact-sheet tiles per row (default {COLS_DEFAULT})")
    ap.add_argument("--text-scale", type=float, default=None,
                    help="IDENTITY guidance scale (default: config.sampling.text_scale)")
    ap.add_argument("--dino-scale", type=float, default=None,
                    help="GEOMETRY guidance scale (default: config.sampling.dino_scale)")
    args = ap.parse_args()

    if args.bootstrap_reps < 1:
        raise SystemExit("--bootstrap-reps must be >= 1")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    print("=" * 72)
    print("EIDOLON IDENTITY RENDER-TO-RENDER RETRIEVAL DIAGNOSTIC")
    print("=" * 72)
    print(f"  config     : {args.config}")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  out-dir    : {out}")
    print(f"  samples    : first {args.samples} of the pool "
          f"(pool-size <= {args.pool_size})")
    print(f"  steps      : {args.num_steps}")
    print(f"  seed       : seed_base {args.seed_base} + sample_idx "
          f"(reset immediately before EVERY generation)")

    config = load_config(str(args.config))
    latent_space = resolve_latent_space(config)
    gamma = resolve_sampling_gamma(config)
    device = torch.device(args.device)
    print(f"  latent_space={latent_space}  gamma={gamma}  device={device}")

    ad = getattr(config, "adapter", None)
    if ad is None or getattr(ad, "name", None) != "eidolon":
        raise SystemExit(
            f"this diagnostic is written for adapter.name == 'eidolon'; "
            f"config has {getattr(ad, 'name', None)!r}"
        )

    sampling = getattr(config, "sampling", None)
    id_scale = args.text_scale if args.text_scale is not None else float(
        getattr(sampling, "text_scale", 3.0)
    )
    geo_scale = args.dino_scale if args.dino_scale is not None else float(
        getattr(sampling, "dino_scale", 2.0)
    )
    # For the eidolon adapter the sampler's "text" branch IS identity-only and its
    # "dino" branch IS geometry-only (production/sample.py dual-CFG block; the arm
    # config's `sampling:` comment), so these two numbers are the identity and
    # geometry guidance scales.
    print(f"  identity_scale(text_scale)={id_scale}  "
          f"geometry_scale(dino_scale)={geo_scale}")

    # ---- identity instrument (CPU; insightface app is loaded lazily on first
    # embed, and onnxruntime is pinned to CPUExecutionProvider by the instrument)
    print("  loading AuraFace instrument provenance...", flush=True)
    instrument_prov = describe_instrument(det_size=CORPUS_DET_SIZE)
    lda_inst = AuraFaceLDA()
    print(f"  instrument: det_size={instrument_prov['det_size']} "
          f"pad_fraction={instrument_prov['pad_fraction']} "
          f"basis_fingerprint={instrument_prov['basis_fingerprint']} "
          f"embedding={instrument_prov['embedding']}")

    model = build_model(config, ckpt_path, device)
    print(f"  model: {sum(p.numel() for p in model.parameters()):,} params")

    vae = load_vae_decoder(device=device) if latent_space else None

    pool = load_pool(config, latent_space, args.pool_size)
    if not pool:
        raise SystemExit(
            "validation loader yielded ZERO samples -- refusing to measure a "
            "silent zero (check the stratum dir / basis guard)."
        )
    print(f"  loaded validation pool: {len(pool)} identities")

    n_render = int(args.samples)
    if n_render < 2:
        raise SystemExit("--samples must be >= 2 (R@1 needs at least one OTHER render)")
    if n_render > len(pool):
        raise SystemExit(
            f"--samples {n_render} exceeds the loaded pool ({len(pool)}); "
            f"lower --samples or raise --pool-size."
        )
    print(f"  rendering the first {n_render} samples")

    # ---- the real GT pool: every rendered sample's own stored identity vector
    # (`auraface_lda.npy`, via the loader), L2-normalized.
    gt_unit = np.stack([l2n(pool[i]["identity_emb"].detach().cpu().numpy())
                        for i in range(n_render)])              # (n_render, 64)
    gt_ids = [str(pool[i]["image_id"]) for i in range(n_render)]
    gt_raw_norms = [
        float(np.linalg.norm(pool[i]["identity_emb"].detach().cpu().numpy()))
        for i in range(n_render)
    ]

    # Generation shape: this arm is FLUX-AE LATENT space, so sample 16ch latents at
    # the model grid and VAE-decode (production/sample.py ValidationSampler.generate,
    # production/validate.py).
    if latent_space:
        in_channels, spatial = 16, config.model.input_size
    else:
        in_channels, spatial = 3, 1024
    shape = (1, in_channels, spatial, spatial)
    sampler = EulerSampler(num_steps=args.num_steps)

    results = {
        "diagnostic": "eidolon_identity_render_to_render",
        "question": (
            "does the identity stream bind strongly enough to retrieve the right "
            "person among OTHER RENDERS (equal render quality), vs among REAL faces"
        ),
        "config": str(args.config),
        "checkpoint": str(ckpt_path),
        "checkpoint_basename": ckpt_path.name,
        "samples_requested": n_render,
        "pool_size": len(pool),
        "rendered_sample_indices": list(range(n_render)),
        "rendered_image_ids": gt_ids,
        "num_steps": args.num_steps,
        "identity_scale": id_scale,
        "geometry_scale": geo_scale,
        "gamma": gamma,
        "latent_space": latent_space,
        "seed_base": args.seed_base,
        "renders": [],
        "retrievals": {},
        "instrument": instrument_prov,
        "conventions": {
            "noise": ("torch.manual_seed(seed_base + sample_idx) reset immediately "
                      "before EVERY generation (same convention as "
                      "diag_eidolon_zg_dilution.py / diag_identity_seed_consistency.py)"),
            "auraface_input": "(H,W,3) BGR uint8 (cv2.imread order, via PNG path)",
            "det_size": CORPUS_DET_SIZE,
            "pad_fraction": instrument_prov.get("pad_fraction"),
            "embedding": "normed_embedding (512-d)",
            "lda": ("clean_auraface (remove pooled_mean, pc1, yaw; L2-renorm) -> "
                    "project_to_lda (lda_basis 512x64) -> L2-NORMALIZED query "
                    "before any cosine (trap 6)"),
            "real_index": ("the stored identity vector the validation loader returns "
                           "for each rendered sample (auraface_lda.npy), L2-normalized"),
            "render_to_render": ("index = OTHER RENDERS' LDA; each query's own render "
                                 "is the target; chance = 1/(n_measured-1)"),
            "render_to_real": ("index = the real GT LDA of the SAME rendered pool; "
                               "chance = 1/(pool-1)"),
            "no_face": ("extract_auraface outcome no_face -> lda_unit=null, excluded "
                        "from every cosine and from R@1; never substituted, always "
                        "reported"),
        },
        "notes": [],
    }

    renders_total = 0
    no_face_total = 0
    padding_total = 0
    ambiguous_total = 0
    render_tiles = []               # (sample_idx, tensor) for the contact sheet
    ok_units = []                   # (sample_idx, unit-norm (64,)) in render order
    ok_cos_vs_helper = []

    for si in range(n_render):
        s = pool[si]
        idv = s["identity_emb"].unsqueeze(0).to(device)
        baseg = s["geometry_emb"].unsqueeze(0).to(device)
        seed = args.seed_base + si
        print(f"\n######## sample {si} (image_id={s['image_id']})  seed={seed} ########")

        # Noise-matched: reset the fixed per-sample seed immediately before the
        # (single) generation for this sample.
        torch.manual_seed(seed)
        with torch.no_grad():
            o = sampler.sample(
                model=model,
                shape=shape,
                identity_emb=idv,
                geometry_emb=baseg,
                device=device,
                text_scale=id_scale,
                dino_scale=geo_scale,
                prediction_type=config.model.prediction_type,
            )
        if latent_space:
            img_t = decode_latents(vae, o)[0]          # (3, H*8, W*8) [-1,1]
        else:
            img_t = o[0].clamp(0, 1) * 2 - 1           # (3, H, W) [-1,1]

        panel_path = out / f"s{si:02d}.png"
        tensor_to_pil(img_t).save(panel_path)
        render_tiles.append((si, img_t))

        m = lda_inst.embed(panel_path)
        renders_total += 1
        rec = {
            "sample_idx": si,
            "image_id": str(s["image_id"]),
            "seed": seed,
            "panel": panel_path.name,
            **m,
        }
        if m["lda_unit"] is None:
            no_face_total += 1
            print(f"   NO FACE ({m['outcome']}) -> excluded, no value substituted "
                  f"  [{panel_path.name}]")
        else:
            if m["used_padding"]:
                padding_total += 1
            if m["ambiguous"]:
                ambiguous_total += 1
            ok_units.append((si, m["lda_unit"]))
            if m["cos_vs_helper_lda"] is not None:
                ok_cos_vs_helper.append(m["cos_vs_helper_lda"])
            print(f"   outcome={m['outcome']:22s} n_faces={m['n_faces']} "
                  f"|lda_raw|={m['lda_norm_raw']:.1f}  [{panel_path.name}]")
        results["renders"].append(_jsonable(rec))

    # ---- (i) RENDER-TO-RENDER -------------------------------------------------
    n_meas = len(ok_units)
    if n_meas < 2:
        raise SystemExit(
            f"only {n_meas} measured render(s) -- R@1 is undefined; refusing to "
            f"report a number. (no_face renders are reported above.)"
        )
    r2r_pool = np.stack([u for _, u in ok_units])            # (n_meas, 64)
    r2r_ranks, r2r_margins, r2r_hits = [], [], []
    for k, (si, u) in enumerate(ok_units):
        rank, own, best_other = rank_of_own(u, r2r_pool, k)
        margin = own - best_other
        r2r_ranks.append(rank)
        r2r_margins.append(margin)
        r2r_hits.append(1.0 if rank == 1 else 0.0)
        rec = results["renders"][si]
        rec["r2r_rank"] = rank
        rec["r2r_cos_to_own_render"] = own
        rec["r2r_best_other_render_cos"] = best_other
        rec["r2r_margin"] = margin
    results["retrievals"]["render_to_render"] = summarize_retrieval(
        r2r_ranks, r2r_margins, r2r_hits,
        chance=1.0 / (n_meas - 1), pool_size=n_meas,
        index_kind="other renders' LDA embeddings",
    )

    # ---- (ii) RENDER-TO-REAL (for comparison) --------------------------------
    r2real_ranks, r2real_margins, r2real_hits = [], [], []
    for si, u in ok_units:
        rank, own, best_other = rank_of_own(u, gt_unit, si)
        margin = own - best_other
        r2real_ranks.append(rank)
        r2real_margins.append(margin)
        r2real_hits.append(1.0 if rank == 1 else 0.0)
        rec = results["renders"][si]
        rec["r2real_rank"] = rank
        rec["r2real_cos_to_own_gt"] = own
        rec["r2real_best_other_gt_cos"] = best_other
        rec["r2real_margin"] = margin
    results["retrievals"]["render_to_real"] = summarize_retrieval(
        r2real_ranks, r2real_margins, r2real_hits,
        chance=1.0 / (n_render - 1), pool_size=n_render,
        index_kind="real GT LDA of the same rendered pool (auraface_lda.npy)",
    )

    # ---- contact sheet --------------------------------------------------------
    cs_path = out / "contact_sheet.png"
    contact_sheet(render_tiles, thumb=args.thumb, cols=args.cols).save(cs_path)
    print(f"\nCONTACT SHEET -> {cs_path}")

    # ---- aggregate + notes ----------------------------------------------------
    results["renders_total"] = renders_total
    results["renders_measured"] = n_meas
    results["no_face_renders"] = no_face_total
    results["no_face_rate"] = (no_face_total / renders_total) if renders_total else None
    results["used_padding_renders"] = padding_total
    results["ambiguous_renders"] = ambiguous_total
    results["cos_query_vs_instrument_lda"] = stat(ok_cos_vs_helper)
    results["gt_raw_norms"] = {
        "min": float(min(gt_raw_norms)) if gt_raw_norms else None,
        "max": float(max(gt_raw_norms)) if gt_raw_norms else None,
    }
    if no_face_total:
        results["notes"].append(
            f"{no_face_total}/{renders_total} renders returned no_face from the "
            "AuraFace detector (a face-filling crop can legitimately do this); "
            "their LDA is null and they are excluded from every cosine and from "
            "R@1. They are reported here explicitly; no value was substituted."
        )
    if ok_cos_vs_helper and min(ok_cos_vs_helper) < 0.99999:
        results["notes"].append(
            "the inline clean->project recipe and the instrument's own lda_coords "
            f"disagree (min cosine {min(ok_cos_vs_helper):.6f}); the basis "
            "artifacts may have moved."
        )
    results["notes"].append(
        "render_to_render is the quality-matched number: every candidate shares the "
        "query's render quality, so the recogniser's real-vs-render domain gap "
        "cancels. render_to_real is the CONFOUNDED number the gate currently "
        "reports (real faces), kept only for direct comparison; at low render "
        "quality all render embeddings collapse toward the mean and it will sit "
        "near chance."
    )
    results["notes"].append(
        "Absolute cosine is not a valid identity threshold on this instrument "
        "(own vs best-other margin is tiny): read rank and margin, and the "
        "bootstrap CI on R@1."
    )
    results["provenance"] = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.uname().nodename,
        "interpreter": sys.executable,
        "torch": torch.__version__,
        "webdataset": WEBDATASET_STATUS,   # "stubbed" -> fail-closed placeholder
        "git_commit": _git_commit(),
        "script": str(Path(__file__).resolve()),
        "basis_dir": str(BASIS_DIR),
    }

    # ---- summary table --------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY -- render-to-render (quality-matched) vs render-to-real")
    print("=" * 72)
    print(f"renders={renders_total}  measured={n_meas}  no_face={no_face_total} "
          f"({0.0 if not renders_total else no_face_total / renders_total:.4f})")
    for key in ("render_to_render", "render_to_real"):
        r = results["retrievals"][key]
        ci = r["bootstrap_R@1"]
        mg = r["margin_own_minus_best_other"]
        r1 = "n/a" if r["R@1"] is None else f"{r['R@1']:.4f}"
        r5 = "n/a" if r["R@5"] is None else f"{r['R@5']:.4f}"
        cil = "n/a" if ci["ci_low"] is None else f"{ci['ci_low']:.4f}"
        cih = "n/a" if ci["ci_high"] is None else f"{ci['ci_high']:.4f}"
        print(f"\n  [{key}]  index={r['index']}")
        print(f"    pool={r['pool_size']}  queries={r['n_queries']}  "
              f"chance_R@1={r['chance_R@1']:.4f}")
        print(f"    R@1={r1}  [95% CI {cil} .. {cih}, {ci['reps']} reps]   R@5={r5}")
        print(f"    median_rank={r['median_rank']}  "
              f"frac_own_ge_best_other={r['frac_own_ge_best_other']}")
        print(f"    margin own-best_other: mean={mg['mean']} median={mg['median']}")
    for note in results["notes"]:
        print(f"  note: {note}")

    results_path = out / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults.json -> {results_path}")
    print(f"contact sheet -> {cs_path}")
    print("\nDONE.")
    return 0


def _jsonable(rec):
    """Make a render record JSON-serializable (numpy -> list)."""
    out = {}
    for k, v in rec.items():
        if k == "lda_unit":
            out["lda_unit"] = None if v is None else [float(x) for x in v]
        elif isinstance(v, np.ndarray):
            out[k] = [float(x) for x in v]
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


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
