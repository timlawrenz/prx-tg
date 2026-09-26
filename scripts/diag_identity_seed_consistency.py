#!/usr/bin/env python3
"""DIAGNOSTIC: has `eidolon-identity-renderer` (Arm EIR) actually LEARNED its
IDENTITY stream?

Why this exists
---------------
The sibling diagnostic `scripts/diag_eidolon_zg_dilution.py` showed that this
arm's geometry stream (z_g, 50-d, cross-attention) does not visibly control head
pose at step 3000 under any dual-CFG setting. So geometry cannot be used to
demonstrate anything about this checkpoint, and any gate that depends on z_g
binding is unmeasurable here.

The identity stream is a different and independent question, and it is
answerable without geometry working:

    Hold the identity conditioning vector AND z_g FIXED.
    Vary ONLY the initial noise seed.
    If the model learned identity, every seed decodes to the SAME person.
    If it ignored the identity stream (or memorised per-image keys), the seeds
    decode to different people.

This is a valid win condition that does not depend on geometry working. It is
the "same identity, different noise" axis, mirroring the gate's existing
"same identity, different z_g" probe.

Test
----
For each requested validation sample index (default [10, 30, 50]) the script
renders one image per noise seed (default 6 seeds, 0..N-1) with
`torch.manual_seed(seed)` reset immediately before EVERY generation, exactly the
noise-matched convention `diag_eidolon_zg_dilution.py` uses. Everything else
(identity vector, z_g, CFG scales, steps, shape, dtype) is byte-identical across
the seeds of a sample.

Measurement
-----------
Every rendered panel is embedded by the project's own SHARED AuraFace identity
instrument (`/home/tim/source/activity/eidolon/tools/auraface` ->
`extract_auraface`), then cleaned and projected into the 64-d LDA identity space
with the SAME recipe `scripts/batch_extract_eidolon_data.py` uses
(`clean_auraface` -> `project_to_lda`). For each sample the script reports:

  * the ground-truth LDA of that sample -- the identity vector the validation
    loader hands the model for it (`auraface_lda.npy`, via the loader);
  * each seed render's own LDA vector;
  * pairwise cosine between renders of the SAME identity (the headline number);
  * cosine from each render to its OWN ground-truth identity;
  * R@1 -- whether each render's nearest neighbour among the pool of LOADED
    validation samples' identity vectors is its own source identity, plus R@5,
    the ranking margin, and the chance rate (1 / pool size);
  * a cross-identity control: cosine from each render to every OTHER sample's
    ground truth (mean / p95 / max). Identity here has a razor-thin margin
    (README of the instrument: own 0.9985 vs best-other 0.9970), so a raw
    cosine is meaningless without this control and without the rank;
  * explicit `no_face` counts. A face-filling crop can return `no_face` from the
    detector; that render is recorded as `ok=false` with its outcome and
    EXCLUDED from every cosine. A guess is NEVER substituted.

Outputs (under --out-dir)
-------------------------
    results.json                          machine-readable, incl. provenance
    s{idx:02d}_seed{s:02d}.png            individual panels
    contact_rows=samples[...]_cols=seeds[...].png
                                          one ROW per sample, one COLUMN per seed

Interpreter (IMPORTANT)
-----------------------
This script needs BOTH the prx-tg rendering stack (torch / diffusers /
safetensors / webdataset) and insightface (what the shared AuraFace instrument
uses). Measured on this host, no venv has both:

    prx-tg/.venv            webdataset YES   insightface NO
    eidolon/.venv           webdataset NO    insightface NO
    stratum-lora/.venv-cuda webdataset NO    insightface YES  <-- run this one

`.venv-cuda` is the only interpreter that can embed with the shared instrument,
and the training venv must not be mutated (that would endanger training
reproducibility). The ONE package it lacks, `webdataset`, is imported at module
scope by `production/data.py` but only ever USED in that module's
webdataset-shard branch (lines ~184-186); the `source="stratum"` validation
branch this diagnostic uses never touches it. So `_ensure_webdataset()` installs
a fail-closed placeholder before that import: any actual attribute use raises
with an explanatory message -- it cannot change behaviour silently, it can only
fail loudly. Run with:

    /home/tim/source/activity/stratum-lora/.venv-cuda/bin/python \\
        scripts/diag_identity_seed_consistency.py --device cuda

(Alternatively `pip install webdataset` into that venv and delete the shim.)

`--check-paths` resolves every config / checkpoint / artifact / model path and
exits WITHOUT touching CUDA or loading a checkpoint, so the wiring can be
verified first. It runs under any interpreter, including prx-tg's own venv.

Recognizer selection (which ONNX file, and why)
-----------------------------------------------
`/mnt/nas-ai-models/models/auraface/` holds exactly:

    glintr100.onnx          260 MB  <- THE RECOGNIZER (512-d embedding)
    scrfd_10g_bnkps.onnx     17 MB  <- the DETECTOR (SCRFD-10GF + 5 keypoints),
                                       also used for the alignment crop
    1k3d68.onnx             144 MB  <- auxiliary: 3D-68 landmarks. NOT identity.
    2d106det.onnx             5 MB  <- auxiliary: 106 2D landmarks. NOT identity.
    genderage.onnx          1.3 MB  <- auxiliary: gender/age. NOT identity.

`glintr100.onnx` is the recognition net inside insightface's `auraface` model
pack: it is the only file that emits the 512-d `normed_embedding` this project's
LDA basis was fit on (`fal/AuraFace-v1` = glintr100, corroborated by
stratum-lora/library/auraface_utils.py, which names `glintr100.onnx` as the
AuraFace ONNX). `2d106det` / `1k3d68` / `genderage` are never loaded by
FaceAnalysis for this pack, and none of them produce an identity embedding. The
instrument is addressed through the shared pack loader
(`tools.auraface.extract.extract_auraface`, whose `_get_app` does
`FaceAnalysis(name="auraface", root="/mnt/nas-ai-models")`), so this script selects
it by using that instrument rather than by naming the file itself -- and records
`model_dir` + `model_sha256` in results.json via `describe_instrument()`.

Detection convention: `det_size = CORPUS_DET_SIZE = None`, i.e. insightface's
default 640x640 -- the CORPUS behaviour, which is what every stored
`auraface_lda.npy` in this project was produced with. Passing a tuple would
reproduce the FFHQ path and would be a DIFFERENT instrument (see
scripts/eidolon_det_size_probe.py), which is why it is not done here.

Provenance of the copied API calls
----------------------------------
* Model construction, checkpoint load, validation-dataloader call WITH the
  identity-basis guard values, latent-space shape selection (input_size*8 /
  resolve_latent_space), VAE decode (`decode_latents`, `load_vae_decoder`,
  `tensor_to_pil`), sampler invocation (`text_scale` / `dino_scale` /
  `prediction_type`), the noise-matched `torch.manual_seed` reset and the
  contact-sheet `grid()` builder:
      scripts/diag_eidolon_zg_dilution.py   <- reused verbatim where possible
* AuraFace -> LDA identity recipe (`auraface_preprocess.npz` pooled_mean /
  pc1_direction / yaw_direction, `auraface_lda.npz` lda_basis / pooled_mean,
  `clean_auraface`, `project_to_lda`, and "L2-normalize the query before any
  cosine" -- trap 6):
      scripts/batch_extract_eidolon_data.py
* The AuraFace instrument itself (model pack, det_size convention, padding
  retry, per-image outcome, `normed_embedding` 512-d):
      /home/tim/source/activity/eidolon/tools/auraface/extract.py
      (`extract_auraface`, `describe_instrument`, `CORPUS_DET_SIZE`)

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
from PIL import Image

# Repo root on sys.path, exactly as scripts/diag_eidolon_zg_dilution.py does.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# The AuraFace identity instrument lives in the sibling eidolon repo. APPEND it,
# not insert(0): the prx-tg repo root must keep winning name resolution for
# `production` (a regular package here) and for the `scripts`/`experiments`
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
    identity-instrument interpreter (.venv-cuda) does not ship webdataset, so
    without this the whole module fails to import for a package it never calls.

    The placeholder is FAIL-CLOSED: fetching any attribute raises, so a future
    code path that does need webdataset cannot be silently mis-served -- it
    fails with a message naming the fix.

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


WEBdataset_STATUS = _ensure_webdataset()

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
DEFAULT_OUT_DIR = REPO / "_diag_identity_seed_consistency"

SAMPLE_INDICES_DEFAULT = [10, 30, 50]
SEEDS_DEFAULT = 6
NUM_STEPS_DEFAULT = 50

# LDA basis artifacts. The same directory scripts/batch_extract_eidolon_data.py
# reads (`ARTIFACT_DIR`) and the same one tools/auraface/extract.py pins hashes
# for (`BASIS_DIR`).
BASIS_DIR = Path("/home/tim/source/activity/eidolon/experiments/geometry_pca/output")
PREPROCESS_NPZ = BASIS_DIR / "auraface_preprocess.npz"
LDA_NPZ = BASIS_DIR / "auraface_lda.npz"

# The AuraFace model pack. `auraface` is an insightface pack name under
# /mnt/nas-ai-models; the instrument resolves
# /mnt/nas-ai-models/models/auraface/ itself.
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
# (which took it from scripts/diag_pose_cfg_sweep.py)
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
    # Training weights, NOT ema (matches diag_eidolon_zg_dilution.py /
    # diag_pose_cfg_sweep.py / visual_debug).
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Validation samples -- copied faithfully from scripts/diag_eidolon_zg_dilution.py,
# with the sample-index list parametrised by the caller.
# ---------------------------------------------------------------------------
def load_samples(config, device, latent_space: bool, sample_indices):
    """Grab `sample_indices` from the deterministic validation dataloader.

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

    maxi = max(sample_indices)
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
# AuraFace -> 64-d LDA identity instrument
# ---------------------------------------------------------------------------
# Recipe copied faithfully from scripts/batch_extract_eidolon_data.py
# (load reference artifacts -> clean -> project). The ONLY addition is
# `l2n()`: `project_to_lda` returns an UNNORMALIZED vector (~norm 150) while the
# identity vectors on disk in this tree are unit-norm (norm 1.0), so every
# cosine is taken after L2-normalizing BOTH sides. Skipping this makes the
# cosine meaningless (~150x scale mismatch) -- documented trap 6 in
# scripts/eidolon_identity_retrieval.py and the instrument README.
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
            det_size=CORPUS_DET_SIZE,   # None == insightface default 640x640,
                                        # the CORPUS convention this project uses
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
# Contact sheet -- copied from scripts/diag_eidolon_zg_dilution.py (grid()).
# One COLUMN per seed, one ROW per sample index.
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
# Ranking helpers
# ---------------------------------------------------------------------------
def rank_of_own(q_unit, pool_unit, own_idx):
    """Rank (1-based) of own identity among the pool, plus (own_cos, best_other_cos)."""
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


# ---------------------------------------------------------------------------
# --check-paths: resolve everything, touch no CUDA, load no checkpoint
# ---------------------------------------------------------------------------
def check_paths(config_path: Path, ckpt_arg, out_dir: Path):
    print("=" * 72)
    print("PATH CHECK (no CUDA, no checkpoint load, no model construction)")
    print("=" * 72)
    print(f"  interpreter : {sys.executable}")
    print(f"  webdataset  : {WEBdataset_STATUS} "
          f"(stubbed == fail-closed placeholder; the stratum branch never uses it)")
    problems = []

    def report(label, path: Path, must_be_file=None):
        exists = path.exists()
        kind = "-"
        if exists:
            kind = "file" if path.is_file() else "dir"
        ok = exists and (must_be_file is None or (path.is_file() == must_be_file))
        print(f"  {'OK ' if ok else 'MISSING'} {label:28s} [{kind:4s}] {path}")
        if not ok:
            problems.append(f"{label}: {path}")
        return ok

    report("config", Path(config_path), must_be_file=True)
    if ckpt_arg:
        report("checkpoint (explicit)", Path(ckpt_arg), must_be_file=True)
    else:
        report("checkpoint dir", DEFAULT_CKPT_DIR)
        try:
            print(f"  -> would pick {find_default_checkpoint(DEFAULT_CKPT_DIR)}")
        except Exception as e:
            print(f"  -> NO CHECKPOINT: {e}")
            problems.append(f"checkpoint dir: {e}")
    report("out dir (parent)", out_dir.parent, must_be_file=False)
    for label, p in (("preprocess npz", PREPROCESS_NPZ), ("lda npz", LDA_NPZ)):
        report(label, p, must_be_file=True)
    report("auraface model dir", AURAFACE_DIR, must_be_file=False)
    for f in ("glintr100.onnx", "scrfd_10g_bnkps.onnx"):
        report(f"auraface/{f}", AURAFACE_DIR / f, must_be_file=True)
    report("eidolon instrument root", EIDOLON_ROOT / "tools" / "auraface",
           must_be_file=False)

    # Try to resolve the config -> stratum_dir without building a model.
    if Path(config_path).exists():
        try:
            cfg = load_config(str(config_path))
            stratum_dir = Path(cfg.data.stratum_dir)
            report("config.data.stratum_dir", stratum_dir, must_be_file=False)
            n = len(os.listdir(stratum_dir)) if stratum_dir.exists() else 0
            print(f"  -> {n} stratum dirs under {stratum_dir}")
            print(f"  -> adapter={getattr(cfg.adapter, 'name', None)} "
                  f"prediction_type={cfg.model.prediction_type} "
                  f"latent_space={resolve_latent_space(cfg)} "
                  f"gamma={resolve_sampling_gamma(cfg)}")
        except Exception as e:
            print(f"  MISSING config load: {type(e).__name__}: {e}")
            problems.append(f"config load: {e}")

    # Instrument import + basis verification (CPU only; no insightface app,
    # because _get_app() is lazy inside tools.auraface.extract).
    try:
        prov = describe_instrument(det_size=CORPUS_DET_SIZE)
        print(f"  OK  instrument: det_size={prov['det_size']} "
              f"model_dir={prov['model_dir']} "
              f"basis_fingerprint={prov['basis_fingerprint']} "
              f"(expected {prov['expected_basis_fingerprint']})")
    except Exception as e:
        print(f"  MISSING instrument: {type(e).__name__}: {e}")
        problems.append(f"instrument: {e}")

    print("-" * 72)
    print(f"PROBLEMS: {len(problems)}")
    for p in problems:
        print(f"  - {p}")
    return 1 if problems else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=str, default=str(CONFIG_PATH),
                    help=f"arm config (default: {CONFIG_PATH})")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="checkpoint .pt (default: most recently modified "
                         "checkpoint_step*.pt / checkpoint_interrupt.pt under "
                         f"{DEFAULT_CKPT_DIR})")
    ap.add_argument("--seeds", type=int, default=SEEDS_DEFAULT,
                    help=f"number of noise seeds per sample (default {SEEDS_DEFAULT})")
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first noise seed; seeds are seed_start .. seed_start+N-1")
    ap.add_argument("--samples", type=int, nargs="+", default=SAMPLE_INDICES_DEFAULT,
                    help=f"validation sample indices (default {SAMPLE_INDICES_DEFAULT})")
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                    help=f"output directory (default {DEFAULT_OUT_DIR})")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num-steps", type=int, default=NUM_STEPS_DEFAULT,
                    help=f"Euler steps (default {NUM_STEPS_DEFAULT}, == arm config)")
    ap.add_argument("--text-scale", type=float, default=None,
                    help="IDENTITY guidance scale (default: config.sampling.text_scale)")
    ap.add_argument("--dino-scale", type=float, default=None,
                    help="GEOMETRY guidance scale (default: config.sampling.dino_scale)")
    ap.add_argument("--check-paths", action="store_true",
                    help="resolve config/checkpoint/artifact/model paths and exit; "
                         "no CUDA, no checkpoint load")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if args.check_paths:
        return check_paths(Path(args.config), args.checkpoint, out)

    out.mkdir(parents=True, exist_ok=True)

    sample_indices = list(args.samples)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    if not seeds:
        raise SystemExit("--seeds must be >= 1")

    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else find_default_checkpoint(DEFAULT_CKPT_DIR)
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    print("=" * 72)
    print("EIDOLON IDENTITY-STREAM LEARNING DIAGNOSTIC (same identity, N noise seeds)")
    print("=" * 72)
    print(f"  config     : {args.config}")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  out-dir    : {out}")
    print(f"  samples    : {sample_indices}")
    print(f"  seeds      : {seeds}")
    print(f"  steps      : {args.num_steps}")

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
    # "dino" branch IS geometry-only (production/sample.py dual-CFG block, and the
    # arm config's `sampling:` comment), so these two numbers are the identity and
    # geometry guidance scales.
    print(f"  identity_scale(text_scale)={id_scale}  "
          f"geometry_scale(dino_scale)={geo_scale}")

    # ---- identity instrument (CPU; insightface app is loaded lazily on first
    # embed, and onnxruntime is pinned to CPUExecutionProvider by the instrument)
    print("  loading AuraFace instrument provenance...", flush=True)
    instrument_prov = describe_instrument(det_size=CORPUS_DET_SIZE)
    lda_inst = AuraFaceLDA()
    print(f"  instrument: det_size={instrument_prov['det_size']} "
          f"basis_fingerprint={instrument_prov['basis_fingerprint']} "
          f"embedding={instrument_prov['embedding']}")

    model = build_model(config, ckpt_path, device)
    print(f"  model: {sum(p.numel() for p in model.parameters()):,} params")

    vae = load_vae_decoder(device=device) if latent_space else None

    samples = load_samples(config, device, latent_space, sample_indices)
    print(f"  loaded {len(samples)} validation samples "
          f"(R@1 pool size = {len(samples)}, chance = {1.0 / len(samples):.4f})")

    # ---- the identity pool: every LOADED sample's own ground-truth identity ----
    pool_ids, pool_vecs = [], []
    for i, s in enumerate(samples):
        pool_ids.append(str(s["image_id"]))
        pool_vecs.append(l2n(s["identity_emb"].detach().cpu().numpy()))
    pool_unit = np.stack(pool_vecs)                      # (N, 64) unit-norm
    # Sanity: the pool vectors on this tree ship unit-norm on disk; record it.
    raw_norms = [float(np.linalg.norm(s["identity_emb"].detach().cpu().numpy()))
                 for s in samples]
    print(f"  pool identity raw norms: min={min(raw_norms):.4f} "
          f"max={max(raw_norms):.4f}")

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
        "diagnostic": "eidolon_identity_seed_consistency",
        "question": ("does the identity stream bind? hold identity + z_g fixed, "
                     "vary only the noise seed"),
        "config": str(args.config),
        "checkpoint": str(ckpt_path),
        "sample_indices": sample_indices,
        "pool_ids": pool_ids,
        "pool_size": len(samples),
        "chance_R@1": 1.0 / len(samples),
        "seeds": seeds,
        "num_steps": args.num_steps,
        "identity_scale": id_scale,
        "geometry_scale": geo_scale,
        "gamma": gamma,
        "latent_space": latent_space,
        "instrument": instrument_prov,
        "conventions": {
            "noise": ("torch.manual_seed(seed) reset immediately before EVERY "
                      "generation (same convention as diag_eidolon_zg_dilution.py)"),
            "auraface_input": "(H,W,3) BGR uint8 (cv2.imread order, via PNG path)",
            "det_size": CORPUS_DET_SIZE,
            "embedding": "normed_embedding (512-d)",
            "lda": ("clean_auraface (remove pooled_mean, pc1, yaw; L2-renorm) -> "
                    "project_to_lda (lda_basis 512x64) -> L2-NORMALIZED query "
                    "before any cosine (trap 6)"),
            "ground_truth": ("the identity vector the validation loader returns "
                             "for the sample (auraface_lda.npy), L2-normalized"),
            "no_face": ("extract_auraface outcome no_face -> lda_unit=null, "
                        "excluded from every cosine; never substituted"),
        },
        "notes": [],
        "per_sample": [],
    }

    renders_total = 0
    no_face_total = 0
    padding_total = 0
    ambiguous_total = 0
    all_pair_cos = []          # within-sample, same identity, different seed
    all_cos_to_gt = []
    all_ranks = []
    all_margins = []
    all_ctrl_cos = []          # cross-identity control (wrong sample's GT)
    all_cos_vs_helper = []
    grid_rows = []

    for si in sample_indices:
        s = samples[si]
        idv = s["identity_emb"].unsqueeze(0).to(device)
        baseg = s["geometry_emb"].unsqueeze(0).to(device)
        gt_unit = pool_unit[si]
        print(f"\n######## sample {si} (image_id={s['image_id']}) ########")

        entry = {
            "sample_idx": si,
            "image_id": str(s["image_id"]),
            "gt_lda_unit": [float(x) for x in gt_unit],
            "gt_norm_raw": raw_norms[si],
            "renders": [],
        }
        row_tensors = []
        render_units = []          # (seed, unit) for ok renders only

        for seed in seeds:
            # Noise-matched: reset the same fixed seed before EVERY generation;
            # identity_emb and z_g are byte-identical across the seeds here.
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
            row_tensors.append(img_t)

            panel_path = out / f"s{si:02d}_seed{seed:02d}.png"
            tensor_to_pil(img_t).save(panel_path)

            m = lda_inst.embed(panel_path)
            renders_total += 1
            rec = {"seed": seed, "panel": panel_path.name, **m}
            if m["lda_unit"] is None:
                no_face_total += 1
                print(f"   seed {seed:02d}: NO FACE ({m['outcome']}) "
                      f"-> excluded, no value substituted  [{panel_path.name}]")
            else:
                if m["used_padding"]:
                    padding_total += 1
                if m["ambiguous"]:
                    ambiguous_total += 1
                u = m["lda_unit"]
                rank, own_cos, best_other = rank_of_own(u, pool_unit, si)
                rec.update({
                    "rank": rank,
                    "cos_to_gt": own_cos,
                    "best_other_cos": best_other,
                    "margin_own_minus_best_other": own_cos - best_other,
                })
                # cross-identity control: this render vs every OTHER sample's GT
                ctrl = np.delete(pool_unit @ u, si)
                rec["cos_to_best_other_gt"] = (
                    float(ctrl.max()) if ctrl.size else None
                )
                render_units.append((seed, u))
                all_cos_to_gt.append(own_cos)
                all_ranks.append(rank)
                all_margins.append(own_cos - best_other)
                if ctrl.size:
                    all_ctrl_cos.append(float(ctrl.max()))
                if m["cos_vs_helper_lda"] is not None:
                    all_cos_vs_helper.append(m["cos_vs_helper_lda"])
                print(f"   seed {seed:02d}: outcome={m['outcome']:22s} "
                      f"n_faces={m['n_faces']} |lda_raw|={m['lda_norm_raw']:.1f} "
                      f"cos_to_gt={own_cos:+.4f} rank={rank}/{len(samples)} "
                      f"margin={own_cos - best_other:+.4f} "
                      f"[{panel_path.name}]")
            entry["renders"].append(_jsonable(rec))

        # ---- pairwise cosine between renders of the SAME identity ----
        pair_cos = []
        for a in range(len(render_units)):
            for b in range(a + 1, len(render_units)):
                c = float(render_units[a][1] @ render_units[b][1])
                pair_cos.append(c)
                all_pair_cos.append(c)
        entry["within_sample_pairwise_cos"] = {
            "pairs": pair_cos,
            "avg": stat(pair_cos),
        }
        entry["cos_to_gt"] = stat([r.get("cos_to_gt") for r in entry["renders"]])
        entry["no_face_renders"] = sum(
            1 for r in entry["renders"] if not r["ok"]
        )
        entry["n_renders"] = len(entry["renders"])
        entry["n_ranked"] = len(render_units)
        entry["gt_rank_all_renders_1"] = (
            all(r.get("rank") == 1 for r in entry["renders"] if r["ok"])
            if render_units else None
        )
        if pair_cos:
            pair_txt = (f"mean {np.mean(pair_cos):+.4f} min {min(pair_cos):+.4f} "
                        f"max {max(pair_cos):+.4f}")
        else:
            pair_txt = "n/a (fewer than 2 measured renders)"
        print(f"   -> same-identity pairwise cos: {pair_txt}"
              f"   no_face={entry['no_face_renders']}/{entry['n_renders']}")

        results["per_sample"].append(entry)
        grid_rows.append(row_tensors)

    # ---- contact sheet: one row per sample, one column per seed ----
    g_path = (
        out
        / f"contact_rows=samples[{','.join(str(i) for i in sample_indices)}]"
          f"_cols=seeds[{','.join(str(s) for s in seeds)}].png"
    )
    grid(grid_rows).save(g_path)
    print(f"\nCONTACT SHEET -> {g_path}")

    # ---- aggregate ----
    ranks = np.asarray(all_ranks, dtype=np.int64) if all_ranks else np.zeros(0, np.int64)
    results["summary"] = {
        "renders_total": renders_total,
        "renders_measured": int(ranks.size),
        "no_face_renders": no_face_total,
        "no_face_rate": (no_face_total / renders_total) if renders_total else None,
        "used_padding_renders": padding_total,
        "ambiguous_renders": ambiguous_total,
        "within_sample_pairwise_cos": stat(all_pair_cos),
        "cos_to_own_gt": stat(all_cos_to_gt),
        "cross_identity_control_cos_best_other_gt": stat(all_ctrl_cos),
        "R@1": (float((ranks == 1).mean()) if ranks.size else None),
        "R@5": (float((ranks <= 5).mean()) if ranks.size else None),
        "median_rank": (float(np.median(ranks)) if ranks.size else None),
        "chance_R@1": 1.0 / len(samples),
        "margin_own_minus_best_other": stat(all_margins),
        "frac_own_ge_best_other": (
            float(np.mean(np.asarray(all_margins) >= 0)) if all_margins else None
        ),
        "cos_query_vs_instrument_lda": stat(all_cos_vs_helper),
    }
    if no_face_total:
        results["notes"].append(
            f"{no_face_total}/{renders_total} renders returned no_face from the "
            "AuraFace detector (a face-filling crop can legitimately do this); "
            "their LDA is null and they are excluded from every cosine and from "
            "R@1. No value was substituted."
        )
    if all_cos_vs_helper and min(all_cos_vs_helper) < 0.99999:
        results["notes"].append(
            "the inline clean->project recipe and the instrument's own "
            "lda_coords disagree (min cosine "
            f"{min(all_cos_vs_helper):.6f}); the basis artifacts may have moved."
        )
    results["notes"].append(
        "R@1 is against the pool of LOADED validation samples "
        f"({len(samples)} identities, chance {1.0 / len(samples):.4f}). Absolute "
        "cosine is NOT a valid identity threshold on this instrument (own vs "
        "best-other margin is ~0.0015): read rank and margin, with the "
        "cross-identity control above as the floor."
    )
    results["provenance"] = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.uname().nodename,
        "interpreter": sys.executable,
        "torch": torch.__version__,
        "webdataset": WEBdataset_STATUS,   # "stubbed" -> fail-closed placeholder
        "git_commit": _git_commit(),
        "script": str(Path(__file__).resolve()),
    }

    # ---- summary table ----
    print("\n" + "=" * 72)
    print("SUMMARY -- same identity, different noise seed")
    print("=" * 72)
    print(f"renders={renders_total}  measured={int(ranks.size)}  "
          f"no_face={no_face_total}  pool={len(samples)}  "
          f"chance_R@1={1.0 / len(samples):.4f}")
    sm = results["summary"]
    for label, key in (
        ("within-sample pairwise cos", "within_sample_pairwise_cos"),
        ("cos to own ground truth", "cos_to_own_gt"),
        ("cross-identity control (best other GT)",
         "cross_identity_control_cos_best_other_gt"),
        ("margin own - best other", "margin_own_minus_best_other"),
    ):
        st = sm[key]
        if st["n"] == 0:
            print(f"  {label:38s} n=0")
        else:
            print(f"  {label:38s} n={st['n']:3d} mean={st['mean']:+.4f} "
                  f"min={st['min']:+.4f} max={st['max']:+.4f} p95={st['p95']:+.4f}")
    r1 = "n/a" if sm["R@1"] is None else f"{sm['R@1']:.4f}"
    print(f"  {'R@1 (own identity at rank 1)':38s} {r1}")
    print(f"  {'median rank':38s} {sm['median_rank']}")
    for note in results["notes"]:
        print(f"  note: {note}")

    results_path = out / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults.json -> {results_path}")
    print(f"contact sheet -> {g_path}")
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
