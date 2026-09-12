#!/usr/bin/env python3
"""VAE CEILING EXPERIMENT — memory-safe, one image at a time, comfy-managed VRAM.

Q: does the FLUX AE (the one that made the precomputed latents) preserve enough
fidelity to clear our photoreal gate BEFORE committing to latent-first?

Tests (all on real FFHQ faces, deterministic subset):
  A) precomputed FLUX-AE latent -> FLUX AE decode   [the actual latent-first path]
     (scale variants x0.3611, x1.0, /0.3611 to lock the exact AE convention;
      the variant with lowest LPIPS vs the real image is the true ceiling)
  B) SDXL VAE encode -> SDXL VAE decode             [independent 2nd AE]
Metrics: LPIPS (alex) at 256px + pixel stats. GPU-light: batch 1, auto-tile,
empty_cache every iteration. Run with the COMFY venv.
"""
import argparse, gc, json, io, os, sys
import numpy as np
import torch
from PIL import Image

OUT = "/home/tim/source/activity/prx-tg/research/results/vae_ceiling"
os.makedirs(OUT, exist_ok=True)
FFHQ_RAW = "/mnt/nas-ai-models/training-data/ffhq/raw"
AE_FLUX = "/mnt/models/vae/ae.safetensors"
AE_SDXL = "/mnt/models/vae/SDXL/sdxl_vae.safetensors"
REPO = "AbstractPhil/ffhq_flux_latents_repaired"
SCALES = [0.3611, 1.0, 1 / 0.3611]


def fetch_latents(n, offset=0):
    """Fetch n latent rows via HF datasets-server API (no big download)."""
    import urllib.request
    url = (f"https://datasets-server.huggingface.co/rows?dataset={REPO}"
           f"&config=default&split=train_00&offset={offset}&length={n}")
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    rows = []
    for row in d.get("rows", []):
        r_ = row.get("row", {})
        arr = np.asarray(r_["latent"], dtype=np.float32)
        while arr.ndim > 4:
            arr = arr[0]
        rows.append(arr)
    return rows


def lpips_model():
    import lpips
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return lpips.LPIPS(net="alex").eval().to(dev)


def lpips_d(model, a, b):
    with torch.no_grad():
        return float(model(a, b).mean().item())


def to_lpips_t(im):
    im = im.resize((256, 256), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return t.to(dev)


def t_to_pil(t):
    t = t.detach().clone().float().cpu()
    while t.ndim > 3:
        t = t[0]
    if t.shape[0] == 3 and t.shape[-1] != 3:  # NCHW
        a = t.permute(1, 2, 0).numpy()
    elif t.shape[-1] == 3:  # NHWC
        a = t.numpy()
    else:
        # fallback: treat as 1-channel grayscale-ish
        a = t.squeeze(0).numpy()
    a = np.clip(a, 0, 1) * 255 if a.max() <= 1.5 else np.clip(a, 0, 255)
    return Image.fromarray(a.astype(np.uint8))


def pixel_stats(t):
    t = t.detach().float().cpu()
    while t.ndim > 3:
        t = t[0]
    if t.shape[0] == 3 and t.shape[-1] != 3:
        a = t.permute(1, 2, 0).numpy()
    else:
        a = t.numpy()
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return {"lum": float(lum.mean()), "contrast": float(lum.std())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--skip-sdxl", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, "/mnt/fscache/essdee/ComfyUI")
    import comfy.sd as csd
    import comfy.model_management as mm
    import comfy.utils as cu

    print(f"[mem] before: {mm.get_free_memory() / 1e6:.0f} MB free")
    vae_flux = csd.VAE(sd=cu.load_torch_file(AE_FLUX))
    vae_sdxl = None if args.skip_sdxl else csd.VAE(sd=cu.load_torch_file(AE_SDXL))
    loss = lpips_model()
    print("[vae] FLUX AE + SDXL AE loaded, LPIPS ready")

    lats = fetch_latents(args.n, args.offset)
    print(f"[data] fetched {len(lats)} latents (shape {lats[0].shape})")

    results = []
    for i, arr in enumerate(lats):
        idx = args.offset + i
        src = os.path.join(FFHQ_RAW, f"{idx:05d}.png")
        if os.path.exists(src):
            real = Image.open(src).convert("RGB")
            srclabel = os.path.basename(src)
        else:
            print(f"[warn] missing {src}; skipping")
            continue
        row = {"idx": idx, "source": srclabel}
        recons = {}

        # --- A) FLUX decode, scale variants ---
        t_lat = torch.from_numpy(arr).float().unsqueeze(0).to("cuda")  # (1,16,64,64)
        for sc in SCALES:
            try:
                z = (t_lat * sc) if sc != 1.0 else t_lat
                dec = vae_flux.decode(z.float())
                recons[f"flux_s{sc:g}"] = t_to_pil(dec)
                del dec, z; gc.collect()
            except Exception as e:
                print(f"[flux s={sc:g}] ERROR {type(e).__name__}: {str(e)[:120]}")
        del t_lat; gc.collect()

        # --- B) SDXL encode->decode (comfy API wants B,H,W,C) ---
        if vae_sdxl is not None:
            try:
                x = torch.from_numpy(np.asarray(real.resize((1024, 1024), Image.LANCZOS),
                                               dtype=np.float32)).unsqueeze(0).to("cuda")  # (1,H,W,3) fp32 [0,255]
                z_t = vae_sdxl.encode(x)
                z = z_t[0] if isinstance(z_t, tuple) else z_t
                dec = vae_sdxl.decode(z)
                recons["sdxl"] = t_to_pil(dec)
                del x, z_t, z, dec; gc.collect()
            except Exception as e:
                print(f"[sdxl] ERROR {type(e).__name__}: {str(e)[:140]}")

        # save + score
        real.save(os.path.join(OUT, f"{idx:03d}_real.png"))
        rt = to_lpips_t(real)

        # --- C) FLUX AE encode->decode at FULL 1024² (the true resolution ceiling) ---
        # comfy applies an internal scale; test scale variants at full res too
        try:
            xfull = torch.from_numpy(np.asarray(real.resize((1024, 1024), Image.LANCZOS),
                                                dtype=np.float32)).unsqueeze(0).to("cuda")  # (1,1024,1024,3)
            zf = vae_flux.encode(xfull)
            zf_t = zf[0] if isinstance(zf, tuple) else zf
            for sc in SCALES:
                zz = (zf_t * sc) if sc != 1.0 else zf_t
                decf = vae_flux.decode(zz)
                recons[f"flux_full1024_s{sc:g}"] = t_to_pil(decf)
                del decf, zz; gc.collect()
            del xfull, zf, zf_t; gc.collect()
        except Exception as e:
            print(f"[flux_full1024] ERROR {type(e).__name__}: {str(e)[:140]}")

        for name, im in recons.items():
            im.save(os.path.join(OUT, f"{idx:03d}_{name}.png"))
            it_ = to_lpips_t(im)
            row[f"{name}_lpips"] = lpips_d(loss, rt, it_)
            ps = pixel_stats(it_)
            row[f"{name}_lum"] = ps["lum"]
            row[f"{name}_contrast"] = ps["contrast"]
            del it_; gc.collect()
        del rt; gc.collect()

        results.append(row)
        print(f"[{i+1}/{len(lats)}] {srclabel}: " +
              " ".join(f"{k}={v:.3f}" for k, v in row.items() if isinstance(v, float)))
        torch.cuda.empty_cache()

    with open(os.path.join(OUT, "ceiling_metrics.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("\n=== SUMMARY (LPIPS mean/min/max, n=%d) ===" % len(results))
    for name in ["flux_s0.3611", "flux_s1", "flux_s2.7689", "flux_full1024_s0.3611",
                 "flux_full1024_s1", "flux_full1024_s2.7689", "sdxl"]:
        vals = [r.get(f"{name}_lpips") for r in results if r.get(f"{name}_lpips") is not None]
        if vals:
            print(f"{name:16s} mean={np.mean(vals):.4f} min={np.min(vals):.4f} max={np.max(vals):.4f}")
    print(f"\nartifacts -> {OUT}")


if __name__ == "__main__":
    main()