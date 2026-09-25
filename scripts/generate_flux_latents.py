#!/usr/bin/env python3
"""Encode all 70k FFHQ stratum images to FLUX latents, stored alongside satellite data.

Reads:  /mnt/nas-ai-models/training-data/ffhq/stratum/<id>/pixel.npy   (3,1024,1024) f16 [0,1]
Writes: /mnt/nas-ai-models/training-data/ffhq/stratum/<id>/flux_latent.npy  (16,128,128) f16
        via atomic tmp+rename; resumable (skips existing flux_latent.npy).
Uses:   /mnt/models/vae/ae.safetensors (FLUX AE) via comfy's VAE loader.

Run via GPU scheduler (4090 only; FLUX AE is CUDA). Example:
  request --gpu 4090 --project prx-tg-flux-latents --vram 12 --duration 12h --job-id ...
"""
import argparse, gc, os, sys, time
import numpy as np
import torch

STRATUM = "/mnt/nas-ai-models/training-data/ffhq/stratum"
AE_PATH = "/mnt/models/vae/ae.safetensors"
sys.path.insert(0, "/mnt/fscache/essdee/ComfyUI")

import comfy.sd as csd
import comfy.utils as cu
import comfy.model_management as mm


def encode_pixels(vae, pix, batch=16):
    """pix: (B,3,1024,1024) float32 [0,1] CUDA. Returns (B,16,128,128) float16 CUDA."""
    x = pix.permute(0, 2, 3, 1).contiguous()  # comfy wants (B,H,W,C)
    out = []
    for i in range(0, len(x), batch):
        z = vae.encode(x[i:i+batch])
        if isinstance(z, tuple):
            z = z[0]
        out.append(z)
    return torch.cat(out, dim=0).to(torch.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only first N dirs (0=all)")
    ap.add_argument("--skip", type=int, default=0, help="skip first N already-valid dirs (resume)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sleep-s", type=float, default=0.0, help="insert sleep between dirs (rate-limit NAS)")
    ap.add_argument("--nvidia-free-floor", type=float, default=5.0, help="GB; pause if other procs consume below this")
    ap.add_argument("--root", type=str, default=STRATUM,
                    help="dataset root holding per-sample dirs (default: FFHQ stratum). "
                         "Each dir must contain pixel.npy (3,1024,1024) f16.")
    ap.add_argument("--request-from", type=str, default="", help="scheduler job id (heartbeat uses it)")
    args = ap.parse_args()

    ROOT = os.path.abspath(args.root)
    if not os.path.isdir(ROOT):
        print(f"[fatal] root does not exist: {ROOT}")
        sys.exit(2)

    dirs = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
    print(f"[init] root={ROOT}")
    print(f"[init] {len(dirs)} sample dirs")

    vae = csd.VAE(sd=cu.load_torch_file(AE_PATH))
    print("[vae] FLUX AE loaded")

    done, skipped, failed = 0, 0, 0
    started = time.time()
    for i, d in enumerate(dirs):
        if args.limit and i >= args.limit:
            break
        dp = os.path.join(ROOT, d)
        out_np = os.path.join(dp, "flux_latent.npy")
        tmp_np = out_np + ".tmp.npy"  # np.save appends .npy; use .tmp.npy so replace target is clean
        if os.path.exists(out_np):  # resume
            skipped += 1
            continue
        pxl = os.path.join(dp, "pixel.npy")
        if not os.path.exists(pxl):
            failed += 1
            continue
        try:
            pix = np.load(pxl, mmap_mode="r").astype(np.float32)  # (3,1024,1024)
            if pix.shape != (3, 1024, 1024):
                failed += 1
                continue
            t = torch.from_numpy(pix).unsqueeze(0).to("cuda")  # (1,3,1024,1024)
            with torch.no_grad():
                z = encode_pixels(vae, t, batch=args.batch)[0]  # (16,128,128) f16
            np.save(tmp_np, z.cpu().numpy())
            os.replace(tmp_np, out_np)
            done += 1
            del t, z
        except Exception as e:
            failed += 1
            if os.path.exists(tmp_np):
                os.remove(tmp_np)
            if i % 200 == 0:
                print(f"[err {d}] {type(e).__name__}: {str(e)[:150]}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        if args.sleep_s:
            time.sleep(args.sleep_s)
        if (i + 1) % 500 == 0:
            el = time.time() - started
            rate = (done + skipped) / max(el, 1)
            print(f"[prog] {i+1}/{len(dirs)} done={done} skip={skipped} fail={failed} rate={rate:.1f}/s ({el/60:.1f}m)")
    print(f"[done] processed; done={done} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()