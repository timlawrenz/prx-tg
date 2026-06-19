#!/usr/bin/env python3
"""
Standalone text-to-image generation with a prx-tg NanoDiT checkpoint.

Usage:
    python scripts/txt2img.py \
        --checkpoint experiments/2026-06-04_2116/checkpoints/checkpoint_step0015000.pt \
        --prompt "A professional headshot of a woman with curly brown hair"

    python scripts/txt2img.py \
        --checkpoint experiments/2026-06-04_2116/checkpoints/checkpoint_step0015000.pt \
        --prompt "A cinematic portrait of a man" \
        --steps 35 --text-scale 4.0 --seed 42 --output out.png

Requirements:
    torch, transformers, pyyaml, pillow
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

# ── Add project root so we can import production.model ──────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from production.model import NanoDiT


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def strip_orig_mod(state_dict: dict) -> dict:
    """Strip torch.compile's _orig_mod. prefix from state dict keys."""
    return {
        k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
        for k, v in state_dict.items()
    }


def build_model(config: dict, device: torch.device) -> NanoDiT:
    """Build NanoDiT from config dict, matching run_checkpoint_validation.py."""
    mc = config.get("model", {})

    kwargs = {
        "input_size": mc.get("input_size", 1024),
        "patch_size": mc.get("patch_size", 16),
        "in_channels": mc.get("in_channels", 3),
        "hidden_size": mc.get("hidden_size", 768),
        "depth": mc.get("depth", 18),
        "num_heads": mc.get("num_heads", 12),
        "mlp_ratio": mc.get("mlp_ratio", 4.0),
        "use_gradient_checkpointing": False,
    }

    # Optional: REPA
    tc = config.get("training", {})
    if tc.get("repa", {}).get("enabled"):
        kwargs["repa_block_idx"] = tc["repa"].get("block_index", -1)

    # Optional: TREAD
    if tc.get("tread", {}).get("enabled"):
        kwargs["tread_route_start"] = tc["tread"].get("route_start", 1)
        kwargs["tread_route_end"] = tc["tread"].get("route_end", -1)
        kwargs["tread_routing_prob"] = tc["tread"].get("routing_probability", 0.5)

    # Optional: bottleneck / pose
    kwargs["bottleneck_size"] = mc.get("bottleneck_size", 0)
    kwargs["num_pose_joints"] = mc.get("num_pose_joints", 133)
    kwargs["pose_confidence_threshold"] = mc.get("pose_confidence_threshold", 0.05)

    model = NanoDiT(**kwargs)
    model = model.to(device)
    model.eval()
    return model


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> NanoDiT:
    """Load prx-tg checkpoint and return model with EMA weights applied."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = build_model(config, device)
    model.load_state_dict(strip_orig_mod(ckpt["model"]))

    # Apply EMA weights manually (avoids importing EMAModel)
    ema_sd = strip_orig_mod(ckpt["ema"])
    model_sd = model.state_dict()
    for key in ema_sd:
        if key in model_sd:
            model_sd[key].copy_(ema_sd[key])

    print(f"  Step: {ckpt.get('step', '?')}  |  "
          f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Text encoding (T5)
# ═══════════════════════════════════════════════════════════════════════════════

class T5Encoder:
    """Lazy-loading T5-large encoder for prx-tg text conditioning."""

    def __init__(self, device: torch.device):
        self.device = device
        self.tokenizer = None
        self.encoder = None

    def _load(self):
        if self.encoder is not None:
            return
        print("Loading T5-large encoder...", end=" ", flush=True)
        t0 = time.time()
        # Try local files first to avoid HF download
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("t5-large", local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("t5-large")

        try:
            self.encoder = T5EncoderModel.from_pretrained(
                "t5-large", torch_dtype=torch.float16, local_files_only=True
            )
        except Exception:
            self.encoder = T5EncoderModel.from_pretrained(
                "t5-large", torch_dtype=torch.float16
            )
        self.encoder.to(self.device)
        self.encoder.eval()
        print(f"done ({time.time() - t0:.1f}s)")

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (hidden_states, attention_mask) both float32, shape (1, 512, 1024)."""
        self._load()
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        attn_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attn_mask)
            hidden = outputs.last_hidden_state.float()  # (1, 512, 1024)

        return hidden, attn_mask


# ═══════════════════════════════════════════════════════════════════════════════
# Euler sampling (rectified flow)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample(
    model: NanoDiT,
    shape: tuple,
    text_emb: torch.Tensor,
    text_mask: torch.Tensor,
    text_scale: float = 3.0,
    dino_scale: float = 0.0,  # 0 = text-only mode
    num_steps: int = 35,
    prediction_type: str = "x_prediction",
    device: torch.device = None,
) -> torch.Tensor:
    """
    Euler integration for rectified flow (t=1.0 → t=0.0).

    Returns pixel-space RGB image in [-1, 1], shape (1, 3, H, W).
    """
    B = shape[0]
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    # Start from pure noise
    zt = torch.randn(shape, device=device)

    # Dummy DINO embeddings (zero influence when dino_scale=0)
    dino_emb = torch.zeros(B, 1024, device=device)
    dino_patches = torch.zeros(B, 256, 1024, device=device)  # safe default

    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_curr  # negative (moving toward data)

        t_batch = torch.full((B,), t_curr, device=device)

        # Dual CFG: 3 passes
        drop_all = torch.ones(B, dtype=torch.bool, device=device)
        keep_all = torch.zeros(B, dtype=torch.bool, device=device)

        # 1. Unconditional
        v_uncond = model(
            zt, t_batch, dino_emb, text_emb, dino_patches, text_mask,
            cfg_drop_text=drop_all, cfg_drop_dino_cls=drop_all, cfg_drop_dino_patches=drop_all,
            tread_enabled=False,
        )

        # 2. Text-only
        v_text = model(
            zt, t_batch, dino_emb, text_emb, dino_patches, text_mask,
            cfg_drop_text=keep_all, cfg_drop_dino_cls=drop_all, cfg_drop_dino_patches=drop_all,
            tread_enabled=False,
        )

        # 3. DINO-only (skipped if dino_scale=0, but still computed for correct
        #    null-conditioning path — the model sees null DINO + dropped text)
        v_dino = model(
            zt, t_batch, dino_emb, text_emb, dino_patches, text_mask,
            cfg_drop_text=drop_all, cfg_drop_dino_cls=keep_all, cfg_drop_dino_patches=keep_all,
            tread_enabled=False,
        )

        v_pred = v_uncond + text_scale * (v_text - v_uncond) + dino_scale * (v_dino - v_uncond)

        if prediction_type == "x_prediction":
            t_val = t_curr.clamp(min=0.05)
            v_euler = (zt - v_pred) / t_val
        else:
            v_euler = v_pred

        zt = zt + v_euler * dt

    return zt


# ═══════════════════════════════════════════════════════════════════════════════
# Image output
# ═══════════════════════════════════════════════════════════════════════════════

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert (3, H, W) in [-1, 1] to PIL Image."""
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    arr = tensor.cpu().float().numpy()
    arr = (arr * 255).astype("uint8")
    arr = arr.transpose(1, 2, 0)  # CHW → HWC
    return Image.fromarray(arr)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def find_config(checkpoint_path: str) -> dict | None:
    """
    Try to find the config file for a checkpoint.
    Looks in parent → parent's parent for config.yaml.
    """
    cp = Path(checkpoint_path).resolve()
    for ancestor in [cp.parent.parent, cp.parent.parent.parent]:
        config_path = ancestor / "config.yaml"
        if config_path.exists():
            return yaml.safe_load(config_path.read_text())
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Text-to-image generation with prx-tg NanoDiT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -c ckpt.pt -p "A portrait of a woman"
  %(prog)s -c ckpt.pt -p "A man with a beard" --steps 50 --text-scale 4.0
  %(prog)s -c ckpt.pt -p prompt.txt --seed 123 --output result.png
        """,
    )
    parser.add_argument("-c", "--checkpoint", required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("-p", "--prompt", required=True,
                        help="Text prompt, or path to a .txt file containing the prompt")
    parser.add_argument("--config",
                        help="Path to config.yaml (auto-detected from checkpoint dir if omitted)")
    parser.add_argument("--steps", type=int, default=0,
                        help="Number of sampling steps (default: from config, or 35)")
    parser.add_argument("--text-scale", type=float, default=None,
                        help="Text CFG scale (default: from config, or 3.0)")
    parser.add_argument("--dino-scale", type=float, default=None,
                        help="DINO CFG scale (default: from config, or 0.0 = text-only)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: random)")
    parser.add_argument("--width", type=int, default=1024,
                        help="Output width in pixels (default: 1024)")
    parser.add_argument("--height", type=int, default=1024,
                        help="Output height in pixels (default: 1024)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (default: txt2img_<timestamp>.png)")
    parser.add_argument("--device", default="cuda",
                        help="Device (default: cuda, fallback: cpu)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON with metadata to stdout")
    args = parser.parse_args()

    # ── Device ──────────────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU", file=sys.stderr)
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Config ──────────────────────────────────────────────────────────
    config = None
    if args.config:
        config = yaml.safe_load(Path(args.config).read_text())
    else:
        config = find_config(args.checkpoint)

    if config is None:
        print("Error: Could not find config.yaml. Use --config to specify.",
              file=sys.stderr)
        sys.exit(1)

    mc = config.get("model", {})
    sc = config.get("sampling", {})

    # ── Prompt (file or string) ─────────────────────────────────────────
    prompt_path = Path(args.prompt)
    if prompt_path.suffix == ".txt" and prompt_path.exists():
        prompt = prompt_path.read_text().strip()
    else:
        prompt = args.prompt

    # ── Seed ────────────────────────────────────────────────────────────
    seed = args.seed if args.seed is not None else torch.seed()
    torch.manual_seed(seed)

    # ── Load model ──────────────────────────────────────────────────────
    model = load_model(args.checkpoint, config, device)

    # ── Encode text ─────────────────────────────────────────────────────
    t5 = T5Encoder(device)
    text_emb, text_mask = t5.encode(prompt)

    # ── Sample ──────────────────────────────────────────────────────────
    prediction_type = mc.get("prediction_type", "x_prediction")
    num_steps = args.steps if args.steps else sc.get("num_steps", 35)
    text_scale = args.text_scale if args.text_scale is not None else sc.get("text_scale", 3.0)
    dino_scale = args.dino_scale if args.dino_scale is not None else sc.get("dino_scale", 0.0)

    print(f"Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print(f"Steps: {num_steps}  |  text_scale: {text_scale}  |  dino_scale: {dino_scale}")
    print(f"Resolution: {args.width}×{args.height}  |  Seed: {seed}")

    t0 = time.time()
    shape = (1, 3, args.height, args.width)

    result = sample(
        model, shape,
        text_emb=text_emb, text_mask=text_mask,
        text_scale=text_scale, dino_scale=dino_scale,
        num_steps=num_steps, prediction_type=prediction_type,
        device=device,
    )

    elapsed = time.time() - t0
    print(f"Generated in {elapsed:.1f}s ({elapsed / num_steps:.1f}s/step)")

    # ── Save ────────────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"txt2img_{ts}.png")

    img = tensor_to_pil(result[0])
    img.save(output_path)
    print(f"Saved: {output_path.resolve()}")

    # ── JSON metadata ───────────────────────────────────────────────────
    if args.json:
        meta = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "prompt": prompt,
            "seed": seed,
            "steps": num_steps,
            "text_scale": text_scale,
            "dino_scale": dino_scale,
            "width": args.width,
            "height": args.height,
            "time_seconds": round(elapsed, 1),
            "output": str(output_path.resolve()),
        }
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
