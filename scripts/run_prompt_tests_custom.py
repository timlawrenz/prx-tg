#!/usr/bin/env python3
"""Batch prompt test — matches txt2img.py approach exactly for reliable checkpoint comparison."""
import argparse
import torch
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from production.model import NanoDiT
from transformers import AutoTokenizer, T5EncoderModel
from PIL import Image
import numpy as np


def strip_orig_mod(sd):
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
            for k, v in sd.items()}


class T5Encoder:
    def __init__(self, device):
        self.device = device
        self.tokenizer = None
        self.encoder = None

    def _load(self):
        if self.encoder is not None:
            return
        print("Loading T5-large encoder...", end=" ", flush=True)
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
        print("done")

    def encode(self, text):
        self._load()
        inputs = self.tokenizer(
            text, max_length=512, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        attn_mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attn_mask)
            hidden = outputs.last_hidden_state.float()
        return hidden, attn_mask


@torch.no_grad()
def sample(model, shape, text_emb, text_mask, text_scale, dino_scale,
           num_steps, prediction_type, device):
    B = shape[0]
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    zt = torch.randn(shape, device=device)
    dino_emb = torch.zeros(B, 1024, device=device)
    dino_patches = torch.zeros(B, 256, 1024, device=device)

    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_curr
        t_batch = torch.full((B,), t_curr, device=device)
        drop_all = torch.ones(B, dtype=torch.bool, device=device)
        keep_all = torch.zeros(B, dtype=torch.bool, device=device)

        v_uncond = model(
            zt, t_batch, dino_emb, text_emb, dino_patches, text_mask,
            cfg_drop_text=drop_all, cfg_drop_dino_cls=drop_all,
            cfg_drop_dino_patches=drop_all, tread_enabled=False,
        )
        v_text = model(
            zt, t_batch, dino_emb, text_emb, dino_patches, text_mask,
            cfg_drop_text=keep_all, cfg_drop_dino_cls=drop_all,
            cfg_drop_dino_patches=drop_all, tread_enabled=False,
        )
        v_dino = model(
            zt, t_batch, dino_emb, text_emb, dino_patches, text_mask,
            cfg_drop_text=drop_all, cfg_drop_dino_cls=keep_all,
            cfg_drop_dino_patches=keep_all, tread_enabled=False,
        )
        v_pred = v_uncond + text_scale * (v_text - v_uncond) + dino_scale * (v_dino - v_uncond)

        if prediction_type == "x_prediction":
            t_val = t_curr.clamp(min=0.05)
            v_euler = (zt - v_pred) / t_val
        else:
            v_euler = v_pred

        zt = zt + v_euler * dt
    return zt


def tensor_to_pil(tensor):
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    arr = tensor.cpu().float().numpy()
    arr = (arr * 255).astype("uint8")
    arr = arr.transpose(1, 2, 0)
    return Image.fromarray(arr)


PROMPTS = [
    "A professional headshot of a middle-aged woman with curly brown hair, wearing a navy blazer, smiling warmly.",
    "A cinematic portrait of a young man with a slight beard, wearing a casual grey t-shirt, soft studio lighting.",
    "A close-up photograph of an elderly woman with kind eyes and deep wrinkles, wearing round glasses.",
    "A passport photo of a teenage boy with short blonde hair, freckles, and a neutral expression.",
    "A candid photo of a smiling woman with dark skin and a shaved head, wearing silver hoop earrings.",
]

# Fixed seeds per prompt index — same seed every checkpoint run for comparability
FIXED_SEEDS = [123123123, 456456456, 789789789, 111222333, 444555666]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", "-o", default="results/prompt_tests")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--text-scale", type=float, default=5.0)
    parser.add_argument("--dino-scale", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = yaml.safe_load(Path(args.config).read_text())
    mc = config["model"]

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"Step: {ckpt.get('step', 'N/A')}")

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
    tc = config.get("training", {})
    if tc.get("repa", {}).get("enabled"):
        kwargs["repa_block_idx"] = tc["repa"].get("block_index", -1)
    if tc.get("tread", {}).get("enabled"):
        kwargs["tread_route_start"] = tc["tread"].get("route_start", 1)
        kwargs["tread_route_end"] = tc["tread"].get("route_end", -1)
        kwargs["tread_routing_prob"] = tc["tread"].get("routing_probability", 0.5)
    kwargs["bottleneck_size"] = mc.get("bottleneck_size", 0)
    kwargs["num_pose_joints"] = mc.get("num_pose_joints", 133)
    kwargs["pose_confidence_threshold"] = mc.get("pose_confidence_threshold", 0.05)

    model = NanoDiT(**kwargs)
    model.load_state_dict(strip_orig_mod(ckpt["model"]))

    # Apply EMA directly (matching txt2img.py)
    ema_sd = strip_orig_mod(ckpt["ema"])
    model_sd = model.state_dict()
    for key in ema_sd:
        if key in model_sd:
            model_sd[key].copy_(ema_sd[key])

    model = model.to(device)
    model.eval()

    t5 = T5Encoder(device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nGenerating {len(PROMPTS)} prompts "
          f"(steps={args.steps}, text_scale={args.text_scale}, dino_scale={args.dino_scale})")

    for i, prompt in enumerate(PROMPTS):
        seed = FIXED_SEEDS[i]
        torch.manual_seed(seed)

        t0 = time.time()
        text_emb, text_mask = t5.encode(prompt)
        text_emb = text_emb.to(device)
        text_mask = text_mask.to(device)

        shape = (1, 3, 1024, 1024)
        result = sample(
            model, shape, text_emb, text_mask,
            text_scale=args.text_scale, dino_scale=args.dino_scale,
            num_steps=args.steps,
            prediction_type=mc.get("prediction_type", "x_prediction"),
            device=device,
        )

        img = tensor_to_pil(result[0])
        save_path = out_dir / f"test_prompt_{i+1}.png"
        img.save(save_path)
        (out_dir / f"test_prompt_{i+1}.txt").write_text(prompt)

        elapsed = time.time() - t0
        print(f"[{i+1}/{len(PROMPTS)}] seed={seed} {elapsed:.1f}s → {save_path.name}  "
              f"\"{prompt[:60]}...\"")

    print(f"\nSaved {len(PROMPTS)} images to {out_dir}/")


if __name__ == "__main__":
    main()
