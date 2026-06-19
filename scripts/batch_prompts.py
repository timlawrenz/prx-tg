#!/usr/bin/env python3
"""Thin batch wrapper around scripts/txt2img.py — loads model once, runs N prompts."""
import sys, time, yaml, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse everything from txt2img.py
from txt2img import (
    build_model, load_model, T5Encoder, sample, tensor_to_pil, strip_orig_mod, find_config,
)

import torch

PROMPTS = [
    "A professional headshot of a middle-aged woman with curly brown hair, wearing a navy blazer, smiling warmly.",
    "A cinematic portrait of a young man with a slight beard, wearing a casual grey t-shirt, soft studio lighting.",
    "A close-up photograph of an elderly woman with kind eyes and deep wrinkles, wearing round glasses.",
    "A passport photo of a teenage boy with short blonde hair, freckles, and a neutral expression.",
    "A candid photo of a smiling woman with dark skin and a shaved head, wearing silver hoop earrings.",
]

FIXED_SEEDS = [123123123, 456456456, 789789789, 111222333, 444555666]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("-o", "--output-dir", default="results/prompt_tests")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--text-scale", type=float, default=5.0)
    parser.add_argument("--dino-scale", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Config (auto-detect or explicit)
    config = find_config(args.checkpoint) if not args.config else \
             yaml.safe_load(Path(args.config).read_text())
    if config is None:
        print("Error: Could not find config.yaml. Use --config to specify.", file=sys.stderr)
        sys.exit(1)

    mc = config["model"]

    # Load model once
    model = load_model(args.checkpoint, config, device)

    # Load T5 once
    t5 = T5Encoder(device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n{len(PROMPTS)} prompts, steps={args.steps}, text_scale={args.text_scale}, "
          f"dino_scale={args.dino_scale}")

    for i, prompt in enumerate(PROMPTS):
        seed = FIXED_SEEDS[i]
        torch.manual_seed(seed)

        t0 = time.time()
        text_emb, text_mask = t5.encode(prompt)

        shape = (1, 3, 1024, 1024)
        result = sample(
            model, shape, text_emb, text_mask,
            text_scale=args.text_scale, dino_scale=args.dino_scale,
            num_steps=args.steps,
            prediction_type=mc.get("prediction_type", "x_prediction"),
            device=device,
        )

        img = tensor_to_pil(result[0])
        path = out_dir / f"prompt_{i+1:02d}.png"
        img.save(path)
        (out_dir / f"prompt_{i+1:02d}.txt").write_text(prompt)

        elapsed = time.time() - t0
        print(f"[{i+1}/{len(PROMPTS)}] seed={seed:>9d}  {elapsed:5.1f}s  {prompt[:70]}...")


if __name__ == "__main__":
    main()
