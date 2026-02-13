#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from pathlib import Path

CAPTION_PROMPT = (
    "Generate a single, dense paragraph describing this image for a text-to-image training dataset. "
    "Write in a strictly dry, objective, and descriptive tone. "
    "Do not use flowery language, subjective interpretations, or lists. "
    "Describe only what is visible: subject (including specific body build, muscle definition, skin texture, and visible anatomical landmarks), "
    "precise pose (mechanics of limb positioning, hand placement), clothing/accessories, lighting, background, "
    "composition/framing, and camera angle. "
    "Do not guess measurements (height, weight) or internal anatomy not visible. "
    "Do not include any conversational filler, preambles (like 'The image shows...'), or meta-commentary. "
    "Start the description immediately."
)

DINO_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
GEMMA_MODEL_ID = "google/gemma-3-27b-it"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def iter_images(input_dir: Path):
    for p in sorted(input_dir.iterdir(), key=lambda x: x.name):
        if p.is_file():
            yield p


def read_completed_paths(output_jsonl: Path, idx_path: Path):
    completed = set()

    if idx_path.exists():
        with idx_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    completed.add(line)
        return completed

    if not output_jsonl.exists():
        return completed

    # Fallback: scan existing JSONL to rebuild the index.
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = obj.get("image_path")
            if isinstance(p, str) and p:
                completed.add(p)

    return completed


def ensure_single_paragraph(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = " ".join(part for part in text.split("\n") if part.strip())
    return " ".join(text.split()).strip()


def preprocess_vit_like(image, size: int = 224):
    import torch
    from torchvision import transforms

    tfm = transforms.Compose(
        [
            transforms.Resize(size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return {"pixel_values": tfm(image).unsqueeze(0)}


def pick_device(device: str):
    import torch

    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dinov3(device, model_id: str):
    # Prefer the documented approach for DINOv3.
    from transformers import AutoImageProcessor, AutoModel, pipeline

    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.to(device)
        model.eval()
        return {"kind": "automodel", "processor": processor, "model": model}
    except Exception:
        # Fallback to the documented pipeline, which can support newer/remote model types.
        dev = 0 if device.type == "cuda" else -1
        fe = pipeline(model=model_id, task="image-feature-extraction", device=dev)
        return {"kind": "pipeline", "feature_extractor": fe}


def compute_dinov3_embedding(dino, device, image):
    import torch

    if dino["kind"] == "pipeline":
        feats = dino["feature_extractor"](image)
        # Pipeline returns nested lists; pick first vector and mean-pool if needed.
        x = feats[0]
        while isinstance(x, list) and x and isinstance(x[0], list):
            # [seq, hidden] -> mean over seq
            seq = x
            hidden = len(seq[0])
            x = [sum(tok[i] for tok in seq) / len(seq) for i in range(hidden)]
        return x

    processor = dino["processor"]
    model = dino["model"]

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        emb = outputs.pooler_output[0]
    else:
        emb = outputs.last_hidden_state[0, 0]

    return emb.detach().cpu().float().tolist()


def load_caption_pipeline(device, model_id: str):
    import torch
    from transformers import pipeline

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    dev = 0 if device.type == "cuda" else -1
    return pipeline("image-text-to-text", model=model_id, device=dev, torch_dtype=dtype)


def generate_captions(caption_pipe, images: list, max_new_tokens: int) -> list[str]:
    # Use the tokenizer's chat template to construct the correct prompt with image tokens.
    # This avoids guessing the correct <image> token string.
    
    chat = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": CAPTION_PROMPT}
        ]}
    ]
    
    # Render the prompt to a string. 
    # Note: apply_chat_template handles the insertion of the correct image token.
    try:
        prompt = caption_pipe.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        # Fallback if tokenizer doesn't support image type in chat template (older versions)
        # But Gemma 3 should support it.
        eprint(f"warning: apply_chat_template failed ({e}), falling back to manual <image> token.")
        prompt = "<image>\n" + CAPTION_PROMPT

    # The pipeline expects a generator of dicts for batching
    def input_generator():
        for img in images:
            yield {"images": img, "text": prompt}

    outputs = caption_pipe(
        input_generator(),
        max_new_tokens=max_new_tokens,
        batch_size=len(images)
    )
    
    results = []
    for out in outputs:
        if isinstance(out, list) and out:
            item = out[0]
            if isinstance(item, dict) and "generated_text" in item:
                text = item["generated_text"]
            else:
                text = str(item)
        else:
            text = str(out)

        # Strip the input prompt if it was echoed in the output
        # With chat template, the prompt is complex, so we might need a better way to strip.
        # Gemma 3 usually returns just the new text? Or echoes?
        # If it echoes, it will start with the prompt.
        
        # Simple heuristic: remove the prompt string if it matches
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
            
        # Also strip typical chat headers if they remain (e.g. "model\n")
        if text.startswith("model\n"):
            text = text[6:].lstrip()

        results.append(ensure_single_paragraph(text))

    return results

    outputs = caption_pipe(
        input_generator(),
        max_new_tokens=max_new_tokens,
        batch_size=len(images)
    )
    
    results = []
    for out in outputs:
        if isinstance(out, list) and out:
            item = out[0]
            if isinstance(item, dict) and "generated_text" in item:
                text = item["generated_text"]
            else:
                text = str(item)
        else:
            text = str(out)

        # Strip the input prompt if it was echoed in the output
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        results.append(ensure_single_paragraph(text))

    return results


def parse_args(argv):
    p = argparse.ArgumentParser(description="Generate JSONL dataset from data/approved images")
    p.add_argument("--input-dir", default="data/approved", help="Directory of approved images")
    p.add_argument(
        "--output",
        default="data/derived/approved_image_dataset.jsonl",
        help="Output JSONL path",
    )
    p.add_argument("--limit", type=int, default=0, help="Process at most N images (0=all)")
    p.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto|cpu|cuda|cuda:0|...",
    )
    p.add_argument(
        "--dinov3-model",
        default=DINO_MODEL_ID,
        help="Hugging Face model id for DINOv3 embeddings",
    )
    p.add_argument(
        "--caption-model",
        default=GEMMA_MODEL_ID,
        help="Hugging Face model id for caption generation",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N images (0 disables)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip images already present in output",
    )
    p.add_argument(
        "--caption-max-new-tokens",
        type=int,
        default=500,
        help="Max new tokens for caption generation",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for caption generation",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress messages",
    )
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    idx_path = output_path.with_suffix(output_path.suffix + ".idx")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = set()
    if not args.no_resume:
        completed = read_completed_paths(output_path, idx_path)
        if completed:
            eprint(f"resume: loaded {len(completed)} completed image_path entries")

    try:
        from PIL import Image
    except ModuleNotFoundError as ex:
        raise SystemExit("Missing dependency: pillow (PIL). Install requirements first.") from ex

    device = pick_device(args.device)
    eprint(f"device: {device}")

    if args.verbose:
        eprint("verbose: loading DINOv3 model...")
    dino = load_dinov3(device, args.dinov3_model)
    if args.verbose:
        eprint("verbose: DINOv3 loaded")

    if args.verbose:
        eprint("verbose: loading caption model...")
    caption_pipe = load_caption_pipeline(device, args.caption_model)
    if args.verbose:
        eprint("verbose: caption model loaded")

    processed = 0
    skipped = 0
    emitted = 0
    started = time.time()
    
    batch_buffer = []

    def flush_batch(batch, out_f, idx_f):
        nonlocal emitted
        if not batch:
            return

        images = [item[1] for item in batch]
        rels = [item[0] for item in batch]
        
        # Compute DINO embeddings (sequentially for now as it's fast)
        embeddings = []
        for i, rel in enumerate(rels):
            if args.verbose:
                eprint(f"verbose: computing DINOv3 embedding for {rel}...")
            embeddings.append(compute_dinov3_embedding(dino, device, images[i]))

        # Generate captions (batched)
        if args.verbose:
            eprint(f"verbose: generating captions for batch of {len(images)}...")
        
        captions = generate_captions(caption_pipe, images, args.caption_max_new_tokens)
        
        for i, rel in enumerate(rels):
            if args.verbose:
                eprint(f"verbose: caption generated for {rel}")
            
            record = {
                "image_path": rel,
                "dinov3_embedding": embeddings[i],
                "caption": captions[i],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            idx_f.write(rel + "\n")
            emitted += 1
            
        out_f.flush()
        idx_f.flush()

    with output_path.open("a", encoding="utf-8") as out_f, idx_path.open(
        "a", encoding="utf-8"
    ) as idx_f:
        for img_path in iter_images(input_dir):
            rel = img_path.as_posix()
            processed += 1

            if completed and rel in completed:
                skipped += 1
                if args.progress_every and processed % args.progress_every == 0:
                    eprint(
                        f"progress: processed={processed} emitted={emitted} skipped={skipped} current={rel}"
                    )
                continue

            try:
                with Image.open(img_path) as im:
                    image = im.convert("RGB")
            except Exception:
                skipped += 1
                continue
            
            batch_buffer.append((rel, image))
            
            if len(batch_buffer) >= args.batch_size:
                flush_batch(batch_buffer, out_f, idx_f)
                batch_buffer = []

                if args.progress_every and processed % args.progress_every == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    eprint(
                        f"progress: processed={processed} emitted={emitted} skipped={skipped} "
                        f"rate={emitted/elapsed:.3f}/s current={rel}"
                    )

                if args.limit and emitted >= args.limit:
                    break
        
        # Flush remaining
        if batch_buffer and (not args.limit or emitted < args.limit):
            flush_batch(batch_buffer, out_f, idx_f)

    eprint(f"done: processed={processed} emitted={emitted} skipped={skipped}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
