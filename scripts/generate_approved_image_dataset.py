#!/usr/bin/env python3

import argparse
import json
import os
import signal
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
T5_MODEL_ID = "t5-large"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def iter_images(input_dir: Path):
    for p in sorted(input_dir.iterdir(), key=lambda x: x.name):
        if p.is_file():
            yield p


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


def load_t5_tokenizer(model_id: str):
    """Load T5 tokenizer for computing attention masks."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


def compute_t5_attention_mask(tokenizer, caption: str) -> list[int]:
    """
    Tokenize caption with T5 and return attention mask.
    
    Returns a list of 77 integers (1 for valid tokens, 0 for padding).
    """
    tokens = tokenizer(
        caption,
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    mask = tokens["attention_mask"][0].tolist()
    
    # Verify correctness
    assert len(mask) == 77, f"Expected mask length 77, got {len(mask)}"
    assert all(v in (0, 1) for v in mask), "Mask must contain only 0s and 1s"
    
    return mask


def get_image_dimensions(image_path: Path) -> tuple[int, int] | None:
    """
    Read image dimensions from file header without full decode.
    
    Returns (width, height) tuple or None if image cannot be opened.
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size  # PIL returns (width, height)
    except Exception as e:
        eprint(f"warning: could not read dimensions for {image_path}: {e}")
        return None


def load_existing_records(output_path: Path) -> dict[str, dict]:
    """
    Load existing JSONL records into a dict keyed by image_path.
    
    Returns empty dict if file doesn't exist.
    Handles malformed lines gracefully (skips with warning).
    """
    if not output_path.exists():
        return {}
    
    records = {}
    with output_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                image_path = obj.get("image_path")
                if isinstance(image_path, str) and image_path:
                    records[image_path] = obj
                else:
                    eprint(f"warning: line {line_num} missing valid image_path, skipping")
            except json.JSONDecodeError as e:
                eprint(f"warning: line {line_num} is malformed JSON ({e}), skipping")
    
    return records


def needs_field(record: dict, field_name: str) -> bool:
    """
    Check if a field is missing or null/empty in a record.
    
    Returns True if field needs to be computed.
    """
    if field_name not in record:
        return True
    value = record[field_name]
    if value is None:
        return True
    # For lists/arrays, check if empty
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return True
    # For strings, check if empty
    if isinstance(value, str) and not value:
        return True
    return False


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


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Generate/enrich JSONL dataset from approved images with DINOv3 embeddings, Gemma captions, and metadata"
    )
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
        help="Delete output and regenerate everything from scratch",
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
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Track temp file for signal handler
    temp_file = None
    
    def merge_and_exit(signum=None, frame=None):
        """Graceful shutdown: merge .tmp into output.jsonl before exiting."""
        eprint("\nInterrupted! Merging partial progress into output file...")
        
        # Close temp file if open
        if temp_file and not temp_file.closed:
            temp_file.close()
        
        # Perform merge if temp file has data
        if temp_path.exists() and temp_path.stat().st_size > 0:
            final_temp = output_path.with_suffix(output_path.suffix + ".final")
            
            # Load both files
            final_records = {}
            if output_path.exists():
                final_records = load_existing_records(output_path)
                eprint(f"  loaded {len(final_records)} records from {output_path}")
            
            all_temp_records = load_existing_records(temp_path)
            eprint(f"  loaded {len(all_temp_records)} records from {temp_path}")
            
            # Merge (temp records win)
            for path, record in all_temp_records.items():
                final_records[path] = record
            
            # Write merged output
            with final_temp.open("w", encoding="utf-8") as f:
                for record in final_records.values():
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            # Atomic rename
            final_temp.rename(output_path)
            temp_path.unlink()
            
            eprint(f"  merged to {output_path} ({len(final_records)} total records)")
        else:
            eprint("  no partial progress to merge")
        
        eprint("Exiting.")
        sys.exit(0)
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, merge_and_exit)

    # Handle --no-resume: delete output and temp file, start fresh
    if args.no_resume:
        if output_path.exists():
            eprint(f"--no-resume: deleting existing {output_path}")
            output_path.unlink()
        if temp_path.exists():
            eprint(f"--no-resume: deleting existing {temp_path}")
            temp_path.unlink()

    # Load existing records from both output and temp file
    existing_records = load_existing_records(output_path)
    if existing_records:
        eprint(f"loaded {len(existing_records)} existing records from {output_path}")
    
    # Also load from temp file (partial progress from previous run)
    temp_records = load_existing_records(temp_path)
    if temp_records:
        eprint(f"loaded {len(temp_records)} partial records from {temp_path}")
        # Merge temp records into existing (temp takes precedence as it's newer)
        for path, record in temp_records.items():
            existing_records[path] = record

    try:
        from PIL import Image
    except ModuleNotFoundError as ex:
        raise SystemExit("Missing dependency: pillow (PIL). Install requirements first.") from ex

    device = pick_device(args.device)
    eprint(f"device: {device}")

    # Load models
    if args.verbose:
        eprint("verbose: loading T5 tokenizer...")
    t5_tokenizer = load_t5_tokenizer(T5_MODEL_ID)
    if args.verbose:
        eprint("verbose: T5 tokenizer loaded")

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

    # Counters for progress tracking
    processed_new = 0
    enriched = 0
    skipped = 0
    total_processed = 0
    started = time.time()

    # Batch buffer for caption generation
    batch_buffer = []
    
    # Open temp file for incremental writes
    temp_file = temp_path.open("a", encoding="utf-8")
    records_written_this_session = 0

    def flush_batch():
        """Process batch: compute embeddings and captions."""
        nonlocal processed_new, batch_buffer, records_written_this_session
        if not batch_buffer:
            return

        images = [item["image"] for item in batch_buffer]
        records = [item["record"] for item in batch_buffer]

        # Compute DINO embeddings for records that need them
        for i, item in enumerate(batch_buffer):
            if item["needs_dino"]:
                if args.verbose:
                    eprint(f"verbose: computing DINOv3 embedding for {item['record']['image_path']}...")
                records[i]["dinov3_embedding"] = compute_dinov3_embedding(dino, device, images[i])

        # Generate captions for records that need them (batched)
        needs_caption_indices = [i for i, item in enumerate(batch_buffer) if item["needs_caption"]]
        if needs_caption_indices:
            if args.verbose:
                eprint(f"verbose: generating captions for batch of {len(needs_caption_indices)}...")
            caption_images = [images[i] for i in needs_caption_indices]
            captions = generate_captions(caption_pipe, caption_images, args.caption_max_new_tokens)
            for idx, caption in zip(needs_caption_indices, captions):
                records[idx]["caption"] = caption
                if args.verbose:
                    eprint(f"verbose: caption generated for {records[idx]['image_path']}")

        # Compute T5 masks and write records incrementally
        for item in batch_buffer:
            rec = item["record"]
            if item["needs_caption"] and "caption" in rec:
                rec["t5_attention_mask"] = compute_t5_attention_mask(t5_tokenizer, rec["caption"])
            
            # Write to temp file immediately (incremental progress)
            required_fields = ["image_path", "dinov3_embedding", "caption", "t5_attention_mask", "height", "width"]
            if all(field in rec and rec[field] is not None for field in required_fields):
                temp_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                temp_file.flush()
                records_written_this_session += 1
            else:
                eprint(f"warning: skipping incomplete record {rec.get('image_path', '?')}")

        batch_buffer = []

    try:
        # Process all images
        for img_path in iter_images(input_dir):
            rel = img_path.as_posix()
            total_processed += 1

            # Check if record exists
            if rel in existing_records:
                record = existing_records[rel].copy()
            else:
                record = {"image_path": rel}

            # Determine what needs to be computed
            needs_dino = needs_field(record, "dinov3_embedding")
            needs_caption = needs_field(record, "caption")
            needs_t5_mask = needs_field(record, "t5_attention_mask")
            needs_dimensions = needs_field(record, "height") or needs_field(record, "width")

            # Check if fully complete (skip)
            if not (needs_dino or needs_caption or needs_t5_mask or needs_dimensions):
                skipped += 1
                if args.verbose:
                    eprint(f"verbose: skipping complete record {rel}")
                if args.progress_every and total_processed % args.progress_every == 0:
                    eprint(
                        f"progress: {processed_new} new, {enriched} enriched, {skipped} skipped (total: {total_processed})"
                    )
                continue

            # Determine operation type for tracking
            is_new = needs_dino or needs_caption
            if is_new:
                processed_new += 1
            else:
                enriched += 1

            # Load image if needed for DINOv3/caption
            image = None
            if needs_dino or needs_caption:
                try:
                    with Image.open(img_path) as im:
                        image = im.convert("RGB")
                except Exception as e:
                    eprint(f"warning: could not load image {rel}: {e}")
                    continue

            # Compute metadata (fast operations, do immediately)
            if needs_dimensions:
                dims = get_image_dimensions(img_path)
                if dims:
                    record["width"], record["height"] = dims
                else:
                    # Skip if we can't get dimensions
                    eprint(f"warning: skipping {rel} due to dimension read failure")
                    continue

            if needs_t5_mask and not needs_caption:
                # Can compute T5 mask now if caption already exists
                if "caption" in record and record["caption"]:
                    record["t5_attention_mask"] = compute_t5_attention_mask(t5_tokenizer, record["caption"])

            # If needs DINOv3 or caption, add to batch
            if needs_dino or needs_caption:
                batch_buffer.append({
                    "record": record,
                    "image": image,
                    "needs_dino": needs_dino,
                    "needs_caption": needs_caption,
                })

                if len(batch_buffer) >= args.batch_size:
                    flush_batch()
            else:
                # No image loading needed, write record immediately
                required_fields = ["image_path", "dinov3_embedding", "caption", "t5_attention_mask", "height", "width"]
                if all(field in record and record[field] is not None for field in required_fields):
                    temp_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    temp_file.flush()
                    records_written_this_session += 1

            # Progress reporting
            if args.progress_every and total_processed % args.progress_every == 0:
                elapsed = max(time.time() - started, 1e-6)
                rate = (processed_new + enriched) / elapsed
                eprint(
                    f"progress: {processed_new} new, {enriched} enriched, {skipped} skipped "
                    f"(total: {total_processed}) rate={rate:.2f}/s"
                )

            # Limit check
            if args.limit and (processed_new + enriched) >= args.limit:
                break

        # Flush remaining batch
        if batch_buffer:
            flush_batch()

    finally:
        # Always close temp file
        temp_file.close()

    # Atomic commit: merge temp and original, write to new temp, rename
    eprint(f"finalizing: merging {len(existing_records)} existing + {records_written_this_session} new records...")
    
    final_temp = output_path.with_suffix(output_path.suffix + ".final")
    
    # Reload temp file to get all records (including what was there before + new)
    all_temp_records = load_existing_records(temp_path)
    
    # Merge: start with original records, overlay temp records
    final_records = {}
    if output_path.exists():
        final_records = load_existing_records(output_path)
    
    # Overlay temp records (newer data wins)
    for path, record in all_temp_records.items():
        final_records[path] = record
    
    # Write final output
    with final_temp.open("w", encoding="utf-8") as f:
        for record in final_records.values():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # Atomic rename
    final_temp.rename(output_path)
    
    # Clean up temp file
    if temp_path.exists():
        temp_path.unlink()
    
    eprint(
        f"done: {processed_new} new, {enriched} enriched, {skipped} skipped "
        f"(total processed: {total_processed}, final output: {len(final_records)} records)"
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
