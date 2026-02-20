#!/usr/bin/env python3

import argparse
import json
import numpy as np
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
FLUX_VAE_MODEL_ID = "black-forest-labs/FLUX.1-dev"
SDXL_VAE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"  # Lighter alternative

# Aspect ratio buckets at 1024px equivalent area (all dims divisible by 64)
ASPECT_BUCKETS = [
    (1024, 1024),  # Square, ratio 1.0
    (832, 1216),   # Portrait, ratio ~0.68
    (1216, 832),   # Landscape, ratio ~1.46
    (768, 1280),   # Tall portrait, ratio 0.6
    (1280, 768),   # Wide landscape, ratio ~1.67
    (704, 1344),   # Very tall, ratio ~0.52
    (1344, 704),   # Very wide, ratio ~1.91
]


def parse_bucket_dims(aspect_bucket: str) -> tuple[int, int] | None:
    """Parse "832x1216" or "bucket_832x1216" into (w, h)."""
    if not aspect_bucket or not isinstance(aspect_bucket, str):
        return None
    s = aspect_bucket
    if s.startswith("bucket_"):
        s = s[len("bucket_"):]
    if "x" not in s:
        return None
    try:
        w_str, h_str = s.split("x", 1)
        return int(w_str), int(h_str)
    except Exception:
        return None


def parse_buckets_arg(buckets: str) -> list[tuple[int, int]]:
    """Parse comma-separated bucket list into [(w,h), ...]."""
    if not buckets:
        return []
    out: list[tuple[int, int]] = []
    for part in buckets.split(","):
        dims = parse_bucket_dims(part.strip())
        if dims:
            out.append(dims)
    return out


def load_bucketed_image(image_path: Path, bucket_w: int, bucket_h: int):
    """Resize-to-cover + center-crop to bucket_w x bucket_h."""
    import math
    from PIL import Image

    with Image.open(image_path) as im:
        img = im.convert("RGB")
    w, h = img.size

    if w <= 0 or h <= 0:
        return img

    scale = max(bucket_w / w, bucket_h / h)
    new_w = int(math.ceil(w * scale))
    new_h = int(math.ceil(h * scale))

    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), resample=Image.BICUBIC)

    left = max(0, (new_w - bucket_w) // 2)
    top = max(0, (new_h - bucket_h) // 2)
    return img.crop((left, top, left + bucket_w, top + bucket_h))


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def compute_aspect_ratio(width: int, height: int) -> float:
    """Compute aspect ratio as width / height."""
    return width / height if height > 0 else 1.0


def assign_aspect_bucket(width: int, height: int) -> str:
    """
    Assign image to closest aspect ratio bucket.
    
    Returns bucket name as string (e.g., "1024x1024").
    """
    target_ratio = compute_aspect_ratio(width, height)
    
    # Find bucket with closest aspect ratio
    best_bucket = None
    best_diff = float('inf')
    
    for bucket_w, bucket_h in ASPECT_BUCKETS:
        bucket_ratio = compute_aspect_ratio(bucket_w, bucket_h)
        diff = abs(bucket_ratio - target_ratio)
        
        if diff < best_diff:
            best_diff = diff
            best_bucket = (bucket_w, bucket_h)
    
    return f"{best_bucket[0]}x{best_bucket[1]}"


def compute_image_id(image_path: Path) -> str:
    """Extract image_id from path (basename without extension)."""
    return image_path.stem


def save_npy(array: np.ndarray, path: Path, dtype: np.dtype):
    """Save numpy array to .npy file with type conversion."""
    import numpy as np
    array_converted = array.astype(dtype)
    np.save(path, array_converted)


def load_npy(path: Path) -> np.ndarray | None:
    """Load numpy array from .npy file. Returns None if missing/corrupt."""
    import numpy as np
    try:
        if not path.exists():
            return None
        return np.load(path)
    except Exception as e:
        eprint(f"warning: could not load {path}: {e}")
        return None


def extract_dinov3_to_npy(record: dict, output_dir: Path) -> bool:
    """
    Extract inline DINOv3 embedding from record and save to .npy file.
    
    Returns True if successful, False otherwise.
    """
    import numpy as np
    
    embedding = record.get("dinov3_embedding")
    if not embedding:
        return False
    
    image_path = Path(record["image_path"])
    image_id = compute_image_id(image_path)
    npy_path = output_dir / f"{image_id}.npy"
    
    try:
        array = np.array(embedding, dtype=np.float32)
        if array.shape != (1024,):
            eprint(f"warning: unexpected DINOv3 shape {array.shape} for {image_id}")
            return False
        
        save_npy(array, npy_path, np.float32)
        return True
    except Exception as e:
        eprint(f"warning: failed to extract DINOv3 for {image_id}: {e}")
        return False


def extract_dinov3_patches_to_npy(record: dict, output_dir: Path) -> bool:
    """
    Extract inline DINOv3 patch embeddings from record and save to .npy file.
    
    Returns True if successful, False otherwise.
    """
    import numpy as np
    
    patches = record.get("dinov3_patches")
    if patches is None:
        return False
    
    image_path = Path(record["image_path"])
    image_id = compute_image_id(image_path)
    npy_path = output_dir / f"{image_id}.npy"
    
    try:
        # patches should already be a numpy array from compute_dinov3_patches
        if not isinstance(patches, np.ndarray):
            array = np.array(patches, dtype=np.float32)
        else:
            array = patches.astype(np.float32)
        
        # Expected shape: (196, 1024) for DINOv3-L with 14x14 spatial patch grid
        if array.shape != (196, 1024):
            eprint(f"warning: unexpected DINOv3 patches shape {array.shape} for {image_id}, expected (196, 1024)")
            return False
        
        save_npy(array, npy_path, np.float32)
        return True
    except Exception as e:
        eprint(f"warning: failed to extract DINOv3 patches for {image_id}: {e}")
        return False


def save_vae_latent(latent: np.ndarray, image_id: str, output_dir: Path):
    """Save VAE latent to .npy file."""
    npy_path = output_dir / f"{image_id}.npy"
    save_npy(latent, npy_path, np.float16)


def save_t5_hidden(hidden: np.ndarray, image_id: str, output_dir: Path):
    """Save T5 hidden states to .npy file."""
    npy_path = output_dir / f"{image_id}.npy"
    save_npy(hidden, npy_path, np.float16)


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


def compute_dinov3_patches(dino, device, image, target_width: int = None, target_height: int = None):
    """Extract DINOv3 patch embeddings with dynamic resolution for spatial alignment.
    
    Args:
        dino: DINOv3 model dict
        device: torch device
        image: PIL Image
        target_width: Target width (bucket dimension). If None, uses default 224.
        target_height: Target height (bucket dimension). If None, uses default 224.
    
    Returns:
        numpy array of shape (num_patches, 1024) where num_patches varies by resolution.
        Returns None for pipeline (not supported).
        
    CRITICAL: Uses dynamic resolution (no center-crop) to preserve spatial alignment!
    - Rounds target dims to nearest multiple of 14 (DINOv3 patch size)
    - Feeds full aspect-correct image to DINO (no cropping!)
    - Example: 1216×832 bucket → 1218×826 DINO → 87×59 = 5133 patches
    
    Note: DINOv3 outputs variable tokens based on input size:
        - 1 CLS token (index 0)
        - num_patches = (dino_h // 14) * (dino_w // 14) spatial patches
        - 4 register tokens at end (exclude these)
    """
    import torch
    import numpy as np

    if dino["kind"] == "pipeline":
        # Pipeline doesn't provide easy access to patch embeddings
        return None

    processor = dino["processor"]
    model = dino["model"]

    # Compute DINO input size: round target dims to nearest multiple of 14
    if target_width is not None and target_height is not None:
        dino_w = round(target_width / 14) * 14
        dino_h = round(target_height / 14) * 14
        
        # Use processor with dynamic size, NO center-crop (preserves spatial alignment!)
        inputs = processor(
            images=image,
            size={"height": dino_h, "width": dino_w},
            do_center_crop=False,  # CRITICAL: preserve spatial alignment
            do_resize=True,
            return_tensors="pt"
        )
    else:
        # Fallback to default behavior (for backward compatibility)
        inputs = processor(images=image, return_tensors="pt")
        dino_h, dino_w = 224, 224
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    # Calculate expected number of patches
    num_patches = (dino_h // 14) * (dino_w // 14)
    
    # Extract spatial patch tokens only
    # last_hidden_state shape: (batch=1, num_tokens, hidden=1024)
    # tokens: [CLS, patch_1, ..., patch_N, register_1, ..., register_4]
    # We want indices [1:num_patches+1] to get all spatial patches, excluding CLS and registers
    patches = outputs.last_hidden_state[0, 1:num_patches+1, :]  # (num_patches, 1024)
    
    return patches.detach().cpu().float().numpy()


def compute_dinov3_both(dino, device, image, target_width: int = None, target_height: int = None):
    """Extract both CLS token and patch embeddings in a single forward pass with dynamic resolution.
    
    Args:
        dino: DINOv3 model dict
        device: torch device
        image: PIL Image
        target_width: Target width (bucket dimension). If None, uses default 224.
        target_height: Target height (bucket dimension). If None, uses default 224.
    
    Returns:
        tuple: (cls_embedding, patches) where:
            - cls_embedding: list of 1024 floats (CLS token)
            - patches: numpy array (num_patches, 1024) or None if pipeline
    
    This is more efficient than calling compute_dinov3_embedding() and 
    compute_dinov3_patches() separately since it only runs the model once.
    
    CRITICAL: Uses dynamic resolution (no center-crop) to preserve spatial alignment!
    """
    import torch
    import numpy as np

    if dino["kind"] == "pipeline":
        # Pipeline: only CLS available
        feats = dino["feature_extractor"](image)
        x = feats[0]
        while isinstance(x, list) and x and isinstance(x[0], list):
            seq = x
            hidden = len(seq[0])
            x = [sum(tok[i] for tok in seq) / len(seq) for i in range(hidden)]
        return x, None

    processor = dino["processor"]
    model = dino["model"]

    # Compute DINO input size: round target dims to nearest multiple of 14
    if target_width is not None and target_height is not None:
        dino_w = round(target_width / 14) * 14
        dino_h = round(target_height / 14) * 14
        
        # Use processor with dynamic size, NO center-crop (preserves spatial alignment!)
        inputs = processor(
            images=image,
            size={"height": dino_h, "width": dino_w},
            do_center_crop=False,  # CRITICAL: preserve spatial alignment
            do_resize=True,
            return_tensors="pt"
        )
    else:
        # Fallback to default behavior (for backward compatibility)
        inputs = processor(images=image, return_tensors="pt")
        dino_h, dino_w = 224, 224
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract CLS token
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        cls_emb = outputs.pooler_output[0]
    else:
        cls_emb = outputs.last_hidden_state[0, 0]
    
    cls_list = cls_emb.detach().cpu().float().tolist()

    # Calculate expected number of patches
    num_patches = (dino_h // 14) * (dino_w // 14)
    
    # Extract patches (skip CLS at 0, exclude register tokens at end)
    patches = outputs.last_hidden_state[0, 1:num_patches+1, :]  # (num_patches, 1024)
    patches_np = patches.detach().cpu().float().numpy()
    
    return cls_list, patches_np


def load_t5_tokenizer(model_id: str):
    """Load T5 tokenizer for computing attention masks."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


def compute_t5_attention_mask(tokenizer, caption: str) -> list[int]:
    """
    Tokenize caption with T5 and return attention mask.
    
    Returns a list of up to 512 integers (1 for valid tokens, 0 for padding).
    T5 can handle sequences up to 512 tokens, unlike CLIP's 77-token limit.
    """
    tokens = tokenizer(
        caption,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    mask = tokens["attention_mask"][0].tolist()
    
    # Verify correctness
    assert len(mask) == 512, f"Expected mask length 512, got {len(mask)}"
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


def ensure_bucket_info(record: dict, image_path: Path) -> bool:
    """Ensure record has width/height (original) and aspect_bucket; returns True if modified."""
    changed = False

    w = record.get("width")
    h = record.get("height")
    if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
        dims = get_image_dimensions(image_path)
        if dims:
            record["width"], record["height"] = dims
            changed = True

    if isinstance(record.get("width"), int) and isinstance(record.get("height"), int):
        ab = record.get("aspect_bucket")
        if not isinstance(ab, str) or not parse_bucket_dims(ab):
            record["aspect_bucket"] = assign_aspect_bucket(record["width"], record["height"])
            changed = True

    return changed


def load_flux_vae(device, compile_encoder: bool = False):
    """Load Flux VAE encoder component only.
    
    Args:
        device: torch device to load model on
        compile_encoder: If True, compile encoder with torch.compile for 20-30% speedup
                        WARNING: torch.compile causes memory corruption after ~100-200 images
                        ("Expected curr_block->next == nullptr" error). Disabled by default.
    
    Returns:
        VAE model with optimized encoder (if compile_encoder=True)
    """
    try:
        from diffusers import AutoencoderKL
        import torch
        
        eprint(f"loading Flux VAE from {FLUX_VAE_MODEL_ID}...")
        # Load VAE component only (subfolder="vae")
        # Use bfloat16 for faster computation on modern GPUs (RTX 3090+)
        vae = AutoencoderKL.from_pretrained(
            FLUX_VAE_MODEL_ID,
            subfolder="vae",
            torch_dtype=torch.bfloat16
        )
        vae.eval()
        vae = vae.to(device)
        
        # DISABLED BY DEFAULT: torch.compile causes memory allocator corruption
        # after processing many images in a single run
        if compile_encoder and hasattr(torch, 'compile'):
            eprint("  WARNING: torch.compile is unstable for long-running VAE encoding")
            eprint("  compiling VAE encoder with torch.compile (first run will be slow)...")
            vae.encoder = torch.compile(vae.encoder, mode="reduce-overhead")
            eprint("  ✓ VAE encoder compiled")
        
        return vae
    except Exception as e:
        eprint(f"error: failed to load Flux VAE: {e}")
        raise


def preprocess_vae_input(image):
    """Prepare PIL image for VAE encoding."""
    import torch
    from torchvision import transforms

    # VAE expects [-1, 1] range
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Rescale [0,1] to [-1,1]
    ])

    return tfm(image).unsqueeze(0)  # (1, 3, H, W)


def encode_vae_latent(image_path: Path, vae_encoder, aspect_bucket: str | None = None) -> np.ndarray | None:
    """Encode bucketed crop to VAE latent space; returns (16, H//8, W//8) float16."""
    try:
        import torch
        import numpy as np
        from PIL import Image

        expected_shape = None
        dims = parse_bucket_dims(aspect_bucket) if aspect_bucket else None
        if dims:
            bucket_w, bucket_h = dims
            img = load_bucketed_image(image_path, bucket_w, bucket_h)
            expected_shape = (16, bucket_h // 8, bucket_w // 8)
        else:
            with Image.open(image_path) as im:
                img = im.convert("RGB")
            w, h = img.size
            expected_shape = (16, h // 8, w // 8)

        tensor = preprocess_vae_input(img)

        device = next(vae_encoder.parameters()).device
        dtype = next(vae_encoder.parameters()).dtype  # Match model dtype (bfloat16)
        tensor = tensor.to(device, dtype=dtype)

        with torch.no_grad():
            latent_dist = vae_encoder.encode(tensor)
            latent = latent_dist.latent_dist.sample()

        latent_fp16 = latent.squeeze(0).to(torch.float16).cpu()
        result = latent_fp16.numpy().astype(np.float16)

        if expected_shape and result.shape != expected_shape:
            eprint(
                f"warning: VAE latent shape mismatch for {image_path}: got {result.shape}, expected {expected_shape}"
            )
            return None

        return result

    except Exception as e:
        import traceback
        eprint(f"warning: VAE encoding failed for {image_path}:")
        eprint(f"  Error: {type(e).__name__}: {e}")
        eprint(f"  Traceback: {traceback.format_exc()}")
        return None


def load_t5_encoder():
    """Load full T5-Large encoder model."""
    from transformers import T5EncoderModel
    import torch
    
    eprint(f"loading T5-Large encoder from {T5_MODEL_ID}...")
    model = T5EncoderModel.from_pretrained(T5_MODEL_ID, torch_dtype=torch.float16)
    model.eval()
    return model


def compute_t5_hidden_states(caption: str, tokenizer, encoder) -> np.ndarray | None:
    """
    Encode caption to T5 hidden states.
    
    Returns (512, 1024) float16 array or None on error.
    """
    try:
        import torch
        import numpy as np
        
        # Tokenize with T5's max length (512 tokens, not CLIP's 77)
        inputs = tokenizer(
            caption,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        device = next(encoder.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
        
        # Convert to numpy float16, squeeze batch dimension
        result = hidden_states.squeeze(0).cpu().numpy().astype(np.float16)
        
        # Validate shape (512 tokens, 1024 hidden dims)
        if result.shape != (512, 1024):
            eprint(f"warning: T5 hidden state shape mismatch: got {result.shape}, expected (512, 1024)")
            return None
        
        return result
        
    except Exception as e:
        eprint(f"warning: T5 encoding failed for caption: {e}")
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
    
    # Special handling for t5_attention_mask: detect truncated 77-token masks
    if field_name == "t5_attention_mask":
        if not isinstance(value, list) or len(value) == 0:
            return True
        # If mask is 77 tokens and all 1s, it's truncated and needs regeneration
        if len(value) == 77 and all(v == 1 for v in value):
            return True
        # If mask is not 512 tokens, regenerate with correct length
        if len(value) != 512:
            return True
    
    # For lists/arrays, check if empty
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return True
    # For strings, check if empty
    if isinstance(value, str) and not value:
        return True
    return False


def load_caption_pipeline(device, model_id: str):
    """
    Return a simple dict with Ollama configuration.
    No longer loads HuggingFace transformers.
    
    Args:
        device: Ignored (Ollama manages its own resources)
        model_id: Ignored (we use gemma3:27b from Ollama)
    
    Returns:
        dict with ollama_url and model_name
    """
    return {
        "ollama_url": "http://192.168.86.162:11434/api/generate",
        "model_name": "gemma3:27b"
    }


def generate_captions(caption_pipe, images: list, max_new_tokens: int) -> list[str]:
    """
    Generate captions using Ollama API.
    
    Args:
        caption_pipe: dict with ollama_url and model_name
        images: list of PIL Images
        max_new_tokens: max tokens to generate
    
    Returns:
        list of caption strings
    """
    import base64
    from io import BytesIO
    import requests
    
    url = caption_pipe["ollama_url"]
    model = caption_pipe["model_name"]
    
    results = []
    
    for img in images:
        # Convert PIL Image to base64
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Prepare Ollama request
        payload = {
            "model": model,
            "prompt": CAPTION_PROMPT,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": max_new_tokens
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            caption = result.get('response', '')
            
            # Clean up the caption
            caption = ensure_single_paragraph(caption)
            results.append(caption)
            
        except Exception as e:
            eprint(f"warning: Ollama caption generation failed: {e}")
            results.append("")  # Empty caption on failure
    
    return results


def needs_dinov3_extraction(record: dict, dinov3_dir: Path) -> bool:
    """Check if DINOv3 needs extraction from inline to .npy file."""
    # Has inline embedding but missing .npy file
    if "dinov3_embedding" in record:
        image_path = Path(record["image_path"])
        image_id = compute_image_id(image_path)
        npy_path = dinov3_dir / f"{image_id}.npy"
        return not npy_path.exists()
    return False


def needs_dinov3_patches_on_disk(record: dict, dinov3_patches_dir: Path) -> bool:
    """Check if DINOv3 patches are missing from disk."""
    image_path = Path(record["image_path"])
    image_id = compute_image_id(image_path)
    npy_path = dinov3_patches_dir / f"{image_id}.npy"
    return not npy_path.exists()


def needs_dinov3_cls_on_disk(record: dict, dinov3_dir: Path) -> bool:
    """Check if DINOv3 CLS token is missing from disk."""
    image_path = Path(record["image_path"])
    image_id = compute_image_id(image_path)
    npy_path = dinov3_dir / f"{image_id}.npy"
    return not npy_path.exists()


def needs_vae_latent(record: dict, vae_dir: Path) -> bool:
    """Check if VAE latent is missing OR stale for the record's aspect_bucket."""
    import numpy as np

    image_path = Path(record["image_path"])
    image_id = compute_image_id(image_path)
    npy_path = vae_dir / f"{image_id}.npy"

    if not npy_path.exists():
        return True

    dims = parse_bucket_dims(record.get("aspect_bucket"))
    # If we can't validate bucket consistency, force regeneration (true-native contract)
    if not dims:
        return True

    bw, bh = dims
    expected = (16, bh // 8, bw // 8)

    try:
        arr = np.load(npy_path, mmap_mode="r")
        if arr.shape != expected:
            return True
        if arr.dtype != np.float16:
            return True
        return False
    except Exception:
        return True


def needs_t5_hidden(record: dict, t5_dir: Path) -> bool:
    """Check if T5 hidden state .npy file is missing."""
    image_path = Path(record["image_path"])
    image_id = compute_image_id(image_path)
    npy_path = t5_dir / f"{image_id}.npy"
    return not npy_path.exists()


def needs_migration(record: dict) -> bool:
    """Check if record needs migration to Stage 2 format."""
    return record.get("format_version") != 2


def check_disk_space(path: Path, required_gb: int = 20) -> bool:
    """
    Check if sufficient disk space is available.
    
    Returns True if at least required_gb GB is free, False otherwise.
    """
    import shutil
    
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024 ** 3)
        
        if free_gb < required_gb:
            eprint(f"error: insufficient disk space at {path}")
            eprint(f"  required: {required_gb} GB, available: {free_gb:.1f} GB")
            return False
        
        eprint(f"disk space check: {free_gb:.1f} GB available at {path}")
        return True
        
    except Exception as e:
        eprint(f"warning: could not check disk space at {path}: {e}")
        return True  # Don't block if check fails


def verify_stage2_data(output_path: Path, base_dir: Path) -> dict:
    """
    Verify Stage 2 data integrity.
    
    Returns dict with counts: {valid, invalid, missing} per embedding type.
    """
    import numpy as np
    
    dinov3_dir = base_dir / "dinov3"
    vae_dir = base_dir / "vae_latents"
    t5_dir = base_dir / "t5_hidden"
    
    results = {
        "dinov3": {"valid": 0, "invalid": 0, "missing": 0},
        "vae": {"valid": 0, "invalid": 0, "missing": 0},
        "t5": {"valid": 0, "invalid": 0, "missing": 0},
    }
    
    if not output_path.exists():
        eprint(f"error: output file {output_path} does not exist")
        return results
    
    records = load_existing_records(output_path)
    eprint(f"verifying {len(records)} records...")
    
    for record in records.values():
        image_path = Path(record["image_path"])
        image_id = compute_image_id(image_path)
        
        # Check DINOv3
        dinov3_path = dinov3_dir / f"{image_id}.npy"
        if not dinov3_path.exists():
            results["dinov3"]["missing"] += 1
        else:
            try:
                arr = np.load(dinov3_path)
                if arr.shape == (1024,) and arr.dtype == np.float32:
                    results["dinov3"]["valid"] += 1
                else:
                    results["dinov3"]["invalid"] += 1
                    eprint(f"  invalid DINOv3 for {image_id}: shape={arr.shape}, dtype={arr.dtype}")
            except Exception as e:
                results["dinov3"]["invalid"] += 1
                eprint(f"  corrupt DINOv3 for {image_id}: {e}")
        
        # Check VAE
        vae_path = vae_dir / f"{image_id}.npy"
        if not vae_path.exists():
            results["vae"]["missing"] += 1
        else:
            try:
                arr = np.load(vae_path)
                expected = None
                dims = parse_bucket_dims(record.get("aspect_bucket"))
                if dims:
                    bw, bh = dims
                    expected = (16, bh // 8, bw // 8)

                is_valid = (
                    arr.ndim == 3
                    and arr.shape[0] == 16
                    and arr.dtype == np.float16
                    and (expected is None or arr.shape == expected)
                )

                if is_valid:
                    results["vae"]["valid"] += 1
                else:
                    results["vae"]["invalid"] += 1
                    if expected is not None:
                        eprint(
                            f"  invalid VAE for {image_id}: shape={arr.shape}, dtype={arr.dtype}, expected_shape={expected}"
                        )
                    else:
                        eprint(f"  invalid VAE for {image_id}: shape={arr.shape}, dtype={arr.dtype}")
            except Exception as e:
                results["vae"]["invalid"] += 1
                eprint(f"  corrupt VAE for {image_id}: {e}")
        
        # Check T5
        t5_path = t5_dir / f"{image_id}.npy"
        if not t5_path.exists():
            results["t5"]["missing"] += 1
        else:
            try:
                arr = np.load(t5_path)
                if arr.shape == (512, 1024) and arr.dtype == np.float16:
                    results["t5"]["valid"] += 1
                else:
                    results["t5"]["invalid"] += 1
                    eprint(f"  invalid T5 for {image_id}: shape={arr.shape}, dtype={arr.dtype}")
            except Exception as e:
                results["t5"]["invalid"] += 1
                eprint(f"  corrupt T5 for {image_id}: {e}")
    
    return results


def analyze_aspect_buckets(input_dir: Path, output_path: Path, num_buckets: int, bucket_area: int, bucket_quantum: int):
    """Analyze dataset aspect ratios and propose a quantized bucket list."""
    import math
    import numpy as np

    existing = load_existing_records(output_path) if output_path.exists() else {}

    log_ratios = []
    for img_path in iter_images(input_dir):
        rel = img_path.as_posix()
        rec = existing.get(rel, {})
        w = rec.get("width")
        h = rec.get("height")

        if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
            dims = get_image_dimensions(img_path)
            if not dims:
                continue
            w, h = dims

        log_ratios.append(math.log(w / h))

    if not log_ratios:
        eprint("error: no valid width/height found to analyze")
        return 1

    x = np.asarray(log_ratios, dtype=np.float64)

    k = max(1, int(num_buckets))
    qs = np.linspace(0.0, 1.0, k)
    centers = np.quantile(x, qs)

    for _ in range(30):
        d = np.abs(x[:, None] - centers[None, :])
        assign = np.argmin(d, axis=1)
        new_centers = centers.copy()
        for i in range(k):
            mask = assign == i
            if np.any(mask):
                new_centers[i] = float(np.mean(x[mask]))
        if float(np.max(np.abs(new_centers - centers))) < 1e-6:
            break
        centers = new_centers

    centers = np.sort(centers)
    bucket_ratios = np.exp(centers)

    buckets: list[tuple[int, int]] = []
    for r in bucket_ratios:
        w = math.sqrt(bucket_area * r)
        h = math.sqrt(bucket_area / r)
        bw = max(bucket_quantum, int(round(w / bucket_quantum)) * bucket_quantum)
        bh = max(bucket_quantum, int(round(h / bucket_quantum)) * bucket_quantum)
        buckets.append((bw, bh))

    # Deduplicate after quantization
    uniq: list[tuple[int, int]] = []
    for b in buckets:
        if b not in uniq:
            uniq.append(b)
    buckets = uniq

    br = np.asarray([math.log(bw / bh) for bw, bh in buckets], dtype=np.float64)
    d2 = np.abs(x[:, None] - br[None, :])
    assign2 = np.argmin(d2, axis=1)
    counts = np.bincount(assign2, minlength=len(buckets))
    mean_abs_log_err = float(np.mean(np.abs(x - br[assign2])))
    median_abs_log_err = float(np.median(np.abs(x - br[assign2])))

    eprint("=== Proposed aspect buckets ===")
    eprint(f"images analyzed: {len(x)}")
    eprint(f"num buckets (requested): {k}, after dedupe: {len(buckets)}")
    eprint(f"mean |log(aspect error)|: {mean_abs_log_err:.4f}")
    eprint(f"median |log(aspect error)|: {median_abs_log_err:.4f}")
    eprint("")

    eprint("ASPECT_BUCKETS = [")
    for (bw, bh), c in sorted(zip(buckets, counts), key=lambda t: (t[0][0] / t[0][1])):
        eprint(f"    ({bw}, {bh}),  # count={int(c)} ratio={bw / bh:.3f}")
    eprint("]")
    eprint("")

    eprint("production/config.yaml data.buckets:")
    for bw, bh in sorted(buckets, key=lambda t: (t[0] / t[1])):
        eprint(f"  - \"bucket_{bw}x{bh}\"")

    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Generate/enrich Stage 2 dataset: JSONL metadata + external .npy embeddings (DINOv3, VAE, T5)"
    )
    p.add_argument("--input-dir", default="data/approved", help="Directory of approved images")
    p.add_argument(
        "--output",
        default="data/derived/approved_image_dataset.jsonl",
        help="Output JSONL path (metadata only in Stage 2)",
    )
    p.add_argument(
        "--output-base-dir",
        default="data/derived",
        help="Base directory for .npy embedding files (creates dinov3/, vae_latents/, t5_hidden/ subdirs)",
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
        "--pass",
        dest="pass_filter",
        choices=["all", "dinov3", "vae", "t5", "migrate"],
        default="all",
        help="Run specific pass: all (default), dinov3 (extract only), vae (encode VAE latents), t5 (encode T5 hidden), migrate (JSONL format only)",
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
        help="Delete output JSONL and .npy directories, regenerate everything from scratch",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Verify Stage 2 data integrity: check for missing/corrupt .npy files and report",
    )
    p.add_argument(
        "--analyze-buckets",
        action="store_true",
        help="Analyze aspect ratios and propose a bucket list (prints output and exits)",
    )
    p.add_argument(
        "--num-buckets",
        type=int,
        default=len(ASPECT_BUCKETS),
        help="Number of aspect buckets to propose in --analyze-buckets mode",
    )
    p.add_argument(
        "--bucket-area",
        type=int,
        default=1024 * 1024,
        help="Target pixel area for derived buckets (default: 1024*1024)",
    )
    p.add_argument(
        "--bucket-quantum",
        type=int,
        default=64,
        help="Quantize derived bucket dims to multiples of this (default: 64)",
    )
    p.add_argument(
        "--buckets",
        default="",
        help="Override buckets as comma-separated list (e.g., 1024x1024,832x1216,...)",
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
    base_dir = Path(args.output_base_dir)

    # Optional bucket override
    if args.buckets:
        parsed = parse_buckets_arg(args.buckets)
        if parsed:
            global ASPECT_BUCKETS
            ASPECT_BUCKETS = parsed
            eprint(f"using overridden ASPECT_BUCKETS: {ASPECT_BUCKETS}")

    # Analysis-only mode
    if args.analyze_buckets:
        return analyze_aspect_buckets(input_dir, output_path, args.num_buckets, args.bucket_area, args.bucket_quantum)

    # Create output directories
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dinov3_dir = base_dir / "dinov3"
    dinov3_patches_dir = base_dir / "dinov3_patches"
    vae_dir = base_dir / "vae_latents"
    t5_dir = base_dir / "t5_hidden"
    
    dinov3_dir.mkdir(parents=True, exist_ok=True)
    dinov3_patches_dir.mkdir(parents=True, exist_ok=True)
    vae_dir.mkdir(parents=True, exist_ok=True)
    t5_dir.mkdir(parents=True, exist_ok=True)

    # Verification mode
    if args.verify:
        eprint("=== Stage 2 Data Verification ===")
        results = verify_stage2_data(output_path, base_dir)
        
        eprint("\nDINOv3 embeddings:")
        eprint(f"  valid: {results['dinov3']['valid']}")
        eprint(f"  invalid: {results['dinov3']['invalid']}")
        eprint(f"  missing: {results['dinov3']['missing']}")
        
        eprint("\nVAE latents:")
        eprint(f"  valid: {results['vae']['valid']}")
        eprint(f"  invalid: {results['vae']['invalid']}")
        eprint(f"  missing: {results['vae']['missing']}")
        
        eprint("\nT5 hidden states:")
        eprint(f"  valid: {results['t5']['valid']}")
        eprint(f"  invalid: {results['t5']['invalid']}")
        eprint(f"  missing: {results['t5']['missing']}")
        
        return 0

    # Check disk space
    if not check_disk_space(base_dir, required_gb=20):
        return 1

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

    # Handle --no-resume: delete output, temp, and .npy directories
    if args.no_resume:
        if output_path.exists():
            eprint(f"--no-resume: deleting existing {output_path}")
            output_path.unlink()
        if temp_path.exists():
            eprint(f"--no-resume: deleting existing {temp_path}")
            temp_path.unlink()
        
        import shutil
        for npy_dir in [dinov3_dir, vae_dir, t5_dir]:
            if npy_dir.exists():
                eprint(f"--no-resume: deleting {npy_dir}")
                shutil.rmtree(npy_dir)
                npy_dir.mkdir(parents=True, exist_ok=True)

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

    # Determine which passes to run
    run_dinov3 = args.pass_filter in ["all", "dinov3"]
    run_vae = args.pass_filter in ["all", "vae"]
    run_t5 = args.pass_filter in ["all", "t5"]
    run_migrate = args.pass_filter in ["all", "migrate"]

    # Load models conditionally based on pass
    t5_tokenizer = None
    t5_encoder = None
    dino = None
    caption_pipe = None
    vae_encoder = None

    if run_dinov3 or run_migrate:
        # Need tokenizer for T5 attention mask
        if args.verbose:
            eprint("verbose: loading T5 tokenizer...")
        t5_tokenizer = load_t5_tokenizer(T5_MODEL_ID)

    if run_dinov3:
        # Need DINOv3 for new images and Gemma for captions
        if args.verbose:
            eprint("verbose: loading DINOv3 model...")
        dino = load_dinov3(device, args.dinov3_model)
        if args.verbose:
            eprint("verbose: loading caption model...")
        caption_pipe = load_caption_pipeline(device, args.caption_model)

    if run_vae:
        if args.verbose:
            eprint("verbose: loading Flux VAE...")
        vae_encoder = load_flux_vae(device, compile_encoder=False)  # Disabled: causes memory corruption

    if run_t5:
        if args.verbose:
            eprint("verbose: loading T5-Large encoder...")
        t5_encoder = load_t5_encoder()
        t5_encoder = t5_encoder.to(device)
        if not t5_tokenizer:
            t5_tokenizer = load_t5_tokenizer(T5_MODEL_ID)

    # Counters for progress tracking (4 types)
    migrated = 0  # Stage 1 → Stage 2 with all embeddings
    enriched = 0  # Partial Stage 2 records with missing embeddings filled
    extracted = 0  # Only DINOv3 extraction (already format v2)
    skipped = 0  # Fully complete records
    total_processed = 0
    started = time.time()

    # Batch buffer for caption generation (only used in dinov3 pass)
    batch_buffer = []
    
    # Open temp file for incremental writes
    temp_file = temp_path.open("a", encoding="utf-8")
    records_written_this_session = 0

    def flush_batch():
        """Process batch: compute embeddings and captions (Stage 1 pass)."""
        nonlocal batch_buffer, records_written_this_session, migrated, enriched
        if not batch_buffer:
            return

        images = [item["image"] for item in batch_buffer]
        records = [item["record"] for item in batch_buffer]

        # Compute DINO embeddings for records that need them
        for i, item in enumerate(batch_buffer):
            if item["needs_dino"]:
                needs_cls = item.get("needs_dino_cls", True)
                needs_patches = item.get("needs_dino_patches", True)
                
                # Extract bucket dimensions for spatial alignment
                bucket_w, bucket_h = None, None
                aspect_bucket = records[i].get("aspect_bucket")
                if aspect_bucket:
                    dims = parse_bucket_dims(aspect_bucket)
                    if dims:
                        bucket_w, bucket_h = dims
                
                if needs_cls and needs_patches:
                    # Need both - use optimized single forward pass
                    if args.verbose:
                        bucket_str = f" (bucket: {bucket_w}x{bucket_h})" if bucket_w else ""
                        eprint(f"verbose: computing DINOv3 CLS + patches for {item['record']['image_path']}{bucket_str}...")
                    cls_emb, patches = compute_dinov3_both(dino, device, images[i], bucket_w, bucket_h)
                    records[i]["dinov3_embedding"] = cls_emb
                    if patches is not None:
                        records[i]["dinov3_patches"] = patches
                        
                elif needs_cls:
                    # Only need CLS
                    if args.verbose:
                        eprint(f"verbose: computing DINOv3 CLS for {item['record']['image_path']}...")
                    records[i]["dinov3_embedding"] = compute_dinov3_embedding(dino, device, images[i])
                    
                elif needs_patches:
                    # Only need patches
                    if args.verbose:
                        bucket_str = f" (bucket: {bucket_w}x{bucket_h})" if bucket_w else ""
                        eprint(f"verbose: computing DINOv3 patches for {item['record']['image_path']}{bucket_str}...")
                    patches = compute_dinov3_patches(dino, device, images[i], bucket_w, bucket_h)
                    if patches is not None:
                        records[i]["dinov3_patches"] = patches

        # Generate captions for records that need them (batched)
        needs_caption_indices = [i for i, item in enumerate(batch_buffer) if item["needs_caption"]]
        if needs_caption_indices:
            if args.verbose:
                eprint(f"verbose: generating captions for batch of {len(needs_caption_indices)}...")
            caption_images = [images[i] for i in needs_caption_indices]
            captions = generate_captions(caption_pipe, caption_images, args.caption_max_new_tokens)
            for idx, caption in zip(needs_caption_indices, captions):
                records[idx]["caption"] = caption

        # Write records with Stage 2 format
        for item in batch_buffer:
            rec = item["record"]
            
            # Compute T5 mask if caption exists and mask is missing/invalid
            if "caption" in rec and t5_tokenizer and needs_field(rec, "t5_attention_mask"):
                rec["t5_attention_mask"] = compute_t5_attention_mask(t5_tokenizer, rec["caption"])
            
            # Extract DINOv3 to .npy if present inline
            if "dinov3_embedding" in rec:
                extract_dinov3_to_npy(rec, dinov3_dir)
            # Extract patches if present
            if "dinov3_patches" in rec:
                extract_dinov3_patches_to_npy(rec, dinov3_patches_dir)
            
            # Migrate to Stage 2 format
            image_path = Path(rec["image_path"])
            image_id = compute_image_id(image_path)
            rec["image_id"] = image_id
            
            if "width" in rec and "height" in rec:
                rec["aspect_bucket"] = assign_aspect_bucket(rec["width"], rec["height"])
            
            rec["format_version"] = 2
            
            # Remove inline embeddings (already extracted to .npy files)
            if "dinov3_embedding" in rec:
                del rec["dinov3_embedding"]
            if "dinov3_patches" in rec:
                del rec["dinov3_patches"]
            
            # Write to temp file
            temp_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            temp_file.flush()
            records_written_this_session += 1

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

            # Compute image_id
            image_id = compute_image_id(img_path)
            if "image_id" not in record:
                record["image_id"] = image_id

            # Check what's needed
            needs_dino_extract = run_dinov3 and needs_dinov3_extraction(record, dinov3_dir)
            needs_dino_patches_disk = run_dinov3 and needs_dinov3_patches_on_disk(record, dinov3_patches_dir)
            needs_vae_gen = run_vae and needs_vae_latent(record, vae_dir)
            needs_t5_gen = run_t5 and needs_t5_hidden(record, t5_dir)
            needs_format_migration = run_migrate and needs_migration(record)
            needs_t5_mask = needs_field(record, "t5_attention_mask")  # Check for missing/truncated masks

            # Check if Stage 1 record (needs full processing)
            is_stage1 = "dinov3_embedding" in record and record.get("format_version") != 2

            # Determine if fully complete
            is_complete = (
                record.get("format_version") == 2
                and not needs_dino_extract
                and not needs_dino_patches_disk  # Also check patches
                and not needs_vae_gen
                and not needs_t5_gen
                and not needs_format_migration
                and not needs_t5_mask  # Also check t5_attention_mask is valid
            )

            if is_complete:
                skipped += 1
                if args.verbose:
                    eprint(f"verbose: skipping complete record {rel}")
                if args.progress_every and total_processed % args.progress_every == 0:
                    elapsed = time.time() - started
                    rate = (migrated + enriched + extracted) / elapsed if elapsed > 0 else 0
                    eprint(
                        f"progress: {migrated} migrated, {enriched} enriched, {extracted} extracted, {skipped} skipped (total: {total_processed}) rate={rate:.2f}/s"
                    )
                continue

            # Handle DINOv3 pass (Stage 1 processing or extraction)
            if run_dinov3:
                needs_dino = needs_field(record, "dinov3_embedding")
                needs_caption = needs_field(record, "caption")
                needs_dimensions = needs_field(record, "height") or needs_field(record, "width")

                # Also check if either CLS or patches are missing on disk
                # (even if inline embedding exists, we might need to compute patches)
                needs_dino_cls_disk = needs_dinov3_cls_on_disk(record, dinov3_dir)
                needs_dino_patches_disk = needs_dinov3_patches_on_disk(record, dinov3_patches_dir)
                needs_any_dino = needs_dino or needs_dino_cls_disk or needs_dino_patches_disk

                if needs_any_dino or needs_caption or needs_dimensions:
                    # Ensure width/height + aspect_bucket exist before any image-based pass
                    ensure_bucket_info(record, img_path)

                    img = None
                    if needs_any_dino or needs_caption:
                        try:
                            dims = parse_bucket_dims(record.get("aspect_bucket"))
                            if dims:
                                img = load_bucketed_image(img_path, dims[0], dims[1])
                            else:
                                with Image.open(img_path) as im:
                                    img = im.convert("RGB")
                        except Exception as e:
                            eprint(f"warning: cannot open {img_path}: {e}")
                            continue

                    # Add to batch for processing
                    batch_buffer.append({
                        "image_path": img_path,
                        "image": img,
                        "record": record,
                        "needs_dino": needs_any_dino,
                        "needs_dino_cls": needs_dino or needs_dino_cls_disk,
                        "needs_dino_patches": needs_dino_patches_disk,
                        "needs_caption": needs_caption,
                    })

                    if len(batch_buffer) >= args.batch_size:
                        flush_batch()
                        
                    if is_stage1:
                        migrated += 1
                    else:
                        enriched += 1

                elif needs_dino_extract:
                    # Just extract existing inline embedding
                    extract_dinov3_to_npy(record, dinov3_dir)
                    # Also extract patches if available
                    if "dinov3_patches" in record:
                        extract_dinov3_patches_to_npy(record, dinov3_patches_dir)
                    extracted += 1

            # Flush any pending batch before VAE/T5 passes
            # This ensures captions are generated before T5 encoding attempts
            if batch_buffer and (run_vae or run_t5):
                flush_batch()

            # Handle VAE pass
            if run_vae and needs_vae_gen:
                if args.verbose:
                    eprint(f"verbose: encoding VAE latent for {image_id}...")

                meta_changed = ensure_bucket_info(record, img_path)
                latent = encode_vae_latent(img_path, vae_encoder, record.get("aspect_bucket"))
                if latent is not None:
                    save_vae_latent(latent, image_id, vae_dir)
                    enriched += 1
                else:
                    eprint(f"warning: VAE encoding failed for {image_id}")

                # Persist metadata updates (for --pass vae runs)
                if meta_changed and not run_dinov3 and not run_migrate:
                    temp_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    temp_file.flush()
                    records_written_this_session += 1

                # Clear CUDA cache to prevent memory accumulation
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Handle T5 pass
            if run_t5 and needs_t5_gen:
                caption = record.get("caption")
                if caption:
                    if args.verbose:
                        eprint(f"verbose: encoding T5 hidden states for {image_id}...")
                    
                    hidden = compute_t5_hidden_states(caption, t5_tokenizer, t5_encoder)
                    if hidden is not None:
                        save_t5_hidden(hidden, image_id, t5_dir)
                        enriched += 1
                    else:
                        eprint(f"warning: T5 encoding failed for {image_id}")
                else:
                    eprint(f"warning: no caption for {image_id}, skipping T5 encoding")

            # Handle migration pass
            if run_migrate and needs_format_migration:
                # Update record to Stage 2 format
                if "dinov3_embedding" in record:
                    extract_dinov3_to_npy(record, dinov3_dir)
                    del record["dinov3_embedding"]
                if "dinov3_patches" in record:
                    extract_dinov3_patches_to_npy(record, dinov3_patches_dir)
                    del record["dinov3_patches"]
                
                if "width" in record and "height" in record:
                    record["aspect_bucket"] = assign_aspect_bucket(record["width"], record["height"])
                
                # Regenerate t5_attention_mask if needed (truncated 77-token masks)
                if "caption" in record and t5_tokenizer and needs_field(record, "t5_attention_mask"):
                    record["t5_attention_mask"] = compute_t5_attention_mask(t5_tokenizer, record["caption"])
                
                record["format_version"] = 2
                
                # Write updated record
                temp_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                temp_file.flush()
                records_written_this_session += 1
                migrated += 1
            
            # Handle standalone t5_attention_mask regeneration
            # (for records that are otherwise complete but have truncated masks)
            elif needs_t5_mask and "caption" in record and t5_tokenizer:
                if args.verbose:
                    eprint(f"verbose: regenerating t5_attention_mask for {rel}")
                record["t5_attention_mask"] = compute_t5_attention_mask(t5_tokenizer, record["caption"])
                temp_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                temp_file.flush()
                records_written_this_session += 1
                enriched += 1

            # Progress reporting
            if args.progress_every and total_processed % args.progress_every == 0:
                elapsed = time.time() - started
                rate = (migrated + enriched + extracted) / elapsed if elapsed > 0 else 0
                eprint(
                    f"progress: {migrated} migrated, {enriched} enriched, {extracted} extracted, {skipped} skipped (total: {total_processed}) rate={rate:.2f}/s"
                )

            # Limit check
            if args.limit and (migrated + enriched + extracted) >= args.limit:
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
    
    elapsed = time.time() - started
    eprint(
        f"done: {migrated} migrated, {enriched} enriched, {extracted} extracted, {skipped} skipped "
        f"(total processed: {total_processed}, final output: {len(final_records)} records) in {elapsed:.1f}s"
    )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
