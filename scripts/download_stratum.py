#!/usr/bin/env python3
"""Download stratum-ffhq from HuggingFace for local use.

Downloads selected layers to a local directory in the stratum format
(per-image directories with metadata.json + .npy files).

Usage:
    python scripts/download_stratum.py --output ./stratum-ffhq --layers caption,dinov3,t5,pose
    python scripts/download_stratum.py --output ./stratum-ffhq --layers all
    python scripts/download_stratum.py --output ./stratum-ffhq --layers all --max-samples 1000
"""

import argparse
import json
import io
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

# Layers available in stratum-ffhq
AVAILABLE_LAYERS = ["caption", "dinov3", "t5", "pose", "seg", "depth", "normal"]


def parse_args():
    p = argparse.ArgumentParser(description="Download stratum-ffhq from HuggingFace")
    p.add_argument("--repo", default="timlawrenz/stratum-ffhq",
                   help="HF dataset repo (default: timlawrenz/stratum-ffhq)")
    p.add_argument("--output", required=True,
                   help="Output directory for stratum dataset")
    p.add_argument("--layers", default="all",
                   help="Comma-separated layer names or 'all' (default: all)")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Maximum samples to download (0 = all)")
    p.add_argument("--token", default="",
                   help="HF token for gated/private repos")
    return p.parse_args()


def decode_npy_blob(data: bytes, shape: tuple, dtype) -> np.ndarray:
    """Decode a raw numpy blob (from npy_tar) into ndarray."""
    try:
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    except Exception:
        # Maybe it's a real .npy file with header
        return np.load(io.BytesIO(data))


def download_layer(repo: str, layer: str, token: Optional[str]) -> dict:
    """Load a single layer from HuggingFace and return dict of image_id → data."""
    from datasets import load_dataset

    print(f"  Loading layer '{layer}' from {repo}...")
    ds = load_dataset(repo, layer, split="train", streaming=True, token=token or None)

    result = {}
    for i, row in enumerate(tqdm(ds, desc=f"  {layer}")):
        image_id = str(row.get("image_id", i))
        result[image_id] = dict(row)
    return result


def build_stratum_dataset(output_dir: Path, layers_data: dict[str, dict[str, dict]], max_samples: int):
    """Write downloaded layers into stratum per-image directory format."""
    # Collect all image_ids
    all_ids = set()
    for layer_name, layer_dict in layers_data.items():
        all_ids.update(layer_dict.keys())

    image_ids = sorted(all_ids)
    if max_samples > 0:
        image_ids = image_ids[:max_samples]

    print(f"\nWriting {len(image_ids)} samples to {output_dir}...")

    for image_id in tqdm(image_ids, desc="Writing"):
        img_dir = output_dir / image_id
        img_dir.mkdir(parents=True, exist_ok=True)

        # Collect metadata from all layers
        metadata = {"image_id": image_id}
        caption = ""

        for layer_name, layer_dict in layers_data.items():
            row = layer_dict.get(image_id, {})

            if layer_name == "caption":
                caption = str(row.get("caption", ""))
                metadata["caption"] = caption
                (img_dir / "caption.txt").write_text(caption, encoding="utf-8")

            elif layer_name == "dinov3":
                cls_data = row.get("dinov3_cls")
                patches_data = row.get("dinov3_patches")

                if cls_data is not None:
                    if isinstance(cls_data, bytes):
                        arr = np.frombuffer(cls_data, dtype=np.float16)
                        if len(arr) == 1024:
                            np.save(img_dir / "dinov3_cls.npy", arr)
                        else:
                            np.save(img_dir / "dinov3_cls.npy",
                                    np.frombuffer(cls_data, dtype=np.float16).reshape(1024))

                if patches_data is not None:
                    if isinstance(patches_data, bytes):
                        arr = np.frombuffer(patches_data, dtype=np.float16)
                        num_patches = len(arr) // 1024
                        np.save(img_dir / "dinov3_patches.npy",
                                arr.reshape(num_patches, 1024))

            elif layer_name == "t5":
                hidden_data = row.get("t5_hidden")
                mask_data = row.get("t5_mask")

                if hidden_data is not None:
                    if isinstance(hidden_data, bytes):
                        np.save(img_dir / "t5_hidden.npy",
                                np.frombuffer(hidden_data, dtype=np.float16).reshape(512, 1024))

                if mask_data is not None:
                    if isinstance(mask_data, bytes):
                        np.save(img_dir / "t5_mask.npy",
                                np.frombuffer(mask_data, dtype=np.uint8).reshape(512))

            elif layer_name == "pose":
                pose_data = row.get("pose")
                if pose_data is not None and isinstance(pose_data, bytes):
                    np.save(img_dir / "pose.npy",
                            np.frombuffer(pose_data, dtype=np.float16).reshape(133, 3))

            elif layer_name in ("seg", "depth", "normal"):
                # Optional layers — save if present
                key_map = {"seg": "seg.npy", "depth": "depth.npy", "normal": "normal.npy"}
                val = row.get(layer_name)
                if val is not None and isinstance(val, bytes):
                    np.save(img_dir / key_map[layer_name],
                            np.frombuffer(val, dtype=np.uint8 if layer_name == "seg" else np.float16))

            # Carry metadata fields
            for k in ("width", "height", "aspect_bucket", "source_path"):
                if k in row and row[k] is not None:
                    metadata[k] = row[k]

        # Write metadata.json
        # Convert non-serializable values
        clean_meta = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                clean_meta[k] = v
            elif isinstance(v, (np.integer,)):
                clean_meta[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean_meta[k] = float(v)
            else:
                clean_meta[k] = str(v)

        with (img_dir / "metadata.json").open("w") as f:
            json.dump(clean_meta, f, indent=2)

    print(f"Done. Dataset written to {output_dir}")
    print(f"  Samples: {len(image_ids)}")
    print(f"  Layers: {list(layers_data.keys())}")
    print(f"\nUse with prx-tg:")
    print(f"  data:")
    print(f"    source: stratum")
    print(f"    stratum_dir: {output_dir}")
    print(f"    stratum_source: local")


def main():
    args = parse_args()

    # Install check
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library required. Install: pip install datasets")
        sys.exit(1)

    layers = AVAILABLE_LAYERS if args.layers == "all" else [l.strip() for l in args.layers.split(",")]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading stratum-ffhq")
    print(f"  Repo: {args.repo}")
    print(f"  Layers: {layers}")
    print(f"  Output: {output_dir}")
    if args.max_samples:
        print(f"  Max samples: {args.max_samples}")

    # Download each layer
    layers_data = {}
    for layer in layers:
        try:
            layers_data[layer] = download_layer(args.repo, layer, args.token or None)
        except Exception as e:
            print(f"  Warning: failed to download layer '{layer}': {e}")
            continue

    if not layers_data:
        print("Error: no layers downloaded successfully")
        sys.exit(1)

    build_stratum_dataset(output_dir, layers_data, args.max_samples)


if __name__ == "__main__":
    main()