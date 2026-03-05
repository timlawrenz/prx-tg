#!/usr/bin/env python3
"""Create WebDataset tar shards from the Stage 2 derived dataset.

This packages "ready" samples (caption + attention mask + DINOv3/VAE/T5 .npy files)
into tar files following WebDataset naming conventions.

By default, samples are grouped by aspect_bucket and sharded within each bucket.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Counters:
    total_records: int = 0
    ready_records: int = 0
    skipped_incomplete: int = 0
    written_samples: int = 0
    written_shards: int = 0


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def is_mask_ok(mask) -> bool:
    if not isinstance(mask, list):
        return False
    # Accept both 77-token (old) and 512-token (new) masks
    if len(mask) not in (77, 512):
        return False
    return all(v in (0, 1) for v in mask)


def shard_name(idx: int) -> str:
    return f"shard-{idx:06d}.tar"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create WebDataset tar shards from derived embeddings.")
    p.add_argument(
        "--input-jsonl",
        default=os.path.join("data", "derived", "approved_image_dataset.jsonl"),
        help="Stage 2 metadata JSONL (default: %(default)s)",
    )
    p.add_argument(
        "--derived-dir",
        default=os.path.join("data", "derived"),
        help="Base directory containing dinov3/, vae_latents/, t5_hidden/ (default: %(default)s)",
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join("data", "shards"),
        help="Output directory for shards (default: %(default)s)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Write at most N ready samples total (0=all) (default: %(default)s)",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Max samples per shard file (default: %(default)s)",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle ready samples before applying --limit (loads all ready records into RAM)",
    )
    p.add_argument("--seed", type=int, default=1337, help="RNG seed used with --shuffle (default: %(default)s)")
    p.add_argument(
        "--bucket",
        action="append",
        default=None,
        help="Only include a specific aspect bucket (e.g. 832x1216). Can be repeated.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing shard tar files")
    p.add_argument("--dry-run", action="store_true", help="Do not write shards; just report what would happen")
    p.add_argument(
        "--include-images",
        action="store_true",
        help="Include resized RGB images as image.npy (for pixel-space training)",
    )
    p.add_argument(
        "--image-dir",
        default=None,
        help="Override base directory for source images (default: use image_path from JSONL as-is)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N ready records scanned (0 to disable) (default: %(default)s)",
    )
    return p.parse_args(argv)


def iter_ready_records(
    jsonl_path: Path,
    derived_dir: Path,
    counters: Counters,
    allowed_buckets: set[str] | None,
    progress_every: int,
    include_images: bool = False,
    image_dir: str | None = None,
):
    dino_dir = derived_dir / "dinov3"
    dino_patches_dir = derived_dir / "dinov3_patches"
    vae_dir = derived_dir / "vae_latents"
    t5_dir = derived_dir / "t5_hidden"

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            counters.total_records += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                counters.skipped_incomplete += 1
                continue

            image_id = obj.get("image_id")
            if not image_id:
                ip = obj.get("image_path")
                if isinstance(ip, str) and ip:
                    image_id = Path(ip).stem
            if not image_id:
                counters.skipped_incomplete += 1
                continue

            bucket = obj.get("aspect_bucket")
            if not isinstance(bucket, str) or not bucket:
                counters.skipped_incomplete += 1
                continue
            if allowed_buckets is not None and bucket not in allowed_buckets:
                continue

            caption_ok = isinstance(obj.get("caption"), str) and obj.get("caption").strip() != ""
            mask = obj.get("t5_attention_mask")
            mask_ok = is_mask_ok(mask)

            dino_path = dino_dir / f"{image_id}.npy"
            dino_patch_path = dino_patches_dir / f"{image_id}.npy"
            vae_path = vae_dir / f"{image_id}.npy"
            t5_path = t5_dir / f"{image_id}.npy"

            if not (caption_ok and mask_ok and dino_path.is_file() and dino_patch_path.is_file() and vae_path.is_file() and t5_path.is_file()):
                counters.skipped_incomplete += 1
                continue

            counters.ready_records += 1
            if progress_every and (counters.ready_records % progress_every == 0):
                eprint(
                    f"progress: scanned_total={counters.total_records} ready={counters.ready_records} "
                    f"skipped_incomplete={counters.skipped_incomplete}"
                )

            record_data = {
                "record": obj,
                "image_id": image_id,
                "bucket": bucket,
                "dino_path": dino_path,
                "dino_patch_path": dino_patch_path,
                "vae_path": vae_path,
                "t5_path": t5_path,
                "t5_mask": mask,
            }

            if include_images:
                image_path_str = obj.get("image_path", "")
                if image_dir:
                    # Override directory: use image_dir + filename from image_path
                    fname = Path(image_path_str).name if image_path_str else f"{image_id}.jpg"
                    img_path = Path(image_dir) / fname
                else:
                    img_path = Path(image_path_str)
                if not img_path.is_file():
                    counters.skipped_incomplete += 1
                    counters.ready_records -= 1
                    continue
                record_data["image_path"] = img_path

            yield record_data


def add_bytes(tf: tarfile.TarFile, arcname: str, data: bytes):
    ti = tarfile.TarInfo(name=arcname)
    ti.size = len(data)
    tf.addfile(ti, io.BytesIO(data))


def add_file(tf: tarfile.TarFile, arcname: str, path: Path):
    # Use addfile to keep control over arcname and avoid following symlinks.
    st = path.stat()
    ti = tarfile.TarInfo(name=arcname)
    ti.size = st.st_size
    with path.open("rb") as f:
        tf.addfile(ti, f)


def load_bucketed_image_as_npy(image_path: Path, bucket: str) -> bytes | None:
    """Load image, resize-to-cover + center-crop to bucket dims, return as float16 npy bytes."""
    import math
    import numpy as np

    try:
        from PIL import Image
    except ImportError:
        eprint("error: Pillow is required for --include-images (pip install Pillow)")
        return None

    parts = bucket.split("x")
    if len(parts) != 2:
        return None
    bucket_w, bucket_h = int(parts[0]), int(parts[1])

    try:
        with Image.open(image_path) as im:
            img = im.convert("RGB")
    except Exception as e:
        eprint(f"warning: failed to load {image_path}: {e}")
        return None

    w, h = img.size
    if w <= 0 or h <= 0:
        return None

    # Resize-to-cover
    scale = max(bucket_w / w, bucket_h / h)
    new_w = int(math.ceil(w * scale))
    new_h = int(math.ceil(h * scale))
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), resample=Image.BICUBIC)

    # Center crop
    left = max(0, (new_w - bucket_w) // 2)
    top = max(0, (new_h - bucket_h) // 2)
    img = img.crop((left, top, left + bucket_w, top + bucket_h))

    # Convert to float16 numpy: (3, H, W), range [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1).astype(np.float16)  # (3, H, W)

    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def write_shards(
    output_dir: Path,
    bucket: str,
    samples: list[dict],
    shard_size: int,
    overwrite: bool,
    dry_run: bool,
    counters: Counters,
    include_images: bool = False,
):
    bucket_dir = output_dir / f"bucket_{bucket}"
    if not dry_run:
        bucket_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    for start in range(0, len(samples), shard_size):
        chunk = samples[start : start + shard_size]
        tar_path = bucket_dir / shard_name(shard_idx)

        if tar_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing shard: {tar_path} (pass --overwrite)")

        if dry_run:
            counters.written_shards += 1
            counters.written_samples += len(chunk)
            shard_idx += 1
            continue

        if tar_path.exists():
            tar_path.unlink()

        with tarfile.open(tar_path, mode="w") as tf:
            for s in chunk:
                image_id = s["image_id"]

                # 1) JSON metadata
                meta = dict(s["record"])
                meta["image_id"] = image_id
                meta["aspect_bucket"] = bucket
                meta_bytes = (json.dumps(meta, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                add_bytes(tf, f"{image_id}.json", meta_bytes)

                # 2) Existing .npy derivatives (copied verbatim)
                add_file(tf, f"{image_id}.dinov3.npy", s["dino_path"])
                add_file(tf, f"{image_id}.dinov3_patches.npy", s["dino_patch_path"])
                add_file(tf, f"{image_id}.vae.npy", s["vae_path"])
                add_file(tf, f"{image_id}.t5h.npy", s["t5_path"])

                # 3) Attention mask as .npy
                try:
                    import numpy as np

                    mask_arr = np.array(s["t5_mask"], dtype=np.uint8)
                    buf = io.BytesIO()
                    np.save(buf, mask_arr)
                    add_bytes(tf, f"{image_id}.t5m.npy", buf.getvalue())
                except Exception as e:
                    raise RuntimeError(f"Failed writing mask for {image_id}: {e}")

                # 4) Optional RGB image as .npy (for pixel-space training)
                if include_images and "image_path" in s:
                    img_bytes = load_bucketed_image_as_npy(s["image_path"], bucket)
                    if img_bytes:
                        add_bytes(tf, f"{image_id}.image.npy", img_bytes)

        counters.written_shards += 1
        counters.written_samples += len(chunk)
        shard_idx += 1


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    jsonl_path = Path(args.input_jsonl)
    derived_dir = Path(args.derived_dir)
    output_dir = Path(args.output_dir)

    if not jsonl_path.is_file():
        eprint(f"error: input jsonl not found: {jsonl_path}")
        return 2

    for sub in ("dinov3", "dinov3_patches", "vae_latents", "t5_hidden"):
        p = derived_dir / sub
        if not p.is_dir():
            eprint(f"error: missing derived subdir: {p}")
            return 2

    allowed_buckets = set(args.bucket) if args.bucket else None

    counters = Counters()

    ready = list(
        iter_ready_records(jsonl_path, derived_dir, counters, allowed_buckets=allowed_buckets, progress_every=args.progress_every,
                           include_images=args.include_images, image_dir=args.image_dir)
    )

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(ready)

    if args.limit and args.limit > 0:
        ready = ready[: args.limit]

    by_bucket: dict[str, list[dict]] = {}
    for s in ready:
        by_bucket.setdefault(s["bucket"], []).append(s)

    eprint(
        f"ready_samples={len(ready)} buckets={len(by_bucket)} "
        f"shard_size={args.shard_size} dry_run={args.dry_run} output_dir={output_dir}"
    )

    for bucket in sorted(by_bucket.keys()):
        samples = by_bucket[bucket]
        eprint(f"bucket {bucket}: samples={len(samples)}")
        write_shards(
            output_dir=output_dir,
            bucket=bucket,
            samples=samples,
            shard_size=args.shard_size,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            counters=counters,
            include_images=args.include_images,
        )

    eprint(
        f"done: total_records={counters.total_records} ready_records={counters.ready_records} "
        f"skipped_incomplete={counters.skipped_incomplete} written_samples={counters.written_samples} "
        f"written_shards={counters.written_shards}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
