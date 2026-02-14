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
    if not isinstance(mask, list) or len(mask) != 77:
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
):
    dino_dir = derived_dir / "dinov3"
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
            vae_path = vae_dir / f"{image_id}.npy"
            t5_path = t5_dir / f"{image_id}.npy"

            if not (caption_ok and mask_ok and dino_path.is_file() and vae_path.is_file() and t5_path.is_file()):
                counters.skipped_incomplete += 1
                continue

            counters.ready_records += 1
            if progress_every and (counters.ready_records % progress_every == 0):
                eprint(
                    f"progress: scanned_total={counters.total_records} ready={counters.ready_records} "
                    f"skipped_incomplete={counters.skipped_incomplete}"
                )

            yield {
                "record": obj,
                "image_id": image_id,
                "bucket": bucket,
                "dino_path": dino_path,
                "vae_path": vae_path,
                "t5_path": t5_path,
                "t5_mask": mask,
            }


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


def write_shards(
    output_dir: Path,
    bucket: str,
    samples: list[dict],
    shard_size: int,
    overwrite: bool,
    dry_run: bool,
    counters: Counters,
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

    for sub in ("dinov3", "vae_latents", "t5_hidden"):
        p = derived_dir / sub
        if not p.is_dir():
            eprint(f"error: missing derived subdir: {p}")
            return 2

    allowed_buckets = set(args.bucket) if args.bucket else None

    counters = Counters()

    ready = list(
        iter_ready_records(jsonl_path, derived_dir, counters, allowed_buckets=allowed_buckets, progress_every=args.progress_every)
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
        )

    eprint(
        f"done: total_records={counters.total_records} ready_records={counters.ready_records} "
        f"skipped_incomplete={counters.skipped_incomplete} written_samples={counters.written_samples} "
        f"written_shards={counters.written_shards}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
