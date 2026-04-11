#!/usr/bin/env python3
"""Benchmark mmap vs WebDataset data loading on Strix Halo.

Measures per-batch loading time for both data paths to quantify the
zero-copy advantage on unified memory architectures.

Usage:
    # Compare both loaders (requires data/derived/ and data/shards/ to exist)
    python scripts/benchmark_dataloader.py

    # Mmap only (for Strix Halo without WebDataset shards)
    python scripts/benchmark_dataloader.py --mmap-only

    # Specific batch size
    python scripts/benchmark_dataloader.py --batch-size 8 --num-batches 50
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark_loader(loader, name: str, num_batches: int, device: str):
    """Time num_batches iterations from a dataloader."""
    import torch

    print(f"\n--- {name} ---")
    iterator = iter(loader)

    # Warmup (2 batches)
    warmup = min(2, num_batches)
    for _ in range(warmup):
        batch = next(iterator)
        # Move to device (simulates training)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                _ = v.to(device)

    # Timed batches
    times = []
    batch_sizes = []
    for i in range(num_batches):
        t0 = time.perf_counter()
        batch = next(iterator)
        # Move to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                _ = v.to(device)
        if device != 'cpu' and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        batch_sizes.append(batch['image_data'].shape[0])

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    avg_bs = sum(batch_sizes) / len(batch_sizes)
    total_images = sum(batch_sizes)
    total_time = sum(times)

    print(f"  Batches: {num_batches}")
    print(f"  Avg batch size: {avg_bs:.1f}")
    print(f"  Per-batch: {avg*1000:.2f} ± {std*1000:.2f} ms")
    print(f"  Images/sec: {total_images / total_time:.1f}")
    print(f"  Min: {min(times)*1000:.2f} ms, Max: {max(times)*1000:.2f} ms")

    # Memory bandwidth estimate
    # Rough: image(3×H×W×4B) + patches(N×1024×4B) + t5(512×1024×4B) + misc
    # At 1024px: ~184 MB per batch of 8
    bytes_per_image = (3 * 1024 * 1024 * 4 + 4096 * 1024 * 4 + 512 * 1024 * 4 + 1024 * 4 + 133 * 3 * 4)
    bw = (avg_bs * bytes_per_image) / avg / 1e9
    print(f"  Est. bandwidth: {bw:.1f} GB/s")

    return {'name': name, 'avg_ms': avg * 1000, 'std_ms': std * 1000,
            'images_per_sec': total_images / total_time}


def main():
    parser = argparse.ArgumentParser(description='Benchmark data loaders')
    parser.add_argument('--config', default='production/config_strix_halo.yaml')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--num-batches', type=int, default=20, help='Number of batches to time')
    parser.add_argument('--mmap-only', action='store_true', help='Only test mmap loader')
    parser.add_argument('--device', default='cpu', help='Device to move batches to')
    args = parser.parse_args()

    from production.config_loader import load_config
    config = load_config(args.config)

    if args.batch_size:
        config.training.batch_size = args.batch_size

    print("=" * 60)
    print("Data Loader Benchmark")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Device: {args.device}")
    print(f"Batches to time: {args.num_batches}")

    results = []

    # Test mmap loader
    if config.data.zero_copy:
        from production.data_mmap import get_mmap_dataloader
        try:
            mmap_loader = get_mmap_dataloader(config)
            r = benchmark_loader(mmap_loader, "Mmap (zero-copy)", args.num_batches, args.device)
            results.append(r)
        except Exception as e:
            print(f"\n  Mmap loader failed: {e}")
            print(f"  (Requires data/derived/ with .npy files and metadata JSONL)")

    # Test WebDataset loader
    if not args.mmap_only:
        config.data.zero_copy = False
        try:
            from production.data import get_production_dataloader
            wds_loader = get_production_dataloader(config, device=args.device)
            r = benchmark_loader(wds_loader, "WebDataset (tar shards)", args.num_batches, args.device)
            results.append(r)
        except Exception as e:
            print(f"\n  WebDataset loader failed: {e}")
            print(f"  (Requires data/shards/ with tar files)")

    # Comparison
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        mmap_r = results[0]
        wds_r = results[1]
        speedup = wds_r['avg_ms'] / mmap_r['avg_ms']
        print(f"  Mmap:       {mmap_r['avg_ms']:.2f} ms/batch ({mmap_r['images_per_sec']:.1f} img/s)")
        print(f"  WebDataset: {wds_r['avg_ms']:.2f} ms/batch ({wds_r['images_per_sec']:.1f} img/s)")
        print(f"  Speedup:    {speedup:.2f}x")
    elif len(results) == 1:
        print(f"\n  Only one loader tested: {results[0]['avg_ms']:.2f} ms/batch")


if __name__ == '__main__':
    main()
