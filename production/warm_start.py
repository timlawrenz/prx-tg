#!/usr/bin/env python3
"""Warm-start a prx-tg model from a checkpoint whose I/O projections differ.

WHY THIS EXISTS
---------------
`production/train.py:load_checkpoint()` calls `load_state_dict()` with the
default `strict=True`, so a **pixel-space** model cannot resume from a
**latent-space** checkpoint. The transformer body is architecture-identical
(same depth/hidden_size/conditioning), but three modules cannot transfer:

    module          latent (P1)          pixel (P2)
    x_embedder      Conv2d(16, 768, 2)   Conv2d(3, 768, 16)
    final_proj      Linear(768, 64)      Linear(768, 768)
    output_conv     Conv2d(16, 16, 3)    Conv2d(3, 3, 3)

This module transfers every tensor whose SHAPE MATCHES and re-initialises only
the ones that cannot, then REPORTS exactly what moved — so the provenance of a
warm start is auditable rather than assumed ("182 of 188 tensors transferred").

DELIBERATELY NOT TRANSFERRED
----------------------------
- **Optimizer state**: Muon/AdamW moments are shaped by the parameter set and
  tied to a schedule that is being restarted. A warm start gets a FRESH
  optimizer (user decision 2026-09-20).
- **RNG state**: a new run starts its own stream.
- **EMA buffer**: `EMAModel.__init__` clones the model's weights at
  construction time. Loading weights afterwards would leave the EMA holding the
  RANDOM init, so `reseed_ema()` must be called after the warm load to make the
  EMA start AT the warm state.
"""
from __future__ import annotations

from pathlib import Path

import torch


def _strip_orig_mod(name: str) -> str:
    """Drop the `_orig_mod.` prefix that torch.compile adds to state-dict keys."""
    return name[len("_orig_mod."):] if name.startswith("_orig_mod.") else name


def warm_start_from(model, ckpt_path: str | Path, source: str = "ema",
                    force_reinit: tuple[str, ...] = ("x_embedder", "final_proj",
                                                     "output_conv"),
                    verbose: bool = True) -> dict:
    """Load every shape-matching tensor from `ckpt_path` into `model`.

    Args:
        model: the freshly constructed target model (may be torch.compile-wrapped;
            `_orig_mod.` prefixes are normalised on both sides).
        ckpt_path: checkpoint to warm from.
        source: "ema" (default) = `ckpt['ema']['ema_params']`, the weights this
            project samples and releases from; "model" = `ckpt['model']`, the raw
            final training weights.
        force_reinit: module prefixes that are re-initialised ENTIRELY, even where
            a tensor's shape happens to match. Needed because shape alone is not
            the right test for the I/O projections: `x_embedder.proj.bias` is
            (768,) in BOTH latent and pixel space, so a pure shape rule would
            transfer the patch-embed bias while re-initialising its weight,
            leaving the module half-warm. Pass () for the pure shape rule.
        verbose: print the accounting.

    Returns:
        report dict: source, step, n_transferred, n_skipped, n_missing,
        transferred, skipped [(name, from_shape, to_shape, reason)], missing.
    """
    ckpt_path = Path(ckpt_path)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if source == "ema":
        ema = ck.get("ema")
        sd = ema.get("ema_params") if isinstance(ema, dict) else None
        if not sd:
            raise ValueError(
                f"{ckpt_path} has no ck['ema']['ema_params']; "
                "pass source='model' to warm from the raw weights instead."
            )
    elif source == "model":
        sd = ck.get("model")
        if not sd:
            raise ValueError(f"{ckpt_path} has no ck['model']")
    else:
        raise ValueError(f"source must be 'ema' or 'model', got {source!r}")

    src = {_strip_orig_mod(k): v for k, v in sd.items()}

    to_load: dict[str, torch.Tensor] = {}
    transferred: list[str] = []
    skipped: list[tuple[str, tuple | None, tuple, str]] = []
    missing: list[str] = []

    for name, param in model.state_dict().items():
        key = _strip_orig_mod(name)
        src_t = src.get(key)

        if any(key.startswith(p) for p in force_reinit):
            skipped.append((key,
                            tuple(src_t.shape) if src_t is not None else None,
                            tuple(param.shape),
                            "forced re-init (I/O projection module)"))
            continue
        if src_t is None:
            missing.append(key)
            continue
        if tuple(src_t.shape) != tuple(param.shape):
            skipped.append((key, tuple(src_t.shape), tuple(param.shape),
                            "shape mismatch"))
            continue
        to_load[name] = src_t.to(device=param.device, dtype=param.dtype)
        transferred.append(key)

    model.load_state_dict(to_load, strict=False)

    report = {
        "checkpoint": str(ckpt_path),
        "source": source,
        "step": ck.get("step"),
        "n_transferred": len(transferred),
        "n_skipped": len(skipped),
        "n_missing": len(missing),
        "transferred": sorted(transferred),
        "skipped": sorted(skipped),
        "missing": sorted(missing),
    }

    if verbose:
        print(f"\n{'='*66}\nWARM START from {ckpt_path} (source={source}, "
              f"step={ck.get('step')})")
        print(f"  transferred : {len(transferred)}")
        print(f"  re-init'd   : {len(skipped)}")
        for name, frm, to, reason in sorted(skipped):
            print(f"      {name}: {frm} -> {to}   [{reason}]")
        if missing:
            print(f"  MISSING from checkpoint: {len(missing)}")
            for name in sorted(missing)[:20]:
                print(f"      {name}")
        print(f"{'='*66}\n")

    return report


def reseed_ema(ema, model) -> int:
    """Copy current model weights into the EMA buffer; reset its warmup step.

    Must be called AFTER a warm load: `EMAModel.__init__` clones the model's
    weights when it is constructed, so without this the EMA would hold the
    random init while the model holds the warm start, and every early
    validation would sample a random model.

    Returns the number of EMA buffers re-seeded.
    """
    n = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and name in ema.ema_params:
                ema.ema_params[name].copy_(param.data)
                n += 1
    ema.step = 0
    return n
