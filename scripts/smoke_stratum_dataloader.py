#!/usr/bin/env python3
"""Smoke-test the stratum dataloader WIRING, then load real data through it.

Why this exists
---------------
Commit 1b5a03b wired the identity-basis guard by passing `basis_fingerprint=`
to `get_stratum_dataloader`, whose parameter is named
`expected_basis_fingerprint`. Python raises TypeError on an unexpected keyword,
so EVERY stratum run — not just eidolon arms — failed at dataloader
construction. The guard's own unit tests passed, because they called the guard
function directly and never went through the call site.

Lesson encoded here: test the SEAM, not just the function. Two checks:

  1. STATIC — every keyword name passed at each call site in production/data.py
     must exist in the target function's signature. This is what catches a
     renamed parameter without needing a GPU or a dataset.
  2. RUNTIME — build the dataloader for real (both roots), latent eidolon mode,
     and assert the shapes/norms the model will actually receive.

Exit non-zero on any failure.
"""
import ast
import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FFHQ = "/mnt/nas-ai-models/training-data/ffhq/stratum"
HEGRE = "/mnt/nas-ai-models/training-data/eidolon/hegre_corpus"
BASIS = "120e1c5a1dc4f423"

failures = []


def check_static_wiring() -> None:
    """Every kwarg at each get_stratum_dataloader(...) call site must be real."""
    from production.data_stratum import get_stratum_dataloader

    params = set(inspect.signature(get_stratum_dataloader).parameters)
    src = (REPO / "production" / "data.py").read_text()
    tree = ast.parse(src)

    call_sites = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name != "get_stratum_dataloader":
            continue
        call_sites += 1
        for kw in node.keywords:
            if kw.arg is None:      # **kwargs splat — cannot check statically
                continue
            if kw.arg not in params:
                failures.append(
                    f"STATIC: production/data.py:{node.lineno} passes "
                    f"`{kw.arg}=` but get_stratum_dataloader has no such parameter "
                    f"(declared: {sorted(params)})")

    if call_sites == 0:
        failures.append("STATIC: found no get_stratum_dataloader call site to check")
    else:
        print(f"  static: {call_sites} call site(s) checked against "
              f"{len(params)} declared parameters")


def check_runtime(root: str, label: str, expect_ch: int = 16) -> None:
    """Build for real and assert what the model will receive."""
    from production.data_stratum import get_stratum_dataloader

    if not Path(root).is_dir():
        failures.append(f"RUNTIME: {label} root missing: {root}")
        return

    ds = get_stratum_dataloader(
        stratum_dir=root,
        batch_size=4,
        shuffle=False,
        target_latent_size=128,
        max_samples=4,
        adapter_name="eidolon",
        latent_mode=True,
        load_dino_patches=False,
        load_seg=False,
        load_geometry_3d=False,
        load_matting=False,
        expected_basis_fingerprint=BASIS,
    )

    n = min(4, len(ds._dirs))
    if n == 0:
        failures.append(f"RUNTIME: {label} yielded no samples")
        return

    for d in ds._dirs[:n]:
        s = ds._load_sample_eidolon(d)
        img, ident, geo = s["image_data"], s["identity_emb"], s["geometry_emb"]

        if tuple(img.shape) != (expect_ch, 128, 128):
            failures.append(
                f"RUNTIME: {label} image_data is {tuple(img.shape)}, expected "
                f"({expect_ch}, 128, 128) — a latent arm is being fed pixels")
        norm = float(ident.norm())
        if abs(norm - 1.0) > 0.02:
            failures.append(
                f"RUNTIME: {label} identity norm {norm:.4f} outside the unit-norm "
                f"band — wrong basis or unnormalised slot")
        if tuple(geo.shape) != (50,):
            failures.append(f"RUNTIME: {label} geometry_emb is {tuple(geo.shape)}, expected (50,)")

    # collate must accept it (the seam the trainer actually crosses)
    from production.data_stratum import _collate
    b = _collate([ds._load_sample_eidolon(d) for d in ds._dirs[:n]])
    bs = b["image_data"].shape[0]
    print(f"  runtime {label:6s}: {n} samples, batch image_data "
          f"{tuple(b['image_data'].shape)}, identity "
          f"{tuple(b['identity_emb'].shape)} norm "
          f"{float(b['identity_emb'].norm(dim=-1).mean()):.4f}, geometry "
          f"{tuple(b['geometry_emb'].shape)}")
    if bs != n:
        failures.append(f"RUNTIME: {label} collate returned batch {bs}, expected {n}")


if __name__ == "__main__":
    print("=== 1. static wiring check (the seam that broke) ===")
    check_static_wiring()
    print("=== 2. runtime load, latent + eidolon ===")
    check_runtime(FFHQ, "ffhq")
    check_runtime(HEGRE, "hegre")

    print()
    if failures:
        print(f"FAIL ({len(failures)}):")
        for f in failures:
            print("  ✗", f)
        sys.exit(1)
    print("PASS — wiring closed, latent eidolon path loads on both roots")
