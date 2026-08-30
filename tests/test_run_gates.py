"""Unit tests for the G0 gate producer (Phase 1)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "harness"))
import run_gates  # noqa: E402


def _write_png(tmp_path, name, arr):
    img = Image.fromarray((arr * 255).astype(np.uint8), "RGB")
    p = tmp_path / name
    img.save(p)
    return p


def test_noise_vs_smooth_separated(tmp_path):
    rng = np.random.default_rng(0)
    noisy = np.clip(rng.normal(0.5, 0.15, (128, 128, 3)), 0, 1).astype(np.float32)
    smooth = np.full((128, 128, 3), 0.5, dtype=np.float32)
    p_noisy = _write_png(tmp_path, "noisy.png", noisy)
    p_smooth = _write_png(tmp_path, "smooth.png", smooth)

    vn = run_gates.measure_image(p_noisy)
    vs = run_gates.measure_image(p_smooth)
    # Sensor-noise floor: noisy image must have (much) higher log10 std residual
    assert vn["g0a_sensor_noise_floor_full"] > vs["g0a_sensor_noise_floor_full"]
    assert np.isfinite(list(vn.values())).all()
    assert np.isfinite(list(vs.values())).all()


def test_cli_end_to_end(tmp_path, capsys):
    rng = np.random.default_rng(1)
    for i in range(3):
        arr = np.clip(rng.normal(0.5, 0.1, (64, 64, 3)), 0, 1).astype(np.float32)
        _write_png(tmp_path, f"p{i:02d}.png", arr)
    out = tmp_path / "gates.jsonl"
    rc = run_gates.main(["--images", str(tmp_path), "--out", str(out)])
    captured = capsys.readouterr().out
    assert rc == 0
    assert "3 images" in captured
    lines = [l for l in out.read_text().strip().splitlines()]
    ids = [eval(l)["gate_id"] for l in lines]  # noqa: S307 — test-local, trusted content
    assert "g0a_sensor_noise_floor_full" in ids
    assert "g0b_spectral_slope" in ids
    assert "g0d_local_contrast_full_p95" in ids
    # every line carries the aggregation contract the tick consumes
    for l in lines:
        rec = eval(l)  # noqa: S307
        assert {"gate_id", "value", "n"} <= set(rec)
        assert rec["n"] == 3
    assert (tmp_path / "gates_detail.jsonl").is_file()


def test_cli_no_images_errors(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = run_gates.main(["--images", str(empty), "--out", str(tmp_path / "g.jsonl")])
    assert rc == 2
    assert "no PNG/JPG" in capsys.readouterr().err
