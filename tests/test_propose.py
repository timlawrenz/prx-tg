"""Unit tests for the gated proposal command (M4)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "harness"))
import propose  # noqa: E402


def base_registry(tmp_path):
    return {
        "schema_version": 1,
        "champion": {"slug": "base", "checkpoint": "release/base.safetensors", "measured": {}},
        "selection_progress": 0,
        "exploration": {"every_n": 5, "novelty_bonus": 0.25},
        "falsification": {"strike_limit": 3},
        "cost_tiers_gpu_hours": {"low": 30, "med": 60},
        "calibration_file": "research/avenues/gates_calibration.json",
        "candidates": [],
    }


def good_candidate(config_path, evidence=("g0a_sensor_noise_floor_full",)):
    return {
        "id": "new-arm", "prior": "med", "measurability": "med", "cost": "med",
        "evidence_parts": list(evidence),
        "declaration": {
            "scope": "s", "differs_from": "champion", "output_semantics": "o",
            "provenance": "p", "abstention": "a",
            "qualification_gate": "default_champion_gate",
            "expected_gpu_hours": 30, "config": str(config_path), "arm_issue": 0,
        },
    }


@pytest.fixture
def world(tmp_path):
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps(base_registry(tmp_path)))
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: {}\n")
    return reg_path, cfg


def run_propose(reg_path, cands, write=False, require_new=False, cal=None):
    cands_path = reg_path.parent / "cands.json"
    cands_path.write_text(json.dumps(cands))
    argv = ["--registry", str(reg_path), "--candidates", str(cands_path)]
    if write:
        argv += ["--write"]
    if require_new:
        argv += ["--require-new-evidence-part"]
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = propose.main(argv)
    return rc, json.loads(buf.getvalue())


def test_accepts_valid_candidate(world, tmp_path):
    reg_path, cfg = world
    rc, report = run_propose(reg_path, [good_candidate(cfg)], write=True)
    assert rc == 0
    assert report["accepted"] == ["new-arm"]
    after = json.loads(reg_path.read_text())
    assert after["candidates"][0]["id"] == "new-arm"
    assert after["candidates"][0]["state"] == "registered"


def test_rejects_missing_declaration_field(world):
    reg_path, cfg = world
    c = good_candidate(cfg)
    del c["declaration"]["provenance"]
    rc, report = run_propose(reg_path, [c])
    assert rc == 1
    assert "provenance" in report["rejected"][0]["reasons"][0]


def test_rejects_missing_config(world, tmp_path):
    reg_path, cfg = world
    c = good_candidate(cfg)
    c["declaration"]["config"] = str(tmp_path / "nope.yaml")
    rc, report = run_propose(reg_path, [c])
    assert rc == 1
    assert any("not on disk" in r for r in report["rejected"][0]["reasons"])


def test_rejects_unregistered_gate(world):
    reg_path, cfg = world
    c = good_candidate(cfg)
    c["declaration"]["qualification_gate"] = "fancy_new_gate"
    rc, report = run_propose(reg_path, [c])
    assert rc == 1
    assert any("gate register" in r for r in report["rejected"][0]["reasons"])


def test_rejects_duplicate_id(world):
    reg_path, cfg = world
    run_propose(reg_path, [good_candidate(cfg)], write=True)
    rc, report = run_propose(reg_path, [good_candidate(cfg)])
    assert rc == 1
    assert any("already exists" in r for r in report["rejected"][0]["reasons"])


def test_require_new_evidence_part(world):
    reg_path, cfg = world
    reg = json.loads(reg_path.read_text())
    reg["candidates"].append({
        "id": "established", "state": "terminal_validated",
        "evidence_parts": ["g0a_sensor_noise_floor_full"],
    })
    reg_path.write_text(json.dumps(reg))
    # old part → rejected
    rc, report = run_propose(reg_path, [good_candidate(cfg)], require_new=True)
    assert rc == 1
    assert any("no NEW evidence part" in r for r in report["rejected"][0]["reasons"])
    # new part → accepted
    rc2, report2 = run_propose(reg_path,
                               [good_candidate(cfg, evidence=("g0b_spectral_slope",))],
                               require_new=True)
    assert rc2 == 0
    assert report2["accepted"] == ["new-arm"]


def test_rejects_bad_enums_and_hours(world):
    reg_path, cfg = world
    c = good_candidate(cfg)
    c["prior"] = "extreme"
    c["declaration"]["expected_gpu_hours"] = -5
    rc, report = run_propose(reg_path, [c])
    assert rc == 1
    reasons = report["rejected"][0]["reasons"]
    assert any("prior" in r for r in reasons)
    assert any("positive number" in r for r in reasons)


def test_rejected_candidates_never_written(world):
    reg_path, cfg = world
    run_propose(reg_path, [good_candidate(cfg)], write=True)  # accepted
    before = reg_path.read_bytes()
    c = good_candidate(cfg)
    c["id"] = "bad-one"
    c["declaration"]["config"] = "missing.yaml"
    rc, _ = run_propose(reg_path, [c], write=True)
    assert rc == 1
    assert reg_path.read_bytes() == before  # registry untouched on rejection