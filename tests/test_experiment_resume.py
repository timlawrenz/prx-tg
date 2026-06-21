"""
Regression tests for experiment resume path.

Tests the create_experiment_dir function in production/train_production.py
to ensure it correctly reuses existing experiment directories when --resume
is provided, for both nested (slug/runs/timestamp) and legacy structures.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from production.train_production import create_experiment_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TempExperiments:
    """Create a temp directory that mimics the 'experiments/' structure.
    
    Because create_experiment_dir does `Path('experiments').resolve()`, we
    monkeypatch that to point at a temp dir.
    """
    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path / "experiments"
        self.root.mkdir()
        monkeypatch.setattr(
            "production.train_production.Path",
            _make_fake_path_class(self.root),
        )


def _make_fake_path_class(exp_root):
    """Return a Path subclass whose .resolve() redirects 'experiments' to exp_root."""
    orig_path = Path

    class FakePath(orig_path):
        def resolve(self, strict=False):
            if str(self) == "experiments":
                return exp_root
            return orig_path.resolve(self, strict=strict)

    return FakePath


def _make_nested_run(exp_root: Path, slug: str, ts: str, step: int = 3500):
    """Create a mock slug/runs/timestamp experiment directory."""
    run_dir = exp_root / slug / "runs" / ts
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("model:\n  hidden_size: 768\n")
    (run_dir / "metadata.json").write_text(json.dumps({
        "timestamp": "2026-06-20T11:25:00",
        "config_path": str(run_dir / "config.yaml"),
        "command": "train_production.py --config ...",
        "git_commit": "abc1234",
        "git_dirty": False,
    }))
    ckpt = ckpt_dir / f"checkpoint_step{step:07d}.pt"
    ckpt.write_text("fake checkpoint")
    return run_dir, ckpt


def _make_legacy_run(exp_root: Path, ts: str, step: int = 3500):
    """Create a mock legacy experiments/{timestamp} directory."""
    run_dir = exp_root / ts
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("model:\n  hidden_size: 768\n")
    (run_dir / "metadata.json").write_text(json.dumps({
        "timestamp": "2026-06-20T11:25:00",
        "config_path": str(run_dir / "config.yaml"),
        "command": "train_production.py --config ...",
        "git_commit": "abc1234",
        "git_dirty": False,
    }))
    ckpt = ckpt_dir / f"checkpoint_step{step:07d}.pt"
    ckpt.write_text("fake checkpoint")
    return run_dir, ckpt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNestedResume:
    """Resume with nested structure: experiments/{slug}/runs/{timestamp}/"""

    def test_reuses_existing_directory(self, tmp_path, monkeypatch):
        slug = "spatial-window-baseline"
        ts = "2026-06-20_1125"
        exp_root = tmp_path / "experiments"
        exp_root.mkdir()
        FakePath = _make_fake_path_class(exp_root)
        monkeypatch.setattr("production.train_production.Path", FakePath)

        run_dir, ckpt = _make_nested_run(exp_root, slug, ts)

        result = create_experiment_dir(
            str(run_dir / "config.yaml"), resume_path=str(ckpt)
        )
        assert result.resolve() == run_dir.resolve()

    def test_adds_resume_entry_to_metadata(self, tmp_path, monkeypatch):
        slug = "spatial-window-baseline"
        ts = "2026-06-20_1125"
        exp_root = tmp_path / "experiments"
        exp_root.mkdir()
        FakePath = _make_fake_path_class(exp_root)
        monkeypatch.setattr("production.train_production.Path", FakePath)

        run_dir, ckpt = _make_nested_run(exp_root, slug, ts)

        create_experiment_dir(
            str(run_dir / "config.yaml"), resume_path=str(ckpt)
        )
        data = json.loads((run_dir / "metadata.json").read_text())
        assert "resumes" in data
        assert len(data["resumes"]) == 1

    def test_no_duplicate_dirs(self, tmp_path, monkeypatch):
        slug = "spatial-window-baseline"
        ts = "2026-06-20_1125"
        exp_root = tmp_path / "experiments"
        exp_root.mkdir()
        FakePath = _make_fake_path_class(exp_root)
        monkeypatch.setattr("production.train_production.Path", FakePath)

        run_dir, ckpt = _make_nested_run(exp_root, slug, ts)
        before = len(list((exp_root / slug / "runs").iterdir()))

        create_experiment_dir(
            str(run_dir / "config.yaml"), resume_path=str(ckpt)
        )
        after = len(list((exp_root / slug / "runs").iterdir()))
        assert after == before, f"Before={before}, After={after}"


class TestLegacyResume:
    """Resume with legacy flat structure: experiments/{timestamp}/"""

    def test_reuses_existing_directory(self, tmp_path, monkeypatch):
        ts = "2026-06-20_1125"
        exp_root = tmp_path / "experiments"
        exp_root.mkdir()
        FakePath = _make_fake_path_class(exp_root)
        monkeypatch.setattr("production.train_production.Path", FakePath)

        run_dir, ckpt = _make_legacy_run(exp_root, ts)

        result = create_experiment_dir(
            str(run_dir / "config.yaml"), resume_path=str(ckpt)
        )
        assert result.resolve() == run_dir.resolve()

    def test_adds_resume_entry_to_metadata(self, tmp_path, monkeypatch):
        ts = "2026-06-20_1125"
        exp_root = tmp_path / "experiments"
        exp_root.mkdir()
        FakePath = _make_fake_path_class(exp_root)
        monkeypatch.setattr("production.train_production.Path", FakePath)

        run_dir, ckpt = _make_legacy_run(exp_root, ts)

        create_experiment_dir(
            str(run_dir / "config.yaml"), resume_path=str(ckpt)
        )
        data = json.loads((run_dir / "metadata.json").read_text())
        assert "resumes" in data


class TestFallbackForUnrecognizedPath:
    """When resume path is unrecognized, fall back to creating a new directory."""

    def test_creates_new_dir_for_bogus_resume_path(self, tmp_path, monkeypatch):
        exp_root = tmp_path / "experiments"
        exp_root.mkdir()
        FakePath = _make_fake_path_class(exp_root)
        monkeypatch.setattr("production.train_production.Path", FakePath)

        # Create a bogus resume path with no recognizable structure
        bogus_run = tmp_path / "weird_structure"
        bogus_ckpt = bogus_run / "checkpoints" / "checkpoint.pt"
        bogus_ckpt.parent.mkdir(parents=True)
        bogus_ckpt.write_text("fake")

        # Need a valid config — create one at experiments/some_config.yaml
        config = exp_root / "some_config.yaml"
        config.write_text("model:\n  hidden_size: 768\n")

        # Should fall through resume detection and create a new dir
        result = create_experiment_dir(str(config), resume_path=str(bogus_ckpt))
        assert result.exists()
        # Config is at experiments/some_config.yaml (not under a slug),
        # so slug detection doesn't trigger — falls back to legacy experiments/{ts}/
        assert result.parent.name == "experiments"


class TestSlugDetection:
    """Non-resume: config at experiments/{slug}/config.yaml creates slug/runs/{ts}/."""

    def test_creates_nested_run_directory(self, tmp_path, monkeypatch):
        slug = "my-ablation"
        exp_root = tmp_path / "experiments"
        exp_root.mkdir()
        FakePath = _make_fake_path_class(exp_root)
        monkeypatch.setattr("production.train_production.Path", FakePath)

        slug_dir = exp_root / slug
        slug_dir.mkdir()
        config = slug_dir / "config.yaml"
        config.write_text("model:\n  hidden_size: 768\n")

        result = create_experiment_dir(str(config))
        assert result.parent.name == "runs"
        assert result.parent.parent.name == slug
