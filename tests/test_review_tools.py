"""Unit tests for the blind-review tooling (pool + aggregator)."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "harness"))
import review_pool  # noqa: E402
import review_aggregate  # noqa: E402


def _img_dir(tmp_path, name, color, n=6):
    d = tmp_path / name
    d.mkdir()
    for i in range(n):
        Image.fromarray(np.full((64, 64, 3), color, dtype=np.uint8)).save(d / f"{i:03d}.png")
    return d


def _write_votes(path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(v) for v in pairs) + "\n")


def test_pool_deterministic_and_blind(tmp_path):
    a = _img_dir(tmp_path, "a", (255, 0, 0))
    b = _img_dir(tmp_path, "b", (0, 255, 0))
    real = _img_dir(tmp_path, "real", (0, 0, 255))
    out = tmp_path / "review"
    rc = review_pool.main(["--a-dir", str(a), "--b-dir", str(b),
                           "--real-dir", str(real), "--block", "realism",
                           "--n-pairs", "10", "--seed", "42", "--out-dir", str(out)])
    assert rc == 0
    pool = json.loads((out / "realism" / "pool.json").read_text())
    assert pool["n_ab"] == 8 and pool["n_calibration"] == 2
    assert pool["block"] == "realism"
    # every pair has a rendered unlabeled png + hidden ground truth in JSON only
    for p in pool["pairs"]:
        assert Path(p["image"]).is_file()
        assert "left_model" in p and "right_model" in p
        assert "left_is_real" in p
    # determinism: rebuild with same seed -> identical pair structure
    rc2 = review_pool.main(["--a-dir", str(a), "--b-dir", str(b),
                            "--real-dir", str(real), "--block", "realism",
                            "--n-pairs", "10", "--seed", "42", "--out-dir", str(out)])
    pool2 = json.loads((out / "realism" / "pool.json").read_text())
    assert [p["pair_id"] for p in pool2["pairs"]] == [p["pair_id"] for p in pool["pairs"]]


def test_aggregate_happy_path(tmp_path):
    a = _img_dir(tmp_path, "a", (255, 0, 0))
    b = _img_dir(tmp_path, "b", (0, 255, 0))
    real = _img_dir(tmp_path, "real", (0, 0, 255))
    out = tmp_path / "review"
    review_pool.main(["--a-dir", str(a), "--b-dir", str(b), "--real-dir", str(real),
                      "--block", "realism", "--n-pairs", "10", "--seed", "7",
                      "--out-dir", str(out)])
    pool = json.loads((out / "realism" / "pool.json").read_text())
    by_id = {p["pair_id"]: p for p in pool["pairs"]}

    # Rater 1: always picks B; rater 2: always picks B; calibration always real
    v1, v2 = [], []
    for p in pool["pairs"]:
        if p["kind"] == "ab":
            pick = "L" if p["left_model"] == "b" else "R"
            v1.append({"pair_id": p["pair_id"], "rater": "t", "choice": pick})
            v2.append({"pair_id": p["pair_id"], "rater": "u", "choice": pick})
        else:
            pick = "L" if p["left_is_real"] else "R"
            v1.append({"pair_id": p["pair_id"], "rater": "t", "choice": pick})
            v2.append({"pair_id": p["pair_id"], "rater": "u", "choice": pick})
    vp1 = tmp_path / "v1.jsonl"
    vp2 = tmp_path / "v2.jsonl"
    _write_votes(vp1, v1)
    _write_votes(vp2, v2)

    out_json = tmp_path / "votes.json"
    rc = review_aggregate.main(["--pool", str(out / "realism" / "pool.json"),
                                "--votes", str(vp1), "--votes", str(vp2),
                                "--out", str(out_json)])
    assert rc == 0
    r = json.loads(out_json.read_text())
    # pooled across both raters: 8 ab pairs × 2 raters = 16 votes, all for B
    assert r["wins"] == 16 and r["losses"] == 0 and r["n_pairs"] == 16
    assert r["wilson_lb"] > 0.5
    assert r["calibration"]["pass"] is True
    assert r["session_valid"] is True
    assert r["raters"] == ["t", "u"]
    assert r["inter_rater_agreement"] == 1.0


def test_calibration_failure_invalidates_session(tmp_path):
    a = _img_dir(tmp_path, "a", (255, 0, 0))
    b = _img_dir(tmp_path, "b", (0, 255, 0))
    real = _img_dir(tmp_path, "real", (0, 0, 255))
    out = tmp_path / "review"
    review_pool.main(["--a-dir", str(a), "--b-dir", str(b), "--real-dir", str(real),
                      "--block", "realism", "--n-pairs", "10", "--seed", "7",
                      "--out-dir", str(out)])
    pool = json.loads((out / "realism" / "pool.json").read_text())
    votes = []
    for p in pool["pairs"]:
        if p["kind"] == "ab":
            pick = "L" if p["left_model"] == "b" else "R"
        else:
            # rater picks the MODEL side on calibration pairs (bad rater)
            pick = "R" if p["left_is_real"] else "L"
        votes.append({"pair_id": p["pair_id"], "rater": "t", "choice": pick})
    vp = tmp_path / "v1.jsonl"
    _write_votes(vp, votes)
    rc = review_aggregate.main(["--pool", str(out / "realism" / "pool.json"),
                                "--votes", str(vp), "--out", str(tmp_path / "votes.json")])
    assert rc == 4  # session invalid, distinct exit code
    r = json.loads((tmp_path / "votes.json").read_text())
    assert r["session_valid"] is False
    assert r["calibration"]["real_win_rate"] == 0.0


def test_unknown_pair_rejected(tmp_path):
    pool = {"block": "realism", "pairs": [{"pair_id": "realism_ab_0000", "kind": "ab",
                                           "left_model": "a", "right_model": "b",
                                           "left_is_real": False, "right_is_real": False}]}
    pp = tmp_path / "pool.json"
    pp.write_text(json.dumps(pool))
    vp = tmp_path / "v.jsonl"
    _write_votes(vp, [{"pair_id": "nonexistent", "rater": "t", "choice": "L"}])
    rc = review_aggregate.main(["--pool", str(pp), "--votes", str(vp),
                                "--out", str(tmp_path / "votes.json")])
    assert rc == 2