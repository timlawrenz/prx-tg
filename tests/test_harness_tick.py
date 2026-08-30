"""Unit tests for the research-loop tick state machine (M3).

Mirrors the stratum-ffhq harness regression suite names:
strike path, one-active invariant, exploration slot, novelty bonus, ties,
blocked skipping, needs-human hold, champion advance, mutation guard,
calibration hard error, determinism.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "harness"))
import tick  # noqa: E402


# ---------- fixtures ----------

def cand(cid, state="registered", prior="high", meas="high", cost="med",
         evidence=("g0a_sensor_noise_floor",), gate="default_champion_gate", strikes=0):
    return {
        "id": cid, "state": state, "prior": prior, "measurability": meas,
        "cost": cost, "strikes": strikes, "evidence_parts": list(evidence),
        "declaration": {
            "scope": "t", "differs_from": "champion", "output_semantics": "o",
            "provenance": "p", "abstention": "a", "qualification_gate": gate,
            "expected_gpu_hours": 30, "config": "experiments/x/config.yaml",
            "arm_issue": 0,
        },
        "verdicts": [],
    }


def registry(cands, selection_progress=0, every_n=5, strike_limit=3, calib="cal.json"):
    return {
        "schema_version": 1,
        "champion": {"slug": "base", "checkpoint": "release/base.safetensors", "measured": {}},
        "selection_progress": selection_progress,
        "exploration": {"every_n": every_n, "novelty_bonus": 0.25},
        "falsification": {"strike_limit": strike_limit},
        "cost_tiers_gpu_hours": {"low": 30, "med": 60},
        "calibration_file": calib,
        "candidates": cands,
    }


CALIB = {
    "schema_version": 1,
    "gates": {
        "g0a_sensor_noise_floor": {"n": 978, "p5": -6.0, "p95": -2.0},
        "g0b_spectral_slope": {"n": 978, "p5": -2.5, "p95": -1.5},
        "g0c_skin_texture_energy": {"n": 978, "p5": 0.01, "p95": 0.5},
    },
}

IN_BAND = [
    {"gate_id": "g0a_sensor_noise_floor", "value": -4.0, "n": 240},
    {"gate_id": "g0b_spectral_slope", "value": -2.0, "n": 240},
    {"gate_id": "g0c_skin_texture_energy", "value": 0.2, "n": 240},
]
OUT_OF_BAND = [
    {"gate_id": "g0a_sensor_noise_floor", "value": -9.0, "n": 240},
    {"gate_id": "g0b_spectral_slope", "value": -2.0, "n": 240},
    {"gate_id": "g0c_skin_texture_energy", "value": 0.2, "n": 240},
]
VOTES_PASS = {"wins": 60, "losses": 30, "ties": 10}
VOTES_FAIL = {"wins": 40, "losses": 50, "ties": 10}


@pytest.fixture
def world(tmp_path):
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps(CALIB))
    return tmp_path, str(cal)


def write_reg(p, reg):
    p.write_text(json.dumps(reg))
    return p


def write_gates(d, lines):
    f = d / "gates.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in lines))
    return f


def write_votes(d, votes):
    f = d / "votes.jsonl"
    f.write_text(json.dumps(votes))
    return f


# ---------- tests ----------

def test_wilson_bounds():
    assert tick.wilson_lb(65.0, 100) > 0.5
    assert tick.wilson_lb(45.0, 100) < 0.5


def test_strike_keeps_same_arm_active(world, capsys):
    d, cal = world
    reg = registry([cand("a", state="active")])
    p = write_reg(d / "registry.json", reg)
    rc = tick.main(["--registry", str(p), "--calibration", cal,
                    "--gates", str(write_gates(d, OUT_OF_BAND)), "--write"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["next_action"] == "research_pending"
    assert out["strike"] == 1
    after = json.loads(p.read_text())
    assert after["candidates"][0]["state"] == "active"
    assert after["candidates"][0]["strikes"] == 1
    assert len([c for c in after["candidates"] if c["state"] == "active"]) == 1


def test_third_strike_falsifies_then_activates_next(world, capsys):
    d, cal = world
    reg = registry([cand("a", state="active", strikes=2), cand("b")])
    p = write_reg(d / "registry.json", reg)
    rc = tick.main(["--registry", str(p), "--calibration", cal,
                    "--gates", str(write_gates(d, OUT_OF_BAND)), "--write"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["next_action"] == "falsified_then_select"
    assert out["activated"] == "b"
    after = json.loads(p.read_text())
    assert after["candidates"][0]["state"] == "terminal_falsified"
    assert after["candidates"][1]["state"] == "active"


def test_needs_human_when_votes_missing(world, capsys):
    d, cal = world
    reg = registry([cand("a", state="active")])
    p = write_reg(d / "registry.json", reg)
    rc = tick.main(["--registry", str(p), "--calibration", cal,
                    "--gates", str(write_gates(d, IN_BAND)), "--write"])
    out = json.loads(capsys.readouterr().out)
    assert out["next_action"] == "needs_human"
    after = json.loads(p.read_text())
    assert after["candidates"][0]["strikes"] == 0  # missing votes never strike


def test_champion_advance_on_validated(world, capsys):
    d, cal = world
    reg = registry([cand("a", state="active"), cand("b")])
    p = write_reg(d / "registry.json", reg)
    rc = tick.main(["--registry", str(p), "--calibration", cal,
                    "--gates", str(write_gates(d, IN_BAND)),
                    "--votes", str(write_votes(d, VOTES_PASS)), "--write"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["next_action"] == "validate_then_select"
    assert out["verdict"]["bootstrap_waived_gain"] is True
    after = json.loads(p.read_text())
    assert after["candidates"][0]["state"] == "terminal_validated"
    assert after["champion"]["slug"] == "a"
    assert after["champion"]["measured"]["g0a_sensor_noise_floor"] == -4.0
    assert after["candidates"][1]["state"] == "active"


def test_votes_fail_is_not_better_strike(world, capsys):
    d, cal = world
    reg = registry([cand("a", state="active")])
    p = write_reg(d / "registry.json", reg)
    tick.main(["--registry", str(p), "--calibration", cal,
               "--gates", str(write_gates(d, IN_BAND)),
               "--votes", str(write_votes(d, VOTES_FAIL)), "--write"])
    out = json.loads(capsys.readouterr().out)
    assert out["next_action"] == "research_pending"
    assert out["strike"] == 1


def test_mid_tick_mutation_refused(world, monkeypatch, capsys):
    d, cal = world
    reg = registry([cand("a", state="active")])
    p = write_reg(d / "registry.json", reg)

    real = Path.read_bytes
    calls = {}

    def fake(self):
        if self.name == "registry.json":
            n = calls.get(str(self), 0) + 1
            calls[str(self)] = n
            if n >= 2:  # mutation between load and write
                mutated = json.loads(real(self))
                mutated["candidates"][0]["strikes"] = 99
                return json.dumps(mutated).encode()
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", fake)
    rc = tick.main(["--registry", str(p), "--calibration", cal,
                    "--gates", str(write_gates(d, OUT_OF_BAND)), "--write"])
    err = capsys.readouterr().err
    assert rc == 3
    assert "changed on disk" in err
    # on-disk registry untouched by the refused write
    assert json.loads(p.read_text())["candidates"][0]["strikes"] == 0


def test_explore_slot_forces_lowest_prior(world, capsys):
    d, cal = world
    # selection_progress=4 → next selection index 5 → explore slot
    reg = registry([cand("high-prior", prior="high"),
                    cand("low-prior", prior="low", cost="high")],
                   selection_progress=4)
    p = write_reg(d / "registry.json", reg)
    tick.main(["--registry", str(p), "--calibration", cal, "--write"])
    out = json.loads(capsys.readouterr().out)
    assert out["next_action"] == "activate"
    assert out["selected_via"] == "explore"
    assert out["activated"] == "low-prior"


def test_exploit_picks_highest_eig(world, capsys):
    d, cal = world
    reg = registry([cand("weak", prior="low", cost="high"),
                    cand("strong", prior="high", cost="low")])
    p = write_reg(d / "registry.json", reg)
    tick.main(["--registry", str(p), "--calibration", cal, "--write"])
    out = json.loads(capsys.readouterr().out)
    assert out["selected_via"] == "exploit"
    assert out["activated"] == "strong"
    table = {r["id"]: r for r in out["score_table"]}
    assert table["strong"]["eig"] > table["weak"]["eig"]


def test_novelty_bonus_only_for_new_evidence(world, capsys):
    d, cal = world
    # established: g0a is terminal-validated; g0b is novel
    reg = registry([
        cand("established", state="terminal_validated", evidence=["g0a_sensor_noise_floor"]),
        cand("redundant", evidence=["g0a_sensor_noise_floor"]),
        cand("novel", evidence=["g0b_spectral_slope"]),
    ])
    p = write_reg(d / "registry.json", reg)
    tick.main(["--registry", str(p), "--calibration", cal, "--write"])
    out = json.loads(capsys.readouterr().out)
    table = {r["id"]: r for r in out["score_table"]}
    assert table["novel"]["novelty_bonus"] == 0.25
    assert table["redundant"]["novelty_bonus"] == 0.0
    # identical base scores → bonus decides the tie → novel wins
    assert out["activated"] == "novel"


def test_ties_broken_by_id(world, capsys):
    d, cal = world
    reg = registry([cand("zeta"), cand("alpha")])  # identical params
    p = write_reg(d / "registry.json", reg)
    tick.main(["--registry", str(p), "--calibration", cal, "--write"])
    out = json.loads(capsys.readouterr().out)
    assert out["activated"] == "alpha"  # lexicographically smallest id wins ties


def test_blocked_skipped_in_selection(world, capsys):
    d, cal = world
    reg = registry([cand("blocked-strong", state="blocked"),
                    cand("plain", prior="med", meas="low", cost="high")])
    p = write_reg(d / "registry.json", reg)
    tick.main(["--registry", str(p), "--calibration", cal, "--write"])
    out = json.loads(capsys.readouterr().out)
    assert out["activated"] == "plain"
    assert all(r["id"] != "blocked-strong" for r in out["score_table"])


def test_calibration_missing_hard_error(world, capsys):
    d, _ = world
    reg = registry([cand("a", state="active")])
    reg["calibration_file"] = str(d / "nope.json")
    p = write_reg(d / "registry.json", reg)
    rc = tick.main(["--registry", str(p),
                    "--gates", str(write_gates(d, IN_BAND)), "--write"])
    assert rc == 2
    assert "calibration" in capsys.readouterr().err


def test_band_containment_no_votes_needed(world, capsys):
    d, cal = world
    reg = registry([cand("a", state="active", gate="band_containment")])
    p = write_reg(d / "registry.json", reg)
    rc = tick.main(["--registry", str(p), "--calibration", cal,
                    "--gates", str(write_gates(d, IN_BAND)), "--write"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["next_action"] == "validate_then_select"


def test_phase4_gated_holds(world, capsys):
    d, cal = world
    reg = registry([cand("a", state="active", gate="phase4_gated")])
    p = write_reg(d / "registry.json", reg)
    tick.main(["--registry", str(p), "--calibration", cal,
               "--gates", str(write_gates(d, IN_BAND)), "--write"])
    out = json.loads(capsys.readouterr().out)
    assert out["next_action"] == "needs_human"
    assert out["reason"] == "phase4_gated requires owner ruling"


def test_determinism_same_input_same_verdict(world, capsys):
    d, cal = world
    reg = registry([cand("a", state="active")])
    p = write_reg(d / "registry.json", reg)
    g = write_gates(d, OUT_OF_BAND)
    outs = []
    for _ in range(2):
        tick.main(["--registry", str(p), "--calibration", cal, "--gates", str(g)])
        outs.append(capsys.readouterr().out)
    assert outs[0] == outs[1]


def test_noop_when_nothing_actionable(world, capsys):
    d, cal = world
    reg = registry([cand("done", state="terminal_validated"),
                    cand("killed", state="terminal_falsified")])
    p = write_reg(d / "registry.json", reg)
    rc = tick.main(["--registry", str(p), "--calibration", cal])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["next_action"] == "none"


def test_one_active_invariant_hard_fails(world, capsys):
    d, cal = world
    reg = registry([cand("a", state="active"), cand("b", state="active")])
    p = write_reg(d / "registry.json", reg)
    rc = tick.main(["--registry", str(p), "--calibration", cal])
    assert rc == 2
    assert "invariant" in capsys.readouterr().err
