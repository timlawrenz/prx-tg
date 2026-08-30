"""Shared constants for the research-loop harness — single source of truth.

Imported by registry_validate.py, tick.py, and propose.py. A candidate whose
qualification_gate is not in KNOWN_GATES is rejected: a trick is payable only
if an existing gate can measure it, and new gates must be registered +
calibrated FIRST (this is the mechanism that stops attractive-mask artifacts
at the selection level).
"""

KNOWN_GATES = {
    "default_champion_gate",  # blind win-rate CI>0.5 vs champion + G0 in-band + no G1 regression
    "stability_only",         # velocity/grad stability (thresholds not yet calibrated -> needs_human)
    "band_containment",       # G0 gates in-band only (e.g. ps32-adapt throughput tradeoff)
    "phase4_gated",           # owner ruling required (structural escalations)
}

DECLARATION_FIELDS = [
    "scope", "differs_from", "output_semantics", "provenance", "abstention",
    "qualification_gate", "expected_gpu_hours", "config", "arm_issue",
]

LEVELS = {
    "prior": {"high", "med", "low"},
    "measurability": {"high", "med", "low"},
    "cost": {"low", "med", "high"},
}

STATES = {"registered", "active", "terminal_validated", "terminal_falsified", "blocked"}
