#!/usr/bin/env python3
"""Step 0 of the Eidolon identity instrument: validate the RETRIEVAL machinery
on data that already exists — no VAE, no detection, no model.

STATUS — SUPERSEDED FOR THE PUBLISHED-NUMBER REPLICATION
--------------------------------------------------------
Do NOT use this script's default mode to reproduce the recorded result. Its
`corpus_centroids` index (321 unit-norm LDA centroids) is a DIFFERENT artefact
from the one the recorded number used (`averages/*.lda.npy`, 325 LDA-space
averages), so it measures a different thing: it returned R@1 0.8427. That is not
a failed replication, just the wrong index. The replication lives in
`eidolon_identity_replication.py`, which returns R@1 0.8910 / R@5 0.9502 /
R@10 0.9657 against recorded 0.8879 / 0.9502 / 0.9657.

The `B_cross_shoot` figure this script printed (R@1 0.8201) is also NOT
trustworthy: its queries were drawn from the whole v1 `lda` tree rather than
restricted to approved corpus images, so they can include tainted/stale vectors.
Re-measure with corpus-restricted queries before citing it.

Kept for the `corpus_centroids` and cross-shoot variants once reworked.

THE CONVENTION THAT MAKES THIS MEANINGFUL (trap 6)
--------------------------------------------------
  per-image LDA  (hegre-faces/v1/lda/)     raw coords,  norm ~150.9
  corpus centroid (hegre_corpus/.../      L2-normalized, norm exactly 1.0
                   auraface_lda.npy)
A cosine between them is meaningless until the query is L2-normalized. Scale is
invariant for Fisher J but NOT for a distance or retrieval metric.

Provenance recorded: both basis artifact hashes, both stamps' fingerprints, the
corpus manifest's content fingerprint (a DIFFERENT thing by design — see
tools/hegre_dataset/basis_fingerprint.py), the conventions used, counts, and the
prx-tg git commit. Fails loudly on a basis mismatch or a zero-result query set.
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BASIS_DIR = Path("/home/tim/source/activity/eidolon/experiments/geometry_pca/output")
V1 = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")
CORPUS = Path("/mnt/nas-ai-models/training-data/eidolon/hegre_corpus")
EXPECTED_FP = "120e1c5a1dc4f423"
BASIS_FILES = ("auraface_lda.npz", "auraface_preprocess.npz")


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def basis_provenance() -> dict:
    files = {n: sha16(BASIS_DIR / n) for n in BASIS_FILES}
    h = hashlib.sha256()
    for n in BASIS_FILES:
        h.update(n.encode())
        h.update((BASIS_DIR / n).read_bytes())
    return {"basis_dir": str(BASIS_DIR), "basis_files": files,
            "computed_basis_fingerprint": h.hexdigest()[:16]}


def read_stamp(d: Path) -> dict:
    f = d / "BASIS_FINGERPRINT.json"
    if not f.exists():
        raise SystemExit(f"FATAL: no BASIS_FINGERPRINT.json in {d} — refusing to measure")
    return json.loads(f.read_text())


def corpus_personas_and_centroids():
    """{persona: unit-norm 64-d centroid}. One read per persona: the corpus slot
    is bit-identical across a persona's samples (verified independently)."""
    samples = sorted(d for d in os.listdir(CORPUS) if (CORPUS / d).is_dir())
    per = {}
    for s in samples:
        per.setdefault(s.split("--")[0], []).append(s)
    cents, counts = {}, {}
    for persona, ss in per.items():
        v = np.load(CORPUS / ss[0] / "auraface_lda.npy").astype(np.float64)
        n = np.linalg.norm(v)
        cents[persona] = v / n if n > 0 else v
        counts[persona] = len(ss)
    return cents, counts


def query_files(persona: str, per_persona: int, rng, held_out_set: str | None):
    """Per-image LDA files for a persona, optionally excluding one set."""
    pdir = V1 / "lda" / "faces" / persona
    if not pdir.is_dir():
        return [], []
    sets = sorted(d for d in os.listdir(pdir) if (pdir / d).is_dir())
    if held_out_set is not None:
        sets = [s for s in sets if s != held_out_set]
    pool = []
    for s in sets:
        pool.extend(sorted((pdir / s).glob("*.npy")))
    if not pool:
        return [], sets
    rng.shuffle(pool)
    return pool[:per_persona], sets


def evaluate(qpath: Path, centroids: np.ndarray, names: list[str], idx: int):
    q = np.load(qpath).astype(np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        return None
    q = q / n                                   # trap 6: normalize the QUERY
    sims = centroids @ q                        # centroids are already unit-norm
    order = np.argsort(-sims)
    rank = int(np.where(order == idx)[0][0]) + 1
    return rank, float(sims[idx]), float(sims.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-persona", type=int, default=8)
    ap.add_argument("--max-personas", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="/tmp/eidolon_identity_step0.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    prov = basis_provenance()

    # Fail loud on a stale basis — the exact confound this project already paid for.
    for d in (V1 / "lda", CORPUS):
        st = read_stamp(d)
        if st.get("basis_fingerprint") != EXPECTED_FP:
            raise SystemExit(f"FATAL: {d} stamped {st.get('basis_fingerprint')} != {EXPECTED_FP}")
        if st.get("basis_files", {}).get("auraface_lda.npz") != prov["basis_files"]["auraface_lda.npz"]:
            raise SystemExit(f"FATAL: {d} basis artifact hash disagrees with on-disk basis")
    print(f"[basis] OK  fingerprint={EXPECTED_FP}  files={prov['basis_files']}")

    cents, counts = corpus_personas_and_centroids()
    names = sorted(cents)
    if args.max_personas:
        names = names[: args.max_personas]
    print(f"[index] {len(names)} persona centroids (corpus has {len(counts)} total)")

    persona_counts = [counts[n] for n in names]
    print(f"[index] samples/persona: min={min(persona_counts)} "
          f"median={sorted(persona_counts)[len(persona_counts)//2]} max={max(persona_counts)}")

    # ---- Variant A: index R@1 (reproduces the published number) ----------
    # ---- Variant B: cross-shoot ceiling (query from a held-out SET) ------
    res = {}
    for variant in ("A_index", "B_cross_shoot"):
        C = np.stack([cents[n] for n in names])
        ranks, margins, hits = [], [], []
        n_q = 0
        for i, persona in enumerate(names):
            held = None
            if variant == "B_cross_shoot":
                fs, sets = query_files(persona, 0, rng, None)
                if len(sets) >= 2:
                    held = sets[rng.randrange(len(sets))]   # hold out one shoot
            files, _ = query_files(persona, args.per_persona, rng, held)
            if not files:
                continue
            for f in files:
                r = evaluate(f, C, names, i)
                if r is None:
                    continue
                rank, own, best = r
                ranks.append(rank)
                margins.append(own - best)
                hits.append(rank == 1)
                n_q += 1
        if n_q == 0:
            raise SystemExit(f"FATAL: variant {variant} produced ZERO queries — "
                             "a silent zero is a failure, not a result")
        rk = np.array(ranks)
        res[variant] = {
            "n_queries": n_q,
            "R@1": float((rk == 1).mean()),
            "R@5": float((rk <= 5).mean()),
            "R@10": float((rk <= 10).mean()),
            "median_rank": float(np.median(rk)),
            "mean_own_minus_best_other_sim": float(np.mean(margins)),
            "n_personas": len(names),
            "chance_R@1": 1.0 / len(names),
        }
        m = res[variant]
        print(f"[{variant}] n={n_q}  R@1={m['R@1']:.4f} R@5={m['R@5']:.4f} "
              f"R@10={m['R@10']:.4f}  median_rank={m['median_rank']:.0f}  "
              f"chance={m['chance_R@1']:.4f}")

    res["published_reference"] = {"R@1": 0.8879, "R@5": 0.9502, "R@10": 0.9657,
                                  "chance_R@1": 0.0031,
                                  "source": "2026-09-23 brief §3.2 (persona-average index)"}
    res["provenance"] = {
        **prov,
        "expected_basis_fingerprint": EXPECTED_FP,
        "v1_lda_convention": read_stamp(V1 / "lda").get("projection_convention"),
        "corpus_convention": read_stamp(CORPUS).get("projection_convention"),
        "corpus_manifest_content_fingerprint": json.loads(
            (CORPUS / "_manifest.json").read_text()).get("lda_basis_fingerprint"),
        "note_on_fingerprints": ("basis_fingerprint = sha256(basis artifacts) = the projection; "
                                 "lda_basis_fingerprint = sha256(corpus averages/*.lda.npy) = "
                                 "the content. Different by design."),
        "query_convention": "per-image LDA L2-normalized (raw norm ~150.9)",
        "index_convention": "corpus centroids, unit-norm, 64-d",
        "tail": "FFHQ cannot appear here: 1 image/identity gives no cross-shoot pair",
        "git_commit": subprocess.run(["git", "-C", "/home/tim/source/activity/prx-tg",
                                      "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": args.seed, "per_persona": args.per_persona,
    }
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[out] {args.out}")

    a, b = res["A_index"]["R@1"], res["B_cross_shoot"]["R@1"]
    print(f"\nA_index R@1      = {a:.4f}   (published reference 0.8879)")
    print(f"B_cross_shoot R@1= {b:.4f}   (the renderer's ceiling)")
    print(f"ceiling cost of a shoot change (A - B) = {a - b:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
