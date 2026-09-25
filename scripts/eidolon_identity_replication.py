#!/usr/bin/env python3
"""Replication of the recorded persona-average-index retrieval result, with no
model, no VAE and no detection in the loop.

REPLICATES
    docs/assets/exp/sapiens2-keypoints-study/evidence-20260923/
    test_average_discriminative.py          (eidolon repo)
    recorded: R@1 0.8879 / R@5 0.9502 / R@10 0.9657, chance 0.0031

    Index   : averages/*.lda.npy  (LDA-space persona averages, L2-normalized)
    Queries : ONE corpus image per persona, resolved via metadata.json ->
              lda/faces/{persona}/{set}/{image_id}.npy
    Metrics : cosine AND euclidean R@1/5/10 (the original printed both)

WHY IT MATTERS
    This is the control for the instrument that will measure the renderer. If a
    definitional replication does not return the recorded numbers, the metric
    machinery is wrong and we learn it before a single training step. A first
    attempt using the 321 corpus centroids as the index and 1,284 arbitrary
    queries gave R@1 0.8427 — that was a DIFFERENT measurement (wrong index
    artifact, different query set), not a failure of this one.

CONVENTION (trap 6)
    per-image LDA is raw-scale (norm ~153); the average index is unit-norm.
    The query must be L2-normalized or the cosine is meaningless.

Fails loudly on basis mismatch or an empty query set.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

BASIS_DIR = Path("/home/tim/source/activity/eidolon/experiments/geometry_pca/output")
V1 = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")
CORPUS = Path("/mnt/nas-ai-models/training-data/eidolon/hegre_corpus")
EXPECTED_FP = "120e1c5a1dc4f423"
PUBLISHED = {"R@1": 0.8879, "R@5": 0.9502, "R@10": 0.9657, "chance_R@1": 0.0031}


def l2n(m):
    m = np.atleast_2d(np.asarray(m, dtype=np.float64))
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


def assert_basis():
    files = {n: hashlib.sha256((BASIS_DIR / n).read_bytes()).hexdigest()[:16]
             for n in ("auraface_lda.npz", "auraface_preprocess.npz")}
    for d in (V1 / "lda", CORPUS):
        st = json.loads((d / "BASIS_FINGERPRINT.json").read_text())
        if st.get("basis_fingerprint") != EXPECTED_FP:
            raise SystemExit(f"FATAL: {d} stamped {st.get('basis_fingerprint')} != {EXPECTED_FP}")
        if st["basis_files"]["auraface_lda.npz"] != files["auraface_lda.npz"]:
            raise SystemExit(f"FATAL: {d} basis hash disagrees with on-disk basis")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="/tmp/eidolon_identity_replication.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    files = assert_basis()
    print(f"[basis] OK fingerprint={EXPECTED_FP} files={files}")

    # ---- index: persona averages in LDA space --------------------------
    avg_files = sorted((V1 / "averages").glob("*.lda.npy"))
    names, vecs = [], []
    for f in avg_files:
        v = np.load(f).astype(np.float64)
        if v.shape == (64,):
            names.append(f.name[: -len(".lda.npy")])
            vecs.append(v)
    A = l2n(np.stack(vecs))
    pidx = {p: i for i, p in enumerate(names)}
    print(f"[index] {len(avg_files)} average files -> A{A.shape}  chance={1/len(names):.4f}")

    # ---- queries: one corpus image per persona, via metadata -----------
    buckets = defaultdict(list)
    for d in sorted(CORPUS.iterdir()):
        if d.is_dir():
            buckets[d.name.split("--", 1)[0]].append(d)

    queries, unresolved = [], 0
    for p, ds in buckets.items():
        if p not in pidx:
            continue
        rng.shuffle(ds)
        for d in ds:
            try:
                meta = json.loads((d / "metadata.json").read_text())
                lp = V1 / "lda" / "faces" / p / meta["set"] / f"{meta['image_id']}.npy"
                if lp.exists():
                    v = np.load(lp).astype(np.float64)
                    if v.shape == (64,):
                        queries.append((p, v))
                        break
            except Exception:
                pass
            unresolved += 1
    if not queries:
        raise SystemExit("FATAL: ZERO queries resolved — a silent zero is a failure, not a result")

    Q = l2n(np.stack([v for _, v in queries]))
    truth = np.array([pidx[p] for p, _ in queries])
    print(f"[queries] n={len(Q)} personas={len(set(p for p,_ in queries))} "
          f"(unresolved attempts {unresolved})")

    res = {"replicates": "evidence-20260923/test_average_discriminative.py",
           "recorded": PUBLISHED, "chance_R@1": 1.0 / len(names),
           "n_queries": int(len(Q)), "n_personas_index": len(names),
           "provenance": {"basis_files": files, "basis_fingerprint": EXPECTED_FP,
                          "index_convention": "averages/*.lda.npy, L2-normalized (norm 1.0)",
                          "query_convention": "per-image LDA, L2-normalized (raw norm ~153)",
                          "seed": args.seed,
                          "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}}

    S = Q @ A.T
    rankc = np.argsort(-S, axis=1)
    D = np.linalg.norm(Q[:, None, :] - A[None, :, :], axis=2)
    rankd = np.argsort(D, axis=1)
    for label, rank, dist in (("cosine", rankc, None), ("euclidean", rankd, D)):
        m = {}
        for k in (1, 5, 10):
            m[f"R@{k}"] = float(np.mean([truth[i] in rank[i, :k] for i in range(len(Q))]))
        m["median_rank"] = float(np.median([int(np.where(rank[i] == truth[i])[0][0]) + 1
                                            for i in range(len(Q))]))
        res[label] = m
        print(f"[{label:9s}] R@1={m['R@1']:.4f} R@5={m['R@5']:.4f} R@10={m['R@10']:.4f} "
              f"median_rank={m['median_rank']:.0f}")

    # margin — distance to own average vs nearest other
    own = D[np.arange(len(Q)), truth]
    Dm = D.copy()
    Dm[np.arange(len(Q)), truth] = np.inf
    other = Dm.min(axis=1)
    res["margin"] = {"own_mean": float(own.mean()), "nearest_other_mean": float(other.mean()),
                     "margin_mean": float(np.mean(other - own)),
                     "frac_own_lt_other": float(np.mean(own < other))}
    print(f"[margin  ] own={own.mean():.4f} other={other.mean():.4f} "
          f"gap={np.mean(other-own):.4f} frac(own<other)={np.mean(own<other):.4f}")

    g = res["cosine"]["R@1"] - PUBLISHED["R@1"]
    res["replication_delta_cosine_R@1"] = float(g)
    print(f"\n[REPLICATION] cosine R@1 {res['cosine']['R@1']:.4f} vs recorded "
          f"{PUBLISHED['R@1']:.4f}  delta {g:+.4f}")
    print("[REPLICATION] " + ("REPRODUCED (within 0.01)" if abs(g) < 0.01
                              else "NOT REPRODUCED — investigate before trusting the metric"))
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
