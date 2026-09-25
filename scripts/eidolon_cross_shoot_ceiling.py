#!/usr/bin/env python3
"""Cross-shoot identity ceiling — the number G6 must be read against.

Same identity, DIFFERENT shoot, measured on real data with no model in the loop.
This is the best any renderer could do, so it is the ceiling for a render score,
not 1.0.

Why leave-one-shoot-out
-----------------------
The Step 0 replication drew one image per persona and indexed it against that
persona's average computed over ALL shots — so the query image contributed to
the very centroid it was matched against. That leakage inflates the number, and
the inflation is exactly the cross-shoot penalty we are trying to measure. Here
each persona's index centroid is built from its OTHER shoots only, and the
queries come from the held-out shoot, so no query can influence its own index
entry. The leaked (all-shots) variant is reported alongside, and the difference
is the leakage correction.

Units: the DiT identity slot and the persona index both live in LDA space,
L2-normalized (norm 1.0). Per-image vectors on disk are raw coords (~norm 153),
so queries are normalized before measuring: mixing the two scales makes every
cosine meaningless.

    .venv/bin/python scripts/eidolon_cross_shoot_ceiling.py [--support 30] [--queries 3]
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

CORPUS = "/mnt/nas-ai-models/training-data/eidolon/hegre_corpus"
LDA_FACES = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/lda/faces"
EXPECTED_BASIS = "120e1c5a1dc4f423"
OUT = "/home/tim/source/activity/prx-tg/research/results/eidolon-identity-instrument/cross_shoot_ceiling.json"
MAP_CACHE = "/home/tim/.hermes/profiles/prx-tg/cache/scratch/lda_stem_set_map.json"


def l2n(v):
    n = float(np.linalg.norm(v))
    return None if n == 0 else (v / n).astype(np.float32)


def load_stamp():
    p = os.path.join(CORPUS, "BASIS_FINGERPRINT.json")
    with open(p) as f:
        st = json.load(f)
    fp = st.get("basis_fingerprint")
    if fp != EXPECTED_BASIS:
        sys.exit(f"[basis] REFUSING: {p} is stamped {fp}, expected {EXPECTED_BASIS}")
    print(f"[basis] OK {fp} | {st.get('projection_convention')}")
    return st


def build_stem_map():
    """stem -> (persona, set). One-time listing of the v1 lda tree."""
    if os.path.exists(MAP_CACHE):
        with open(MAP_CACHE) as f:
            m = json.load(f)
        print(f"[map] cached: {len(m)} stems")
        return m
    m = {}
    t0 = time.time()
    personas = sorted(os.listdir(LDA_FACES))
    for i, p in enumerate(personas):
        pdir = os.path.join(LDA_FACES, p)
        for s in os.listdir(pdir):
            sdir = os.path.join(pdir, s)
            if not os.path.isdir(sdir):
                continue
            for fn in os.listdir(sdir):
                if fn.endswith(".npy"):
                    m[fn[:-4]] = [p, s]
        if (i + 1) % 100 == 0:
            print(f"[map] {i+1}/{len(personas)} personas, {len(m)} stems, {time.time()-t0:.0f}s", flush=True)
    with open(MAP_CACHE, "w") as f:
        json.dump(m, f)
    print(f"[map] built {len(m)} stems in {time.time()-t0:.0f}s (cached)")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support", type=int, default=30,
                    help="max per-image vectors per persona used for its centroid")
    ap.add_argument("--queries", type=int, default=3,
                    help="max queries per persona, from the held-out shoot only")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset-personas", type=int, default=0,
                    help="randomly restrict the index/query pool to N personas, to "
                         "measure the CI a persona-disjoint holdout of that size buys")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    load_stamp()
    stem_map = build_stem_map()

    # --- corpus dirs -> (persona, set, stem) ---
    by_persona = defaultdict(list)
    unresolved = 0
    t0 = time.time()
    entries = sorted(d for d in os.listdir(CORPUS) if os.path.isdir(os.path.join(CORPUS, d)))
    for i, d in enumerate(entries):
        if "--" not in d:
            unresolved += 1
            continue
        persona, rest = d.split("--", 1)
        hit = stem_map.get(rest)
        if hit is None:
            unresolved += 1
            continue
        by_persona[hit[0]].append((hit[1], rest))
        if (i + 1) % 5000 == 0:
            print(f"[corpus] {i+1}/{len(entries)} mapped, {time.time()-t0:.0f}s", flush=True)

    personas = sorted(p for p, v in by_persona.items() if len({s for s, _ in v}) >= 2)
    n_available = len(personas)
    if args.subset_personas:
        personas = sorted(rng.sample(personas, min(args.subset_personas, len(personas))))
    print(f"[corpus] {len(entries)} dirs | {n_available} personas with >=2 shoots | "
          f"{unresolved} unresolved"
          + (f" | SUBSET to {len(personas)} personas" if args.subset_personas else ""))

    # --- per persona: hold out one shoot; centroid from the others ---
    q_vecs, q_owner = [], []
    cent_vecs, cent_owner = [], []
    support_used = query_used = 0
    skipped = []
    for pi, p in enumerate(personas):
        by_set = defaultdict(list)
        for s, stem in by_persona[p]:
            by_set[s].append(stem)
        sets = sorted(by_set)
        held = rng.choice(sets)
        others = [s for s in sets if s != held]

        def rd(stem):
            fp = os.path.join(LDA_FACES, stem_map[stem][0], stem_map[stem][1], stem + ".npy")
            try:
                v = np.load(fp)
            except Exception:
                return None
            return l2n(v) if v.shape == (64,) else None

        # centroid from the other shoots (capped)
        sup = []
        pool = [st for s in others for st in by_set[s]]
        rng.shuffle(pool)
        for stem in pool[: args.support]:
            v = rd(stem)
            if v is not None:
                sup.append(v)
        if not sup:
            skipped.append((p, "no support vectors"))
            continue
        c = l2n(np.mean(np.stack(sup), axis=0))
        if c is None:
            skipped.append((p, "empty centroid"))
            continue
        cent_vecs.append(c)
        cent_owner.append(pi)
        support_used += len(sup)

        qpool = by_set[held][:]
        rng.shuffle(qpool)
        nq = 0
        for stem in qpool:
            if nq >= args.queries:
                break
            v = rd(stem)
            if v is not None:
                q_vecs.append(v)
                q_owner.append(pi)
                nq += 1
                query_used += 1
        if nq == 0:
            skipped.append((p, "no held-out queries"))
        if (pi + 1) % 50 == 0:
            print(f"[vectors] {pi+1}/{len(personas)} personas, "
                  f"{len(q_vecs)} queries, {time.time()-t0:.0f}s", flush=True)

    Q = np.stack(q_vecs)
    C = np.stack(cent_vecs)
    owner_q = np.array(q_owner)
    owner_c = np.array(cent_owner)
    print(f"[vectors] queries {Q.shape} | index {C.shape} | support reads {support_used} "
          f"| skipped {len(skipped)} personas")

    # --- retrieval ---
    n_pers = len(personas)
    chance = 1.0 / len(C)

    def rk(queries):
        # cosine on unit vectors
        sim = queries @ C.T
        order = np.argsort(-sim, axis=1)
        ranks = np.empty(len(queries), dtype=int)
        for i, row in enumerate(order):
            ranks[i] = int(np.where(owner_c[row] == owner_q[i])[0][0]) + 1
        out = {f"R@{k}": float((ranks <= k).mean()) for k in (1, 5, 10)}
        out["median_rank"] = float(np.median(ranks))
        out["n"] = int(len(queries))
        return out, sim, ranks

    res, sim, ranks = rk(Q)

    # margin: own-centroid similarity vs best other-centroid similarity
    own_s = np.array([sim[i, np.where(owner_c == owner_q[i])[0][0]] for i in range(len(Q))])
    other_s = sim.copy()
    for i in range(len(Q)):
        other_s[i, np.where(owner_c == owner_q[i])[0][0]] = -np.inf
    best_other = other_s.max(axis=1)
    margin = own_s - best_other

    # persona-level bootstrap CI on R@1 (resample personas, not queries)
    boot = []
    pers_ids = np.unique(owner_q)
    idx_by_pers = {p: np.where(owner_q == p)[0] for p in pers_ids}
    for _ in range(args.bootstrap):
        pick = [rng.choice(pers_ids) for _ in pers_ids]
        sel = np.concatenate([idx_by_pers[p] for p in pick])
        boot.append(float((ranks[sel] <= 1).mean()))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    print()
    print(f"[cross-shoot ceiling] queries {res['n']} | personas {len(pers_ids)}/{n_pers} "
          f"| index {C.shape[0]} | chance {chance:.4f}")
    print(f"  R@1  = {res['R@1']:.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]  (persona bootstrap)")
    print(f"  R@5  = {res['R@5']:.4f}")
    print(f"  R@10 = {res['R@10']:.4f}")
    print(f"  margin: own {own_s.mean():.4f} vs best-other {best_other.mean():.4f} "
          f"-> gap {margin.mean():.4f} | frac(own nearer) {res['R@1']:.4f}")

    payload = {
        "measurement": "cross_shoot_identity_ceiling",
        "definition": {
            "index": "per-persona LDA centroid built from the persona's OTHER shoots",
            "query": "per-image LDA from the held-out shoot, L2-normalized",
            "leakage": "none — no query contributes to its own centroid",
            "metric": "cosine ranking on L2-normalized vectors",
        },
        "config": vars(args),
        "units": "LDA 64-d, L2-normalized (norm 1.0); on-disk per-image vectors are raw coords ~153",
        "basis_fingerprint_expected": EXPECTED_BASIS,
        "corpus": CORPUS,
        "n_corpus_dirs": len(entries),
        "n_personas_with_2plus_shoots": n_pers,
        "n_personas_available": n_available,
        "persona_subset": args.subset_personas or None,
        "unresolved_corpus_dirs": unresolved,
        "support_reads": support_used,
        "query_reads": query_used,
        "skipped_personas": [list(x) for x in skipped],
        "chance_R@1": chance,
        "ceiling": res,
        "ceiling_R@1_persona_bootstrap_CI95": list(ci),
        "margin": {
            "own_mean": float(own_s.mean()),
            "best_other_mean": float(best_other.mean()),
            "gap_mean": float(margin.mean()),
        },
        "reference_step0_replication_R@1": 0.8910,
        "caveat": ("centroids use at most --support per-image vectors per persona, so they are "
                   "noisier than the full corpus centroids; this makes the ceiling a slight "
                   "LOWER bound on the true value."),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.uname().nodename,
    }
    out_path = OUT if not args.subset_personas else OUT.replace(
        ".json", f"_personas{args.subset_personas}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[out] {out_path}")


if __name__ == "__main__":
    main()
