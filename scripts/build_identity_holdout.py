#!/usr/bin/env python3
"""Build the LOCKED identity-level holdout manifest for the eidolon renderer arm.

Why identity-level, and why locked
----------------------------------
The arm's central risk is MIXED with the arm's central claim: a model that has
seen an identity's vector during training can score well on it by recall, not by
learning identity. So the gate must be measured on identities whose vectors were
NEVER fed in training, and the lists must be frozen BEFORE the split (and before
any run) so the holdout cannot be quietly reshaped to flatter a result.

Two holdouts, two different probes:
  FFHQ  10% of identities = 7,000 people, 1 photo each. Excluded from training
        entirely (identity AND imagery). Probe: unseen-identity FIDELITY --
        condition on a real unseen person's vector, render, retrieve -> is it that
        person? Index = the 7,000 held-out identity vectors, so chance ~1.4e-4.
  hegre 10% of personas = 32 people with multiple shots. Excluded from training
        entirely. Probe: unseen-persona GENERALIZATION, cross-shoot R@1 with the
        learned cross-shoot ceiling read AT THIS INDEX SIZE (0.9504 at 32).

Not a never-touched test set: both are used for in-training monitoring as well as
the final verdict. The plan's separate never-touched test slice is a Stage B
reconciliation, flagged rather than invented here.

Deterministic: seed is recorded, output is sorted, and a sha256 over the sorted
lists is stored so a future reader can prove the manifest did not drift.

    .venv/bin/python scripts/build_identity_holdout.py
"""
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

FFHQ_RAW = Path("/mnt/nas-ai-models/training-data/ffhq/raw")
FFHQ_AURA = Path("/mnt/nas-ai-models/training-data/ffhq/auraface")
HEGRE_CORPUS = Path("/mnt/nas-ai-models/training-data/eidolon/hegre_corpus")
HEGRE_LDA_FACES = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/lda/faces")
OUT = Path("/home/tim/source/activity/prx-tg/experiment-configs/eidolon-identity-renderer/holdout.json")
# NOTE: deliberately NOT under data/ -- data/ is gitignored, so a manifest written there
# is invisible to GitHub forever (AGENTS.md 0.1: the exact trap that lost 21 arms of their
# provenance). Soft arm data belongs in the arm git-tracked experiment-configs/ directory.

SEED = 20260925
FFHQ_FRACTION = 0.10
HEGRE_FRACTION = 0.10


def sha256_of(lines):
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode() + b"\n")
    return h.hexdigest()


def main():
    # --- FFHQ: identities == images (1 photo per person), require a stored vector ---
    raws = sorted(p.stem for p in FFHQ_RAW.glob("*.png"))
    with_vector = [s for s in raws if (FFHQ_AURA / f"{s}.npy").exists()]
    rng = random.Random(SEED)
    n_ffhq = int(len(with_vector) * FFHQ_FRACTION)
    ffhq_hold = sorted(rng.sample(with_vector, n_ffhq))
    ffhq_hold_set = set(ffhq_hold)
    ffhq_train = [s for s in with_vector if s not in ffhq_hold_set]

    # --- hegre: identities == personas, need >=2 shoots for a cross-shoot probe ---
    corpus_personas = {}
    for d in sorted(os.listdir(HEGRE_CORPUS)):
        if os.path.isdir(HEGRE_CORPUS / d) and "--" in d:
            corpus_personas.setdefault(d.split("--", 1)[0], 0)
            corpus_personas[d.split("--", 1)[0]] += 1
    # Pool must be EXACTLY the pool the cross-shoot ceiling was measured on, or the
    # gate's reference (0.9504 at 32) is only approximately right. The ceiling builds
    # it as: corpus dirs that RESOLVE into the LDA tree via the stem map, grouped by
    # the tree's persona, keeping personas present in >=2 distinct SETS among their
    # own corpus samples. Reproduced here rather than approximated, because "sets in
    # the tree" (320 personas) is a different and looser pool (313) -- a persona can
    # own >=2 set dirs while its corpus images sit in only one.
    stem_map = {}
    for pdir in HEGRE_LDA_FACES.iterdir():
        if not pdir.is_dir():
            continue
        for sdir in pdir.iterdir():
            if not sdir.is_dir():
                continue
            for fn in sdir.iterdir():
                if fn.suffix == ".npy":
                    stem_map[fn.stem] = (pdir.name, sdir.name)

    sets_by_persona = {}
    unresolved = 0
    for d in sorted(os.listdir(HEGRE_CORPUS)):
        if not os.path.isdir(HEGRE_CORPUS / d) or "--" not in d:
            continue
        hit = stem_map.get(d.split("--", 1)[1])
        if hit is None:
            unresolved += 1
            continue
        sets_by_persona.setdefault(hit[0], set()).add(hit[1])
    eligible = sorted(p for p, v in sets_by_persona.items() if len(v) >= 2)
    # Pin the count to the index size the ceiling was measured at (32), which is also
    # Tim's 10% of 321. A different count would need its own ceiling measurement.
    n_hegre = 32
    hegre_hold = sorted(rng.sample(eligible, n_hegre))

    payload = {
        "manifest": "eidolon-identity-holdout",
        "purpose": "identity-level holdout for the eidolon renderer arm: identities whose "
                   "vectors are never fed in training, so the gate measures generalization "
                   "rather than recall",
        "locked": True,
        "seed": SEED,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "builder": "scripts/build_identity_holdout.py",
        "ffhq": {
            "unit": "identity (== image; FFHQ is 1 photo per person)",
            "pool_total": len(with_vector),
            "holdout_fraction": FFHQ_FRACTION,
            "holdout_count": len(ffhq_hold),
            "train_count": len(ffhq_train),
            "excluded_from_training": "identity AND imagery (clean probe)",
            "probe": "unseen-identity FIDELITY: 1-shot retrieval, index = the holdout "
                     "identity vectors, chance = 1/holdout_count",
            "holdout_sha256": sha256_of(ffhq_hold),
            "holdout": ffhq_hold,
        },
        "hegre": {
            "unit": "persona (multi-shot; requires >=2 shoots among its corpus samples)",
            "unresolved_corpus_dirs": unresolved,
            "pool_total": len(eligible),
            "holdout_fraction": "10% of 321 = 32 personas (pinned to the index size at "
                                "which the cross-shoot ceiling 0.9504 was measured)",
            "holdout_count": len(hegre_hold),
            "excluded_from_training": "entirely (all shots, identity and imagery)",
            "probe": "unseen-persona GENERALIZATION: cross-shoot R@1, index = the holdout "
                     "personas, ceiling must be read at this index size",
            "ceiling_at_this_index_size": 0.9504,
            "ceiling_source": "scripts/eidolon_cross_shoot_ceiling.py --subset-personas 32 "
                              "--support 100 --queries 5 --seed 7",
            "holdout_sha256": sha256_of(hegre_hold),
            "holdout": hegre_hold,
        },
        "notes": [
            "NOT a never-touched test set: both holdouts are monitored in-training as well "
            "as used for the verdict. The plan's separate never-touched slice is a Stage B "
            "reconciliation, deliberately not invented here.",
            "The FFHQ holdout removes 10% of FFHQ imagery from training; that is the "
            "intended cost of a clean fidelity probe.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"FFHQ   pool {len(with_vector):,} identities -> holdout {len(ffhq_hold):,}, "
          f"train {len(ffhq_train):,}  sha256 {payload['ffhq']['holdout_sha256'][:16]}")
    print(f"hegre  pool {len(eligible)} personas   -> holdout {len(hegre_hold)}, "
          f"train {len(eligible) - len(hegre_hold)}  sha256 {payload['hegre']['holdout_sha256'][:16]}")
    print(f"\nout: {OUT}")


if __name__ == "__main__":
    main()
