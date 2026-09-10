"""Stage 3: freeze the eval set.

Writes data/eval/MANIFEST.json with a SHA256 of every file. After this runs,
the eval set is sacred: never regenerated, never trained on, never tuned on
(ENGINEERING.md rule 1). tests/test_eval_frozen.py fails if a byte moves.

Re-running this on a changed eval set is the one way to break the freeze, so it
refuses unless --force is passed and prints what changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "data" / "eval"
MANIFEST = EVAL / "MANIFEST.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build() -> dict:
    files = sorted(
        p for p in EVAL.rglob("*") if p.is_file() and p.name != MANIFEST.name
    )
    spec = json.loads((EVAL / "eval_set.json").read_text())
    turns = spec["turns"]
    return {
        "corpus": spec["corpus"],
        "split": spec["split"],
        "seed": spec["seed"],
        "horizon_ms": spec["horizon_ms"],
        "n_turns": len(turns),
        "n_meetings": len({t["meeting"] for t in turns}),
        "n_speakers": len({t["global_name"] for t in turns}),
        "strata": {
            s: sum(1 for t in turns if t["stratum"] == s)
            for s in sorted({t["stratum"] for t in turns})
        },
        "boundary_source": {
            s: sum(1 for t in turns if t["boundary_source"] == s)
            for s in sorted({t["boundary_source"] for t in turns})
        },
        "files": {
            str(p.relative_to(EVAL)): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in files
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-freeze a changed set")
    args = ap.parse_args()

    fresh = build()
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        changed = [
            k
            for k in set(old["files"]) | set(fresh["files"])
            if old["files"].get(k, {}).get("sha256")
            != fresh["files"].get(k, {}).get("sha256")
        ]
        if not changed:
            print(f"unchanged: {fresh['n_turns']} turns already frozen")
            return 0
        print(f"MANIFEST EXISTS and {len(changed)} file(s) differ, e.g.:")
        for k in sorted(changed)[:5]:
            print(f"  {k}")
        if not args.force:
            print(
                "\nRefusing to re-freeze. The eval set is frozen once and never\n"
                "regenerated (rule 1). Pass --force only if you genuinely intend\n"
                "to invalidate every number measured against the old set."
            )
            return 1

    MANIFEST.write_text(json.dumps(fresh, indent=2, sort_keys=True))
    print(
        f"froze {fresh['n_turns']} turns across {fresh['n_meetings']} meetings, "
        f"{fresh['n_speakers']} speakers"
    )
    print(f"  strata: {fresh['strata']}")
    print(f"  files:  {len(fresh['files'])}")
    print(f"wrote {MANIFEST.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
