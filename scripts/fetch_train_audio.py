"""Fetch audio for the training turns, the same way the eval audio was fetched.

Same range requests, same Silero boundary refinement, same segment shape,
through the same function (scripts/fetch_eval_audio.fetch_turns), so training
and eval audio are cut with one knife. Output is gitignored -- ~1GB of FLAC --
and reproducible from data/train/turns.jsonl.

Tail is short: prosody examples need the end-of-turn pause plus a margin, not
the eval's 2000ms observation horizon. The caller channel is applied later, at
feature time, from the refined boundary.

Run:
    uv run python scripts/fetch_train_audio.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_eval_audio import fetch_turns  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO / "data" / "train"
HORIZON_MS = 1000.0
SEARCH_MS = 500.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse-audio", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=TRAIN_DIR / "audio_set.json")
    args = ap.parse_args()

    turns = [json.loads(line) for line in (TRAIN_DIR / "turns.jsonl").open()]
    if args.limit:
        turns = turns[: args.limit]
    print(f"{len(turns)} training turns")
    rows, deltas, fetched = fetch_turns(
        turns,
        HORIZON_MS,
        SEARCH_MS,
        TRAIN_DIR / "audio",
        args.reuse_audio,
        args.sleep,
        log_every=250,
    )
    d = np.array(deltas)
    n_ann = sum(1 for r in rows if r["boundary_source"] == "annotation")
    print(
        f"\nkept {len(rows)} turns, {fetched / 1e6:.0f}MB fetched; boundary from "
        f"audio {len(rows) - n_ann}, annotation {n_ann}"
    )
    print(
        f"annotated minus audio end: p50={np.percentile(d, 50):+.0f}ms "
        f"p90={np.percentile(d, 90):+.0f}ms"
    )
    out = args.out
    out.write_text(
        json.dumps(
            {"horizon_ms": HORIZON_MS, "boundary_search_ms": SEARCH_MS, "turns": rows}
        )
    )
    print(f"wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
