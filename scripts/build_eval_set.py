"""Build the frozen eval set from AMI annotations.

Stage 1 (this file, `select`): read AMI's NXT annotations, extract clean turns,
and write the selection to data/eval/turns.json. Annotations only -- no audio.

Why the eval set needs audio at all, given section 10 replays gold transcripts:
AMI's forced alignment makes words contiguous inside an utterance. `end(word N)`
equals `start(word N+1)`, and intra-utterance silence is absorbed into word
durations -- a hesitation shows up as a 1190ms-long word "you", not as a gap.
Deriving silence_ms from gold timings would therefore report ~zero silence
during a mid-turn hesitation, no baseline would ever fire early, and
cutoff_rate would come out near zero for everything. The eval would measure
nothing, convincingly. So silence, and the true turn boundary, are grounded in
audio at BUILD time and then frozen. Runs stay byte-identical reproducible.

Eval draws from AMI's official `development` split, which shares zero speakers
with the `training` split (verified from corpusResources/meetings.xml). Rule 1
and the Phase 3 speaker-disjointness assertion are then properties of the
corpus, not of our sampling.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ami import (  # noqa: E402
    MEETINGS_XML,
    SPLIT_MS,
    Turn,
    extract_turns,
    load_meetings,
)

REPO = Path(__file__).resolve().parents[1]

EVAL_SPLIT = "development"

HORIZON_MS = 2000.0
"""How long after the true turn end the eval may keep observing.

A turn is only usable if the speaker stays silent for at least this long after
finishing -- otherwise the tail would capture them resuming, and a false_hold
would be scored against audio in which the turn had not actually ended.

2000ms is twice the slowest baseline section 3 requires (1000ms). Pushing it
higher costs turns quickly (164 at 2000ms, 142 at 3000ms of 198) and biases the
set toward turns followed by long silence -- which are the easy cases.
"""

BOUNDARY_SEARCH_MS = 500.0
"""How far past the annotated end the audio boundary refinement may look.

The refined boundary usually lands LATER than the annotated one (median +60ms),
so the segment must carry this much extra tail or the horizon, measured from the
refined boundary, comes up short. Capping the search at this value makes the
full horizon guaranteed by construction rather than checked afterwards.
"""

TAIL_MARGIN_MS = 100.0
"""Extra guard between the end of the tail and the speaker resuming."""

REQUIRED_TAIL_MS = HORIZON_MS + BOUNDARY_SEARCH_MS + TAIL_MARGIN_MS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="target number of turns")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "eval" / "turns.json")
    args = ap.parse_args()

    if not MEETINGS_XML.exists():
        raise SystemExit(
            f"ERROR: {MEETINGS_XML} not found. Fetch and unzip the AMI manual "
            f"annotations into data/raw/ami/ first (see data/README.md). "
            f"Not downloading anything implicitly."
        )

    meetings = load_meetings()
    eval_meetings = sorted(m for m, v in meetings.items() if v["split"] == EVAL_SPLIT)
    print(f"{EVAL_SPLIT} split: {len(eval_meetings)} meetings")

    raw_pool: list[Turn] = []
    for m in eval_meetings:
        raw_pool.extend(extract_turns(m, meetings[m]))
    print(f"clean turns available: {len(raw_pool)}")

    # Filter BEFORE sampling, so every stratum still fills to the target count.
    pool = [t for t in raw_pool if t.own_resume_gap_ms >= HORIZON_MS + TAIL_MARGIN_MS]
    print(
        f"with >={REQUIRED_TAIL_MS:.0f}ms of clean tail: {len(pool)} "
        f"(dropped {len(raw_pool) - len(pool)})"
    )
    print(f"  by stratum: {dict(Counter(t.stratum for t in pool))}")

    # Equal thirds, so section 3's disfluent and short-answer conditions are
    # present by construction rather than by luck.
    rng = random.Random(args.seed)
    per = args.n // 3
    chosen: list[Turn] = []
    for stratum in ("disfluent", "short", "ordinary"):
        bucket = sorted(
            (t for t in pool if t.stratum == stratum), key=lambda t: t.turn_id
        )
        if len(bucket) < per:
            raise SystemExit(
                f"ERROR: only {len(bucket)} '{stratum}' turns available, need {per}. "
                f"Refusing to silently under-fill a stratum."
            )
        chosen.extend(rng.sample(bucket, per))
    chosen.sort(key=lambda t: t.turn_id)

    print(f"\nselected {len(chosen)} turns")
    print(f"  meetings: {len({t.meeting for t in chosen})}")
    print(f"  speakers: {len({t.global_name for t in chosen})}")
    print(f"  audio files needed: {len({(t.meeting, t.channel) for t in chosen})}")
    durs = sorted(t.end_s - t.start_s for t in chosen)
    p50, p90 = durs[len(durs) // 2], durs[int(len(durs) * 0.9)]
    print(f"  turn duration p50={p50:.1f}s p90={p90:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for t_ in chosen:
        d = asdict(t_)
        if d["own_resume_gap_ms"] == float("inf"):
            d["own_resume_gap_ms"] = -1.0  # speaker never talks again
        rows.append(d)

    args.out.write_text(
        json.dumps(
            {
                "corpus": "AMI Meeting Corpus, manual annotations v1.6.2 (CC BY 4.0)",
                "split": EVAL_SPLIT,
                "seed": args.seed,
                "split_gap_ms": SPLIT_MS,
                "horizon_ms": HORIZON_MS,
                "boundary_search_ms": BOUNDARY_SEARCH_MS,
                "turns": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
