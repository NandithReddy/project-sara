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
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AMI = REPO / "data" / "raw" / "ami"
WORDS = AMI / "words"
MEETINGS_XML = AMI / "corpusResources" / "meetings.xml"

SPLIT_MS = 1000.0
"""Same-speaker gap that separates two utterances. The gap distribution is flat
between 300ms and 1500ms (only 61 of 121,575 gaps fall in that band), so this
threshold is insensitive -- it is a round number, not a tuned one."""

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


@dataclass(frozen=True)
class Turn:
    turn_id: str
    meeting: str
    speaker: str
    channel: int
    global_name: str
    start_s: float
    end_s: float
    n_words: int
    disfluent: bool
    text: str
    words: tuple[tuple[str, float, float], ...]
    """(text, start_s, end_s) per word. The harness replays these at their own
    timings to imitate what a streaming recogniser would have emitted."""
    next_speaker_gap_ms: float
    own_resume_gap_ms: float
    """Silence until THIS speaker talks again, or inf if they never do.

    Caps how much trailing audio the segment may include. Overrunning it would
    capture the speaker resuming, and a false_hold measurement would then be
    scored against audio in which the turn had not actually ended.
    """
    stratum: str


def load_meetings() -> dict:
    """meeting -> {split, speakers: {agent: (channel, global_name)}}."""
    root = ET.parse(MEETINGS_XML).getroot()
    out = {}
    for m in root.iter("meeting"):
        obs = m.get("observation")
        if not obs:
            continue
        out[obs] = {
            "split": m.get("seen_type") or "unlabelled",
            "speakers": {
                s.get("nxt_agent"): (int(s.get("channel")), s.get("global_name"))
                for s in m.findall("speaker")
                if s.get("nxt_agent") and s.get("channel") is not None
            },
        }
    return out


def load_words(path: Path) -> list[dict]:
    """Word stream for one meeting-speaker, minus punctuation and non-speech."""
    root = ET.parse(path).getroot()
    out, disf, gap = [], False, False
    for el in root:
        tag = el.tag.split("}")[-1]
        if tag == "disfmarker":
            disf = True
            continue
        if tag == "gap":
            gap = True
            continue
        if tag != "w":
            continue
        s, e = el.get("starttime"), el.get("endtime")
        if s is None or e is None:
            continue
        if el.get("punc") == "true":
            continue
        out.append(
            {
                "s": float(s),
                "e": float(e),
                "t": (el.text or "").strip(),
                "disf": disf,
                "gap": gap,
            }
        )
        disf = gap = False
    return sorted(out, key=lambda w: (w["s"], w["e"]))


def extract_turns(meeting: str, meta: dict) -> list[Turn]:
    """Clean, unambiguous turns: no overlap, floor demonstrably transfers."""
    by_speaker = {}
    for f in sorted(WORDS.glob(f"{meeting}.*.words.xml")):
        agent = f.name.split(".")[1]
        if agent not in meta["speakers"]:
            continue
        ws = load_words(f)
        if ws:
            by_speaker[agent] = ws
    if len(by_speaker) < 2:
        return []

    utterances = []
    for agent, ws in by_speaker.items():
        cur = [ws[0]]
        for a, b in zip(ws, ws[1:], strict=False):
            if (b["s"] - a["e"]) * 1000 > SPLIT_MS:
                utterances.append((agent, cur))
                cur = [b]
            else:
                cur.append(b)
        utterances.append((agent, cur))
    utterances.sort(key=lambda u: u[1][0]["s"])

    turns = []
    for i, (agent, ws) in enumerate(utterances):
        t0, t1 = ws[0]["s"], ws[-1]["e"]
        if any(w["gap"] for w in ws):
            continue  # untranscribable audio cannot be ground truth

        nxt = next((u for u in utterances[i + 1 :] if u[1][0]["s"] >= t1), None)
        if nxt is None or nxt[0] == agent:
            continue  # nobody else took the floor: the turn end is not observable

        if any(
            other != agent and o[0]["s"] < t1 and o[-1]["e"] > t0
            for other, o in utterances
            if other != agent
        ):
            continue  # someone talked over it

        own_next = next((w["s"] for w in by_speaker[agent] if w["s"] > t1), None)
        own_resume_gap_ms = float("inf") if own_next is None else (own_next - t1) * 1000

        channel, global_name = meta["speakers"][agent]
        n = len(ws)
        disfluent = any(w["disf"] for w in ws)
        turns.append(
            Turn(
                turn_id=f"{meeting}.{agent}.{ws[0]['s']:.2f}",
                meeting=meeting,
                speaker=agent,
                channel=channel,
                global_name=global_name,
                start_s=t0,
                end_s=t1,
                n_words=n,
                disfluent=disfluent,
                text=" ".join(w["t"] for w in ws),
                words=tuple((w["t"], w["s"], w["e"]) for w in ws),
                next_speaker_gap_ms=(nxt[1][0]["s"] - t1) * 1000,
                own_resume_gap_ms=own_resume_gap_ms,
                stratum=(
                    "disfluent" if disfluent else "short" if n <= 3 else "ordinary"
                ),
            )
        )
    return turns


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
