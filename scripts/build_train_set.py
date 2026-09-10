"""Build the training set from AMI's `training` split. Annotations only.

Each example is a word-prefix of a turn with a binary label: 1 if the prefix
is the whole turn (the speaker stopped here and someone else took the floor),
0 if the speaker kept going. Positives come from true turn ends; negatives from
every proper prefix, so the model sees the turn as a streaming recogniser
would -- growing one word at a time -- and learns which growth steps are ends.

Disjointness from data/eval/ is asserted, not assumed (rule 1): no shared
meeting and no shared speaker, or this script refuses to write anything. The
only thing it reads from data/eval/ is the identity fields it needs for that
check. It passes because AMI's official training and development splits share
zero speakers -- a property of the corpus we verified, not of our sampling.

examples.jsonl stores (turn_id, k, label) and no text: the text is rendered
from turns.jsonl with ami.render() on demand, so the two never disagree and the
file stays small.

Punctuation is a rendering choice (`punctuated=True/False`), not a stored one.
Training on annotator punctuation would hand the model the punctuation
baseline's optimism gap: a
streaming recogniser punctuates causally from a hypothesis it keeps revising,
so a model that leans on a trailing "." learns a cue that degrades exactly when
it matters. Phase 4 decides; both renderings are here.

Run:
    uv run python scripts/build_train_set.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ami import (  # noqa: E402
    MEETINGS_XML,
    Turn,
    extract_turns,
    load_meetings,
    render,
)

REPO = Path(__file__).resolve().parents[1]
EVAL_SET = REPO / "data" / "eval" / "eval_set.json"
TRAIN_DIR = REPO / "data" / "train"
TRAIN_SPLIT = "training"


def assert_disjoint(train_turns: list[Turn]) -> tuple[int, int]:
    """Rule 1, executable. Returns (eval_meetings, eval_speakers) for the log."""
    if not EVAL_SET.exists():
        raise SystemExit(
            f"ERROR: {EVAL_SET} not found. The eval set must exist before the "
            f"train set is built, or disjointness cannot be asserted."
        )
    spec = json.loads(EVAL_SET.read_text())
    eval_meetings = {t["meeting"] for t in spec["turns"]}
    eval_speakers = {t["global_name"] for t in spec["turns"]}
    train_meetings = {t.meeting for t in train_turns}
    train_speakers = {t.global_name for t in train_turns}

    shared_m = train_meetings & eval_meetings
    shared_s = train_speakers & eval_speakers
    if shared_m or shared_s:
        raise SystemExit(
            f"ERROR: training data overlaps the frozen eval set -- refusing to "
            f"write anything.\n  shared meetings: {sorted(shared_m)[:5]}\n"
            f"  shared speakers: {sorted(shared_s)[:5]}\n"
            f"Training on eval speakers or sessions invalidates every number "
            f"measured against data/eval/ (ENGINEERING.md rule 1)."
        )
    return len(eval_meetings), len(eval_speakers)


def build_examples(turns: list[Turn]) -> list[dict]:
    out = []
    for t in turns:
        n = len(t.words)
        for k in range(1, n + 1):
            out.append(
                {
                    "turn_id": t.turn_id,
                    "k": k,
                    "n": n,
                    "label": 1 if k == n else 0,
                    "text": render(t.words, k, punctuated=False),
                    "text_punc": render(t.words, k, punctuated=True),
                    "disfluent": t.disfluent,
                    "global_name": t.global_name,
                }
            )
    return out


def label_collisions(examples: list[dict], key: str) -> dict:
    """How often the same rendered text carries both labels.

    This is the ceiling for any text-only model: on these strings no function
    of the text can be right every time, because the same string is sometimes
    an end and sometimes not. Only timing or prosody could separate them.
    """
    labels: dict[str, Counter] = defaultdict(Counter)
    for e in examples:
        labels[e[key]][e["label"]] += 1
    both = {s: c for s, c in labels.items() if len(c) == 2}
    n_examples_in_conflict = sum(sum(c.values()) for c in both.values())
    # Best any text-only classifier can do on the conflicted strings: pick the
    # majority label per string.
    unavoidable_errors = sum(min(c.values()) for c in both.values())
    top = sorted(both.items(), key=lambda kv: -sum(kv[1].values()))[:8]
    return {
        "distinct_strings": len(labels),
        "strings_with_both_labels": len(both),
        "examples_on_conflicted_strings": n_examples_in_conflict,
        "unavoidable_errors": unavoidable_errors,
        "unavoidable_error_rate": unavoidable_errors / len(examples),
        "worst": [{"text": s, "complete": c[1], "incomplete": c[0]} for s, c in top],
    }


def write_review(examples: list[dict], rng: random.Random, per_class: int) -> Path:
    pos = [e for e in examples if e["label"] == 1]
    neg = [e for e in examples if e["label"] == 0]
    sample_pos = rng.sample(pos, per_class)
    sample_neg = rng.sample(neg, per_class)

    def block(title, rows, note):
        lines = [f"## {title}", "", note, ""]
        for e in rows:
            rest = ""
            if e["label"] == 0:
                t = TURNS_BY_ID[e["turn_id"]]
                rest = " ".join(w[0] for w in t.words[e["k"] :])
                rest = f"  ⟶ continues: *{rest}*"
            lines.append(
                f"- `{e['text']}`   (word {e['k']} of {e['n']}"
                f"{', disfluent' if e['disfluent'] else ''}){rest}"
            )
        return "\n".join(lines)

    body = "\n\n".join(
        [
            "# Label review -- Phase 3",
            f"{per_class} random examples per class, seed {SEED}. Rendered "
            "without punctuation, as the model will see them. The done-condition "
            "for this phase is that a human has read these and agrees with the "
            "labels (PROMPT: *bad labels here silently cap your ceiling*).",
            "A label of **complete** means the speaker stopped here and another "
            "speaker took the floor. **Incomplete** means the same speaker kept "
            "going -- shown after ⟶. Note that an incomplete prefix can be a "
            "perfectly grammatical sentence; the label is about whether they "
            "*stopped*, not whether they *could have*.",
            block("Complete (label 1)", sample_pos, "The whole turn."),
            block(
                "Incomplete (label 0)",
                sample_neg,
                "A prefix; the speaker continued.",
            ),
        ]
    )
    path = TRAIN_DIR / "REVIEW.md"
    path.write_text(body + "\n")
    return path


SEED = 20260910
TURNS_BY_ID: dict[str, Turn] = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--review", type=int, default=20, help="examples per class")
    args = ap.parse_args()

    if not MEETINGS_XML.exists():
        raise SystemExit(f"ERROR: {MEETINGS_XML} not found; see data/README.md")

    meetings = load_meetings()
    train_meetings = sorted(m for m, v in meetings.items() if v["split"] == TRAIN_SPLIT)
    print(f"{TRAIN_SPLIT} split: {len(train_meetings)} meetings")

    turns: list[Turn] = []
    for m in train_meetings:
        turns.extend(extract_turns(m, meetings[m]))
    turns.sort(key=lambda t: t.turn_id)
    TURNS_BY_ID.update({t.turn_id: t for t in turns})
    print(f"clean turns: {len(turns)}  speakers: {len({t.global_name for t in turns})}")

    n_em, n_es = assert_disjoint(turns)
    print(
        f"disjoint from eval: 0 of {n_em} meetings shared, 0 of {n_es} speakers shared"
    )

    examples = build_examples(turns)
    pos = sum(e["label"] for e in examples)
    neg = len(examples) - pos
    total = len(examples)
    print(
        f"\nexamples: {total:,}  complete={pos:,} ({pos / total * 100:.1f}%)  "
        f"incomplete={neg:,} ({neg / total * 100:.1f}%)"
    )
    print(f"  negatives per positive: {(len(examples) - pos) / pos:.2f}")
    nw = sorted(len(t.words) for t in turns)
    p50, p90 = nw[len(nw) // 2], nw[int(len(nw) * 0.9)]
    print(f"  words per turn: p50={p50} p90={p90} max={nw[-1]}")

    for key in ("text", "text_punc"):
        c = label_collisions(examples, key)
        print(f"\nlabel collisions on `{key}`:")
        print(
            f"  {c['strings_with_both_labels']:,} of {c['distinct_strings']:,} "
            f"distinct strings carry BOTH labels"
        )
        print(
            f"  unavoidable errors for any text-only model: "
            f"{c['unavoidable_errors']:,} = {c['unavoidable_error_rate'] * 100:.2f}% "
            f"of examples"
        )
        for w in c["worst"][:5]:
            print(
                f"    {w['text']!r:24} complete={w['complete']:4} "
                f"incomplete={w['incomplete']:4}"
            )

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    with (TRAIN_DIR / "turns.jsonl").open("w") as f:
        for t in turns:
            d = asdict(t)
            if d["own_resume_gap_ms"] == float("inf"):
                d["own_resume_gap_ms"] = -1.0
            f.write(json.dumps(d) + "\n")
    stored = ("turn_id", "k", "n", "label", "disfluent", "global_name")
    with (TRAIN_DIR / "examples.jsonl").open("w") as f:
        for e in examples:
            f.write(json.dumps({key: e[key] for key in stored}) + "\n")
    review = write_review(examples, random.Random(args.seed), args.review)

    stats = {
        "corpus": "AMI Meeting Corpus, manual annotations v1.6.2 (CC BY 4.0)",
        "split": TRAIN_SPLIT,
        "seed": args.seed,
        "n_meetings": len(train_meetings),
        "n_turns": len(turns),
        "n_speakers": len({t.global_name for t in turns}),
        "n_examples": len(examples),
        "n_complete": pos,
        "n_incomplete": len(examples) - pos,
        "collisions_text": label_collisions(examples, "text"),
        "collisions_text_punc": label_collisions(examples, "text_punc"),
    }
    (TRAIN_DIR / "STATS.json").write_text(json.dumps(stats, indent=2))
    print("\nwrote data/train/turns.jsonl, examples.jsonl, STATS.json")
    print(f"wrote {review.relative_to(REPO)} -- read it before calling this phase done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
