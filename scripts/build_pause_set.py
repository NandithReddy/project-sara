"""Turn training audio into pause examples: end of turn, or hesitation?

For every training turn: the caller channel (zeroed after the refined end),
Silero for silence, the prosody tracker for features. Each time silence
reaches the gate (200ms) inside the turn's window, that is one example:
  label 1  the speaker did not speak again -- the turn ended here
  label 0  speech resumed -- a mid-turn pause, the thing that breaks timers
Features are the tracker's 14 numbers at the LAST SPEECH FRAME before the
pause -- the prosody of what was just said -- plus the text so far (gold words
ended by then) and the frozen text model's P(complete) on it.

Why not at the gate crossing: the first build did that, and the classifier
scored a perfect 1.000 AP on held-out speakers with energy_db as its dominant
weight. 200ms into an END pause lies inside the region the caller channel
zeroes (true_end + 150ms), so energy read -120dB of digital silence there and
room tone at a mid-turn pause. It had learned to detect the preprocessing.
At the last speech frame the zeroed tail is never in the window.

Label noise, counted rather than hidden: "mid" means the VAD saw speech again,
which includes a laugh or a breath after the final word. `word_follows` says
whether a WORD started after the pause; the eval boundary is VAD-based too,
so the label stays VAD-based for consistency and the count is reported.

Reads data/train/ only. SARA_TRAINING is set so the eval-set guard is live.

Run:
    uv run python scripts/build_pause_set.py
"""

from __future__ import annotations

import os

os.environ["SARA_TRAINING"] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.dataset import EvalTurn  # noqa: E402
from eval.harness import caller_channel  # noqa: E402
from src.audio.prosody import FEATURE_NAMES, ProsodyTracker  # noqa: E402
from src.audio.vad import SileroVAD  # noqa: E402
from src.eot.base import Update  # noqa: E402
from src.eot.gated import DEFAULT_GATE_MS  # noqa: E402
from src.eot.model import TextEOT  # noqa: E402

TRAIN_DIR = REPO / "data" / "train"
SEED = 20260912


def as_turn(r: dict) -> EvalTurn:
    """The caller-channel transform wants an EvalTurn; only true_end_ms matters."""
    return EvalTurn(
        turn_id=r["turn_id"],
        meeting=r["meeting"],
        speaker=r["speaker"],
        global_name=r["global_name"],
        stratum=r["stratum"],
        text=r["text"],
        n_words=r["n_words"],
        disfluent=r["disfluent"],
        audio_path=TRAIN_DIR / r["audio"],
        seg_duration_ms=r["seg_duration_ms"],
        turn_start_ms=r["turn_start_ms"],
        true_end_ms=r["true_end_ms"],
        annotated_end_ms=r["annotated_end_ms"],
        boundary_source=r["boundary_source"],
    )


def pauses_for(turn: EvalTurn, words: list[dict], vad, tracker, text_model, gate_ms):
    audio, sr = sf.read(turn.audio_path, dtype="float32")
    audio = caller_channel(np.asarray(audio, dtype=np.float32).reshape(-1), turn)
    vad.reset()
    tracker.reset()
    frames = vad.push(audio)
    feats = tracker.push(audio)
    assert len(frames) == len(feats)
    speech_after = np.zeros(len(frames), dtype=bool)  # any speech at or after i
    acc = False
    for i in range(len(frames) - 1, -1, -1):
        acc = acc or frames[i].is_speech
        speech_after[i] = acc

    out, armed, last_speech = [], False, None
    for i, (f, (_t, x)) in enumerate(zip(frames, feats, strict=True)):
        if f.t_ms < turn.turn_start_ms:
            continue
        if f.is_speech:
            armed = True  # a pause only counts after some speech in the turn
            last_speech = x  # the prosody of the speech that is about to end
            continue
        if armed and f.silence_ms >= gate_ms:
            armed = False  # one example per pause, at the gate crossing
            resumes = speech_after[i]
            pause_start = f.t_ms - f.silence_ms
            word_follows = any(w["start_ms"] > pause_start for w in words)
            prefix = " ".join(w["t"] for w in words if w["end_ms"] <= f.t_ms)
            text_model.reset()
            p_text = text_model.update(Update(t_ms=f.t_ms, text=prefix))
            out.append(
                {
                    "turn_id": turn.turn_id,
                    "global_name": turn.global_name,
                    "disfluent": turn.disfluent,
                    "t_ms": round(f.t_ms, 1),
                    "pause_start_ms": round(f.t_ms - f.silence_ms, 1),
                    "ms_to_true_end": round(
                        turn.true_end_ms - (f.t_ms - f.silence_ms), 1
                    ),
                    "label": 0 if resumes else 1,
                    "word_follows": bool(word_follows),
                    "prosody": [round(float(v), 4) for v in last_speech],
                    "text": prefix,
                    "n_words_so_far": len(prefix.split()),
                    "n_words": turn.n_words,
                    "p_text": round(float(p_text), 4),
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-set", type=Path, default=TRAIN_DIR / "audio_set.json")
    ap.add_argument("--out", type=Path, default=TRAIN_DIR / "pauses.jsonl")
    ap.add_argument("--gate-ms", type=float, default=DEFAULT_GATE_MS)
    ap.add_argument("--review", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not args.audio_set.exists():
        raise SystemExit(
            f"{args.audio_set} not found; run scripts/fetch_train_audio.py first"
        )
    rows = json.loads(args.audio_set.read_text())["turns"]
    if args.limit:
        rows = rows[: args.limit]
    vad, tracker, text_model = SileroVAD(), ProsodyTracker(), TextEOT()
    examples = []
    for i, r in enumerate(rows, 1):
        examples.extend(
            pauses_for(as_turn(r), r["words"], vad, tracker, text_model, args.gate_ms)
        )
        if i % 500 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] {len(examples):,} pauses", flush=True)

    n1 = sum(e["label"] for e in examples)
    n0 = len(examples) - n1
    turns_with_end = len({e["turn_id"] for e in examples if e["label"] == 1})
    n_mid_turns = len({e["turn_id"] for e in examples if e["label"] == 0})
    tot = max(len(examples), 1)
    print(
        f"\n{len(examples):,} pauses from {len(rows):,} turns: "
        f"end={n1:,} ({n1 / tot * 100:.1f}%)  mid={n0:,} ({n0 / tot * 100:.1f}%)"
    )
    print(
        f"turns with an end pause: {turns_with_end}/{len(rows)}; "
        f"turns with >=1 mid pause: {n_mid_turns}"
    )
    mid = [e for e in examples if e["label"] == 0]
    if mid:
        n_nonword = sum(1 for e in mid if not e["word_follows"])
        print(
            f"mid pauses followed by NO further word (VAD heard a laugh, breath, "
            f"or bleed): {n_nonword}/{len(mid)} = {n_nonword / len(mid) * 100:.1f}% "
            f"-- label noise, kept for consistency with the VAD-based eval boundary"
        )
        d = np.array([-e["ms_to_true_end"] for e in mid])
        print(
            f"mid pauses: p50 {np.median(d):.0f}ms before the end, "
            f"p90 {np.percentile(d, 90):.0f}ms"
        )
    for k, v in (("end", 1), ("mid", 0)):
        ps = [e["p_text"] for e in examples if e["label"] == v]
        if ps:
            print(f"p_text at the gate, {k}: p50={np.median(ps):.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")

    rng = random.Random(SEED)
    lines = [
        "# Pause review -- Phase 7",
        "",
        f"{args.review} random examples per class, seed {SEED}. Each is the "
        "transcript so far at the moment 200ms of silence was reached. **end**: "
        "the speaker did not speak again. **mid**: they did -- a hesitation a "
        "200ms timer would have fired on.",
        "",
    ]
    for label, title in ((1, "End of turn (label 1)"), (0, "Mid-turn pause (label 0)")):
        pool = [e for e in examples if e["label"] == label]
        lines += [f"## {title}", ""]
        for e in rng.sample(pool, min(args.review, len(pool))):
            rest = ""
            if label == 0:
                full = next(r for r in rows if r["turn_id"] == e["turn_id"])[
                    "text"
                ].split()
                rest = f"  ⟶ continues: *{' '.join(full[e['n_words_so_far'] :])[:80]}*"
            lines.append(
                f"- `{e['text']}`  (P_text={e['p_text']:.2f}, "
                f"{e['n_words_so_far']}/{e['n_words']} words){rest}"
            )
        lines.append("")
    (args.out.parent / "PAUSE_REVIEW.md").write_text("\n".join(lines))
    stats = {
        "n_pauses": len(examples),
        "n_end": n1,
        "n_mid": n0,
        "n_turns": len(rows),
        "gate_ms": args.gate_ms,
        "features": list(FEATURE_NAMES),
        "seed": SEED,
    }
    (args.out.parent / "PAUSE_STATS.json").write_text(json.dumps(stats, indent=2))
    print(f"wrote {args.out}, PAUSE_REVIEW.md, PAUSE_STATS.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
