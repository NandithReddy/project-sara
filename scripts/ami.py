"""AMI corpus access shared by the eval and train builders.

Parsing and the turn definition live here so that data/eval/ and data/train/
are cut with the SAME knife. If the definition of a turn drifted between the
two, the model would be trained on one thing and measured on another.

Turn definition: a maximal same-speaker run split on >1000ms same-speaker
gaps, requiring another speaker to take the floor afterwards, with no
cross-speaker overlap and no <gap> element. Punctuation is folded onto the
preceding word as `punc_after`.

Not imported by src/ or eval/: this reads data/raw/, which only the builders
touch.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AMI = REPO / "data" / "raw" / "ami"
WORDS = AMI / "words"
MEETINGS_XML = AMI / "corpusResources" / "meetings.xml"

SPLIT_MS = 1000.0
"""Same-speaker gap that separates two utterances. The gap distribution is flat
between 300ms and 1500ms (only 61 of 121,575 gaps fall in that band), so this
threshold is insensitive -- it is a round number, not a tuned one."""


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
    words: tuple[tuple[str, float, float, str], ...]
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
    """Word stream for one meeting-speaker, minus non-speech elements.

    Punctuation is a separate zero-duration <w punc="true"> element in AMI. It
    is folded onto the preceding word as `punc_after` rather than kept as its
    own token: it has no duration, so it cannot be "emitted" at a time of its
    own, and baseline #3 only ever asks what the transcript currently ends with.
    """
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
            if out:
                out[-1]["punc_after"] = (out[-1]["punc_after"] or "") + (
                    el.text or ""
                ).strip()
            continue
        out.append(
            {
                "s": float(s),
                "e": float(e),
                "t": (el.text or "").strip(),
                "disf": disf,
                "gap": gap,
                "punc_after": None,
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
                words=tuple(
                    (w["t"], w["s"], w["e"], w["punc_after"] or "") for w in ws
                ),
                next_speaker_gap_ms=(nxt[1][0]["s"] - t1) * 1000,
                own_resume_gap_ms=own_resume_gap_ms,
                stratum=(
                    "disfluent" if disfluent else "short" if n <= 3 else "ordinary"
                ),
            )
        )
    return turns


def render(words, k: int, punctuated: bool) -> str:
    """The first k words of a turn as the model will see them.

    One function, shared by the train builder and (in Phase 4) the trainer and
    the live adapter, so "the text the model saw" means the same thing
    everywhere. `words` is the Turn.words tuple: (text, start_s, end_s, punc).
    """
    return " ".join(
        text + (punc if punctuated else "") for text, _s, _e, punc in words[:k]
    )
