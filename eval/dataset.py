"""Read the frozen eval set. The only sanctioned way in.

Enforces ENGINEERING.md rule 1: the eval set is never trained on and never
tuned on. Any training entry point sets SARA_TRAINING=1, and every read here
fails loudly while it is set. The rule is worth nothing if it lives only in
prose.

Never imports src/stt/ (section 10). The eval path replays gold transcripts and
audio that were frozen at build time; it never runs a recogniser, so a run is
reproducible byte for byte and a regression is always a change in our model.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "data" / "eval"
MANIFEST = EVAL_DIR / "MANIFEST.json"
TRAINING_ENV = "SARA_TRAINING"


class EvalSetViolation(RuntimeError):
    """Raised when the frozen eval set is touched by something that must not."""


@dataclass(frozen=True, slots=True)
class EvalTurn:
    """One frozen turn. Times are ms from the start of this turn's audio."""

    turn_id: str
    meeting: str
    speaker: str
    global_name: str
    stratum: str
    text: str
    n_words: int
    disfluent: bool
    audio_path: Path
    seg_duration_ms: float
    turn_start_ms: float
    true_end_ms: float
    annotated_end_ms: float
    boundary_source: str


def assert_not_training() -> None:
    """Rule 1, made executable."""
    if os.environ.get(TRAINING_ENV):
        raise EvalSetViolation(
            f"{TRAINING_ENV} is set, so this process is training. Reading "
            f"data/eval/ during training is a bug: it is the frozen, held-out "
            f"set and touching it invalidates every number measured against it. "
            f"Train on data/train/ instead (ENGINEERING.md rule 1)."
        )


def verify_manifest() -> list[str]:
    """Return the paths whose bytes no longer match the freeze. Empty is good."""
    manifest = json.loads(MANIFEST.read_text())
    bad = []
    for rel, meta in manifest["files"].items():
        path = EVAL_DIR / rel
        if not path.exists():
            bad.append(f"{rel}: missing")
            continue
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != meta["sha256"]:
            bad.append(f"{rel}: sha256 changed")
    return bad


def load_eval_set(verify: bool = False) -> list[EvalTurn]:
    """Load the frozen turns. `verify=True` also re-hashes every file."""
    assert_not_training()
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"{MANIFEST} not found. Build the eval set first:\n"
            f"  uv run python scripts/build_eval_set.py\n"
            f"  uv run python scripts/fetch_eval_audio.py\n"
            f"  uv run python scripts/freeze_eval_set.py"
        )
    if verify and (bad := verify_manifest()):
        raise EvalSetViolation(
            f"The frozen eval set has changed ({len(bad)} file(s)): {bad[:3]}"
        )

    spec = json.loads((EVAL_DIR / "eval_set.json").read_text())
    return [
        EvalTurn(
            turn_id=t["turn_id"],
            meeting=t["meeting"],
            speaker=t["speaker"],
            global_name=t["global_name"],
            stratum=t["stratum"],
            text=t["text"],
            n_words=t["n_words"],
            disfluent=t["disfluent"],
            audio_path=EVAL_DIR / t["audio"],
            seg_duration_ms=t["seg_duration_ms"],
            turn_start_ms=t["turn_start_ms"],
            true_end_ms=t["true_end_ms"],
            annotated_end_ms=t["annotated_end_ms"],
            boundary_source=t["boundary_source"],
        )
        for t in spec["turns"]
    ]
