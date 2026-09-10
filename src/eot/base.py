"""The pluggable end-of-turn interface.

Every EOT implementation -- the silence baselines, the punctuation heuristic,
and eventually the model itself -- satisfies EOTDetector, so the live pipeline
and the eval harness can drive any of them without knowing which one they hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Word:
    """One word with its alignment, in ms on the audio clock."""

    text: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True, slots=True)
class Update:
    """One incremental observation of the user's in-progress turn.

    Every time here is on the AUDIO clock -- ms since the start of the turn's
    audio -- never wall clock. The eval harness replays gold transcripts and has
    no wall clock at all, and `added_latency_ms` is defined against the true turn
    end in audio time. An audio-clock interface therefore measures the same
    quantity on both paths; a wall-clock one would not port to eval.
    """

    t_ms: float
    """Audio time of this observation."""

    text: str
    """Transcript hypothesis so far. May be revised, not only appended to."""

    words: tuple[Word, ...] = ()
    """Aligned words, where the source supplies them. Also subject to revision."""

    silence_ms: float = 0.0
    """Trailing non-speech at `t_ms`. Zero means we are inside speech.

    The live path derives this from the VAD; the eval path derives it from gold
    word end times (`t_ms` minus the last word's end). Both paths supplying the
    same quantity is what lets a silence baseline run unmodified on each, without
    the harness ever importing `src/stt/` (section 10).
    """


@runtime_checkable
class EOTDetector(Protocol):
    """Emits P(turn_complete) for each incremental update.

    Implementations return a probability rather than a bool so that the Phase 5
    threshold sweep can re-threshold a cached run instead of re-running the
    detector once per threshold. Deciding *when to fire* is the caller's job.
    """

    name: str

    def reset(self) -> None:
        """Clear per-turn state. Called before the first update of each turn."""
        ...

    def update(self, u: Update) -> float:
        """Return P(turn_complete) in [0.0, 1.0] for this update."""
        ...
