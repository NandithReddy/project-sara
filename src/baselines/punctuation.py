"""Punctuation heuristic -- section 3's baseline #3.

Declares the turn over as soon as the transcript ends in sentence-final
punctuation. No timer, no audio: purely a property of the text.

Two things bound it on this eval set, both measured rather than assumed:
87.4% of turns end in terminal punctuation, so 12.6% can never fire at all;
and terminal punctuation appears mid-turn in 37 of 198 turns, where it fires
early.

It is also the baseline most flattered by gold transcripts. AMI punctuation is
placed by human annotators who heard the whole recording; a streaming recogniser
punctuates causally, from a partial hypothesis it keeps revising. The Phase 6
real-ASR condition is what turns that caveat into a number.
"""

from __future__ import annotations

from src.eot.base import Update

TERMINAL = (".", "?", "!")


class PunctuationHeuristic:
    """Fires when the transcript so far ends in terminal punctuation."""

    def __init__(self, terminal: tuple[str, ...] = TERMINAL) -> None:
        if not terminal:
            raise ValueError("need at least one terminal punctuation mark")
        self.terminal = terminal
        self.name = "punctuation"

    def reset(self) -> None:
        """No-op: the decision depends only on the current transcript."""

    def update(self, u: Update) -> float:
        return 1.0 if u.text.rstrip().endswith(self.terminal) else 0.0
