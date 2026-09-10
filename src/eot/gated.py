"""A silence gate around any detector: semantics decide, silence confirms.

Fires only when the wrapped detector says the turn is complete AND the speaker
has actually been silent for `min_silence_ms`. Below the gate it reports 0.0
regardless of what the text looks like.

Why this exists, measured in Phase 4: the text-only model's false fires were
half on the first word -- "Okay", "Oh", "Um" -- because in the training data
those are whole turns most of the time, and no function of the text can tell
"Okay." from "Okay so the thing is". A timer can, because it waits. The gate
gives the model that one thing, and only that: it does not add a timeout, so
the decision still comes from the text. The thesis, restated: the semantic
signal should reduce the silence a system needs, not replace it with a guess.

The gate value is not tuned on the eval set (rule 1). The training data has no
audio, so it cannot be tuned on validation either; it is chosen on principle
and reported across a small range in results/tradeoff.csv so its sensitivity
is visible rather than hidden.
"""

from __future__ import annotations

from src.eot.base import EOTDetector, Update

DEFAULT_GATE_MS = 200.0
"""Just past the boundary tolerance (150ms) and the longest ordinary
articulatory pause. Long enough to mean "stopped", far shorter than any timer
that avoids cutoffs (the Silero timer needs 900ms to get under 10%)."""


class SilenceGated:
    """`inner` decides; the speaker must have paused at least `min_silence_ms`."""

    def __init__(self, inner: EOTDetector, min_silence_ms: float = DEFAULT_GATE_MS):
        if min_silence_ms < 0:
            raise ValueError(f"min_silence_ms must be >= 0, got {min_silence_ms!r}")
        self.inner = inner
        self.min_silence_ms = float(min_silence_ms)
        self.name = f"{inner.name}+gate{int(self.min_silence_ms)}"

    def reset(self) -> None:
        self.inner.reset()

    def update(self, u: Update) -> float:
        if u.silence_ms < self.min_silence_ms:
            return 0.0
        return self.inner.update(u)

    # Pass-throughs so the harness reports the model's own numbers.
    @property
    def meta(self) -> dict:
        return getattr(self.inner, "meta", {})

    def inference_latency_ms(self) -> dict:
        fn = getattr(self.inner, "inference_latency_ms", None)
        return fn() if fn else {"n": 0}
