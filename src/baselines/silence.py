"""Fixed silence timeout -- baseline #1, and the one that matters most.

This is the rule production voice agents actually ship: declare the turn over
once trailing silence passes a fixed duration. Section 3 names 500ms and 1000ms
as the industry defaults. If our model cannot beat this, the project has failed
and we say so plainly.
"""

from __future__ import annotations

from src.eot.base import Update

#: The two operating points section 3 requires us to beat.
SECTION_3_TIMEOUTS_MS = (500.0, 1000.0)


class FixedSilenceTimeout:
    """Fires once trailing silence reaches `timeout_ms`.

    A step function, not a probability -- this baseline has no notion of
    confidence. It is swept by constructing instances across timeout values
    rather than by varying a threshold, which is exactly why `update` returns a
    float for the caller to threshold instead of the detector deciding itself.
    """

    def __init__(self, timeout_ms: float) -> None:
        if timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be positive, got {timeout_ms!r}")
        self.timeout_ms = float(timeout_ms)
        self.name = f"fixed_silence_{self.timeout_ms:g}ms"

    def reset(self) -> None:
        """No-op: the decision depends only on the current update."""

    def update(self, u: Update) -> float:
        return 1.0 if u.silence_ms >= self.timeout_ms else 0.0
