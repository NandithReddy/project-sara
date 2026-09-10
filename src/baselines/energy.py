"""Energy-threshold voice activity detection -- the naive silence source.

This is what separates section 3's baseline #1 from baseline #2. Both apply a
fixed timeout; the difference is how "silence" is decided. Baseline #1 is what a
cheap pipeline ships: frame energy against an adaptive noise floor. Baseline #2
uses Silero. Without this module the two baselines would share a silence source
and collapse into the same system.

No parameter here is tuned on the eval set (rule 1). The margin is a
conventional 10dB over the running noise floor, and the noise floor tracker
adapts quickly downward and slowly upward -- the standard shape, because a
sudden quiet stretch is new noise-floor evidence while a sudden loud stretch is
probably speech.
"""

from __future__ import annotations

import numpy as np

from src.audio.vad import FRAME_MS, FRAME_SAMPLES, VadFrame

MARGIN_DB = 10.0
"""How far over the noise floor a frame must sit to count as speech."""

RELEASE_DB = 7.0
"""Lower bar to stay in speech. Hysteresis, for the same reason as Silero's:
every flap resets silence_ms, which is the signal the EOT decision reads."""

FLOOR_DOWN_TAU_MS = 500.0
"""Time constant for adapting the noise floor DOWNWARD, in ms.

Set from the physics, not from the scoreboard. An earlier value of 0.5 per
frame gave a ~64ms time constant, which is not a noise floor at all -- it is a
signal follower that hugs the running minimum, so ordinary room tone then sits
10dB above it and reads as speech. Measured consequence: on false-hold turns the
detector called 54% of post-turn frames speech and trailing silence never passed
672ms. 500ms is the conventional order for a noise-floor estimator.
"""

FLOOR_UP_TAU_MS = 30_000.0
"""Upward time constant -- very slow, so sustained speech cannot lift the floor
and mute the detector."""

FLOOR_DOWN = FRAME_MS / FLOOR_DOWN_TAU_MS
FLOOR_UP = FRAME_MS / FLOOR_UP_TAU_MS

SILENCE_DB = -90.0


def frame_db(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
    return SILENCE_DB if rms <= 0 else max(SILENCE_DB, 20.0 * np.log10(rms))


class EnergyVAD:
    """Frame-synchronous energy VAD with an adaptive noise floor."""

    name = "energy"

    def __init__(
        self, margin_db: float = MARGIN_DB, release_db: float = RELEASE_DB
    ) -> None:
        if release_db > margin_db:
            raise ValueError(
                f"release_db ({release_db}) must not exceed margin_db "
                f"({margin_db}); hysteresis only makes sense downward"
            )
        self.margin_db = float(margin_db)
        self.release_db = float(release_db)
        self.reset()

    def reset(self) -> None:
        self._floor_db: float | None = None
        self._t_ms = 0.0
        self._silence_ms = 0.0
        self._in_speech = False
        self._buffer = np.zeros(0, dtype=np.float32)

    def push(self, audio: np.ndarray) -> list[VadFrame]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        self._buffer = np.concatenate([self._buffer, audio])
        out: list[VadFrame] = []
        while len(self._buffer) >= FRAME_SAMPLES:
            chunk = self._buffer[:FRAME_SAMPLES]
            self._buffer = self._buffer[FRAME_SAMPLES:]
            out.append(self._process(chunk))
        return out

    def _process(self, chunk: np.ndarray) -> VadFrame:
        db = frame_db(chunk)
        if self._floor_db is None:
            self._floor_db = db
        else:
            w = FLOOR_DOWN if db < self._floor_db else FLOOR_UP
            self._floor_db = (1 - w) * self._floor_db + w * db

        over = db - self._floor_db
        self._in_speech = (
            over >= self.release_db if self._in_speech else over >= self.margin_db
        )

        self._t_ms += FRAME_MS
        self._silence_ms = 0.0 if self._in_speech else self._silence_ms + FRAME_MS
        # `prob` is not a probability here; it is the margin over the noise
        # floor, normalised only so the field means "more is more like speech".
        return VadFrame(
            t_ms=self._t_ms,
            prob=float(np.clip(over / (2 * self.margin_db), 0.0, 1.0)),
            is_speech=self._in_speech,
            silence_ms=self._silence_ms,
        )
