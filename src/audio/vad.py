"""Streaming voice activity detection: Silero VAD on onnxruntime.

This is where `Update.silence_ms` comes from on the live path. The eval path
derives the same quantity from gold word end times and never touches this
module (section 10).

Runs the ONNX graph directly rather than through the `silero-vad` package,
which pulls torch + torchaudio -- 16 packages, several GB -- to execute a 2.3MB
model. onnxruntime also does double duty: Phase 4 needs it to hit the <20ms CPU
inference budget in section 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512
"""Silero v5 accepts exactly 512 samples (32ms) per call at 16kHz."""

CONTEXT_SAMPLES = 64
"""Samples of the previous frame prepended to every call. NOT optional.

Omitting this does not raise. The model silently returns P(speech) ~= 0.001 for
every frame of clear speech. Measured on an 11s clip: 0% of frames detected as
speech without the prefix, 68% with it. The tensor fed to the session is
therefore 576 samples wide, not 512.
"""

FRAME_MS = FRAME_SAMPLES / SAMPLE_RATE * 1000.0
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "silero_vad.onnx"


@dataclass(frozen=True, slots=True)
class VadFrame:
    """One 32ms decision, timed at the END of the frame on the audio clock."""

    t_ms: float
    prob: float
    is_speech: bool
    silence_ms: float


class SileroVAD:
    """Frame-synchronous VAD over a stream of arbitrary-length audio blocks.

    Accepts any block size and buffers internally, because a mic callback does
    not hand you 512 samples at a time.

    Uses two thresholds rather than one: speech starts at `threshold` and only
    ends below `neg_threshold`. A single threshold flaps frame-to-frame around
    the boundary, and every flap resets `silence_ms` -- which is precisely the
    signal the EOT decision reads, so flapping there corrupts the metric.
    """

    name = "silero"

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        threshold: float = 0.5,
        neg_threshold: float | None = None,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Silero VAD model not found at {model_path}. It is committed to "
                f"the repo at models/silero_vad.onnx -- see models/README.md. "
                f"Not downloading anything automatically (rule 3)."
            )
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold!r}")

        self.threshold = float(threshold)
        self.neg_threshold = (
            float(neg_threshold) if neg_threshold is not None else self.threshold - 0.15
        )
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Clear all per-turn state. Call before each new turn."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._t_ms = 0.0
        self._silence_ms = 0.0
        self._in_speech = False

    def push(self, audio: np.ndarray) -> list[VadFrame]:
        """Feed mono float32 audio of any length; return one frame per 512 samples.

        Leftover samples are retained and consumed by the next call.
        """
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        self._buffer = np.concatenate([self._buffer, audio])

        frames: list[VadFrame] = []
        while len(self._buffer) >= FRAME_SAMPLES:
            chunk = self._buffer[:FRAME_SAMPLES]
            self._buffer = self._buffer[FRAME_SAMPLES:]
            frames.append(self._process(chunk))
        return frames

    def _process(self, chunk: np.ndarray) -> VadFrame:
        x = np.concatenate([self._context, chunk]).reshape(1, -1)
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = chunk[-CONTEXT_SAMPLES:]

        prob = float(out[0, 0])
        if self._in_speech:
            self._in_speech = prob >= self.neg_threshold
        else:
            self._in_speech = prob >= self.threshold

        self._t_ms += FRAME_MS
        self._silence_ms = 0.0 if self._in_speech else self._silence_ms + FRAME_MS
        return VadFrame(
            t_ms=self._t_ms,
            prob=prob,
            is_speech=self._in_speech,
            silence_ms=self._silence_ms,
        )
