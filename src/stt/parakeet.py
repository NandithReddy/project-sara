"""Streaming STT for the LIVE path only. Never imported by eval/ (section 10).

parakeet-mlx spends a fixed ~360ms inside `add_audio` regardless of chunk size
(results/spike-parakeet-mlx-streaming.md). Calling it inline from the audio loop
would stall the VAD and push every EOT timestamp out by that much -- which would
corrupt the one thing this pipeline exists to display. So it runs on a worker
thread and the audio path never waits for it.

The same spike is why chunks are >=640ms: below roughly 400ms the stream falls
behind real time without bound.

MLX streams are thread-local: a model built on one thread cannot be driven from
another ("RuntimeError: There is no Stream(cpu, 1) in current thread"). So the
worker loads the model itself and every MLX op stays on that one thread.

Heavy imports are deferred to `start()` so this module can be imported (and the
package smoke-tested) on machines without Apple Silicon.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16_000
MIN_CHUNK_MS = 640.0
"""Below ~400ms the fixed per-call cost exceeds the audio duration and the
stream falls behind without bound. 640ms leaves headroom over the measured
~360ms."""

_STOP = object()
_RESET = object()


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """The current transcript guess. There is no finality signal.

    parakeet exposes no `is_final`, and it revises both text and timings of
    already-emitted tokens as context accumulates, so a consumer cannot tell
    which words have settled. `seq` increments whenever the text changes, which
    is the most a caller can rely on.
    """

    text: str = ""
    seq: int = 0


class ParakeetStream:
    """Feeds audio to parakeet-mlx off-thread and exposes the latest hypothesis."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        min_chunk_ms: float = MIN_CHUNK_MS,
        context_size: tuple[int, int] = (256, 256),
    ) -> None:
        if min_chunk_ms < 400:
            raise ValueError(
                f"min_chunk_ms={min_chunk_ms} is below the measured floor. "
                f"Per-call cost is ~360ms regardless of chunk size, so shorter "
                f"chunks fall behind real time without bound."
            )
        self.model_name = model_name
        self.min_chunk_samples = int(SAMPLE_RATE * min_chunk_ms / 1000.0)
        self.context_size = context_size

        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._latest = Hypothesis()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._load_s = 0.0

    def start(self, timeout_s: float = 300.0) -> float:
        """Start the worker, which loads the model. Returns load time in seconds.

        Blocks until the model is ready so the caller does not start feeding a
        stream that is not listening yet.
        """
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise TimeoutError(f"parakeet model did not load within {timeout_s:.0f}s")
        self._raise_if_failed()
        return self._load_s

    def push(self, audio: np.ndarray) -> None:
        """Queue audio. Non-blocking; returns immediately regardless of STT load."""
        self._buffer = np.concatenate(
            [self._buffer, np.asarray(audio, dtype=np.float32).reshape(-1)]
        )
        while len(self._buffer) >= self.min_chunk_samples:
            self._q.put(self._buffer[: self.min_chunk_samples])
            self._buffer = self._buffer[self.min_chunk_samples :]

    def latest(self) -> Hypothesis:
        """Most recent hypothesis. Raises if the worker has died (rule 3)."""
        self._raise_if_failed()
        with self._lock:
            return self._latest

    def _raise_if_failed(self) -> None:
        """Never let a dead worker look like a silent stream of empty text."""
        if self._error is not None:
            raise RuntimeError("parakeet worker thread died") from self._error

    def reset(self) -> None:
        """Start a fresh transcript. Drops any partial chunk still buffering."""
        self._buffer = np.zeros(0, dtype=np.float32)
        self._q.put(_RESET)

    def close(self) -> None:
        self._q.put(_STOP)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _worker(self) -> None:
        try:
            self._run()
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the main thread
            self._error = exc
            self._ready.set()

    def _run(self) -> None:
        import time

        import mlx.core as mx
        from parakeet_mlx import from_pretrained

        # Loaded here, not in start(): MLX streams are thread-local, so the
        # model must be built on the same thread that will drive it.
        t0 = time.perf_counter()
        self._model = from_pretrained(self.model_name)
        self._load_s = time.perf_counter() - t0
        self._ready.set()

        while True:
            with self._model.transcribe_stream(context_size=self.context_size) as tr:
                with self._lock:
                    self._latest = Hypothesis(seq=self._latest.seq + 1)
                while True:
                    item = self._q.get()
                    if item is _STOP:
                        return
                    if item is _RESET:
                        break
                    tr.add_audio(mx.array(item))
                    text = (getattr(tr.result, "text", "") or "").strip()
                    with self._lock:
                        if text != self._latest.text:
                            self._latest = Hypothesis(text, self._latest.seq + 1)
