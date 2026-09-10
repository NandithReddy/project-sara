"""Microphone capture: 16kHz mono float32 blocks.

Deliberately thin. The mic is scaffolding for the live demo (section 2), and no
reported number ever comes through here.
"""

from __future__ import annotations

import queue
import sys
from collections.abc import Iterator

import numpy as np

from src.audio.vad import SAMPLE_RATE

DEFAULT_BLOCK_MS = 32.0


def mic_blocks(
    sample_rate: int = SAMPLE_RATE,
    block_ms: float = DEFAULT_BLOCK_MS,
    device: int | str | None = None,
) -> Iterator[np.ndarray]:
    """Yield mono float32 blocks from the input device until the caller stops.

    Blocks arrive on PortAudio's callback thread and are handed over through a
    queue, because doing real work inside the callback drops audio.
    """
    import sounddevice as sd

    if not any(d["max_input_channels"] > 0 for d in sd.query_devices()):
        raise RuntimeError(
            "No audio input device found. Check System Settings > Sound > Input, "
            "and that the terminal has microphone permission."
        )

    blocks: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status) -> None:
        if status:  # overflow/underflow -- surface it, never swallow it (rule 3)
            print(f"[audio] {status}", file=sys.stderr)
        blocks.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=int(sample_rate * block_ms / 1000.0),
        device=device,
        callback=callback,
    ):
        while True:
            yield blocks.get()
