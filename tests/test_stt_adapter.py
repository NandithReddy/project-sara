"""ParakeetStream buffering and failure behaviour.

These never load the model: it is 2.3GB and Apple-Silicon-only, and none of
this logic depends on it. The model path is exercised by scripts/live.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.stt.parakeet import MIN_CHUNK_MS, SAMPLE_RATE, Hypothesis, ParakeetStream


def test_default_chunk_matches_the_measured_floor():
    """The spike measured ~360ms fixed cost per call; 640ms leaves headroom."""
    assert MIN_CHUNK_MS == 640.0


@pytest.mark.parametrize("bad", [0, 100, 399.9])
def test_rejects_chunks_that_cannot_keep_up_with_real_time(bad):
    with pytest.raises(ValueError, match="below the measured floor"):
        ParakeetStream(min_chunk_ms=bad)


def test_push_buffers_until_a_full_chunk_then_enqueues():
    stt = ParakeetStream()
    half = stt.min_chunk_samples // 2

    stt.push(np.zeros(half, dtype=np.float32))
    assert stt._q.qsize() == 0, "a partial chunk must not be dispatched"

    stt.push(np.zeros(half + 1, dtype=np.float32))
    assert stt._q.qsize() == 1
    assert stt._q.get().shape == (stt.min_chunk_samples,)


def test_push_splits_long_audio_into_several_chunks():
    stt = ParakeetStream()
    stt.push(np.zeros(stt.min_chunk_samples * 3, dtype=np.float32))
    assert stt._q.qsize() == 3


def test_chunk_size_follows_the_sample_rate():
    stt = ParakeetStream(min_chunk_ms=640.0)
    assert stt.min_chunk_samples == int(SAMPLE_RATE * 0.640)


def test_a_dead_worker_fails_loudly_instead_of_returning_empty_text():
    """Rule 3: a crashed STT must not look like a stream of empty transcripts."""
    stt = ParakeetStream()
    stt._latest = Hypothesis("something", 1)
    stt._error = RuntimeError("metal exploded")

    with pytest.raises(RuntimeError, match="worker thread died"):
        stt.latest()
