"""Silero VAD wrapper.

`test_detects_speech_in_real_audio` is the regression guard for the context
prefix described in src/audio/vad.py: delete the prepend and that test drops to
0% speech while everything else here still passes, because the model does not
raise -- it just returns ~0.001 forever.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.audio.vad import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, SileroVAD

REPO = Path(__file__).resolve().parents[1]
JFK_WAV = REPO / "data" / "raw" / "jfk.wav"
needs_audio = pytest.mark.skipif(
    not JFK_WAV.exists(), reason=f"{JFK_WAV} absent (data/raw is gitignored)"
)


@pytest.fixture
def vad():
    return SileroVAD()


def test_missing_model_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="not downloading anything|rule 3"):
        SileroVAD(model_path=tmp_path / "nope.onnx")


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_rejects_out_of_range_threshold(bad):
    with pytest.raises(ValueError, match="must be in"):
        SileroVAD(threshold=bad)


def test_buffers_until_a_full_frame_is_available(vad):
    """A mic callback does not hand you 512 samples at a time."""
    assert vad.push(np.zeros(300, dtype=np.float32)) == []
    frames = vad.push(np.zeros(300, dtype=np.float32))
    assert len(frames) == 1, "512 of the 600 buffered samples should have run"


def test_emits_one_frame_per_512_samples(vad):
    frames = vad.push(np.zeros(FRAME_SAMPLES * 5, dtype=np.float32))
    assert len(frames) == 5
    assert [f.t_ms for f in frames] == pytest.approx(
        [FRAME_MS * i for i in range(1, 6)]
    )


def test_silence_accumulates_over_silent_audio(vad):
    frames = vad.push(np.zeros(FRAME_SAMPLES * 4, dtype=np.float32))
    assert all(not f.is_speech for f in frames)
    assert [f.silence_ms for f in frames] == pytest.approx(
        [FRAME_MS * i for i in range(1, 5)]
    )


def test_reset_clears_the_clock_and_the_buffer(vad):
    vad.push(np.zeros(FRAME_SAMPLES * 3 + 100, dtype=np.float32))
    vad.reset()
    frames = vad.push(np.zeros(FRAME_SAMPLES, dtype=np.float32))
    assert len(frames) == 1, "leftover buffer should have been dropped"
    assert frames[0].t_ms == pytest.approx(FRAME_MS)
    assert frames[0].silence_ms == pytest.approx(FRAME_MS)


@needs_audio
def test_detects_speech_in_real_audio(vad):
    """Regression guard for the 64-sample context prefix.

    Without the prefix this reports 0% speech on 11 seconds of clear speech,
    and reports it silently.
    """
    import soundfile as sf

    pcm, sr = sf.read(JFK_WAV, dtype="float32")
    assert sr == SAMPLE_RATE

    frames = vad.push(pcm)
    speech_ratio = sum(f.is_speech for f in frames) / len(frames)
    assert speech_ratio > 0.5, (
        f"only {speech_ratio:.1%} of frames detected as speech on a clip that is "
        f"almost entirely speech -- is the context prefix still being applied?"
    )
    assert max(f.prob for f in frames) > 0.9


@needs_audio
def test_hysteresis_reduces_speech_silence_flapping():
    """Every flap resets silence_ms, which is the signal EOT reads."""
    import soundfile as sf

    pcm, _ = sf.read(JFK_WAV, dtype="float32")

    def transitions(neg_threshold):
        v = SileroVAD(threshold=0.5, neg_threshold=neg_threshold)
        states = [f.is_speech for f in v.push(pcm)]
        return sum(a != b for a, b in zip(states, states[1:], strict=False))

    assert transitions(0.35) <= transitions(0.5), "hysteresis should not flap more"
