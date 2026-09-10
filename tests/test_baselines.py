"""Energy VAD (baseline #1's silence source) and the punctuation heuristic."""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.vad import FRAME_MS, FRAME_SAMPLES
from src.baselines.energy import FLOOR_DOWN, FLOOR_UP, EnergyVAD, frame_db
from src.baselines.punctuation import PunctuationHeuristic
from src.eot.base import EOTDetector, Update


def tone(n_frames: int, amp: float = 0.2, sr: int = 16_000) -> np.ndarray:
    t = np.arange(n_frames * FRAME_SAMPLES) / sr
    return (amp * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def test_noise_floor_time_constants_are_set_from_duration_not_per_frame():
    """A floor that adapts in ~64ms is a signal follower, not a noise floor."""
    assert FLOOR_DOWN == pytest.approx(FRAME_MS / 500.0)
    assert FLOOR_UP < FLOOR_DOWN / 50


def test_frame_db_floors_digital_silence():
    assert frame_db(np.zeros(FRAME_SAMPLES, dtype=np.float32)) == -90.0
    assert frame_db(np.full(FRAME_SAMPLES, 0.1, dtype=np.float32)) == pytest.approx(
        -20.0, abs=0.1
    )


def test_release_threshold_must_not_exceed_the_attack_threshold():
    with pytest.raises(ValueError, match="hysteresis only makes sense downward"):
        EnergyVAD(margin_db=6.0, release_db=12.0)


def test_digital_silence_never_reads_as_speech():
    v = EnergyVAD()
    frames = v.push(np.zeros(FRAME_SAMPLES * 20, dtype=np.float32))
    assert not any(f.is_speech for f in frames)
    assert frames[-1].silence_ms == pytest.approx(FRAME_MS * 20)


def test_a_loud_tone_over_a_quiet_floor_reads_as_speech():
    v = EnergyVAD()
    v.push(np.zeros(FRAME_SAMPLES * 30, dtype=np.float32) + 1e-4)
    frames = v.push(tone(20))
    assert any(f.is_speech for f in frames), "20dB over the floor should trigger"


def test_speech_resets_the_silence_counter():
    v = EnergyVAD()
    v.push(np.zeros(FRAME_SAMPLES * 30, dtype=np.float32) + 1e-4)
    frames = v.push(tone(10))
    speech = [f for f in frames if f.is_speech]
    assert speech and speech[-1].silence_ms == 0.0


@pytest.mark.parametrize(
    ("text", "fires"),
    [
        ("my order number is", False),
        ("my order number is.", True),
        ("is that right?", True),
        ("stop!", True),
        ("well, um", False),
        ("hello there  ", False),
        ("done.  ", True),
        ("", False),
    ],
)
def test_punctuation_fires_only_on_terminal_punctuation(text, fires):
    d = PunctuationHeuristic()
    assert d.update(Update(t_ms=0.0, text=text)) == (1.0 if fires else 0.0)


def test_punctuation_conforms_to_the_detector_protocol():
    assert isinstance(PunctuationHeuristic(), EOTDetector)


def test_punctuation_ignores_silence_entirely():
    """Baseline #3 is purely textual: no timer, no audio."""
    d = PunctuationHeuristic()
    assert d.update(Update(t_ms=0.0, text="not done", silence_ms=9_999.0)) == 0.0
