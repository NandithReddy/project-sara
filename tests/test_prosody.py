"""The prosody tracker hears what it claims to hear -- on synthetic signals
with known answers, including through the telephony band."""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.prosody import (
    FEATURE_NAMES,
    N_FEATURES,
    ProsodyTracker,
    estimate_f0,
)
from src.audio.telephony import degrade
from src.audio.vad import FRAME_SAMPLES, SAMPLE_RATE

IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def tone(hz, seconds=1.0, amp=0.2, harmonics=1):
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    x = sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, harmonics + 1))
    return (amp * x / np.abs(x).max()).astype(np.float32)


def glide(hz0, hz1, seconds=1.0, amp=0.2):
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    phase = 2 * np.pi * np.cumsum(np.linspace(hz0, hz1, len(t))) / SAMPLE_RATE
    return (amp * np.sin(phase)).astype(np.float32)


def test_feature_vector_has_the_declared_shape():
    tr = ProsodyTracker()
    out = tr.push(np.zeros(FRAME_SAMPLES * 3, dtype=np.float32))
    assert len(out) == 3
    assert all(len(feat) == N_FEATURES for _, feat in out)


def test_f0_of_a_pure_tone():
    f0, peak = estimate_f0(tone(200.0)[: FRAME_SAMPLES * 2])
    assert abs(f0 - 200.0) < 6.0 and peak > 0.9


def test_silence_is_unvoiced():
    tr = ProsodyTracker()
    out = tr.push(np.zeros(FRAME_SAMPLES * 20, dtype=np.float32))
    feat = out[-1][1]
    assert feat[IDX["voiced_fraction"]] == 0.0
    assert feat[IDX["ms_since_voiced"]] > 0


def test_a_falling_pitch_has_a_negative_slope_and_fall():
    tr = ProsodyTracker()
    out = tr.push(glide(220.0, 150.0, seconds=0.8))
    feat = out[-1][1]
    assert feat[IDX["voiced_fraction"]] > 0.8
    assert feat[IDX["f0_slope_st_per_s"]] < -3.0
    assert feat[IDX["last_run_f0_fall_st"]] < -3.0


def test_energy_trailing_off_reads_as_a_drop_and_negative_slope():
    x = tone(180.0, seconds=1.0)
    x[len(x) // 2 :] *= np.linspace(1.0, 0.05, len(x) - len(x) // 2).astype(np.float32)
    tr = ProsodyTracker()
    feat = tr.push(x)[-1][1]
    assert feat[IDX["energy_drop_from_peak_db"]] > 15.0
    assert feat[IDX["energy_slope_db_per_s"]] < 0.0


def test_pitch_survives_the_telephony_band_by_its_harmonics():
    """A 120Hz voice has its fundamental below the 300Hz line; autocorrelation
    still finds the 120Hz period from the harmonics that get through."""
    x = degrade(tone(120.0, seconds=0.5, harmonics=12))
    f0, peak = estimate_f0(x[SAMPLE_RATE // 4 : SAMPLE_RATE // 4 + FRAME_SAMPLES * 2])
    assert abs(f0 - 120.0) < 6.0, f"got {f0:.1f}Hz"
    assert peak > 0.5


def test_final_lengthening_shows_as_last_run_over_mean():
    """Three short voiced bursts then one long one."""
    gap = np.zeros(int(SAMPLE_RATE * 0.12), dtype=np.float32)
    short = tone(200.0, seconds=0.15)
    long_ = tone(200.0, seconds=0.45)
    x = np.concatenate([short, gap, short, gap, short, gap, long_])
    feat = ProsodyTracker().push(x)[-1][1]
    assert feat[IDX["n_runs"]] >= 3
    assert feat[IDX["last_run_over_mean_run"]] > 1.4


def test_reset_clears_history():
    tr = ProsodyTracker()
    tr.push(tone(200.0, seconds=0.5))
    tr.reset()
    feat = tr.push(np.zeros(FRAME_SAMPLES, dtype=np.float32))[-1][1]
    assert feat[IDX["n_runs"]] == 0.0


@pytest.mark.parametrize("n", [1, 7, 40])
def test_deterministic(n):
    x = tone(170.0, seconds=0.3)
    a = ProsodyTracker().push(x)[-1][1]
    b = ProsodyTracker().push(x)[-1][1]
    assert a == b
