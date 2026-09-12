"""The telephony simulation does what a phone line does, and nothing random."""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.telephony import SAMPLE_RATE, degrade, lowpass_fir, mulaw_roundtrip


def tone(hz: float, seconds: float = 1.0, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def rms(x: np.ndarray) -> float:
    core = x[len(x) // 4 : -len(x) // 4]  # ignore filter edges
    return float(np.sqrt(np.mean(np.square(core))))


def db(out: np.ndarray, ref: np.ndarray) -> float:
    return 20 * np.log10(rms(out) / rms(ref))


def test_length_is_preserved_odd_and_even():
    for n in (16_000, 16_001):
        assert len(degrade(np.zeros(n, dtype=np.float32))) == n


def test_deterministic():
    x = tone(1_000.0)
    assert np.array_equal(degrade(x), degrade(x))


def test_silence_stays_silence():
    assert np.abs(degrade(np.zeros(8_000, dtype=np.float32))).max() == 0.0


def test_speech_band_survives():
    x = tone(1_000.0)
    assert db(degrade(x), x) > -3.0


def test_above_narrowband_nyquist_is_gone():
    x = tone(6_000.0)
    assert db(degrade(x), x) < -30.0


def test_below_the_300hz_highpass_is_attenuated():
    x = tone(80.0)
    assert db(degrade(x), x) < -15.0


def test_top_of_passband_edge_rolls_off():
    inside, outside = tone(3_000.0), tone(3_900.0)
    assert db(degrade(inside), inside) > db(degrade(outside), outside) + 10


def test_mulaw_quantises_and_roundtrips_within_tolerance():
    x = np.linspace(-1, 1, 10_001)
    y = mulaw_roundtrip(x)
    assert len(np.unique(np.round(y, 6))) <= 256
    assert np.abs(y - x).max() < 0.05
    assert np.abs(y[np.abs(x) < 0.01] - x[np.abs(x) < 0.01]).max() < 0.002, (
        "companding keeps small signals precise"
    )


def test_lowpass_has_unity_dc_gain_and_needs_odd_taps():
    assert lowpass_fir(3_600.0, SAMPLE_RATE).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="odd"):
        lowpass_fir(3_600.0, SAMPLE_RATE, taps=100)
