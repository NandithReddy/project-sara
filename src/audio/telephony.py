"""Telephony-band degradation: 16kHz wideband -> 8kHz narrowband G.711 -> 16kHz.

Phone audio is the deployment target (section 5), and it strips exactly the
cues a clean-audio model leans on. This applies, in order: a 2x decimation to
8kHz (windowed-sinc low-pass, then drop every other sample); the 300-3400Hz
pass-band of a PSTN channel; a G.711 mu-law encode/decode round trip (8-bit
companding); and a 2x interpolation back to 16kHz, so every downstream
consumer -- Silero and parakeet both expect 16kHz -- sees the rate it wants
while carrying only what a phone line would have delivered.

Deterministic on purpose: no injected line noise, so a run reproduces byte
for byte. That makes this a clean-line telephony simulation, which is the
kinder end of what a real call sounds like. Stated as a limitation.

numpy only. No scipy or soxr on the runtime path (rule 5).
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
NARROW_RATE = 8_000
PASSBAND_HZ = (300.0, 3400.0)
MU = 255.0
TAPS = 201


def lowpass_fir(cutoff_hz: float, fs: float, taps: int = TAPS) -> np.ndarray:
    """Windowed-sinc (Hamming) low-pass, unity DC gain, odd length."""
    if taps % 2 == 0:
        raise ValueError("taps must be odd so the filter is symmetric")
    n = np.arange(taps) - (taps - 1) / 2
    h = 2 * cutoff_hz / fs * np.sinc(2 * cutoff_hz / fs * n)
    h *= np.hamming(taps)
    return (h / h.sum()).astype(np.float64)


def bandpass_fir(lo_hz: float, hi_hz: float, fs: float, taps: int = TAPS) -> np.ndarray:
    """Difference of two symmetric low-passes of equal length is a band-pass."""
    return lowpass_fir(hi_hz, fs, taps) - lowpass_fir(lo_hz, fs, taps)


def _apply(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.convolve(x.astype(np.float64), h, mode="same")


def decimate2(x16: np.ndarray) -> np.ndarray:
    """16kHz -> 8kHz. Anti-alias below the new Nyquist, then keep every other."""
    return _apply(lowpass_fir(3_600.0, SAMPLE_RATE), x16)[::2]


def interpolate2(x8: np.ndarray) -> np.ndarray:
    """8kHz -> 16kHz. Zero-stuff, then image-reject; gain 2 restores level."""
    z = np.zeros(len(x8) * 2, dtype=np.float64)
    z[::2] = x8
    return _apply(lowpass_fir(3_600.0, SAMPLE_RATE) * 2.0, z)


def mulaw_roundtrip(x: np.ndarray, mu: float = MU) -> np.ndarray:
    """G.711 mu-law: compand, quantise to 8 bits, expand. In [-1, 1].

    Sign plus 7-bit magnitude, as the codec does, so zero is exactly
    representable. A 256-level grid centred on 127.5 has no zero code and
    turns digital silence into a -81dBFS DC offset; a test caught it.
    """
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.sign(y) * np.round(np.abs(y) * 127.0) / 127.0
    return np.sign(q) * (np.expm1(np.abs(q) * np.log1p(mu)) / mu)


def degrade(audio16k: np.ndarray) -> np.ndarray:
    """Wideband float32 at 16kHz -> the same length, telephony band, 16kHz."""
    x = np.asarray(audio16k, dtype=np.float32).reshape(-1)
    n = len(x)
    if n % 2:
        x = np.concatenate([x, np.zeros(1, dtype=np.float32)])
    narrow = decimate2(x)
    narrow = _apply(bandpass_fir(*PASSBAND_HZ, NARROW_RATE), narrow)
    narrow = mulaw_roundtrip(narrow)
    wide = interpolate2(narrow)[:n]
    return np.clip(wide, -1.0, 1.0).astype(np.float32)
