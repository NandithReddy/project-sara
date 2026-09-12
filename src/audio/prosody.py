"""Causal prosodic features, one vector per 32ms frame.

The cues a listener uses to hear a turn ending -- a final pitch fall, the last
syllable stretched, energy trailing off -- are exactly what a transcript throws
away, and what the Phase 3 ceiling said no text can recover ("Yeah" is a whole
turn 63% of the time, and the text cannot say which). This tracker computes
them from the waveform alone, causally, so the same code runs on the live mic
and over the frozen eval audio.

Per frame: log energy, and F0 by normalised autocorrelation over a 64ms window
(two frames). Autocorrelation finds the period from the harmonics, so it
survives the 300-3400Hz telephony band even though most fundamentals sit below
it -- the missing-fundamental effect, tested.

Over a trailing 1500ms window: slopes, falls, drops, and the length of the last
voiced run against the mean run, which is the lengthening cue. Fourteen
numbers. Nothing here is tuned on the eval set; the thresholds are
conventional and the model downstream learns the weights.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.audio.vad import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE

WINDOW_MS = 1500.0
F0_MIN_HZ, F0_MAX_HZ = 60.0, 400.0
VOICING_THRESHOLD = 0.55
"""Normalised autocorrelation peak needed to call a frame voiced."""
ENERGY_FLOOR_DB = -55.0
"""Below this a frame cannot be voiced whatever the autocorrelation says."""
SLOPE_MS = 500.0

FEATURE_NAMES = (
    "energy_db",
    "energy_drop_from_peak_db",
    "energy_slope_db_per_s",
    "f0_last_semitones_vs_median",
    "f0_slope_st_per_s",
    "f0_range_st",
    "voiced_fraction",
    "ms_since_voiced",
    "last_run_ms",
    "last_run_over_mean_run",
    "n_runs",
    "last_run_energy_vs_window_db",
    "last_run_f0_fall_st",
    "energy_std_db",
)
N_FEATURES = len(FEATURE_NAMES)


@dataclass(frozen=True, slots=True)
class Frame:
    t_ms: float
    energy_db: float
    f0_hz: float  # 0.0 when unvoiced
    voiced: bool


def _semitones(f0: float, ref: float) -> float:
    return 12.0 * np.log2(max(f0, 1e-6) / max(ref, 1e-6))


def frame_energy_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return float(20.0 * np.log10(rms)) if rms > 0 else -120.0


def estimate_f0(window: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[float, float]:
    """(f0_hz, peak) by normalised autocorrelation over the lag band."""
    x = window.astype(np.float64)
    x = x - x.mean()
    e0 = float(np.dot(x, x))
    if e0 <= 0:
        return 0.0, 0.0
    lo, hi = int(sr / F0_MAX_HZ), int(sr / F0_MIN_HZ)
    hi = min(hi, len(x) - 1)
    if hi <= lo:
        return 0.0, 0.0
    # r(lag) = <x[:-lag], x[lag:]> / sqrt(E(x[:-lag]) E(x[lag:]))
    best_lag, best = 0, 0.0
    for lag in range(lo, hi + 1):
        a, b = x[:-lag], x[lag:]
        denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
        if denom <= 0:
            continue
        r = float(np.dot(a, b) / denom)
        if r > best:
            best, best_lag = r, lag
    return (sr / best_lag if best_lag else 0.0), best


class ProsodyTracker:
    """Push 512-sample frames; read a 14-feature vector after each."""

    def __init__(self, window_ms: float = WINDOW_MS) -> None:
        self.window_frames = max(2, int(round(window_ms / FRAME_MS)))
        self.reset()

    def reset(self) -> None:
        self._prev = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        self._t_ms = 0.0
        self._frames: deque[Frame] = deque(maxlen=self.window_frames)
        self._buffer = np.zeros(0, dtype=np.float32)

    def push(self, audio: np.ndarray) -> list[tuple[float, tuple[float, ...]]]:
        """Feed any length; returns [(t_ms, features)] per completed frame."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        self._buffer = np.concatenate([self._buffer, audio])
        out = []
        while len(self._buffer) >= FRAME_SAMPLES:
            chunk = self._buffer[:FRAME_SAMPLES]
            self._buffer = self._buffer[FRAME_SAMPLES:]
            out.append((self._t_ms + FRAME_MS, self._step(chunk)))
        return out

    def _step(self, chunk: np.ndarray) -> tuple[float, ...]:
        self._t_ms += FRAME_MS
        energy = frame_energy_db(chunk)
        f0, peak = estimate_f0(np.concatenate([self._prev, chunk]))
        voiced = peak >= VOICING_THRESHOLD and energy >= ENERGY_FLOOR_DB
        self._prev = chunk
        self._frames.append(Frame(self._t_ms, energy, f0 if voiced else 0.0, voiced))
        return self.features()

    def features(self) -> tuple[float, ...]:
        fr = list(self._frames)
        if not fr:
            return tuple(0.0 for _ in FEATURE_NAMES)
        t_now = fr[-1].t_ms
        energies = np.array([f.energy_db for f in fr])
        voiced = [f for f in fr if f.voiced]

        # --- energy ------------------------------------------------------
        energy_db = fr[-1].energy_db
        energy_drop = float(energies.max() - energy_db)
        recent = [f for f in fr if t_now - f.t_ms <= SLOPE_MS]
        energy_slope = _slope([f.t_ms for f in recent], [f.energy_db for f in recent])
        energy_std = float(energies.std()) if len(fr) > 1 else 0.0

        # --- voiced runs ---------------------------------------------------
        runs: list[list[Frame]] = []
        for f in fr:
            if (
                f.voiced
                and runs
                and runs[-1]
                and runs[-1][-1].t_ms == f.t_ms - FRAME_MS
            ):
                runs[-1].append(f)
            elif f.voiced:
                runs.append([f])
        n_runs = len(runs)
        last_run = runs[-1] if runs else []
        last_run_ms = len(last_run) * FRAME_MS
        mean_run_ms = float(np.mean([len(r) for r in runs])) * FRAME_MS if runs else 0.0
        last_over_mean = last_run_ms / mean_run_ms if mean_run_ms > 0 else 0.0
        ms_since_voiced = (t_now - voiced[-1].t_ms) if voiced else float(WINDOW_MS)
        last_run_energy = (
            float(np.mean([f.energy_db for f in last_run]) - energies.mean())
            if last_run
            else 0.0
        )

        # --- pitch ---------------------------------------------------------
        f0s = np.array([f.f0_hz for f in voiced])
        if len(f0s):
            median = float(np.median(f0s))
            f0_last = _semitones(voiced[-1].f0_hz, median)
            st = np.array([_semitones(v, median) for v in f0s])
            f0_range = float(st.max() - st.min())
            last300 = [f for f in voiced if t_now - f.t_ms <= 300.0]
            f0_slope = _slope(
                [f.t_ms for f in last300],
                [_semitones(f.f0_hz, median) for f in last300],
            )
            last_run_fall = (
                _semitones(last_run[-1].f0_hz, last_run[0].f0_hz)
                if len(last_run) > 1
                else 0.0
            )
        else:
            f0_last = f0_range = f0_slope = last_run_fall = 0.0

        return (
            float(energy_db),
            energy_drop,
            energy_slope,
            float(f0_last),
            f0_slope,
            f0_range,
            len(voiced) / len(fr),
            float(ms_since_voiced),
            float(last_run_ms),
            float(last_over_mean),
            float(n_runs),
            last_run_energy,
            float(last_run_fall),
            energy_std,
        )


def _slope(ts: list[float], ys: list[float]) -> float:
    """Least-squares slope in units per second; 0 with fewer than 3 points."""
    if len(ts) < 3:
        return 0.0
    t = np.asarray(ts, dtype=np.float64) / 1000.0
    y = np.asarray(ys, dtype=np.float64)
    t = t - t.mean()
    denom = float(np.dot(t, t))
    return float(np.dot(t, y - y.mean()) / denom) if denom > 0 else 0.0
