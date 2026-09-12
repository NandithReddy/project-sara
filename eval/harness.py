"""Run an EOT detector over the frozen eval set and emit metrics.

Replays each frozen turn as a simulated incremental stream: gold words are
released at their aligned timings, imitating what a streaming recogniser would
have emitted, and `silence_ms` comes from a VAD pass over the frozen audio.
Never calls an STT (section 10), so the same run twice produces byte-identical
numbers and a regression is always a change in our model.

Reports every metric in section 3: cutoff_rate, added_latency_ms at p50/p95/p99,
false_hold_rate, plus CPU cost per update against the <20ms budget in section 5.

Reproducibility, precisely. Every decision and every derived metric is
byte-identical across runs -- verified by re-running and diffing. The one
exception is `update_latency_ms`, which is a wall-clock measurement of the
machine rather than a property of the model, and cannot be deterministic. Diff
results excluding that block when checking for a regression.

Silence comes from audio rather than from gold word timings because AMI's
alignment absorbs intra-utterance pauses into word durations -- see
scripts/build_eval_set.py. Using the timings would show no silence during a
hesitation, so nothing would ever fire early and cutoff_rate would be ~0 for
every system.

Text arrives as a TIMELINE: (t_ms, text) events on the audio clock, cumulative.
Gold builds one from word end times. The Phase 6 real-ASR condition builds one
from a cached recogniser run (results/asr/), so the same replay serves both
and the difference between them is exactly the optimism gap.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.dataset import EVAL_DIR, MANIFEST, EvalTurn, load_eval_set
from src.audio.prosody import ProsodyTracker
from src.audio.vad import SileroVAD, VadFrame
from src.eot.base import EOTDetector, Update

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIRE_THRESHOLD = 0.5

BOUNDARY_TOLERANCE_MS = 150.0
"""How close to the true end a fire may land and still not count as a cutoff.

Not a fudge factor and not tuned: it is the resolution of the instrument. The
audio-grounded boundary agrees with the word alignment only to about +/-150ms
(p10 -144ms, p90 +145ms over 198 turns), and VAD frames are 32ms. Calling a
96ms-early fire a "premature cutoff" asserts precision the measurement does not
have.

Both rates are reported. `cutoff_rate` is strict, `cutoff_rate_at_tolerance`
allows this margin, and the gap between them is the population sitting inside
the boundary's own uncertainty. Reporting only one would be choosing a
flattering number.
"""


@dataclass(frozen=True, slots=True)
class TurnResult:
    turn_id: str
    stratum: str
    fired_at_ms: float | None
    true_end_ms: float
    cutoff: bool
    early_by_ms: float | None
    """How far before the true end it fired. Negative means it fired after."""
    added_latency_ms: float | None
    false_hold: bool
    n_updates: int


def _percentile(xs: list[float], p: float) -> float | None:
    return float(np.percentile(xs, p)) if xs else None


ProsodyFrames = list[tuple[float, tuple[float, ...]]]


@dataclass(frozen=True)
class ProsodyCache:
    """Prosodic features for every turn under one audio transform, computed
    once. Must be built with the SAME transform as the silence cache it is
    replayed with -- the conditions runner pairs them."""

    label: str
    feats: dict[str, ProsodyFrames]


def prosody_frames(turn: EvalTurn, transform=None) -> ProsodyFrames:
    audio, sr = sf.read(turn.audio_path, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if transform is not None:
        audio = transform(audio, turn)
    return ProsodyTracker().push(audio)


def precompute_prosody(transform=None, label: str = "prosody") -> ProsodyCache:
    return ProsodyCache(
        label=label,
        feats={t.turn_id: prosody_frames(t, transform) for t in load_eval_set()},
    )


def caller_channel(
    audio: np.ndarray, turn: EvalTurn, sr: int = 16_000, ramp_ms: float = 5.0
) -> np.ndarray:
    """What a phone pipeline hears after the caller stops: nothing.

    AMI headset channels carry the NEXT speaker at low level once the floor
    changes hands, and a full recogniser transcribes it. In the first real-ASR
    pass the transcript kept growing after the true end on 139 of 198 turns
    (median +4 words), and 79 of the punctuation heuristic's fires came from
    those words -- someone else's. A VAD at threshold mostly ignores the bleed;
    an ASR does not. Zeroing the channel from true_end + the boundary tolerance
    removes a party that a caller's channel never carried in the first place.
    Cutoffs were valid either way (pre-end text is the speaker's own); holds and
    latency were not.
    """
    x = np.array(audio, dtype=np.float32, copy=True)
    cut = int((turn.true_end_ms + BOUNDARY_TOLERANCE_MS) / 1000.0 * sr)
    if cut >= len(x):
        return x
    ramp = int(sr * ramp_ms / 1000.0)
    end = min(cut + ramp, len(x))
    x[cut:end] *= np.linspace(1.0, 0.0, end - cut, dtype=np.float32)
    x[end:] = 0.0
    return x


def vad_frames(turn: EvalTurn, vad, transform=None) -> list[VadFrame]:
    """Deterministic silence track for one frozen turn.

    `transform(audio, turn)` reshapes the audio first: the telephony band, the
    caller channel, or both (Phase 6). The boundary stays frozen: the true end
    of a turn is a property of the speech, not of the channel it came down.

    `vad` is any frame-synchronous detector with reset()/push() -- SileroVAD for
    baseline #2, EnergyVAD for baseline #1. The silence source is a property of
    the pipeline, not of the EOT rule, which is exactly what distinguishes those
    two baselines: same timer, different notion of silence.
    """
    audio, sr = sf.read(turn.audio_path, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if transform is not None:
        audio = transform(audio, turn)
    vad.reset()
    return vad.push(audio)


Timeline = list[tuple[float, str]]


def gold_timeline(words: list[dict]) -> Timeline:
    """Perfect streaming STT: each word appears, punctuated, exactly at its end."""
    out, parts = [], []
    for w in words:
        parts.append(w["t"] + (w.get("punc_after") or ""))
        out.append((float(w["end_ms"]), " ".join(parts)))
    return out


def asr_timeline(partials: list[dict], include_compute: bool = True) -> Timeline:
    """A real recogniser's partials, available when the caller would have them:
    audio fed so far plus the time the recogniser took -- or the content gap
    alone, with include_compute=False."""
    out = []
    for p in partials:
        t = float(p["audio_ms"]) + (float(p["compute_ms"]) if include_compute else 0.0)
        out.append((t, p["text"]))
    out.sort(key=lambda e: e[0])
    return out


def replay(
    turn: EvalTurn,
    frames: list[VadFrame],
    timeline: Timeline,
    prosody: ProsodyFrames | None = None,
) -> Iterator[Update]:
    """Yield one Update per VAD frame, on the audio clock.

    Evaluation starts at the turn's first word, not at frame zero, and
    `silence_ms` is clamped to time elapsed SINCE that point.

    Both halves matter. Each segment carries 500ms of pre-roll, and the VAD
    accumulates it as silence, so at the first evaluated frame silence_ms is
    already ~512ms. Fed that, a 500ms timer fires on its first update -- before
    the speaker has been heard at all -- and scores a premature cutoff on
    essentially every turn. Measured: 97.0% cutoff before this clamp, on 39 of
    40 sampled turns arriving with >=500ms of phantom silence.

    Silence before a turn begins is not silence within it. This is the same
    mistake the live loop made in Phase 1 ("a turn that never started cannot
    end"), reappearing in the eval path.
    """
    i, text, pj = 0, "", 0
    for f in frames:
        while i < len(timeline) and timeline[i][0] <= f.t_ms:
            text = timeline[i][1]
            i += 1
        feats: tuple[float, ...] = ()
        if prosody is not None:
            # Same 32ms grid from the same audio: align by time, never by index.
            while pj < len(prosody) and prosody[pj][0] < f.t_ms - 1e-6:
                pj += 1
            if pj < len(prosody) and abs(prosody[pj][0] - f.t_ms) < 1e-6:
                feats = prosody[pj][1]
        if f.t_ms < turn.turn_start_ms:
            continue
        within_turn_ms = f.t_ms - turn.turn_start_ms
        yield Update(
            t_ms=f.t_ms,
            text=text,
            prosody=feats,
            silence_ms=min(f.silence_ms, max(0.0, within_turn_ms)),
        )


def run_turn(
    detector: EOTDetector,
    turn: EvalTurn,
    frames: list[VadFrame],
    timeline: Timeline,
    horizon_ms: float,
    threshold: float = FIRE_THRESHOLD,
    prosody: ProsodyFrames | None = None,
) -> tuple[TurnResult, list[float]]:
    detector.reset()
    deadline = turn.true_end_ms + horizon_ms
    fired_at, n, latencies = None, 0, []

    for u in replay(turn, frames, timeline, prosody):
        if u.t_ms > deadline:
            break
        t0 = time.perf_counter()
        p = detector.update(u)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        n += 1
        if p >= threshold:
            fired_at = u.t_ms
            break

    cutoff = fired_at is not None and fired_at < turn.true_end_ms
    return (
        TurnResult(
            early_by_ms=(None if fired_at is None else turn.true_end_ms - fired_at),
            turn_id=turn.turn_id,
            stratum=turn.stratum,
            fired_at_ms=fired_at,
            true_end_ms=turn.true_end_ms,
            cutoff=cutoff,
            added_latency_ms=(
                None if fired_at is None or cutoff else fired_at - turn.true_end_ms
            ),
            false_hold=fired_at is None,
            n_updates=n,
        ),
        latencies,
    )


def summarise(
    name: str, results: list[TurnResult], latencies: list[float], horizon_ms: float
) -> dict:
    n = len(results)
    added = [r.added_latency_ms for r in results if r.added_latency_ms is not None]
    # What the caller actually waits. A false hold is not "no latency" -- in a
    # real pipeline the fallback timer fires at the horizon, so the caller waits
    # that long. Excluding holds flatters any system that holds a lot: a timer
    # holding 46.5% of turns would otherwise plot as fast and safe.
    with_fallback = [
        horizon_ms if r.false_hold else r.added_latency_ms
        for r in results
        if not r.cutoff
    ]

    def cutoffs(tol: float) -> int:
        return sum(
            1 for r in results if r.early_by_ms is not None and r.early_by_ms > tol
        )

    def strata_block() -> dict:
        out = {}
        for s in sorted({r.stratum for r in results}):
            rs = [r for r in results if r.stratum == s]
            a = [r.added_latency_ms for r in rs if r.added_latency_ms is not None]
            out[s] = {
                "n": len(rs),
                "cutoff_rate": sum(r.cutoff for r in rs) / len(rs),
                "cutoff_rate_at_tolerance": sum(
                    1
                    for r in rs
                    if r.early_by_ms is not None
                    and r.early_by_ms > BOUNDARY_TOLERANCE_MS
                )
                / len(rs),
                "false_hold_rate": sum(r.false_hold for r in rs) / len(rs),
                "added_latency_p50": _percentile(a, 50),
            }
        return out

    return {
        "name": name,
        "n_turns": n,
        "horizon_ms": horizon_ms,
        "cutoff_rate": cutoffs(0.0) / n,
        "cutoff_rate_at_tolerance": cutoffs(BOUNDARY_TOLERANCE_MS) / n,
        "boundary_tolerance_ms": BOUNDARY_TOLERANCE_MS,
        "false_hold_rate": sum(r.false_hold for r in results) / n,
        "added_latency_ms": {
            "n": len(added),
            "p50": _percentile(added, 50),
            "p95": _percentile(added, 95),
            "p99": _percentile(added, 99),
        },
        "added_latency_with_fallback_ms": {
            "n": len(with_fallback),
            "p50": _percentile(with_fallback, 50),
            "p95": _percentile(with_fallback, 95),
            "p99": _percentile(with_fallback, 99),
            "note": "false holds counted at horizon_ms, the fallback timeout",
        },
        "update_latency_ms": {
            "n": len(latencies),
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
            "budget_ms": 20.0,
            "within_budget": bool(_percentile(latencies, 99) or 0) < 20.0,
        },
        "by_stratum": strata_block(),
    }


@dataclass(frozen=True)
class FrameCache:
    """Silence tracks for every turn under one source, computed once.

    The VAD pass is the only slow part of an eval run, and it does not depend
    on the detector. A sweep re-runs detectors dozens of times over the same
    audio, so the frames are computed once and shared.
    """

    source_name: str
    frames: dict[str, list[VadFrame]]


def precompute(silence=None, transform=None, label: str | None = None) -> FrameCache:
    vad = SileroVAD() if silence is None else silence
    return FrameCache(
        source_name=label or getattr(vad, "name", "silero"),
        frames={t.turn_id: vad_frames(t, vad, transform) for t in load_eval_set()},
    )


def evaluate(
    detector: EOTDetector,
    silence=None,
    name: str | None = None,
    threshold: float = FIRE_THRESHOLD,
    cache: FrameCache | None = None,
    transcripts: dict[str, Timeline] | None = None,
    transcript_label: str = "gold",
    prosody_cache: ProsodyCache | None = None,
) -> dict:
    """Run one detector over the whole frozen eval set.

    `silence` selects the silence source (default Silero). `name` overrides the
    result name; without it the source is appended, since the same timer over a
    different silence source is a different baseline. Pass `cache` from
    `precompute()` to skip the VAD pass. `transcripts` maps turn_id to a text
    Timeline; None replays gold.
    """
    turns = load_eval_set()
    spec = json.loads((EVAL_DIR / "eval_set.json").read_text())
    if transcripts is None:
        transcripts = {t["turn_id"]: gold_timeline(t["words"]) for t in spec["turns"]}
    horizon_ms = json.loads(MANIFEST.read_text())["horizon_ms"]

    if cache is None:
        cache = precompute(silence)
    source_name = cache.source_name
    results, latencies = [], []
    for turn in turns:
        frames = cache.frames[turn.turn_id]
        r, lat = run_turn(
            detector,
            turn,
            frames,
            transcripts[turn.turn_id],
            horizon_ms,
            threshold,
            prosody_cache.feats[turn.turn_id] if prosody_cache else None,
        )
        results.append(r)
        latencies.extend(lat)

    run_name = name or f"{detector.name}__{source_name}"
    summary = summarise(run_name, results, latencies, horizon_ms)
    summary["detector"] = detector.name
    summary["silence_source"] = source_name
    summary["transcripts"] = transcript_label
    summary["prosody"] = prosody_cache.label if prosody_cache else None
    summary["threshold"] = threshold
    summary["turns"] = [asdict(r) for r in results]
    return summary


def write_results(summary: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{summary['name']}.json"
    path.write_text(json.dumps(summary, indent=2))
    return path
