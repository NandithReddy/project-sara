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
from src.audio.vad import SileroVAD, VadFrame
from src.eot.base import EOTDetector, Update

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIRE_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class TurnResult:
    turn_id: str
    stratum: str
    fired_at_ms: float | None
    true_end_ms: float
    cutoff: bool
    added_latency_ms: float | None
    false_hold: bool
    n_updates: int


def _percentile(xs: list[float], p: float) -> float | None:
    return float(np.percentile(xs, p)) if xs else None


def vad_frames(turn: EvalTurn, vad: SileroVAD) -> list[VadFrame]:
    """Deterministic silence track for one frozen turn."""
    audio, sr = sf.read(turn.audio_path, dtype="float32")
    vad.reset()
    return vad.push(np.asarray(audio, dtype=np.float32).reshape(-1))


def replay(
    turn: EvalTurn, frames: list[VadFrame], words: list[dict]
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
    for f in frames:
        if f.t_ms < turn.turn_start_ms:
            continue
        text = " ".join(w["t"] for w in words if w["end_ms"] <= f.t_ms)
        within_turn_ms = f.t_ms - turn.turn_start_ms
        yield Update(
            t_ms=f.t_ms,
            text=text,
            silence_ms=min(f.silence_ms, max(0.0, within_turn_ms)),
        )


def run_turn(
    detector: EOTDetector,
    turn: EvalTurn,
    frames: list[VadFrame],
    words: list[dict],
    horizon_ms: float,
    threshold: float = FIRE_THRESHOLD,
) -> tuple[TurnResult, list[float]]:
    detector.reset()
    deadline = turn.true_end_ms + horizon_ms
    fired_at, n, latencies = None, 0, []

    for u in replay(turn, frames, words):
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

    def strata_block() -> dict:
        out = {}
        for s in sorted({r.stratum for r in results}):
            rs = [r for r in results if r.stratum == s]
            a = [r.added_latency_ms for r in rs if r.added_latency_ms is not None]
            out[s] = {
                "n": len(rs),
                "cutoff_rate": sum(r.cutoff for r in rs) / len(rs),
                "false_hold_rate": sum(r.false_hold for r in rs) / len(rs),
                "added_latency_p50": _percentile(a, 50),
            }
        return out

    return {
        "name": name,
        "n_turns": n,
        "horizon_ms": horizon_ms,
        "cutoff_rate": sum(r.cutoff for r in results) / n,
        "false_hold_rate": sum(r.false_hold for r in results) / n,
        "added_latency_ms": {
            "n": len(added),
            "p50": _percentile(added, 50),
            "p95": _percentile(added, 95),
            "p99": _percentile(added, 99),
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


def evaluate(detector: EOTDetector, threshold: float = FIRE_THRESHOLD) -> dict:
    """Run one detector over the whole frozen eval set."""
    turns = load_eval_set()
    spec = json.loads((EVAL_DIR / "eval_set.json").read_text())
    words_by_id = {t["turn_id"]: t["words"] for t in spec["turns"]}
    horizon_ms = json.loads(MANIFEST.read_text())["horizon_ms"]

    vad = SileroVAD()
    results, latencies = [], []
    for turn in turns:
        frames = vad_frames(turn, vad)
        r, lat = run_turn(
            detector, turn, frames, words_by_id[turn.turn_id], horizon_ms, threshold
        )
        results.append(r)
        latencies.extend(lat)

    summary = summarise(detector.name, results, latencies, horizon_ms)
    summary["threshold"] = threshold
    summary["turns"] = [asdict(r) for r in results]
    return summary


def write_results(summary: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{summary['name']}.json"
    path.write_text(json.dumps(summary, indent=2))
    return path
