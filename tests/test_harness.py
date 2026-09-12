"""Replay semantics and metric arithmetic.

`test_preroll_silence_is_not_counted_as_within_turn_silence` guards a bug that
produced a 97.0% cutoff rate: each segment carries 500ms of pre-roll, the VAD
accumulates it, and a 500ms timer fed that pre-roll fires on its first update,
before the speaker has been heard. Silence before a turn begins is not silence
within it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from eval.dataset import EvalTurn
from eval.harness import asr_timeline, gold_timeline, replay, run_turn, summarise
from src.audio.vad import VadFrame
from src.baselines.silence import FixedSilenceTimeout

TURN = EvalTurn(
    turn_id="T.1",
    meeting="M",
    speaker="A",
    global_name="G",
    stratum="ordinary",
    text="hello there",
    n_words=2,
    disfluent=False,
    audio_path=None,  # type: ignore[arg-type]
    seg_duration_ms=3000.0,
    turn_start_ms=500.0,
    true_end_ms=1500.0,
    annotated_end_ms=1500.0,
    boundary_source="audio",
)
WORDS = [
    {"t": "hello", "start_ms": 500.0, "end_ms": 1000.0},
    {"t": "there", "start_ms": 1000.0, "end_ms": 1500.0, "punc_after": "."},
]
GOLD = gold_timeline(WORDS)


def frames(spec):
    """spec: [(t_ms, is_speech, raw_silence_ms)]"""
    return [
        VadFrame(t_ms=t, prob=1.0 if sp else 0.0, is_speech=sp, silence_ms=sil)
        for t, sp, sil in spec
    ]


def test_replay_starts_at_the_first_word_not_at_frame_zero():
    f = frames([(100, False, 100), (400, False, 400), (600, True, 0)])
    assert [u.t_ms for u in replay(TURN, f, GOLD)] == [600]


def test_preroll_silence_is_not_counted_as_within_turn_silence():
    """The 97%-cutoff bug. Raw silence is 512ms; within-turn silence is 12ms."""
    f = frames([(512, False, 512)])
    u = next(iter(replay(TURN, f, GOLD)))
    assert u.silence_ms == pytest.approx(12.0), (
        "silence accumulated before the turn began must not count towards ending it"
    )


def test_a_500ms_timer_does_not_fire_on_its_first_update():
    f = frames([(512, False, 512), (544, False, 544)])
    r, _ = run_turn(FixedSilenceTimeout(500), TURN, f, GOLD, horizon_ms=2000.0)
    assert not r.cutoff and r.fired_at_ms is None


def test_text_is_released_at_word_timings():
    f = frames([(700, True, 0), (1100, True, 0), (1600, False, 100)])
    texts = [u.text for u in replay(TURN, f, GOLD)]
    assert texts == ["", "hello", "hello there."]


def test_firing_before_the_true_end_is_a_cutoff():
    f = frames([(600, False, 600)])
    r, _ = run_turn(FixedSilenceTimeout(100), TURN, f, GOLD, horizon_ms=2000.0)
    assert r.cutoff and r.fired_at_ms == 600 and r.added_latency_ms is None


def test_firing_after_the_true_end_records_added_latency():
    f = frames([(1900, False, 400)])
    r, _ = run_turn(FixedSilenceTimeout(300), TURN, f, GOLD, horizon_ms=2000.0)
    assert not r.cutoff and r.added_latency_ms == pytest.approx(400.0)


def test_never_firing_within_the_horizon_is_a_false_hold():
    f = frames([(t, False, t - 500) for t in range(600, 3600, 100)])
    r, _ = run_turn(FixedSilenceTimeout(99_000), TURN, f, GOLD, horizon_ms=2000.0)
    assert r.false_hold and r.fired_at_ms is None


def test_evaluation_stops_at_the_horizon():
    f = frames([(t, False, 0) for t in range(600, 6000, 100)])
    r, _ = run_turn(FixedSilenceTimeout(99_000), TURN, f, GOLD, horizon_ms=2000.0)
    last = TURN.true_end_ms + 2000.0
    assert r.n_updates == len([t for t in range(600, 6000, 100) if t <= last])


def test_the_silence_track_is_deterministic():
    """Section 10's whole claim: the same eval run twice gives the same numbers.

    The VAD pass over frozen audio is the only place nondeterminism could enter
    the decision path, so it is the thing worth pinning.
    """
    import soundfile as sf

    from eval.dataset import load_eval_set
    from eval.harness import vad_frames
    from src.audio.vad import SileroVAD

    turn = load_eval_set()[0]
    vad = SileroVAD()
    first = vad_frames(turn, vad)
    second = vad_frames(turn, vad)
    assert first == second, "VAD output differed between runs on identical audio"
    assert sf.info(turn.audio_path).samplerate == 16_000


def test_summary_reports_every_section_3_metric():
    f = frames([(1900, False, 400)])
    r, lat = run_turn(FixedSilenceTimeout(300), TURN, f, GOLD, horizon_ms=2000.0)
    other = replace(
        r, turn_id="T.2", cutoff=True, early_by_ms=400.0, added_latency_ms=None
    )
    s = summarise("x", [r, other], lat, 2000.0)
    assert s["cutoff_rate"] == pytest.approx(0.5)
    assert s["added_latency_ms"]["p50"] == pytest.approx(400.0)
    assert set(s["added_latency_ms"]) >= {"p50", "p95", "p99"}
    assert s["update_latency_ms"]["budget_ms"] == 20.0


def test_a_fire_inside_the_boundary_uncertainty_is_reported_both_ways():
    """72 of the punctuation baseline's 124 "cutoffs" were under 100ms early --
    the alignment/audio offset, not the heuristic firing mid-sentence. Both
    rates are reported so neither reading is hidden."""
    f = frames([(1400, False, 100)])  # 100ms before true_end, inside tolerance
    r, lat = run_turn(FixedSilenceTimeout(50), TURN, f, GOLD, horizon_ms=2000.0)
    s = summarise("x", [r], lat, 2000.0)
    assert r.early_by_ms == pytest.approx(100.0)
    assert s["cutoff_rate"] == pytest.approx(1.0), "strictly, it fired early"
    assert s["cutoff_rate_at_tolerance"] == pytest.approx(0.0), (
        "but 100ms is inside the boundary's own uncertainty"
    )


def test_gold_timeline_is_cumulative_and_punctuated_at_word_ends():
    assert GOLD == [(1000.0, "hello"), (1500.0, "hello there.")]


def test_asr_timeline_is_available_when_the_caller_would_have_it():
    partials = [
        {"audio_ms": 640.0, "compute_ms": 360.0, "text": ""},
        {"audio_ms": 1280.0, "compute_ms": 350.0, "text": "hello"},
        {"audio_ms": 1920.0, "compute_ms": 370.0, "text": "hello there."},
    ]
    with_compute = asr_timeline(partials)
    assert with_compute[1] == (1630.0, "hello")
    content_only = asr_timeline(partials, include_compute=False)
    assert content_only[1] == (1280.0, "hello")


def test_replay_uses_the_latest_event_at_or_before_each_frame():
    """A recogniser's partial lands between frames; the next frame sees it,
    the previous one does not, and a revision replaces earlier text."""
    tl = [(700.0, "hell"), (1100.0, "hello there"), (1600.0, "yellow there.")]
    f = frames([(600, True, 0), (700, True, 0), (1200, True, 0), (1700, False, 100)])
    assert [u.text for u in replay(TURN, f, tl)] == [
        "",
        "hell",
        "hello there",
        "yellow there.",
    ]
