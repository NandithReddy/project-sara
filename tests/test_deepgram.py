"""Deepgram client: key handling and message parsing, no network."""

from __future__ import annotations

import pytest

from src.baselines.deepgram import (
    Received,
    flux_events,
    flux_query,
    load_api_key,
    nova3_partials,
    nova_query,
)
from src.eot.base import Update
from src.eot.replayed import (
    ReplayedEOT,
    flux_confidence_series,
    flux_shipped_series,
    nova3_endpoint_series,
)


def test_missing_key_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.setattr("src.baselines.deepgram.KEY_FILE", tmp_path / "none")
    with pytest.raises(RuntimeError, match="rule 3"):
        load_api_key()


def test_env_key_wins_over_file(monkeypatch, tmp_path):
    f = tmp_path / "k"
    f.write_text("from-file\n")
    monkeypatch.setattr("src.baselines.deepgram.KEY_FILE", f)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "from-env")
    assert load_api_key() == "from-env"
    monkeypatch.delenv("DEEPGRAM_API_KEY")
    assert load_api_key() == "from-file"


def test_queries_pin_the_things_the_eval_depends_on():
    q = nova_query(16_000)
    assert q["model"] == "nova-3" and q["interim_results"] == "true"
    assert q["punctuate"] == "true" and q["encoding"] == "linear16"
    f = flux_query(16_000)
    assert f["model"] == "flux-general-en" and f["eot_threshold"] == "0.7"


def r(wall, **m):
    return Received(wall, m)


def test_nova3_text_is_cumulative_across_segments():
    msgs = [
        r(
            300,
            type="Results",
            start=0.0,
            duration=0.5,
            is_final=False,
            speech_final=False,
            channel={"alternatives": [{"transcript": "my order"}]},
        ),
        r(
            900,
            type="Results",
            start=0.0,
            duration=1.0,
            is_final=True,
            speech_final=False,
            channel={"alternatives": [{"transcript": "my order number is"}]},
        ),
        r(
            1400,
            type="Results",
            start=1.0,
            duration=0.4,
            is_final=False,
            speech_final=False,
            channel={"alternatives": [{"transcript": "four four"}]},
        ),
        r(
            2100,
            type="Results",
            start=1.0,
            duration=1.0,
            is_final=True,
            speech_final=True,
            channel={"alternatives": [{"transcript": "four four seven one."}]},
        ),
        r(2200, type="Metadata"),
    ]
    p = nova3_partials(msgs)
    assert [x["text"] for x in p] == [
        "my order",
        "my order number is",
        "my order number is four four",
        "my order number is four four seven one.",
    ]
    assert p[1]["audio_ms"] == 1000.0 and p[1]["compute_ms"] == pytest.approx(
        -100.0 + 100.0
    )  # clamped at 0
    assert p[-1]["speech_final"] is True


def test_flux_events_keep_confidence_on_every_update():
    msgs = [
        r(
            500,
            type="TurnInfo",
            event="StartOfTurn",
            turn_index=0,
            audio_window_end=0.48,
            end_of_turn_confidence=0.02,
            transcript="my",
        ),
        r(
            1500,
            type="TurnInfo",
            event="Update",
            turn_index=0,
            audio_window_end=1.44,
            end_of_turn_confidence=0.31,
            transcript="my order number is",
        ),
        r(
            2300,
            type="TurnInfo",
            event="EndOfTurn",
            turn_index=0,
            audio_window_end=2.24,
            end_of_turn_confidence=0.91,
            transcript="my order number is",
            trigger="model",
        ),
        r(2400, type="Connected"),
    ]
    ev = flux_events(msgs)
    assert [e["eot_confidence"] for e in ev] == [0.02, 0.31, 0.91]
    assert ev[-1]["trigger"] == "model" and ev[-1]["audio_ms"] == 2240.0


def test_flux_confidence_counts_only_while_a_turn_is_open():
    cache = {
        "turns": {
            "A": [
                {
                    "wall_ms": 200,
                    "audio_ms": 190,
                    "event": "Update",
                    "eot_confidence": 0.8,
                },
                {
                    "wall_ms": 500,
                    "audio_ms": 480,
                    "event": "StartOfTurn",
                    "eot_confidence": 0.1,
                },
                {
                    "wall_ms": 700,
                    "audio_ms": 690,
                    "event": "Update",
                    "eot_confidence": 0.6,
                },
                {
                    "wall_ms": 900,
                    "audio_ms": 880,
                    "event": "EndOfTurn",
                    "eot_confidence": 0.9,
                },
                {
                    "wall_ms": 1100,
                    "audio_ms": 1080,
                    "event": "Update",
                    "eot_confidence": 0.85,
                },
            ]
        }
    }
    s = flux_confidence_series(cache)["A"]
    assert s == [(500.0, 0.1), (700.0, 0.6), (900.0, 0.9)], "idle values dropped"
    raw = flux_confidence_series(cache, in_turn_only=False)["A"]
    assert len(raw) == 5
    assert (
        flux_confidence_series(
            {
                "turns": {
                    "B": [
                        {
                            "wall_ms": 300,
                            "audio_ms": 290,
                            "event": "Update",
                            "eot_confidence": 0.95,
                        },
                    ]
                }
            }
        )["B"]
        == []
    ), "no turn ever opened: nothing to act on"


def test_replayed_detector_follows_the_series_and_needs_its_turn():
    class T:
        turn_id = "A"

    cache = {
        "turns": {
            "A": [
                {
                    "wall_ms": 500,
                    "audio_ms": 480,
                    "event": "StartOfTurn",
                    "eot_confidence": 0.1,
                },
                {
                    "wall_ms": 1500,
                    "audio_ms": 1440,
                    "event": "Update",
                    "eot_confidence": 0.6,
                },
                {
                    "wall_ms": 2300,
                    "audio_ms": 2240,
                    "event": "EndOfTurn",
                    "eot_confidence": 0.9,
                },
            ]
        }
    }
    d = ReplayedEOT("flux", flux_confidence_series(cache))
    d.reset()
    d.begin(T())
    assert d.update(Update(t_ms=400, text="")) == 0.0
    assert d.update(Update(t_ms=1600, text="")) == 0.6
    assert d.update(Update(t_ms=9000, text="")) == 0.9
    shipped = ReplayedEOT("flux_shipped", flux_shipped_series(cache))
    shipped.reset()
    shipped.begin(T())
    assert shipped.update(Update(t_ms=1600, text="")) == 0.0
    assert shipped.update(Update(t_ms=2300, text="")) == 1.0

    class U:
        turn_id = "missing"

    with pytest.raises(KeyError, match="rule 3"):
        d.begin(U())


def test_nova3_endpointing_series_fires_at_speech_final():
    cache = {
        "turns": {
            "A": [
                {
                    "audio_ms": 1000,
                    "compute_ms": 200,
                    "text": "x",
                    "speech_final": False,
                },
                {
                    "audio_ms": 2000,
                    "compute_ms": 150,
                    "text": "x y.",
                    "speech_final": True,
                },
            ]
        }
    }
    s = nova3_endpoint_series(cache)["A"]
    assert s == [(2150.0, 1.0)], "events only: a fire is not undone by the next message"


def test_replayed_detector_reports_a_crossing_that_lasted_one_message():
    class T:
        turn_id = "A"

    # 0.80 and 0.78 arrive 19ms apart, inside one 32ms frame: the frame must
    # see the 0.80, or a threshold of 0.8 would never fire on this turn.
    d = ReplayedEOT("x", {"A": [(4662.0, 0.80), (4681.0, 0.78), (4900.0, 0.10)]})
    d.reset()
    d.begin(T())
    assert d.update(Update(t_ms=4640, text="")) == 0.0
    assert d.update(Update(t_ms=4704, text="")) == 0.80
    assert d.update(Update(t_ms=4736, text="")) == 0.78, "then holds the last value"
    assert d.update(Update(t_ms=4928, text="")) == 0.10
