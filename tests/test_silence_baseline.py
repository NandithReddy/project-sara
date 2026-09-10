"""The fixed silence timeout, and the failure it produces.

The last two tests encode section 1's motivating example as executable
assertions: a 500ms timer cuts the user off mid-thought, and a 1000ms timer
survives the pause only by paying a full second of dead air. That tradeoff is
the thing this project exists to measure.
"""

from __future__ import annotations

import pytest

from src.baselines.silence import SECTION_3_TIMEOUTS_MS, FixedSilenceTimeout
from src.eot.base import EOTDetector, Update

# "my order number is... umm... 4471" -- (text, start_ms, end_ms) on the audio
# clock. The 700ms gap after "is" is the hesitation that breaks silence timers.
DISFLUENT_TURN = [
    ("my order number is", 0.0, 1500.0),
    ("umm", 2200.0, 2500.0),
    ("4471", 2900.0, 3400.0),
]
TRUE_TURN_END_MS = 3400.0


def make_updates(segments, end_ms, step_ms=50.0):
    """Build an incremental stream from speech segments, sampled every step_ms."""
    updates = []
    t = 0.0
    while t <= end_ms:
        inside = any(start <= t < end for _, start, end in segments)
        ended = [end for _, _, end in segments if end <= t]
        silence = 0.0 if inside else (t - max(ended) if ended else t)
        text = " ".join(w for w, _, end in segments if end <= t)
        updates.append(Update(t_ms=t, text=text, silence_ms=silence))
        t += step_ms
    return updates


def first_fire_ms(detector, updates, threshold=0.5):
    """Drive a detector over a stream; return audio time of first fire, or None."""
    detector.reset()
    for u in updates:
        if detector.update(u) >= threshold:
            return u.t_ms
    return None


def test_conforms_to_eot_detector_protocol():
    assert isinstance(FixedSilenceTimeout(500), EOTDetector)


def test_fires_at_exactly_the_timeout():
    d = FixedSilenceTimeout(500)
    assert d.update(Update(t_ms=500, text="", silence_ms=500.0)) == 1.0


def test_does_not_fire_just_below_the_timeout():
    d = FixedSilenceTimeout(500)
    assert d.update(Update(t_ms=499, text="", silence_ms=499.9)) == 0.0


def test_unfires_when_speech_resumes():
    """Silence resetting to zero must retract the decision, not latch it."""
    d = FixedSilenceTimeout(500)
    assert d.update(Update(t_ms=600, text="a", silence_ms=600.0)) == 1.0
    assert d.update(Update(t_ms=650, text="a b", silence_ms=0.0)) == 0.0


@pytest.mark.parametrize("bad", [0, -1, -500.0])
def test_rejects_nonpositive_timeout(bad):
    with pytest.raises(ValueError, match="must be positive"):
        FixedSilenceTimeout(bad)


def test_name_identifies_the_operating_point():
    assert FixedSilenceTimeout(500).name == "fixed_silence_500ms"
    assert FixedSilenceTimeout(1000).name == "fixed_silence_1000ms"


def test_section_3_requires_two_operating_points():
    assert SECTION_3_TIMEOUTS_MS == (500.0, 1000.0)


def test_500ms_timer_cuts_off_a_midthought_pause():
    """Section 1's premature cutoff, measured.

    The 700ms hesitation after "my order number is" exceeds the 500ms timer, so
    it fires at 2000ms -- 1400ms before the turn actually ends, and before the
    user has said the digits that were the entire point of the sentence.
    """
    updates = make_updates(DISFLUENT_TURN, end_ms=5000.0)
    fired_at = first_fire_ms(FixedSilenceTimeout(500), updates)

    assert fired_at == 2000.0
    assert fired_at < TRUE_TURN_END_MS, "expected a premature cutoff"

    heard = next(u.text for u in updates if u.t_ms == fired_at)
    assert "4471" not in heard, "the order number was cut off"


def test_1000ms_timer_survives_the_pause_but_pays_for_it():
    """The other half of the tradeoff: no cutoff, at the cost of dead air."""
    updates = make_updates(DISFLUENT_TURN, end_ms=5000.0)
    fired_at = first_fire_ms(FixedSilenceTimeout(1000), updates)

    assert fired_at == 4400.0
    assert fired_at > TRUE_TURN_END_MS, "expected no cutoff"
    assert fired_at - TRUE_TURN_END_MS == 1000.0, "a full second of dead air"
