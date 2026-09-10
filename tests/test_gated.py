"""SilenceGated: the wrapped detector decides, silence confirms."""

from __future__ import annotations

import pytest

from src.eot.base import EOTDetector, Update
from src.eot.gated import DEFAULT_GATE_MS, SilenceGated


class Always:
    """A detector that always reports the same probability."""

    name = "always"

    def __init__(self, p: float):
        self.p, self.resets, self.calls = p, 0, 0

    def reset(self):
        self.resets += 1

    def update(self, u):
        self.calls += 1
        return self.p


def test_conforms_to_the_protocol_and_names_the_gate():
    g = SilenceGated(Always(0.9), 200)
    assert isinstance(g, EOTDetector)
    assert g.name == "always+gate200"


def test_rejects_a_negative_gate():
    with pytest.raises(ValueError, match=">= 0"):
        SilenceGated(Always(0.9), -1)


def test_below_the_gate_nothing_fires_whatever_the_text_says():
    inner = Always(1.0)
    g = SilenceGated(inner, 200)
    assert g.update(Update(t_ms=0, text="Okay.", silence_ms=0.0)) == 0.0
    assert g.update(Update(t_ms=0, text="Okay.", silence_ms=199.9)) == 0.0
    assert inner.calls == 0, "the model is not even consulted below the gate"


def test_at_or_above_the_gate_the_inner_decision_passes_through():
    g = SilenceGated(Always(0.73), 200)
    assert g.update(Update(t_ms=0, text="Okay.", silence_ms=200.0)) == 0.73
    assert g.update(Update(t_ms=0, text="Okay.", silence_ms=900.0)) == 0.73


def test_the_gate_is_not_a_timer():
    """Silence alone never fires: the text still has to say complete."""
    g = SilenceGated(Always(0.0), 200)
    assert g.update(Update(t_ms=0, text="my order number is", silence_ms=5000)) == 0.0


def test_reset_reaches_the_inner_detector():
    inner = Always(0.5)
    SilenceGated(inner).reset()
    assert inner.resets == 1


def test_default_gate_clears_the_boundary_tolerance():
    assert DEFAULT_GATE_MS > 150.0
