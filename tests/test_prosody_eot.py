"""The pause classifier: gate, feature contract, and the arithmetic."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.audio.prosody import FEATURE_NAMES, N_FEATURES
from src.eot.base import EOTDetector, Update
from src.eot.prosody_eot import ProsodyEOT, logit


def write_model(tmp_path, *, w, b=0.0, uses_text=False, features=None):
    n = N_FEATURES + (1 if uses_text else 0)
    m = {
        "kind": "fusion" if uses_text else "prosody",
        "features": list(
            features or (FEATURE_NAMES + (("logit_p_text",) if uses_text else ()))
        ),
        "w": list(w),
        "b": b,
        "mean": [0.0] * n,
        "std": [1.0] * n,
        "uses_text": uses_text,
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(m))
    return p


def feats(**kw):
    v = [0.0] * N_FEATURES
    for k, val in kw.items():
        v[FEATURE_NAMES.index(k)] = val
    return tuple(v)


def test_conforms_to_the_protocol(tmp_path):
    d = ProsodyEOT(write_model(tmp_path, w=[0.0] * N_FEATURES))
    assert isinstance(d, EOTDetector)
    assert d.name == "prosody_prosody+gate200"


def test_missing_model_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="rule 3"):
        ProsodyEOT(tmp_path / "nope.json")


def test_feature_order_mismatch_is_refused(tmp_path):
    bad = list(FEATURE_NAMES)
    bad[0], bad[1] = bad[1], bad[0]
    with pytest.raises(ValueError, match="does not match the tracker"):
        ProsodyEOT(write_model(tmp_path, w=[0.0] * N_FEATURES, features=bad))


def test_below_the_gate_nothing_fires(tmp_path):
    d = ProsodyEOT(write_model(tmp_path, w=[10.0] * N_FEATURES, b=10.0))
    assert d.update(Update(t_ms=0, text="x", prosody=feats(), silence_ms=199.0)) == 0.0


def test_missing_prosody_at_the_gate_is_an_error_not_a_guess(tmp_path):
    d = ProsodyEOT(write_model(tmp_path, w=[0.0] * N_FEATURES))
    with pytest.raises(ValueError, match="needs Update.prosody"):
        d.update(Update(t_ms=0, text="x", silence_ms=250.0))


def test_a_single_weight_drives_the_probability(tmp_path):
    w = [0.0] * N_FEATURES
    w[FEATURE_NAMES.index("last_run_f0_fall_st")] = -1.0  # a fall raises P
    d = ProsodyEOT(write_model(tmp_path, w=w))
    falling = d.update(
        Update(0, "x", prosody=feats(last_run_f0_fall_st=-4.0), silence_ms=250)
    )
    rising = d.update(
        Update(0, "x", prosody=feats(last_run_f0_fall_st=4.0), silence_ms=250)
    )
    assert falling > 0.9 > 0.1 > rising
    assert falling == pytest.approx(1 / (1 + np.exp(-4.0)))


def test_logit_is_clipped_at_the_extremes():
    assert np.isfinite(logit(0.0)) and np.isfinite(logit(1.0))
    assert logit(0.5) == pytest.approx(0.0)
