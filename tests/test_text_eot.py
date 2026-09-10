"""The deliverable's inference wrapper.

Behavioural assertions are deliberately weak -- invariants, not specific
probabilities -- so a retrain does not break the suite. The one behavioural
check is the task itself: a complete sentence must outrank its own obviously
unfinished prefix. A model that fails that is not a model.
"""

from __future__ import annotations

import pytest

from src.eot.base import EOTDetector, Update
from src.eot.model import DEFAULT_MODEL_DIR, TextEOT, strip_punctuation

needs_model = pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / "model.onnx").exists(), reason="model not trained"
)


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("hello there.", "hello there"),
        ("is that right?", "is that right"),
        ("well, um, so", "well um so"),
        ("it's a kick-off L_C_D_ ok.", "it's a kick-off L_C_D_ ok"),
        ("Yeah..", "Yeah"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_strip_punctuation_matches_the_training_rendering(raw, clean):
    assert strip_punctuation(raw) == clean


def test_missing_model_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="rule 3"):
        TextEOT(model_dir=tmp_path)


@pytest.fixture(scope="module")
def det():
    return TextEOT()


@needs_model
def test_conforms_to_the_detector_protocol(det):
    assert isinstance(det, EOTDetector)
    assert det.name.startswith("text_eot_v1")


@needs_model
def test_nothing_said_means_not_complete(det):
    det.reset()
    assert det.update(Update(t_ms=0.0, text="")) == 0.0
    assert det.update(Update(t_ms=32.0, text="...")) == 0.0


@needs_model
def test_output_is_a_probability(det):
    det.reset()
    for text in ("yes", "my order number is", "and then we", "okay so"):
        p = det.update(Update(t_ms=0.0, text=text))
        assert 0.0 <= p <= 1.0


@needs_model
def test_silence_is_ignored_text_only_by_design(det):
    det.reset()
    a = det.update(Update(t_ms=0.0, text="is that right", silence_ms=0.0))
    det.reset()
    b = det.update(Update(t_ms=0.0, text="is that right", silence_ms=5000.0))
    assert a == b


@needs_model
def test_identical_text_hits_the_cache_and_punctuation_does_not_miss_it(det):
    det.reset()
    n0 = det.inference_latency_ms().get("n", 0)
    det.update(Update(t_ms=0.0, text="my order number is"))
    det.update(Update(t_ms=32.0, text="my order number is"))
    det.update(Update(t_ms=64.0, text="my order number is."))  # same after strip
    assert det.inference_latency_ms()["n"] == n0 + 1


@needs_model
def test_a_complete_sentence_outranks_its_unfinished_prefix(det):
    """The task. If this fails, nothing else about the model matters."""
    det.reset()
    prefix = det.update(Update(t_ms=0.0, text="my order number is"))
    det.reset()
    full = det.update(Update(t_ms=0.0, text="my order number is four four seven one"))
    assert full > prefix


@needs_model
def test_inference_stays_inside_the_cpu_budget(det):
    det.reset()
    for i in range(50):
        det.update(Update(t_ms=float(i), text=f"word{i} " * (i % 12 + 1)))
    lat = det.inference_latency_ms()
    assert lat["n"] >= 50
    assert lat["p99"] < lat["budget_ms"], f"p99 {lat['p99']:.2f}ms over 20ms budget"
