"""The training set: disjointness is enforced, labels are what they claim."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from ami import Turn, render  # noqa: E402
from build_train_set import (  # noqa: E402
    EVAL_SET,
    TRAIN_DIR,
    assert_disjoint,
    build_examples,
    label_collisions,
)

needs_train = pytest.mark.skipif(
    not (TRAIN_DIR / "STATS.json").exists(), reason="train set not built"
)


def turn(words, **kw) -> Turn:
    base = dict(
        turn_id="T.A.0.00",
        meeting="T",
        speaker="A",
        channel=0,
        global_name="ZZZ999",
        start_s=0.0,
        end_s=1.0,
        n_words=len(words),
        disfluent=False,
        text=" ".join(w[0] for w in words),
        words=tuple(words),
        next_speaker_gap_ms=100.0,
        own_resume_gap_ms=-1.0,
        stratum="ordinary",
    )
    base.update(kw)
    return Turn(**base)


W = (
    ("my", 0.0, 0.1, ""),
    ("order", 0.1, 0.2, ""),
    ("is", 0.2, 0.3, ""),
    ("here", 0.3, 0.4, "."),
)


def test_render_with_and_without_punctuation():
    assert render(W, 4, punctuated=False) == "my order is here"
    assert render(W, 4, punctuated=True) == "my order is here."
    assert render(W, 2, punctuated=True) == "my order"


def test_every_prefix_becomes_an_example_and_only_the_full_turn_is_positive():
    ex = build_examples([turn(W)])
    assert [e["k"] for e in ex] == [1, 2, 3, 4]
    assert [e["label"] for e in ex] == [0, 0, 0, 1]
    assert ex[-1]["text"] == "my order is here"


def test_a_one_word_turn_yields_exactly_one_positive_and_no_negatives():
    ex = build_examples([turn((("Yeah", 0.0, 0.3, "."),))])
    assert len(ex) == 1 and ex[0]["label"] == 1


def test_disjointness_refuses_a_shared_speaker():
    """Rule 1. Uses a real eval speaker id so the check is not hypothetical."""
    spec = json.loads(EVAL_SET.read_text())
    eval_speaker = spec["turns"][0]["global_name"]
    with pytest.raises(SystemExit, match="overlaps the frozen eval set"):
        assert_disjoint([turn(W, global_name=eval_speaker)])


def test_disjointness_refuses_a_shared_meeting():
    spec = json.loads(EVAL_SET.read_text())
    eval_meeting = spec["turns"][0]["meeting"]
    with pytest.raises(SystemExit, match="overlaps the frozen eval set"):
        assert_disjoint([turn(W, meeting=eval_meeting)])


def test_disjointness_passes_for_a_novel_speaker_and_meeting():
    n_m, n_s = assert_disjoint([turn(W)])
    assert n_m > 0 and n_s > 0


def test_label_collisions_count_unavoidable_errors():
    ex = [
        {"text": "Yeah", "label": 1},
        {"text": "Yeah", "label": 1},
        {"text": "Yeah", "label": 0},
        {"text": "So", "label": 0},
    ]
    c = label_collisions(ex, "text")
    assert c["strings_with_both_labels"] == 1
    assert c["unavoidable_errors"] == 1, "majority is complete; one incomplete lost"
    assert c["worst"][0]["text"] == "Yeah"


@needs_train
def test_built_set_is_disjoint_from_eval_on_disk():
    spec = json.loads(EVAL_SET.read_text())
    eval_speakers = {t["global_name"] for t in spec["turns"]}
    eval_meetings = {t["meeting"] for t in spec["turns"]}
    with (TRAIN_DIR / "turns.jsonl").open() as f:
        for line in f:
            t = json.loads(line)
            assert t["global_name"] not in eval_speakers, t["turn_id"]
            assert t["meeting"] not in eval_meetings, t["turn_id"]


@needs_train
def test_stats_match_the_files():
    stats = json.loads((TRAIN_DIR / "STATS.json").read_text())
    n_turns = sum(1 for _ in (TRAIN_DIR / "turns.jsonl").open())
    n_ex = sum(1 for _ in (TRAIN_DIR / "examples.jsonl").open())
    assert stats["n_turns"] == n_turns
    assert stats["n_examples"] == n_ex
    assert stats["n_complete"] == n_turns, "exactly one positive per turn"
