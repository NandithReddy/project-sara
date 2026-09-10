"""The eval set is frozen and sacred (ENGINEERING.md rule 1), and the eval path
never touches an STT (section 10).

These tests exist to fail loudly. If the freeze test goes red, either someone
regenerated the eval set -- invalidating every number ever measured against it
-- or a file is corrupt. Neither is fixed by editing this file (rule 4).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from eval.dataset import (
    MANIFEST,
    TRAINING_ENV,
    EvalSetViolation,
    load_eval_set,
    verify_manifest,
)

REPO = Path(__file__).resolve().parents[1]
EVAL_PKG = REPO / "eval"


def test_manifest_exists():
    assert MANIFEST.exists(), "eval set is not frozen; run scripts/freeze_eval_set.py"


def test_every_frozen_file_still_matches_its_hash():
    """The freeze. If this fails, the eval set moved under us."""
    assert verify_manifest() == []


def test_manifest_describes_the_set_we_designed():
    m = json.loads(MANIFEST.read_text())
    assert m["split"] == "development", "eval must come from AMI's held-out split"
    assert m["n_turns"] == 198
    assert m["strata"] == {"disfluent": 66, "ordinary": 66, "short": 66}, (
        "strata must stay balanced: section 3 requires disfluent and "
        "short-answer conditions to be represented"
    )


def test_reading_the_eval_set_during_training_fails_loudly():
    """Rule 1 as an executable assertion, not a paragraph of prose."""
    os.environ[TRAINING_ENV] = "1"
    try:
        with pytest.raises(EvalSetViolation, match="frozen, held-out"):
            load_eval_set()
    finally:
        del os.environ[TRAINING_ENV]


def test_eval_package_never_imports_an_stt():
    """Section 10: importing src/stt/ from eval/ is a bug, like rule 1."""
    offenders = []
    for path in EVAL_PKG.rglob("*.py"):
        text = path.read_text()
        for marker in ("src.stt", "from src import stt", "parakeet", "whisper"):
            if marker in text and "never imports" not in text.split(marker)[0][-120:]:
                offenders.append(f"{path.relative_to(REPO)} mentions {marker!r}")
    assert not offenders, f"eval path must not reach an STT: {offenders}"


def test_every_turn_has_audio_and_a_boundary():
    turns = load_eval_set()
    assert len(turns) == 198
    missing = [t.turn_id for t in turns if not t.audio_path.exists()]
    assert not missing, f"missing audio for {len(missing)} turns"
    for t in turns:
        assert 0 < t.true_end_ms < t.seg_duration_ms, t.turn_id
        assert t.boundary_source in {"audio", "annotation"}


def test_turns_have_the_tail_the_horizon_requires():
    """Every turn must allow observation for the full horizon after it ends."""
    horizon = json.loads(MANIFEST.read_text())["horizon_ms"]
    turns = load_eval_set()
    short = [t.turn_id for t in turns if t.seg_duration_ms - t.true_end_ms < horizon]
    assert not short, f"{len(short)} turns lack {horizon}ms of tail: {short[:3]}"
