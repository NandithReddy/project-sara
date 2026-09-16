"""GUIDE.md quotes a few rounded figures in plain words. They come from the
same file the README's numbers do, so they are held to it: if a result
moves, this test says which sentence of the guide is now wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GUIDE = REPO / "GUIDE.md"
NUMBERS = REPO / "results" / "readme_numbers.json"

needs = pytest.mark.skipif(not NUMBERS.exists(), reason="README not generated")


@needs
def test_guide_figures_match_the_results():
    n = json.loads(NUMBERS.read_text())
    text = " ".join(GUIDE.read_text().split())  # the guide wraps at 76 columns
    # "about 17% of turns on clean audio and 24% on phone audio"
    assert f"about {n['timer500_cut']:.0f}% of turns on clean audio" in text
    assert f"and {n['tel_timer500_cut']:.0f}% on phone audio" in text
    # "answers three times faster (256 ms instead of 800 ms)"
    assert f"({n['pg_p50']:.0f} ms instead of {n['timer800_p50']:.0f} ms)" in text
    assert n["timer800_p50"] / n["pg_p50"] == pytest.approx(3.0, abs=0.35), (
        "three times"
    )
    # "push the interruptions up to 20%"
    assert f"up to {n['tel_pg_cut']:.0f}%" in text
    # "interrupts less than the timer on the phone track (10% vs 17%)"
    assert f"({n['tel_v11g03_cut']:.0f}% vs {n['tel_timer800_cut']:.0f}%)" in text
    assert 30 <= n["tel_v11g03_hold"] < 40, "'a third of callers'"
    # Flux: "under 10%", "about 12%", "about half a second"
    assert n["tel_flux_cut"] < 10 and "interrupts under 10% of turns" in text
    assert f"never answers about {n['tel_flux_hold']:.0f}%" in text
    assert 400 <= n["tel_flux_p50"] <= 700, "'about half a second'"
    # "about one word in five is wrong"
    assert 15 <= n["wer_pct"] <= 25
    # the set and the model, as described
    assert f"{n['n_turns']} short recordings" in text
    assert 5_700 <= n["train_turns"] < 5_800, "'5,700 other meeting turns'"
    assert round(n["v11_params_m"]) == 11, "'11 million numbers'"


def test_guide_is_linked_from_the_readme_and_names_real_files():
    readme = (REPO / "README.md").read_text()
    assert "GUIDE.md" in readme
    text = GUIDE.read_text()
    for rel in (
        "scripts/run_baselines.py",
        "scripts/live.py",
        "scripts/deepgram_eval.py",
        "scripts/transcribe_eval.py",
        "src/baselines/punctuation.py",
        "src/eot/base.py",
        "results/tradeoff.md",
        "results/conditions.md",
        "ENGINEERING.md",
    ):
        assert rel in text and (REPO / rel).exists(), rel
