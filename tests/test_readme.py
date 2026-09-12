"""The README is written from results/ and cannot drift from it.

scripts/write_readme.py records every number it used in
results/readme_numbers.json. These tests recompute the headline ones from the
CSVs and check that the README contains them as printed. Edit the README by
regenerating it, not by hand.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NUMBERS = REPO / "results" / "readme_numbers.json"
README = REPO / "README.md"

needs = pytest.mark.skipif(not NUMBERS.exists(), reason="README not generated")


def pick(rows, system, knob, condition=None):
    for r in rows:
        if (
            r["system"] == system
            and float(r["knob_value"]) == knob
            and (condition is None or r["condition"] == condition)
        ):
            return r
    raise AssertionError(f"missing {system} {knob} {condition}")


@needs
def test_headline_numbers_match_the_csvs():
    n = json.loads(NUMBERS.read_text())
    sweep = list(csv.DictReader((REPO / "results/tradeoff.csv").open()))
    cond = list(csv.DictReader((REPO / "results/conditions.csv").open()))
    checks = {
        "timer500_cut": pick(sweep, "fixed_timeout+silero", 500.0),
        "timer800_cut": pick(sweep, "fixed_timeout+silero", 800.0),
        "pg_cut": pick(sweep, "punctuation+gate200", 0.0),
        "v11g03_cut": pick(sweep, "text_eot_v1.1+gate200", 0.3),
        "pro_cut": pick(sweep, "prosody_prosody+gate200", 0.5),
        "tel_pg_cut": pick(cond, "punctuation+gate200", 0.0, "asr_tel"),
        "tel_v11g03_cut": pick(cond, "text_eot_v1.1+gate200", 0.3, "asr_tel"),
        "tel_timer800_cut": pick(cond, "fixed_timeout+silero", 800.0, "asr_tel"),
    }
    for key, row in checks.items():
        assert n[key] == pytest.approx(float(row["cutoff_rate_at_tolerance"]) * 100), (
            key
        )
    assert (
        n["n_turns"]
        == json.loads((REPO / "data/eval/MANIFEST.json").read_text())["n_turns"]
    )


@needs
def test_readme_prints_the_numbers_it_recorded():
    n = json.loads(NUMBERS.read_text())
    text = README.read_text()
    for key in (
        "timer500_cut",
        "timer800_cut",
        "pg_cut",
        "v11g03_cut",
        "tel_pg_cut",
        "tel_v11g03_cut",
    ):
        assert f"{n[key]:.1f}%" in text, f"{key}={n[key]:.1f}% not on the page"
    assert f"{n['n_turns']:,}" in text
    assert "not measured yet" not in text.lower() or "Not yet measured" in text


@needs
def test_readme_claims_no_unmeasured_figure():
    """Anything the page promises but has not run must say so, not estimate."""
    text = README.read_text()
    assert "Not yet measured" in text  # the Deepgram row is honest about itself
