"""The tradeoff artefacts: schema and shape of results/tradeoff.csv.

Does not re-run the sweep (16s). Checks the committed CSV, which is the record
the chart and the reading are built from.
"""

from __future__ import annotations

import csv
import math

import pytest

from eval.sweep import CSV_PATH, FIELDS, PNG_PATH, THRESHOLDS, TIMEOUTS_MS, row

needs_csv = pytest.mark.skipif(not CSV_PATH.exists(), reason="sweep not run yet")


def test_row_maps_every_field_from_a_summary():
    summary = {
        "silence_source": "silero",
        "n_turns": 198,
        "cutoff_rate": 0.1,
        "cutoff_rate_at_tolerance": 0.09,
        "false_hold_rate": 0.02,
        "added_latency_ms": {"p50": 500.0, "p95": 700.0, "p99": 900.0},
        "added_latency_with_fallback_ms": {"p50": 512.0, "p95": 2000.0, "p99": 2000.0},
    }
    r = row("fixed_timeout+silero", "timeout_ms", 500.0, summary)
    assert set(r) == set(FIELDS)
    assert r["latency_fallback_p95"] == 2000.0
    assert r["knob_value"] == 500.0


@needs_csv
def test_csv_has_the_declared_schema():
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == FIELDS
        rows = list(reader)
    assert rows, "empty sweep"


@needs_csv
def test_every_system_has_the_expected_number_of_points():
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    by_system = {}
    for r in rows:
        by_system.setdefault(r["system"], []).append(r)
    assert set(by_system) == {
        "fixed_timeout+silero",
        "fixed_timeout+energy",
        "punctuation",
        "text_eot_v1",
    }
    assert len(by_system["fixed_timeout+silero"]) == len(TIMEOUTS_MS)
    assert len(by_system["fixed_timeout+energy"]) == len(TIMEOUTS_MS)
    assert len(by_system["punctuation"]) == 1, "no knob, so exactly one point"
    assert len(by_system["text_eot_v1"]) == len(THRESHOLDS), "sweeps on threshold"


@needs_csv
def test_no_metric_is_missing_or_nan():
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in (
            "cutoff_rate_at_tolerance",
            "false_hold_rate",
            "latency_fallback_p50",
        ):
            assert r[k] not in ("", "None"), f"{r['system']} {r['knob_value']}: {k}"
            assert not math.isnan(float(r[k]))


@needs_csv
def test_rates_are_rates_and_latency_respects_the_fallback():
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        assert 0.0 <= float(r["cutoff_rate_at_tolerance"]) <= 1.0
        assert 0.0 <= float(r["false_hold_rate"]) <= 1.0
        assert float(r["latency_fallback_p95"]) <= 2000.0 + 1e-6


@needs_csv
def test_the_chart_exists_next_to_the_data():
    assert PNG_PATH.exists() and PNG_PATH.stat().st_size > 10_000
