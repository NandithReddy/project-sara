"""The conditions artefacts and the Deepgram caches they replay.

Does not re-run the conditions (minutes). Checks the committed CSV and the
committed caches, which are the record the chart and the reading are built
from: a cache that does not cover the frozen set, or that carries a different
manifest, would make every number downstream of it wrong.
"""

from __future__ import annotations

import csv
import hashlib
import json

import pytest

from eval.conditions import ASR_DIR, CFIELDS, CHARTED, CONDITIONS, CSV_PATH, PNG_PATH
from eval.dataset import MANIFEST, load_eval_set
from eval.sweep import THRESHOLDS

needs_csv = pytest.mark.skipif(not CSV_PATH.exists(), reason="conditions not run yet")


def load_rows():
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == CFIELDS
        return list(reader)


@needs_csv
def test_every_condition_is_present_and_charted_ones_have_the_chart():
    rows = load_rows()
    assert {r["condition"] for r in rows} == set(CONDITIONS)
    assert set(CHARTED) <= set(CONDITIONS)
    assert PNG_PATH.exists() and PNG_PATH.stat().st_size > 10_000


@needs_csv
def test_text_systems_run_under_every_transcript_source():
    rows = load_rows()
    for cond in CONDITIONS:
        systems = {r["system"] for r in rows if r["condition"] == cond}
        assert "punctuation+gate200" in systems, cond
        assert any(s.startswith("text_eot_v") and "+gate" in s for s in systems), cond


@needs_csv
def test_deepgram_systems_sit_only_where_deepgram_heard_the_audio():
    rows = load_rows()
    by = {}
    for r in rows:
        by.setdefault((r["condition"], r["system"]), []).append(r)
    for cond in ("nova3_16k", "nova3_tel"):
        assert len(by[(cond, "flux")]) == 1, cond
        assert len(by[(cond, "flux_conf")]) == len(THRESHOLDS), cond
        assert len(by[(cond, "nova3_speech_final")]) == 1, cond
    for cond in ("gold_16k", "gold_tel", "asr_16k", "asr_tel", "asr_16k_content"):
        for s in ("flux", "flux_conf", "nova3_speech_final"):
            assert (cond, s) not in by, f"{s} has no business under {cond}"


@needs_csv
def test_rates_are_rates_everywhere():
    for r in load_rows():
        assert 0.0 <= float(r["cutoff_rate_at_tolerance"]) <= 1.0
        assert 0.0 <= float(r["false_hold_rate"]) <= 1.0
        assert float(r["latency_fallback_p95"]) <= 2000.0 + 1e-6


# ---- the Deepgram caches themselves --------------------------------------

DEEPGRAM_CACHES = [
    ASR_DIR / f"{b}_{c}.json" for b in ("nova3", "flux") for c in ("16k", "tel")
]
needs_caches = pytest.mark.skipif(
    not all(p.exists() for p in DEEPGRAM_CACHES), reason="Deepgram passes not run"
)


@needs_caches
def test_deepgram_caches_cover_the_frozen_set_and_name_its_manifest():
    ids = {t.turn_id for t in load_eval_set()}
    sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    for path in DEEPGRAM_CACHES:
        spec = json.loads(path.read_text())
        assert spec["eval_manifest_sha256"] == sha, path.name
        assert spec["tail"] == "caller", path.name
        assert set(spec["turns"]) == ids, path.name
        assert spec["n_closed_early"] == 0, path.name
        for k in ("model", "query", "date", "workers", "wall_s"):
            assert k in spec, f"{path.name} lacks {k}"
        assert "key" not in json.dumps(spec["query"]).lower()


@needs_caches
def test_nova3_cache_is_the_parakeet_partial_format_plus_finality():
    for cond in ("16k", "tel"):
        spec = json.loads((ASR_DIR / f"nova3_{cond}.json").read_text())
        assert spec["model"] == "nova-3" and spec["condition"] == cond
        for parts in spec["turns"].values():
            assert parts, "a turn with no results at all"
            for p in parts:
                assert {
                    "audio_ms",
                    "compute_ms",
                    "text",
                    "is_final",
                    "speech_final",
                } <= set(p)
                assert p["compute_ms"] >= 0.0
            # A segment's is_final can arrive after an interim that already covers
            # later audio (Deepgram closes the earlier segment); the harness
            # orders by availability, which never runs backwards.
            avail = [p["audio_ms"] + p["compute_ms"] for p in parts]
            assert all(b >= a - 1.0 for a, b in zip(avail, avail[1:], strict=False))


@needs_caches
def test_flux_cache_carries_confidence_on_every_update():
    for cond in ("16k", "tel"):
        spec = json.loads((ASR_DIR / f"flux_{cond}.json").read_text())
        assert spec["model"] == "flux-general-en" and spec["condition"] == cond
        assert spec["query"]["eot_threshold"] == "0.7", "the shipped default"
        events = {"Update", "StartOfTurn", "EagerEndOfTurn", "TurnResumed", "EndOfTurn"}
        for evs in spec["turns"].values():
            assert evs, "a turn with no TurnInfo at all"
            for e in evs:
                assert e["event"] in events
                assert 0.0 <= e["eot_confidence"] <= 1.0
                assert e["wall_ms"] >= 0.0 and e["audio_ms"] >= 0.0
