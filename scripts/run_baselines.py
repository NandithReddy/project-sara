"""Run every section 3 baseline AND the model over the frozen eval set.

One table, as Phase 4 asks: the model's row sits next to the baselines it has
to beat. If it cannot beat the fixed timeout, the project has failed and we say
so plainly (section 3).

The model row reports two latencies. `cpu p99` is what the harness measures on
every update() -- most hit the text cache and cost microseconds. `infer p99`
is the model-only cost per real inference, the honest number against the 20ms
budget.

Baselines #1 and #2 apply the same timer over different silence sources -- naive
energy thresholding versus Silero. That difference is the whole distinction
between them; sharing a source would make them one baseline reported twice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.harness import (  # noqa: E402
    RESULTS,
    evaluate,
    precompute_prosody,
    write_results,
)
from src.audio.vad import SileroVAD  # noqa: E402
from src.baselines.energy import EnergyVAD  # noqa: E402
from src.baselines.punctuation import PunctuationHeuristic  # noqa: E402
from src.baselines.silence import (  # noqa: E402
    SECTION_3_TIMEOUTS_MS,
    FixedSilenceTimeout,
)
from src.eot.gated import DEFAULT_GATE_MS, SilenceGated  # noqa: E402
from src.eot.model import TextEOT  # noqa: E402
from src.eot.prosody_eot import ProsodyEOT  # noqa: E402
from src.eot.replayed import (  # noqa: E402
    ReplayedEOT,
    flux_shipped_series,
    nova3_endpoint_series,
)

# 800ms is what the Phase 1 live loop shipped with, so the demo and the
# measurement talk about the same operating point.
TIMEOUTS_MS = sorted({*SECTION_3_TIMEOUTS_MS, 800.0})


def main() -> int:
    runs = []
    for source_factory in (EnergyVAD, SileroVAD):
        for timeout in TIMEOUTS_MS:
            runs.append((FixedSilenceTimeout(timeout), source_factory(), None))
    # Baseline #3 reads only text, so the silence source cannot affect it.
    runs.append((PunctuationHeuristic(), SileroVAD(), "punctuation"))
    # The deliverable, bare and gated. Raises if untrained: rule 3, no silent
    # substitution. The gated punctuation heuristic is the fair fight for the
    # gated model: same gate, no model.
    runs.append(
        (SilenceGated(PunctuationHeuristic(), DEFAULT_GATE_MS), SileroVAD(), None)
    )
    runs.append((TextEOT(), SileroVAD(), None))
    runs.append((SilenceGated(TextEOT(), DEFAULT_GATE_MS), SileroVAD(), None))
    # Phase 7: the pause classifier, prosody alone and fused with the text model.
    runs.append((ProsodyEOT(kind="prosody"), SileroVAD(), None))
    runs.append((ProsodyEOT(kind="fusion"), SileroVAD(), None))
    # Baseline #4: Deepgram Flux as shipped, and Nova-3's own endpointing,
    # replayed from their cached responses on the same audio (caller channel,
    # wideband). Cloud, own recogniser; the silence source is not consulted.
    flux = json.loads((RESULTS / "asr" / "flux_16k.json").read_text())
    nova3 = json.loads((RESULTS / "asr" / "nova3_16k.json").read_text())
    runs.append((ReplayedEOT("flux", flux_shipped_series(flux)), SileroVAD(), "flux"))
    runs.append(
        (
            ReplayedEOT("nova3_speech_final", nova3_endpoint_series(nova3)),
            SileroVAD(),
            "nova3_speech_final",
        )
    )

    print("prosody features ...", flush=True)
    prosody = precompute_prosody()

    rows = []
    for detector, silence, name in runs:
        # Text-only systems (and gates on them) carry no silence-source suffix:
        # the source cannot change their transcript, only the gate reads it.
        if name is None and not isinstance(detector, FixedSilenceTimeout):
            name = detector.name
        label = name or f"{detector.name}__{silence.name}"
        print(f"running {label} ...", flush=True)
        summary = evaluate(detector, silence=silence, name=name, prosody_cache=prosody)
        if hasattr(detector, "inference_latency_ms"):
            summary["inference_latency_ms"] = detector.inference_latency_ms()
            summary["model"] = detector.meta.get("base_model") or detector.meta.get(
                "kind"
            )
        write_results(summary)
        rows.append(summary)

    print(
        f"\n{'system':>30} {'cutoff':>8} {'cut>150':>8} {'hold':>7} "
        f"{'lat p50':>9} {'p95':>8} {'p99':>8} {'infer p99':>10}"
    )
    print("-" * 96)
    for s in rows:
        a = s["added_latency_ms"]
        fmt = lambda v: f"{v:8.0f}" if v is not None else "       -"  # noqa: E731
        infer = s.get("inference_latency_ms", {}).get("p99")
        print(
            f"{s['name']:>30} {s['cutoff_rate'] * 100:7.1f}% "
            f"{s['cutoff_rate_at_tolerance'] * 100:7.1f}% "
            f"{s['false_hold_rate'] * 100:6.1f}% "
            f"{fmt(a['p50'])} {fmt(a['p95'])} {fmt(a['p99'])} "
            f"{f'{infer:7.2f}ms' if infer is not None else '         -'}"
        )
    print(
        f"\ncutoff = fired before the true end. cut>150 allows the "
        f"{rows[0]['boundary_tolerance_ms']:.0f}ms the boundary itself is "
        f"uncertain to."
    )
    print("results/ written. No number here is estimated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
