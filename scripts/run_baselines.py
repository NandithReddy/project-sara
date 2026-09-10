"""Run every section 3 baseline over the frozen eval set and write results/.

These are the numbers everything else is measured against. If our model cannot
beat the fixed timeout, the project has failed and we say so plainly (section 3).

Baselines #1 and #2 apply the same timer over different silence sources -- naive
energy thresholding versus Silero. That difference is the whole distinction
between them; sharing a source would make them one baseline reported twice.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.harness import evaluate, write_results  # noqa: E402
from src.audio.vad import SileroVAD  # noqa: E402
from src.baselines.energy import EnergyVAD  # noqa: E402
from src.baselines.punctuation import PunctuationHeuristic  # noqa: E402
from src.baselines.silence import (  # noqa: E402
    SECTION_3_TIMEOUTS_MS,
    FixedSilenceTimeout,
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

    rows = []
    for detector, silence, name in runs:
        label = name or f"{detector.name}__{silence.name}"
        print(f"running {label} ...", flush=True)
        summary = evaluate(detector, silence=silence, name=name)
        write_results(summary)
        rows.append(summary)

    print(
        f"\n{'baseline':>30} {'cutoff':>8} {'cut>150':>8} {'hold':>7} "
        f"{'lat p50':>9} {'p95':>8} {'p99':>8}"
    )
    print("-" * 84)
    for s in rows:
        a = s["added_latency_ms"]
        fmt = lambda v: f"{v:8.0f}" if v is not None else "       -"  # noqa: E731
        print(
            f"{s['name']:>30} {s['cutoff_rate'] * 100:7.1f}% "
            f"{s['cutoff_rate_at_tolerance'] * 100:7.1f}% "
            f"{s['false_hold_rate'] * 100:6.1f}% "
            f"{fmt(a['p50'])} {fmt(a['p95'])} {fmt(a['p99'])}"
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
