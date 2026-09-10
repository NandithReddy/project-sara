"""Run the section 3 baselines over the frozen eval set and write results/.

These are the numbers everything else is measured against. If our model cannot
beat the fixed timeout, the project has failed and we say so plainly (section 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.harness import evaluate, write_results  # noqa: E402
from src.baselines.silence import (  # noqa: E402
    SECTION_3_TIMEOUTS_MS,
    FixedSilenceTimeout,
)

# 800ms is the value the Phase 1 live loop shipped with; included so the demo
# and the measurement talk about the same operating point.
TIMEOUTS_MS = (*SECTION_3_TIMEOUTS_MS, 800.0)


def main() -> int:
    rows = []
    for timeout in sorted(TIMEOUTS_MS):
        detector = FixedSilenceTimeout(timeout)
        print(f"running {detector.name} ...", flush=True)
        summary = evaluate(detector)
        path = write_results(summary)
        rows.append(summary)
        print(f"  wrote {path.relative_to(Path.cwd())}")

    print(
        f"\n{'baseline':>22} {'cutoff':>8} {'hold':>7} "
        f"{'lat p50':>9} {'p95':>8} {'p99':>8} {'cpu p99':>9}"
    )
    print("-" * 76)
    for s in rows:
        a, u = s["added_latency_ms"], s["update_latency_ms"]
        f = lambda v: f"{v:8.0f}" if v is not None else "       -"  # noqa: E731
        print(
            f"{s['name']:>22} {s['cutoff_rate'] * 100:7.1f}% "
            f"{s['false_hold_rate'] * 100:6.1f}% "
            f"{f(a['p50'])} {f(a['p95'])} {f(a['p99'])} {u['p99']:8.3f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
