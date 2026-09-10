"""Talk to your laptop and have it talk back.

Run:
    uv run python scripts/live.py                        # fixed 800ms silence timer
    uv run python scripts/live.py --eot text             # the text EOT model, bare
    uv run python scripts/live.py --eot gated            # model + 200ms silence gate
    uv run python scripts/live.py --eot punct-gated      # the best point on the chart
    uv run python scripts/live.py --timeout-ms 500 --seconds 45

With --eot text the decision reads the transcript only. Two consequences you
will hear: the model cannot fire until parakeet has emitted any text, which
takes ~1.7s of speech (Phase 1c), so a very short answer gets no reply until
the fallback; and it fires on single words like "Okay" that are usually a
whole turn in the training data. Both are measured in results/, not surprises.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.silence import FixedSilenceTimeout  # noqa: E402
from src.pipeline.live import REPLY, run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eot",
        choices=("timeout", "text", "gated", "punct-gated"),
        default="timeout",
        help="end-of-turn rule: fixed silence timer; the text model bare; the "
        "model behind a 200ms silence gate; or the punctuation heuristic behind "
        "the same gate (the best measured point, results/tradeoff.md)",
    )
    ap.add_argument("--timeout-ms", type=float, default=800.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--reply", default=REPLY)
    args = ap.parse_args()

    if args.eot == "timeout":
        eot = FixedSilenceTimeout(args.timeout_ms)
    elif args.eot == "punct-gated":
        from src.baselines.punctuation import PunctuationHeuristic
        from src.eot.gated import SilenceGated

        eot = SilenceGated(PunctuationHeuristic())
    else:
        from src.eot.model import TextEOT  # loads the ONNX; raises if untrained

        eot = TextEOT()
        if args.eot == "gated":
            from src.eot.gated import SilenceGated

            eot = SilenceGated(eot)

    return run(
        eot=eot,
        seconds=args.seconds,
        threshold=args.threshold,
        reply=args.reply,
    )


if __name__ == "__main__":
    raise SystemExit(main())
