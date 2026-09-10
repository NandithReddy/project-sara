"""Phase 1: talk to your laptop and have it talk back.

Run:
    uv run python scripts/live.py
    uv run python scripts/live.py --timeout-ms 500 --seconds 45
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
    ap.add_argument("--timeout-ms", type=float, default=800.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--reply", default=REPLY)
    args = ap.parse_args()

    return run(
        eot=FixedSilenceTimeout(args.timeout_ms),
        seconds=args.seconds,
        threshold=args.threshold,
        reply=args.reply,
    )


if __name__ == "__main__":
    raise SystemExit(main())
