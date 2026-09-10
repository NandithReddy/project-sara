"""Phase 1b: watch the VAD and the fixed-timeout EOT rule run on your voice.

Prints, with millisecond audio-clock timestamps, every speech/silence
transition and the exact moment the 800ms silence rule would end your turn.
There is no STT and no reply yet -- those are 1c.

Run:
    uv run python scripts/vad_live.py
    uv run python scripts/vad_live.py --timeout-ms 500 --seconds 20

Or replay a file through the identical path, no mic needed:
    uv run python scripts/vad_live.py --wav data/raw/jfk.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.capture import mic_blocks  # noqa: E402
from src.audio.vad import SAMPLE_RATE, SileroVAD  # noqa: E402
from src.baselines.silence import FixedSilenceTimeout  # noqa: E402
from src.eot.base import Update  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout-ms", type=float, default=800.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument(
        "--wav", type=Path, help="replay a 16kHz mono WAV instead of the mic"
    )
    args = ap.parse_args()

    vad = SileroVAD(threshold=args.threshold)
    eot = FixedSilenceTimeout(args.timeout_ms)
    eot.reset()

    if args.wav:
        import soundfile as sf

        if not args.wav.exists():
            print(f"ERROR: {args.wav} not found.", file=sys.stderr)
            return 2
        pcm, file_sr = sf.read(args.wav, dtype="float32", always_2d=True)
        if file_sr != SAMPLE_RATE or pcm.shape[1] != 1:
            print(
                f"ERROR: need 16kHz mono, got {file_sr}Hz / {pcm.shape[1]}ch. "
                f"No resampling here on purpose -- convert it explicitly.",
                file=sys.stderr,
            )
            return 2
        step = int(SAMPLE_RATE * 0.032)
        source = (pcm[i : i + step, 0] for i in range(0, len(pcm), step))
        print(f"Replaying {args.wav} ({len(pcm) / SAMPLE_RATE:.2f}s)")
    else:
        source = mic_blocks()
        print(f"Listening for {args.seconds:.0f}s. Speak, then pause. Ctrl-C to stop.")

    print(f"VAD threshold {args.threshold}  |  EOT rule {eot.name}")
    print("-" * 72)

    was_speech = False
    fired = False
    turns = 0
    wall0 = time.perf_counter()

    try:
        for block in source:
            for f in vad.push(block):
                if f.is_speech != was_speech:
                    label = "SPEECH" if f.is_speech else "silence"
                    print(f"[{f.t_ms:8.0f}ms] {label:>7}  p={f.prob:.3f}")
                    was_speech = f.is_speech
                    if f.is_speech:
                        fired = False  # new speech -- the turn is live again

                p = eot.update(Update(t_ms=f.t_ms, text="", silence_ms=f.silence_ms))
                if p >= 0.5 and not fired:
                    fired = True
                    turns += 1
                    print(
                        f"[{f.t_ms:8.0f}ms] *** END OF TURN *** "
                        f"after {f.silence_ms:.0f}ms of silence"
                    )
                else:
                    print(
                        f"  p(speech)={f.prob:5.3f}  silence={f.silence_ms:6.0f}ms   ",
                        end="\r",
                        flush=True,
                    )

            if not args.wav and time.perf_counter() - wall0 > args.seconds:
                break
    except KeyboardInterrupt:
        pass

    print(f"\n{'-' * 72}\nturns ended: {turns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
