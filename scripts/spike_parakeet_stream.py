"""SPIKE — throwaway. Does parakeet-mlx's streaming API give us what section 5 assumes?

NOT project code. Nothing imports this. It exists to answer three questions with
real numbers before Phase 1 commits to parakeet-mlx:

  1. How far behind real time does the FIRST partial land?
  2. Do partials get REVISED (earlier text rewritten), or only appended to?
  3. Are token timestamps STABLE across revisions, or do they move?

Question 3 is the one that matters most. ENGINEERING.md section 5 justifies choosing
parakeet-mlx over whisper.cpp on "native word-level timestamps". If those
timestamps shift every time a partial is revised, the justification is wrong and
the fallback is Deepgram Nova-3.

Audio is fed PACED at 1x real time, so "lag" means what it means in a live
pipeline: wall-clock now, minus how much audio the mic has produced so far.

Run:
    uv run --with parakeet-mlx python scripts/spike_parakeet_stream.py \
        --wav data/raw/jfk.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


def flatten_tokens(result):
    """All aligned tokens across sentences, in order."""
    tokens = []
    for sentence in getattr(result, "sentences", []) or []:
        for tok in getattr(sentence, "tokens", []) or []:
            tokens.append(
                (tok.text, round(float(tok.start), 3), round(float(tok.end), 3))
            )
    return tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--chunk-ms", type=int, default=320)
    ap.add_argument("--context", type=int, nargs=2, default=(256, 256))
    ap.add_argument(
        "--no-pace",
        action="store_true",
        help="feed as fast as possible; measures raw throughput, not lag",
    )
    args = ap.parse_args()

    # Rule 3: fail loudly rather than substituting anything.
    if not args.wav.exists():
        print(
            f"ERROR: {args.wav} not found. Fetch the sample first:\n"
            f"  curl -sSL -o {args.wav} https://github.com/ggerganov/"
            f"whisper.cpp/raw/master/samples/jfk.wav",
            file=sys.stderr,
        )
        return 2

    import mlx.core as mx
    import soundfile as sf
    from parakeet_mlx import from_pretrained

    print(f"model      : {args.model}")
    t0 = time.perf_counter()
    model = from_pretrained(args.model)
    load_s = time.perf_counter() - t0
    print(f"load time  : {load_s:.2f}s")

    sr = model.preprocessor_config.sample_rate
    # parakeet_mlx.audio.load_audio shells out to ffmpeg. Our file is already
    # 16kHz mono PCM, so read it directly rather than adding a system dependency.
    pcm, file_sr = sf.read(args.wav, dtype="float32", always_2d=True)
    if file_sr != sr:
        print(
            f"ERROR: {args.wav} is {file_sr}Hz, model wants {sr}Hz. "
            f"No resampling here on purpose — resample it explicitly.",
            file=sys.stderr,
        )
        return 2
    if pcm.shape[1] != 1:
        print(f"ERROR: expected mono, got {pcm.shape[1]} channels.", file=sys.stderr)
        return 2
    audio = mx.array(pcm[:, 0])
    total_s = len(audio) / sr
    chunk_n = int(sr * args.chunk_ms / 1000)
    print(f"sample rate: {sr} Hz")
    print(f"audio      : {args.wav} ({total_s:.2f}s)")
    print(
        f"chunk      : {args.chunk_ms}ms ({chunk_n} samples)   "
        f"context={tuple(args.context)}"
    )
    print(f"pacing     : {'OFF (max speed)' if args.no_pace else 'ON (1x real time)'}")
    print("=" * 78)

    prev_text = ""
    prev_tokens: list[tuple[str, float, float]] = []
    first_partial_lag = None
    n_updates = n_append = n_revise = 0
    ts_mutations = 0
    mutation_examples: list[str] = []
    compute_times: list[float] = []

    with model.transcribe_stream(context_size=tuple(args.context)) as tr:
        wall0 = time.perf_counter()
        for i in range(0, len(audio), chunk_n):
            chunk = audio[i : i + chunk_n]
            audio_end_s = min((i + len(chunk)) / sr, total_s)

            # Pace: don't hand over audio the mic wouldn't have produced yet.
            if not args.no_pace:
                ahead = audio_end_s - (time.perf_counter() - wall0)
                if ahead > 0:
                    time.sleep(ahead)

            t_start = time.perf_counter()
            tr.add_audio(chunk)
            result = tr.result
            text = (getattr(result, "text", "") or "").strip()
            tokens = flatten_tokens(result)
            t_done = time.perf_counter()

            compute_times.append((t_done - t_start) * 1000)
            wall = t_done - wall0
            lag_ms = (wall - audio_end_s) * 1000

            if not text:
                continue

            n_updates += 1
            if first_partial_lag is None:
                first_partial_lag = lag_ms

            # Append-only, or was earlier text rewritten?
            if text.startswith(prev_text):
                kind, n_append = "APPEND", n_append + 1
            else:
                kind, n_revise = "REVISE", n_revise + 1

            # Did any already-emitted token's timestamps move?
            moved = 0
            for j in range(min(len(prev_tokens), len(tokens))):
                if prev_tokens[j] != tokens[j]:
                    moved += 1
                    if len(mutation_examples) < 5:
                        mutation_examples.append(
                            f"    tok[{j}] {prev_tokens[j]!r} -> {tokens[j]!r}"
                        )
            ts_mutations += moved

            flag = f"  TS-MOVED x{moved}" if moved else ""
            print(
                f"[{n_updates:>3}] wall=+{wall:6.3f}s  audio={audio_end_s:5.2f}s  "
                f"lag={lag_ms:+7.1f}ms  {kind}{flag}"
            )
            print(f"      text: {text!r}")
            new_toks = tokens[len(prev_tokens) :]
            if new_toks:
                shown = " ".join(f"{t!r}[{s:.2f}-{e:.2f}]" for t, s, e in new_toks[-6:])
                print(f"      +tok: {shown}")

            prev_text, prev_tokens = text, tokens

    print("=" * 78)
    print(f"final text        : {prev_text!r}")
    print(f"final token count : {len(prev_tokens)}")
    print()
    print("--- ANSWERS ---")
    print(
        f"1. first partial lag      : {first_partial_lag:+.1f}ms"
        if first_partial_lag is not None
        else "1. first partial lag      : NEVER PRODUCED A PARTIAL"
    )
    print(
        f"2. partials revised?      : {n_revise} revised / {n_updates} updates "
        f"({n_append} pure appends)"
    )
    print(
        f"3. timestamps stable?     : {ts_mutations} mutations "
        f"of already-emitted tokens"
    )
    if mutation_examples:
        print("   examples:")
        for ex in mutation_examples:
            print(ex)
    if compute_times:
        s = sorted(compute_times)
        p50 = s[len(s) // 2]
        p95 = s[int(len(s) * 0.95) - 1]
        print(
            f"   per-chunk compute     : p50={p50:.1f}ms  p95={p95:.1f}ms  "
            f"max={s[-1]:.1f}ms  (chunk={args.chunk_ms}ms)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
