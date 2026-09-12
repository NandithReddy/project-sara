"""Transcribe the frozen eval audio with the live-path STT, and cache it.

The Phase 6 real-ASR condition. Section 10's exception: a system under test
may read eval AUDIO. It runs here in scripts/, never in eval/ -- eval/ reads
the cached timeline this writes and never imports src/stt/. Cached under
results/asr/ with model, library version, date, machine and the eval manifest
hash, as section 3 requires for a black-box system: the number is not
bit-reproducible across vendor or hardware changes, so the raw responses are
what reproduce.

By default the recogniser hears the CALLER CHANNEL -- the frozen audio zeroed
from true_end + 150ms (eval.harness.caller_channel) -- because AMI headsets
carry the next speaker at low level and a recogniser transcribes it. The first
pass on the raw tail is kept as results/asr/parakeet_*_rawtail.json.

Feeds each turn's audio in 640ms chunks (the measured floor; shorter falls
behind real time) and records, after every chunk, the full hypothesis and how
long the chunk took. A partial is AVAILABLE at audio_end_ms + compute_ms: that
is when a caller would have it. Both are stored so the replay can also show
the content gap alone, with compute set to zero.

Run:
    uv run python scripts/transcribe_eval.py --condition 16k
    uv run python scripts/transcribe_eval.py --condition tel
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.dataset import MANIFEST, load_eval_set  # noqa: E402
from eval.harness import caller_channel  # noqa: E402
from src.audio.telephony import degrade  # noqa: E402

MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
CHUNK_MS = 640.0
CONTEXT = (256, 256)
OUT_DIR = REPO / "results" / "asr"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=("16k", "tel"), required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--tail",
        choices=("caller", "raw"),
        default="caller",
        help="'caller': zero the channel after the turn ends, as a phone line "
        "would be. 'raw': the headset as recorded, which carries the next "
        "speaker -- the first pass, kept as *_rawtail.json for the record.",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or OUT_DIR / f"parakeet_{args.condition}.json"

    import mlx.core as mx
    from parakeet_mlx import from_pretrained

    turns = load_eval_set()
    if args.limit:
        turns = turns[: args.limit]
    t0 = time.perf_counter()
    model = from_pretrained(MODEL)
    sr = model.preprocessor_config.sample_rate
    print(
        f"model ready in {time.perf_counter() - t0:.1f}s; {len(turns)} turns; "
        f"condition={args.condition}"
    )

    chunk_n = int(sr * CHUNK_MS / 1000.0)
    result, total_audio_s, total_compute_s, first_lags, revisions = {}, 0.0, 0.0, [], 0
    for i, turn in enumerate(turns, 1):
        audio, file_sr = sf.read(turn.audio_path, dtype="float32")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if file_sr != sr:
            raise SystemExit(f"{turn.audio_path}: {file_sr}Hz, model wants {sr}Hz")
        if args.tail == "caller":
            audio = caller_channel(audio, turn)
        if args.condition == "tel":
            audio = degrade(audio)
        total_audio_s += len(audio) / sr

        partials, prev = [], ""
        with model.transcribe_stream(context_size=CONTEXT) as tr:
            for s in range(0, len(audio), chunk_n):
                chunk = audio[s : s + chunk_n]
                t1 = time.perf_counter()
                tr.add_audio(mx.array(chunk))
                text = (getattr(tr.result, "text", "") or "").strip()
                compute_ms = (time.perf_counter() - t1) * 1000.0
                total_compute_s += compute_ms / 1000.0
                audio_ms = min(s + len(chunk), len(audio)) / sr * 1000.0
                if text and not prev:
                    first_lags.append(audio_ms - turn.turn_start_ms)
                if prev and not text.startswith(prev):
                    revisions += 1
                partials.append(
                    {
                        "audio_ms": round(audio_ms, 1),
                        "compute_ms": round(compute_ms, 1),
                        "text": text,
                    }
                )
                prev = text
        result[turn.turn_id] = partials
        if i % 20 == 0 or i == len(turns):
            print(
                f"  [{i}/{len(turns)}] audio {total_audio_s:.0f}s, "
                f"compute {total_compute_s:.0f}s",
                flush=True,
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_sha = __import__("hashlib").sha256(MANIFEST.read_bytes()).hexdigest()
    out.write_text(
        json.dumps(
            {
                "condition": args.condition,
                "tail": args.tail,
                "model": MODEL,
                "parakeet_mlx": pkg_version("parakeet-mlx"),
                "mlx": pkg_version("mlx"),
                "date": dt.date.today().isoformat(),
                "machine": platform.platform(),
                "chunk_ms": CHUNK_MS,
                "context": list(CONTEXT),
                "eval_manifest_sha256": manifest_sha,
                "n_turns": len(result),
                "total_audio_s": round(total_audio_s, 1),
                "total_compute_s": round(total_compute_s, 1),
                "turns": result,
            },
            indent=1,
        )
    )
    lags = np.array(first_lags) if first_lags else np.array([np.nan])
    rtf = total_compute_s / max(total_audio_s, 1e-9)
    print(
        f"\n{len(result)} turns, {total_audio_s:.0f}s audio in "
        f"{total_compute_s:.0f}s compute ({rtf:.2f}x realtime)"
    )
    print(
        f"first partial after turn start: p50={np.nanmedian(lags):.0f}ms "
        f"p90={np.nanpercentile(lags, 90):.0f}ms; turns with NO text: "
        f"{sum(1 for p in result.values() if not p[-1]['text'])}; "
        f"revisions: {revisions}"
    )
    shown = out.relative_to(REPO) if out.is_relative_to(REPO) else out
    print(f"wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
