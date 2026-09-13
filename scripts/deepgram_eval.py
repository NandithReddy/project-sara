"""Run Deepgram over the frozen eval turns and cache every response.

  --backend nova3   streaming STT -> results/asr/nova3_<cond>.json, the same
                    partial format as the parakeet caches, so every text
                    system replays it unchanged (plus speech_final events)
  --backend flux    turn detection -> results/asr/flux_<cond>.json, confidence
                    on every update, so the threshold curve is swept offline

Each turn's audio is the CALLER CHANNEL (zeroed after true_end + 150ms), and
for --condition tel additionally band-limited and mu-law coded, exactly as for
parakeet. Sent paced at 1x real time over `--workers` concurrent connections.
Cost at list price: ~$0.15 per pass (20 minutes of audio).

The section 10 exception: a black box reads eval audio, here, in scripts/.
eval/ never imports src/baselines/deepgram.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.dataset import MANIFEST, load_eval_set  # noqa: E402
from eval.harness import caller_channel  # noqa: E402
from src.audio.telephony import degrade  # noqa: E402
from src.baselines import deepgram as dg  # noqa: E402

OUT_DIR = REPO / "results" / "asr"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("nova3", "flux"), required=True)
    ap.add_argument("--condition", choices=("16k", "tel"), required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or OUT_DIR / f"{args.backend}_{args.condition}.json"

    key = dg.load_api_key()
    turns = load_eval_set()
    if args.limit:
        turns = turns[: args.limit]
    sr = 16_000
    query = dg.nova_query(sr) if args.backend == "nova3" else dg.flux_query(sr)
    url = dg._url(dg.NOVA_URL if args.backend == "nova3" else dg.FLUX_URL, query)

    jobs, total_s = [], 0.0
    for t in turns:
        audio, file_sr = sf.read(t.audio_path, dtype="float32")
        if file_sr != sr:
            raise SystemExit(f"{t.audio_path}: {file_sr}Hz")
        audio = caller_channel(np.asarray(audio, dtype=np.float32).reshape(-1), t)
        if args.condition == "tel":
            audio = degrade(audio)
        total_s += len(audio) / sr
        jobs.append((url, key, audio, sr))
    print(
        f"{args.backend} {args.condition}: {len(jobs)} turns, {total_s:.0f}s audio, "
        f"{args.workers} concurrent, paced 1x -> ~{total_s / args.workers / 60:.1f} min"
    )

    t0 = dt.datetime.now()
    results = asyncio.run(dg.run_many(jobs, args.workers))
    elapsed = (dt.datetime.now() - t0).total_seconds()

    cache, n_closed_early, n_msgs, failed = {}, 0, 0, []
    for t, received in zip(turns, results, strict=True):
        n_msgs += len(received)
        if any(m.message.get("type") == "_closed" for m in received):
            n_closed_early += 1
        if any(m.message.get("type") == "_error" for m in received):
            failed.append((t.turn_id, received[-1].message.get("reason")))
            continue
        cache[t.turn_id] = (
            dg.nova3_partials(received)
            if args.backend == "nova3"
            else dg.flux_events(received)
        )
    empty = [tid for tid, v in cache.items() if not v]
    if failed or empty:
        for tid, why in failed[:5]:
            print(f"  {tid}: {why}")
        raise SystemExit(
            f"{len(failed)} turns failed after retries, {len(empty)} returned nothing "
            f"usable: not writing a cache with holes in it (rule 3). Re-run."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "backend": args.backend,
                "condition": args.condition,
                "tail": "caller",
                "model": query["model"],
                "query": query,
                "date": dt.date.today().isoformat(),
                "machine": platform.platform(),
                "eval_manifest_sha256": hashlib.sha256(
                    MANIFEST.read_bytes()
                ).hexdigest(),
                "n_turns": len(cache),
                "total_audio_s": round(total_s, 1),
                "wall_s": round(elapsed, 1),
                "workers": args.workers,
                "n_messages": n_msgs,
                "n_closed_early": n_closed_early,
                "turns": cache,
            },
            indent=1,
        )
    )
    shown = out.relative_to(REPO) if out.is_relative_to(REPO) else out
    print(
        f"{len(cache)} turns, {n_msgs} messages in {elapsed:.0f}s wall; "
        f"{n_closed_early} streams closed early; wrote {shown}"
    )
    if args.backend == "flux":
        ends = sum(
            1 for v in cache.values() if any(e["event"] == "EndOfTurn" for e in v)
        )
        print(f"turns with a Flux EndOfTurn event: {ends}/{len(cache)}")
    else:
        sf_ = sum(1 for v in cache.values() if any(p["speech_final"] for p in v))
        empty_final = sum(1 for v in cache.values() if v and not v[-1]["text"])
        print(
            f"speech_final on {sf_}/{len(cache)} turns; empty final text: {empty_final}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
