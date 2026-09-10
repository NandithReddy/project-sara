"""Stage 2: fetch per-turn audio and freeze the eval set.

Fetches only the seconds each turn needs, using HTTP range requests against the
AMI mirror (the files are uncompressed 16kHz mono PCM, so a byte offset maps
directly to a timestamp). 198 turns cost ~45MB instead of the ~2.4GB the 60
whole files would.

Grounds the turn boundary in audio. AMI's alignment stretches words to absorb
silence, so the last word's endtime overruns the real end of speech by an
unknown amount -- and added_latency_ms is measured FROM that boundary, so the
error would land directly in the headline metric. An offline VAD pass over the
segment recovers the real speech offset.

Known limitation, stated rather than hidden: the boundary is refined with
Silero, and Silero is also baseline #2. The ground truth therefore shares a
model with one system under test. The two uses differ -- offline over full
context here, streaming with a timeout there -- and the printed delta
distribution quantifies how far the refinement moves the boundary at all.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.vad import SileroVAD  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
SAMPLE_RATE = 16_000
PRE_ROLL_MS = 500.0
UA = "project-sara/eval-builder (research; AMI CC BY 4.0)"


def audio_url(meeting: str, channel: int) -> str:
    return f"{MIRROR}/{meeting}/audio/{meeting}.Headset-{channel}.wav"


def http_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end}", "User-Agent": UA}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.status not in (200, 206):
            raise RuntimeError(f"{url} returned HTTP {r.status}, expected 206")
        return r.read()


def wav_layout(url: str) -> tuple[int, int, int]:
    """Return (data_offset, sample_rate, block_align) by reading the header."""
    head = http_range(url, 0, 4095)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise RuntimeError(f"{url} is not a RIFF/WAVE file")
    pos, sr, ba, bits, data_off = 12, None, None, None, None
    while pos + 8 <= len(head):
        cid = head[pos : pos + 4]
        size = struct.unpack("<I", head[pos + 4 : pos + 8])[0]
        if cid == b"fmt ":
            _, ch, sr, _, ba, bits = struct.unpack("<HHIIHH", head[pos + 8 : pos + 24])
            if ch != 1 or bits != 16:
                raise RuntimeError(f"{url}: expected mono 16-bit, got {ch}ch {bits}bit")
        elif cid == b"data":
            data_off = pos + 8
            break
        pos += 8 + size + (size & 1)
    if data_off is None or sr is None:
        raise RuntimeError(f"{url}: no data chunk in first 4KB")
    return data_off, sr, ba


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=Path, default=REPO / "data/eval/turns.json")
    ap.add_argument("--out", type=Path, default=REPO / "data/eval")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--sleep", type=float, default=0.15, help="politeness delay")
    ap.add_argument(
        "--reuse-audio",
        action="store_true",
        help="decode existing FLAC instead of refetching (unchanged segments)",
    )
    args = ap.parse_args()

    spec = json.loads(args.turns.read_text())
    turns = spec["turns"][: args.limit] if args.limit else spec["turns"]
    horizon_ms = spec["horizon_ms"]
    search_ms = spec["boundary_search_ms"]

    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    vad = SileroVAD()

    layouts: dict[str, tuple[int, int, int]] = {}
    rows, deltas, fetched = [], [], 0

    for i, t in enumerate(turns, 1):
        url = audio_url(t["meeting"], t["channel"])
        if url not in layouts:
            layouts[url] = wav_layout(url)
            time.sleep(args.sleep)
        data_off, sr, ba = layouts[url]
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"{url}: {sr}Hz, expected {SAMPLE_RATE}Hz")

        seg_start = max(0.0, t["start_s"] - PRE_ROLL_MS / 1000.0)
        seg_end = t["end_s"] + (horizon_ms + search_ms) / 1000.0
        b0 = data_off + int(seg_start * sr) * ba
        b1 = data_off + int(seg_end * sr) * ba - 1

        path = audio_dir / f"{t['turn_id']}.flac"
        if args.reuse_audio and path.exists():
            audio, _sr = sf.read(path, dtype="float32")
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        else:
            raw = http_range(url, b0, b1)
            fetched += len(raw)
            pcm = np.frombuffer(raw[: len(raw) - len(raw) % ba], dtype="<i2")
            audio = (pcm.astype(np.float32) / 32768.0).copy()
            sf.write(path, audio, sr, format="FLAC", subtype="PCM_16")

        # Ground the boundary: last speech frame at or before the annotated end.
        vad.reset()
        frames = vad.push(audio)
        ann_end_in_seg_ms = (t["end_s"] - seg_start) * 1000.0
        # Raw probability, not the hysteresis-smoothed is_speech flag. Hysteresis
        # exists to stop the LIVE path flapping; here it would just push the
        # boundary late by its release time, and that bias would land straight
        # in added_latency_ms. It also keeps the ground truth one step further
        # from baseline #2, which does use the smoothed flag.
        speech = [
            f
            for f in frames
            if f.prob >= vad.threshold and f.t_ms <= ann_end_in_seg_ms + search_ms
        ]
        if speech:
            true_end_ms = speech[-1].t_ms
            boundary_source = "audio"
            deltas.append(ann_end_in_seg_ms - true_end_ms)
        else:
            # Quiet single-word backchannels ("Yep", "Okay") can sit under the
            # VAD threshold. Dropping them would systematically remove the
            # quietest short answers -- the hardest cases for a VAD baseline --
            # and make the eval easier for the systems under test. Keep them on
            # the annotated boundary and mark the source so the difference can
            # be audited rather than hidden.
            true_end_ms = ann_end_in_seg_ms
            boundary_source = "annotation"
            print(
                f"  [{i}/{len(turns)}] {t['turn_id']}: below VAD threshold "
                f"({t['n_words']}w {t['text'][:24]!r}) -- annotated boundary"
            )

        rows.append(
            {
                **{k: v for k, v in t.items() if k != "words"},
                "audio": f"audio/{path.name}",
                "words": [
                    {
                        "t": w[0],
                        "start_ms": round((w[1] - seg_start) * 1000.0, 1),
                        "end_ms": round((w[2] - seg_start) * 1000.0, 1),
                    }
                    for w in t["words"]
                ],
                "seg_start_s": round(seg_start, 3),
                "seg_duration_ms": round(len(audio) / sr * 1000.0, 1),
                "turn_start_ms": round((t["start_s"] - seg_start) * 1000.0, 1),
                "annotated_end_ms": round(ann_end_in_seg_ms, 1),
                "true_end_ms": round(true_end_ms, 1),
                "boundary_source": boundary_source,
            }
        )
        if i % 25 == 0 or i == len(turns):
            print(f"  [{i}/{len(turns)}] {fetched / 1e6:.1f}MB fetched")
        time.sleep(args.sleep)

    d = np.array(deltas)
    n_ann = sum(1 for r in rows if r["boundary_source"] == "annotation")
    print(f"\nkept {len(rows)}/{len(turns)} turns, {fetched / 1e6:.1f}MB fetched")
    print(f"boundary from audio: {len(rows) - n_ann}, from annotation: {n_ann}")
    print("annotated_end minus audio-grounded end (positive = alignment overruns):")
    print(
        f"  p10={np.percentile(d, 10):+.0f}ms  p50={np.percentile(d, 50):+.0f}ms  "
        f"p90={np.percentile(d, 90):+.0f}ms  max={d.max():+.0f}ms"
    )
    print(f"  turns where the alignment overruns by >200ms: {(d > 200).sum()}")

    out = args.out / "eval_set.json"
    out.write_text(
        json.dumps(
            {k: v for k, v in spec.items() if k != "turns"} | {"turns": rows}, indent=2
        )
    )
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
