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


def http_range(url: str, start: int, end: int, attempts: int = 4) -> bytes:
    """One range request, retried with backoff. A transient error in one of
    several thousand requests must not kill a half-hour fetch; a persistent
    one still fails loudly with the URL and range."""
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end}", "User-Agent": UA}
    )
    last: Exception | None = None
    for k in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status}, expected 206")
                return r.read()
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as e:
            last = e
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"{url} bytes={start}-{end}: {attempts} attempts failed: {last}")


def wav_layout(url: str) -> tuple[int, int, int, int]:
    """Return (data_offset, sample_rate, block_align, channels) from the header."""
    head = http_range(url, 0, 4095)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise RuntimeError(f"{url} is not a RIFF/WAVE file")
    pos, sr, ba, bits, data_off = 12, None, None, None, None
    while pos + 8 <= len(head):
        cid = head[pos : pos + 4]
        size = struct.unpack("<I", head[pos + 4 : pos + 8])[0]
        if cid == b"fmt ":
            _, ch, sr, _, ba, bits = struct.unpack("<HHIIHH", head[pos + 8 : pos + 24])
            if bits != 16:
                raise RuntimeError(f"{url}: expected 16-bit PCM, got {bits}bit")
        elif cid == b"data":
            data_off = pos + 8
            break
        pos += 8 + size + (size & 1)
    if data_off is None or sr is None:
        raise RuntimeError(f"{url}: no data chunk in first 4KB")
    return data_off, sr, ba, ch


def fetch_turns(
    turns: list[dict],
    horizon_ms: float,
    search_ms: float,
    audio_dir: Path,
    reuse_audio: bool,
    workers: int = 6,
    log_every: int = 25,
) -> tuple[list[dict], list[float], int]:
    """Fetch, boundary-refine and store each turn. Returns (rows, deltas, bytes).

    Shared by the eval and training fetches so the boundary is defined once.
    Bytes are fetched by a small thread pool -- every range request is a fresh
    TLS connection to the mirror, and sequentially that was ~3s per turn --
    while the refinement and the row order stay strictly sequential, so the
    output is byte-identical to the single-threaded version.
    """
    from concurrent.futures import ThreadPoolExecutor

    audio_dir.mkdir(parents=True, exist_ok=True)
    vad = SileroVAD()

    urls = sorted({audio_url(t["meeting"], t["channel"]) for t in turns})
    with ThreadPoolExecutor(workers) as pool:
        layouts = dict(zip(urls, pool.map(wav_layout, urls), strict=True))
    for url, (_off, sr, _ba, _ch) in layouts.items():
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"{url}: {sr}Hz, expected {SAMPLE_RATE}Hz")

    # A few AMI headset files are stereo. If the two channels are the same
    # signal (dual-mono) the left one is the headset; if they differ we do not
    # know which is the speaker, and the turns are skipped, out loud.
    dual_mono, skipped_urls = set(), set()
    for url, (off, sr, ba, ch) in layouts.items():
        if ch == 1:
            continue
        if ch != 2:
            skipped_urls.add(url)
            continue
        raw = http_range(url, off + 60 * sr * ba, off + 62 * sr * ba - 1)
        x = np.frombuffer(raw[: len(raw) - len(raw) % ba], dtype="<i2")
        x = x.reshape(-1, 2).astype(np.float64)
        corr = (
            float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])
            if x[:, 0].std() > 0 and x[:, 1].std() > 0
            else 0.0
        )
        (dual_mono if corr > 0.99 else skipped_urls).add(url)
        print(
            f"  stereo: {url.rsplit('/', 1)[1]} L/R correlation {corr:.3f} -> "
            f"{'dual-mono, using L' if url in dual_mono else 'SKIPPING its turns'}"
        )
    if skipped_urls:
        n_skip = sum(
            1 for t in turns if audio_url(t["meeting"], t["channel"]) in skipped_urls
        )
        print(
            f"  SKIPPING {n_skip} turns from {len(skipped_urls)} stereo file(s) "
            f"whose channels differ"
        )
        turns = [
            t
            for t in turns
            if audio_url(t["meeting"], t["channel"]) not in skipped_urls
        ]

    def plan(t: dict):
        url = audio_url(t["meeting"], t["channel"])
        data_off, sr, ba, _ch = layouts[url]
        seg_start = max(0.0, t["start_s"] - PRE_ROLL_MS / 1000.0)
        seg_end = t["end_s"] + (horizon_ms + search_ms) / 1000.0
        b0 = data_off + int(seg_start * sr) * ba
        b1 = data_off + int(seg_end * sr) * ba - 1
        return url, b0, b1, ba, sr, seg_start

    def get_audio(t: dict):
        """(audio, bytes_fetched) -- from disk if reusable, else the mirror."""
        url, b0, b1, ba, sr, _ = plan(t)
        path = audio_dir / f"{t['turn_id']}.flac"
        if reuse_audio and path.exists():
            audio, _sr = sf.read(path, dtype="float32")
            return np.asarray(audio, dtype=np.float32).reshape(-1), 0, False
        raw = http_range(url, b0, b1)
        pcm = np.frombuffer(raw[: len(raw) - len(raw) % ba], dtype="<i2")
        if layouts[url][3] == 2:
            pcm = pcm.reshape(-1, 2)[:, 0]  # verified dual-mono above
        return (pcm.astype(np.float32) / 32768.0).copy(), len(raw), True

    rows, deltas, fetched = [], [], 0
    batch = 96
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(turns), batch):
            chunk = turns[start : start + batch]
            for t, (audio, nbytes, fresh) in zip(
                chunk, pool.map(get_audio, chunk), strict=True
            ):
                _url, _b0, _b1, _ba, sr, seg_start = plan(t)
                fetched += nbytes
                path = audio_dir / f"{t['turn_id']}.flac"
                if fresh:
                    sf.write(path, audio, sr, format="FLAC", subtype="PCM_16")

                vad.reset()
                frames = vad.push(audio)
                ann_end_in_seg_ms = (t["end_s"] - seg_start) * 1000.0
                speech = [
                    f
                    for f in frames
                    if f.prob >= vad.threshold
                    and f.t_ms <= ann_end_in_seg_ms + search_ms
                ]
                if speech:
                    true_end_ms = speech[-1].t_ms
                    boundary_source = "audio"
                    deltas.append(ann_end_in_seg_ms - true_end_ms)
                else:
                    true_end_ms = ann_end_in_seg_ms
                    boundary_source = "annotation"

                rows.append(
                    {
                        **{k: v for k, v in t.items() if k != "words"},
                        "audio": f"audio/{path.name}",
                        "words": [
                            {
                                "t": w[0],
                                "start_ms": round((w[1] - seg_start) * 1000.0, 1),
                                "end_ms": round((w[2] - seg_start) * 1000.0, 1),
                                "punc_after": w[3],
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
                i = len(rows)
                if i % log_every == 0 or i == len(turns):
                    print(
                        f"  [{i}/{len(turns)}] {fetched / 1e6:.1f}MB fetched",
                        flush=True,
                    )
    return rows, deltas, fetched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=Path, default=REPO / "data/eval/turns.json")
    ap.add_argument("--out", type=Path, default=REPO / "data/eval")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=6, help="concurrent range requests")
    ap.add_argument(
        "--reuse-audio",
        action="store_true",
        help="decode existing FLAC instead of refetching (unchanged segments)",
    )
    args = ap.parse_args()

    spec = json.loads(args.turns.read_text())
    turns = spec["turns"][: args.limit] if args.limit else spec["turns"]
    rows, deltas, fetched = fetch_turns(
        turns,
        spec["horizon_ms"],
        spec["boundary_search_ms"],
        args.out / "audio",
        args.reuse_audio,
        args.workers,
    )

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
