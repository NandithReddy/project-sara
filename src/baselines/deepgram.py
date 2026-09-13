"""Deepgram as a system under test: Nova-3 streaming STT and Flux turn detection.

Lives in src/baselines/, not src/stt/ (section 10): these are black boxes that
read eval AUDIO, and their raw responses are cached under results/asr/ with
model, parameters, date and request ids, so the numbers survive a vendor
update and never have to be paid for twice. eval/ replays the caches and never
imports this module.

Audio is sent paced at 1x real time, so a message's wall-clock arrival is its
availability on the audio clock -- network and model latency included, which
is what a caller experiences. Each message also carries the server's own
audio position, so the content-only view (compute delay removed) exists too.

The API key is read from DEEPGRAM_API_KEY or ~/.config/sara/deepgram_api_key
and is never printed, logged or written anywhere else. Absent, this refuses
to run (rule 3).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KEY_FILE = Path.home() / ".config" / "sara" / "deepgram_api_key"
NOVA_URL = "wss://api.deepgram.com/v1/listen"
FLUX_URL = "wss://api.deepgram.com/v2/listen"
NOVA_MODEL = "nova-3"
FLUX_MODEL = "flux-general-en"
CHUNK_MS = 100.0
"""Send interval. Deepgram recommends small frames; 100ms keeps pacing tight."""


def load_api_key() -> str:
    key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    if not key:
        raise RuntimeError(
            "No Deepgram key: set DEEPGRAM_API_KEY or write it to "
            f"{KEY_FILE} (chmod 600). Not mocking anything (rule 3)."
        )
    return key


def nova_query(sample_rate: int) -> dict:
    return {
        "model": NOVA_MODEL,
        "encoding": "linear16",
        "sample_rate": str(sample_rate),
        "channels": "1",
        "interim_results": "true",
        "punctuate": "true",
        "endpointing": "10",
    }


def flux_query(
    sample_rate: int, eot_threshold: float = 0.7, eot_timeout_ms: int = 5000
) -> dict:
    return {
        "model": FLUX_MODEL,
        "encoding": "linear16",
        "sample_rate": str(sample_rate),
        "eot_threshold": str(eot_threshold),
        "eot_timeout_ms": str(eot_timeout_ms),
    }


@dataclass(frozen=True)
class Received:
    wall_ms: float  # since the first audio byte was sent
    message: dict


# ---- parsing: pure functions, tested without a network --------------------


def nova3_partials(received: list[Received]) -> list[dict]:
    """Cumulative text timeline from Nova-3 Results messages.

    Deepgram's interim results are per segment: an is_final result closes a
    segment and later results describe the next one. The caller's transcript
    so far is every closed segment plus the current interim, which is what a
    pipeline would show. speech_final marks Deepgram's own endpointing.
    """
    finals: list[str] = []
    out = []
    for r in received:
        m = r.message
        if m.get("type") != "Results":
            continue
        alt = (m.get("channel") or {}).get("alternatives") or [{}]
        transcript = (alt[0].get("transcript") or "").strip()
        audio_ms = (float(m.get("start", 0.0)) + float(m.get("duration", 0.0))) * 1000.0
        if m.get("is_final"):
            if transcript:
                finals.append(transcript)
            text = " ".join(finals)
        else:
            text = " ".join(finals + ([transcript] if transcript else []))
        out.append(
            {
                "audio_ms": round(audio_ms, 1),
                "compute_ms": round(max(0.0, r.wall_ms - audio_ms), 1),
                "text": text,
                "is_final": bool(m.get("is_final")),
                "speech_final": bool(m.get("speech_final")),
            }
        )
    return out


def flux_events(received: list[Received]) -> list[dict]:
    """Flux TurnInfo timeline: confidence on every update, plus its own events."""
    out = []
    for r in received:
        m = r.message
        if m.get("type") != "TurnInfo":
            continue
        out.append(
            {
                "wall_ms": round(r.wall_ms, 1),
                "audio_ms": round(float(m.get("audio_window_end", 0.0)) * 1000.0, 1),
                "event": m.get("event"),
                "turn_index": m.get("turn_index"),
                "eot_confidence": float(m.get("end_of_turn_confidence", 0.0)),
                "trigger": m.get("trigger"),
                "transcript": m.get("transcript", ""),
            }
        )
    return out


# ---- streaming: needs `websockets`; imported lazily so the module loads without it


def _url(base: str, query: dict) -> str:
    from urllib.parse import urlencode

    return f"{base}?{urlencode(query)}"


async def stream_audio(
    url: str, key: str, audio: np.ndarray, sample_rate: int
) -> list[Received]:
    """Open one connection, pace the audio at 1x, collect every message."""
    import websockets

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    chunk = int(sample_rate * CHUNK_MS / 1000.0) * 2
    headers = {"Authorization": f"Token {key}"}
    received: list[Received] = []
    try:
        ctx = websockets.connect(url, additional_headers=headers, max_size=None)
    except TypeError:  # websockets < 13
        ctx = websockets.connect(url, extra_headers=headers, max_size=None)
    async with ctx as ws:
        t0 = time.perf_counter()

        async def sender():
            for i in range(0, len(pcm), chunk):
                await ws.send(pcm[i : i + chunk])
                target = (i + chunk) / 2 / sample_rate
                ahead = target - (time.perf_counter() - t0)
                if ahead > 0:
                    await asyncio.sleep(ahead)
            await ws.send(json.dumps({"type": "CloseStream"}))

        async def receiver():
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    received.append(
                        Received((time.perf_counter() - t0) * 1000.0, json.loads(raw))
                    )
            except Exception as exc:  # noqa: BLE001 -- the close is reported below
                received.append(
                    Received(
                        (time.perf_counter() - t0) * 1000.0,
                        {"type": "_closed", "reason": str(exc)},
                    )
                )

        # A server that never closes after CloseStream would hang the pass:
        # the audio's own length plus a generous tail, then the caller retries.
        budget = len(pcm) / 2 / sample_rate + 30.0
        await asyncio.wait_for(asyncio.gather(sender(), receiver()), timeout=budget)
    return received


async def run_many(
    jobs: list[tuple[str, str, np.ndarray, int]], workers: int, attempts: int = 3
) -> list[list[Received]]:
    """Every job gets `attempts` tries (handshake refusals, dropped sockets); a
    job that still fails returns a single {"type": "_error"} record so the
    caller can refuse to write a cache with holes in it."""
    sem = asyncio.Semaphore(workers)

    async def one(url, key, audio, sr):
        async with sem:
            last = None
            for i in range(attempts):
                try:
                    return await stream_audio(url, key, audio, sr)
                except Exception as exc:  # noqa: BLE001 -- reported, not hidden
                    last = exc
                    await asyncio.sleep(2.0 * (i + 1))
            return [Received(0.0, {"type": "_error", "reason": repr(last)})]

    return await asyncio.gather(*(one(*j) for j in jobs))
