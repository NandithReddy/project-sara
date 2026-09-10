"""Phase 1 live loop: mic -> VAD -> STT -> EOT -> TTS.

Demonstration only. No number reported by this project comes from this file
(section 10) -- the eval harness replays gold transcripts and never runs an STT.

Prints, on every event, both clocks:
  audio  -- ms of audio processed, the clock the EOT decision actually sees
  wall   -- ms since the loop started, the clock the user actually feels
They diverge when the machine falls behind real time, and that divergence is
exactly what the parakeet spike was measuring.
"""

from __future__ import annotations

import math
import subprocess
import time
from collections.abc import Iterator

import numpy as np

from src.audio.capture import DEFAULT_BLOCK_MS, mic_blocks
from src.audio.vad import SileroVAD
from src.eot.base import EOTDetector, Update
from src.stt.parakeet import ParakeetStream

REPLY = "Got it, thanks."
SAY_BIN = "/usr/bin/say"


def speak(text: str) -> subprocess.Popen:
    """Start TTS. Returns as soon as the process is spawned.

    The timestamp we log is when `say` was launched, not when a speaker
    membrane moved -- true playback onset happens inside CoreAudio and is not
    observable from here. Logged honestly as "TTS spawned" for that reason.
    """
    return subprocess.Popen(
        [SAY_BIN, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def run(
    eot: EOTDetector,
    seconds: float = 60.0,
    threshold: float = 0.5,
    reply: str = REPLY,
    source: Iterator[np.ndarray] | None = None,
) -> int:
    """Run the loop until `seconds` elapse or the user interrupts."""
    vad = SileroVAD(threshold=threshold)
    stt = ParakeetStream()

    print(f"loading {stt.model_name} ...")
    load_s = stt.start()
    print(f"model ready in {load_s:.2f}s")

    eot.reset()
    blocks = source if source is not None else mic_blocks()

    print(f"EOT rule: {eot.name}")
    print(f"Speak, then pause. {seconds:.0f}s, Ctrl-C to stop.")
    print("-" * 78)

    wall0 = time.perf_counter()

    def stamp(audio_ms: float) -> str:
        wall_ms = (time.perf_counter() - wall0) * 1000.0
        return f"[audio {audio_ms:7.0f}ms | wall {wall_ms:7.0f}ms]"

    turn_active = False
    last_seq = -1
    turns = 0
    audio_ms = 0.0

    try:
        for block in blocks:
            stt.push(block)

            for f in vad.push(block):
                audio_ms = f.t_ms
                if f.is_speech and not turn_active:
                    turn_active = True
                    print(f"{stamp(audio_ms)}  speech starts")

                if not turn_active:
                    continue

                hyp = stt.latest()
                if hyp.seq != last_seq and hyp.text:
                    last_seq = hyp.seq
                    print(f"{stamp(audio_ms)}  partial: {hyp.text!r}")

                if (
                    eot.update(Update(audio_ms, hyp.text, silence_ms=f.silence_ms))
                    < 0.5
                ):
                    continue

                # --- end of turn ---
                turn_active = False
                turns += 1
                fired_wall = time.perf_counter()
                print(
                    f"{stamp(audio_ms)}  *** EOT FIRED *** "
                    f"after {f.silence_ms:.0f}ms silence"
                )
                print(f"{stamp(audio_ms)}  heard: {stt.latest().text!r}")

                proc = speak(reply)
                print(
                    f"{stamp(audio_ms)}  TTS spawned "
                    f"(+{(time.perf_counter() - fired_wall) * 1000:.0f}ms "
                    f"after the decision): {reply!r}"
                )
                proc.wait()
                spoken_s = time.perf_counter() - fired_wall

                # We have no echo cancellation, so the mic captured our own
                # reply. Drop roughly the blocks that arrived while speaking,
                # then start the next turn clean. Half-duplex gating; a real
                # deployment needs AEC.
                for _ in range(math.ceil(spoken_s * 1000.0 / DEFAULT_BLOCK_MS)):
                    next(blocks, None)

                vad.reset()
                stt.reset()
                eot.reset()
                last_seq = -1
                print("-" * 78)

            if time.perf_counter() - wall0 > seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stt.close()

    print(f"\n{'-' * 78}\nturns completed: {turns}")
    return 0
