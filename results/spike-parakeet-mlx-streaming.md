# Spike: parakeet-mlx streaming API

**Date:** 2026-08-23 · **Script:** `scripts/spike_parakeet_stream.py`

Ran before Phase 1 to check whether `parakeet-mlx` delivers what CLAUDE.md §5
assumed when it was chosen over `whisper.cpp`. **It does not, on two of three
counts.** Every number below was produced by an actual run; nothing is estimated.

## Setup

| | |
|---|---|
| Machine | Apple M4, 16 GB, macOS 26.5.2 (MLX default device: `gpu`) |
| Versions | `parakeet-mlx` 0.5.2, `mlx` 0.32.1 |
| Model | `mlx-community/parakeet-tdt-0.6b-v3` (cold load incl. download: 88.5s) |
| Audio | `data/raw/jfk.wav` — 11.00s, 16 kHz mono PCM, public domain (US Gov work) |
| Method | Audio fed **paced at 1x real time**. `lag` = wall clock − audio produced so far. |

Final transcript was **correct** at every chunk size ≥ 320ms with default context:
`"And so, my fellow Americans, ask not what your country can do for you, ask what
you can do for your country."`

## Measured

| chunk | context | first-partial lag | revised / updates | timestamp mutations | per-chunk compute p50 |
|---|---|---|---|---|---|
| 320ms | (256,256) | **+2686ms** (grew to +3733ms) | 16 / 34 | 299 | 357.2ms |
| 640ms | (256,256) | +1076ms | 9 / 18 | 160 | 357.6ms |
| 1280ms | (256,256) | +294ms | 3 / 8 | 66 | 368.9ms |
| 320ms | (256,32) | +573ms | 14 / 28 | 135 | 286.9ms |

## Findings

**1. Compute per `add_audio` is ~360ms regardless of chunk size.** It is a fixed
cost per update, not proportional to audio length. Consequences: the update rate
is capped at **~2.8 updates/sec**, and any chunk shorter than ~400ms falls behind
real time *without bound* — at 320ms chunks, lag grew monotonically from 2.7s to
3.7s over 11s of audio. Use chunks ≥ 640ms.

**2. Timestamps are on an 80ms grid.** Across 44 distinct timestamp values in an
unrevised whole-file pass, **zero** were off a 0.08s grid; the smallest gap
between any two distinct values is exactly 0.08s. This is the TDT frame rate
(10ms × 8× subsampling), not a property of the audio.

**3. Timestamps of already-emitted tokens mutate constantly during streaming** —
66 mutations even in the most favourable config. The model re-decodes as context
accumulates, so both text and timings of earlier tokens are rewritten:

```
tok[0] (' And', 0.24, 0.32) -> (' And', 0.32, 0.40)
tok[1] (' so',  0.56, 0.64) -> (' so',  0.56, 0.80)
tok[2] (' am',  0.80, 0.88) -> (' my',  0.96, 1.04)
```

There is **no finality signal** — `transcriber.result` returns the whole current
hypothesis with no `is_final` marker, so a consumer cannot tell which tokens have
settled. This, not raw precision, is the substantive gap versus Deepgram Nova-3.

**4. Tokens are subword, not words.** 12 of 38 tokens are continuation pieces
(`'f'`,`'ell'`,`'ow'` → "fellow"; `'Amer'`,`'ic'`,`'ans'`). Word-level timings
require reassembling on leading-space boundaries. §5's phrase "native word-level
timestamps" is inaccurate as written.

**5. Right context is load-bearing for accuracy.** Cutting it from 256 to 32
frames wrecked the transcript: *"My American, not what your cut drink can do for
you a can do for your country."* Do not tune latency down this way.

## Bearing on §5

The §5 justification — that parakeet-mlx beats whisper.cpp because its decoder
emits precise timings while Whisper's are DTW-approximated — **does not survive
contact with the streaming API**: an 80ms grid with timings that move under
revision is not the precise instrument that argument assumed.

This does **not** invalidate the eval, because §10 keeps the eval path on
force-aligned gold transcripts and off any live STT. It bears on the live demo
path and on Phase 6's real-ASR condition.
