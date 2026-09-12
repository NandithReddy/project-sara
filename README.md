# SARA — Semantic End-of-Turn Detection for Voice Agents

Production voice agents decide that you have stopped speaking by waiting for
silence, typically 500–1000ms. This is wrong in both directions: it cuts you off
when you pause mid-thought ("my order number is… umm…"), and it leaves dead air
after a short complete answer ("yes"). Humans don't do this — we predict turn
completion from linguistic content, not from how long the silence has run.

This project builds a streaming end-of-turn (EOT) classifier that consumes a
partial transcript and emits `P(turn_complete)` on every update, replacing the
fixed silence timer in a cascade voice pipeline.

---

## Status: Phase 6 of 9

**The baselines are measured, the tradeoff curve exists, and the first model
loses to a 800ms silence timer.** The chart, the CSV behind it, and the
reading of both are in [`results/`](results/tradeoff.md). Every number there is
from a run over the frozen eval set; nothing is estimated.

| Phase | | |
|---|---|---|
| 0 | Scaffold, pinned toolchain, test suite | done |
| 1 | Live loop with a fixed-timeout baseline | done |
| 2 | Frozen eval set + harness + baseline numbers | done — 198 turns, 3 baselines |
| 3 | Training data | done — 41,287 examples, labels reviewed |
| 4 | EOT model, text only | **v1 and v1.1 measured: neither beats the timer; the silence gate does** |
| 5 | The tradeoff curve | done — baselines and v1 on it |
| 6 | Telephony band + real-ASR conditions | done — the gold winner does not survive; the model degrades least |
| 7 | Prosody, only if Phase 5 shows headroom | not started |
| 8 | Inbound telephony adapter (optional) | not started |
| 9 | Results write-up | not started |

The README you are reading is a placeholder for Phase 9, which rewrites it from
the contents of `results/`. Until then, every claim here is either a design
decision or a measurement with a file behind it.

---

## What has actually been measured

**The tradeoff curve** — premature-cutoff rate against added latency for two
silence timers, a punctuation heuristic, and the text-only model, on 198 held
out turns: [`results/tradeoff.png`](results/tradeoff.png), read in
[`results/tradeoff.md`](results/tradeoff.md). The headline: the industry
default 500ms timer interrupts 17% of turns; the punctuation heuristic answers
3× faster at the same cutoff rate but its p95 is the fallback wall; and two
iterations of the text-only model are dominated by the timer, for reasons that
are measured rather than guessed. A 200ms silence gate on the punctuation
heuristic was the first system into the chart's empty bottom-left on gold
transcripts — and [`results/conditions.md`](results/conditions.md) shows it
does not survive a real recogniser (11.1% → 19.7% cutoff), while the model
behind the same gate loses 2.5 points and, on the telephony condition, has
fewer cutoffs than the timer for the first time.

**The streaming STT** — whether `parakeet-mlx` can serve on the live path. Full record in
[`results/spike-parakeet-mlx-streaming.md`](results/spike-parakeet-mlx-streaming.md).

The STT was originally chosen over `whisper.cpp` on the argument that its TDT
decoder emits precise native timestamps, while Whisper's are approximated from
cross-attention. Since the entire metric of this project is *when* a decision
fired relative to the true turn end, timestamp error propagates directly into
the headline number — the measuring instrument has to be more precise than the
effect being measured.

Measured on an M4 against 11.00s of 16kHz mono audio, fed paced at 1x real time:

- Compute is a **fixed ~360ms per update**, independent of chunk size. The
  update rate is therefore capped near 2.8/sec, and any chunk under ~400ms
  falls behind real time without bound — at 320ms chunks, lag grew from 2.7s to
  3.7s over 11s of audio.
- Timestamps land on an **80ms grid**: 0 of 44 distinct values were off it.
  That is the TDT frame rate, not a property of the audio.
- Timestamps of already-emitted tokens **mutate under revision**, and there is
  no `is_final` signal, so a consumer cannot tell which tokens have settled.

**The original justification does not survive the measurement.** An 80ms grid
with timings that move is not the precise instrument the argument assumed. This
is recorded rather than quietly corrected, and the STT choice is being revisited
on honest grounds. Eval validity is unaffected — see the path separation below.

---

## Design decisions worth knowing

**The eval path and the live path never share a transcript source.** The eval
harness never calls an STT. It replays force-aligned gold transcripts as a
simulated incremental stream, releasing words at their aligned timestamps. Runs
are therefore deterministic and free, and a regression is always a change in the
model — never STT jitter, network latency, or a vendor shipping a new version
underneath us. The live pipeline uses a real streaming STT and is for
demonstration only; it is never the source of a reported number.

**The eval set is frozen.** Once created it is never trained on, tuned on, or
regenerated. Any code path that reads `data/eval/` during training is a bug, and
there is an assertion that fails loudly.

**A single operating point is not a result.** The deliverable is the full
tradeoff curve of premature-cutoff rate against added latency, for this model
and for every baseline, on the same eval set under the same conditions. Reported
metrics are always `cutoff_rate`, `added_latency_ms` at p50/p95/p99, and
`false_hold_rate` — never a mean alone.

**Inference must run on CPU in under 20ms per update.** A model that needs a GPU
at inference is useless in a phone pipeline. This is a constraint, not a goal,
and it is measured on every eval run. Training may use a GPU; inference may not.

**Known limitation.** Gold transcripts have no word errors, no revised partials,
and perfect timings. Real ASR has all three, and the gap flatters us by an amount
not yet measured. Phase 6 adds a real-ASR condition to quantify it. Until then,
every number carries that caveat.

Full rationale for all of the above is in
[`ENGINEERING.md`](ENGINEERING.md).

---

## Baselines to beat

Implemented before the model, so there is something to lose against:

1. Fixed silence timeout at 500ms and 1000ms — the industry default
2. Silero VAD plus a fixed timeout
3. A punctuation heuristic over the STT transcript
4. Deepgram Flux — commercial fused STT + EOT, eval set only, never in the live
   loop

If the model cannot beat the fixed timeout, that is the finding and it gets
reported as such.

---

## Running it

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The toolchain uses a
uv-managed standalone CPython rather than a system or Homebrew interpreter, so
a `brew upgrade` cannot break the venv underneath the project.

```bash
make install   # create .venv and install pinned deps
make test      # pytest
make lint      # ruff check + format --check
```

There is no application to run yet. That arrives in Phase 1.

---

## Layout

```
src/
  audio/       mic capture, VAD, resampling
  stt/         streaming transcription — live path only
  eot/         the deliverable: model + inference
  pipeline/    live loop
  baselines/   silence timer, punctuation heuristic, black-box systems
eval/
  harness.py   runs a model over the eval set, emits metrics
  sweep.py     threshold sweep, produces the tradeoff curve
data/
  raw/         downloaded corpora (gitignored)
  train/
  eval/        frozen; never touched by training
results/       metrics JSON and charts, committed
```

`eval/harness.py` and `eval/sweep.py` are currently docstring-only stubs naming
the phase that implements them.

---

## Scope

Inbound calls only; no outbound dialing. Any human test subject is told they are
speaking to an automated system and that audio is recorded, before the call
starts. Dataset licenses are recorded in [`data/README.md`](data/README.md)
before download.
