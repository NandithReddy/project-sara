# CLAUDE.md — Semantic End-of-Turn Detection for Voice Agents

This file is read automatically by Claude Code at the start of every session.
It defines what we are building, how we work, and what is forbidden.

---

## 1. The Problem

Production voice agents decide "the user has finished speaking" by waiting for
silence — typically 500–1000ms. This is wrong in two directions:

- **Premature cutoff:** user pauses mid-thought ("my order number is… umm…")
  and the agent interrupts.
- **Dead air:** user gives a short complete answer ("yes") and the agent still
  waits out the full silence timer.

Humans do not do this. We predict turn completion from *linguistic content* —
syntax, semantics, intonation, filler words — not from silence duration alone.

## 2. What We Are Building

A **streaming end-of-turn (EOT) classifier** that consumes a partial transcript
(and later, prosodic features) and emits P(turn_complete) on every update.
It replaces the fixed silence timer in a cascade voice pipeline.

Full pipeline for demo purposes: `mic/phone → VAD → streaming STT → EOT → LLM → TTS`

**The EOT model is the deliverable.** The rest of the pipeline is scaffolding
to demonstrate and measure it. Do not over-invest in the scaffolding.

## 3. Success Criteria

The project succeeds if we can produce **one chart**: the tradeoff curve of
premature-cutoff rate vs. added latency, for our model against baselines,
measured on held-out real conversational audio.

A single operating point is not a result. A curve is.

### Primary metrics (report all, always)

| Metric | Definition |
|---|---|
| `cutoff_rate` | % of turns where model fired before the true end of the user's turn |
| `added_latency_ms` | ms between true turn end and model firing (only for non-cutoff turns) |
| `p50 / p95 / p99` | Latency distribution. **Never report mean alone.** |
| `false_hold_rate` | % of turns where model never fired and hit the fallback timeout |

### Baselines we must beat (implement these first)

1. **Fixed silence timeout** at 500ms and 1000ms (the industry default)
2. **Silero VAD + fixed timeout**
3. **Punctuation heuristic** on the STT transcript

If we cannot beat baseline #1, the project has failed and we say so plainly.

## 4. Non-Negotiable Engineering Rules

These exist because violating them silently invalidates the entire project.

1. **The eval set is frozen and sacred.** Once `data/eval/` is created, it is
   never trained on, never tuned on, never regenerated. Any code path that
   reads `data/eval/` during training is a bug. Add an assertion that enforces
   this and fail loudly.

2. **Never fabricate numbers.** Do not write estimated, illustrative, or
   placeholder metrics into a README, chart, or log line. If a number has not
   been produced by an actual run on actual data, it does not get written down.
   Say "not measured yet."

3. **Never silently mock.** If data is missing or an API key is absent, fail
   loudly with a clear error. Do not substitute synthetic data and continue.
   Every synthetic path must be explicitly named `--synthetic` and must print
   a warning banner.

4. **Never delete or weaken a failing test to make the suite green.** Fix the
   code or report the failure.

5. **Ask before adding a dependency.** Justify it in one line first. We keep
   the runtime dependency list small enough to run on CPU.

6. **Every phase ends with a command I can run** and see output from. No phase
   is "done" until I have run it myself.

7. **Commit at the end of each phase** with a message describing what was
   measured, not just what was written.

## 5. Tech Constraints

- **Python 3.12**, `uv` or `venv`. Pinned versions in `requirements.txt`.
- **EOT inference must run on CPU in <20ms per update.** This is a hard
  constraint, not a goal. A model that needs a GPU at inference is useless
  in a phone pipeline. Measure and log this on every eval run.
- **Sample rate:** develop at 16kHz, but the eval must include an 8kHz
  telephony-band condition. Phone audio is the real deployment target and it
  degrades linguistic cues that clean-audio models rely on.
- Training may use GPU (Colab / rented hour). Inference may not.
- Prefer small encoder models (DistilBERT-class or smaller) over anything
  large. Latency budget dominates accuracy here.

## 6. Repo Layout

```
.
├── CLAUDE.md
├── README.md               # written LAST, from real numbers
├── requirements.txt
├── src/
│   ├── audio/              # mic capture, VAD, resampling
│   ├── stt/                # streaming transcription adapter
│   ├── eot/                # THE DELIVERABLE — model + inference
│   ├── pipeline/           # live loop: mic → … → speaker
│   └── baselines/          # silence timer, punctuation heuristic
├── data/
│   ├── raw/                # downloaded corpora (gitignored)
│   ├── train/
│   └── eval/               # FROZEN. Never touched by training.
├── eval/
│   ├── harness.py          # runs a model over eval set, emits metrics
│   └── sweep.py            # threshold sweep → tradeoff curve
├── scripts/
├── tests/
└── results/                # metrics JSON + charts, committed
```

## 7. How I Want You To Work

- **Explain the plan before writing code.** For any phase, state the approach
  in 5 bullets and wait for my go-ahead.
- **Small diffs.** I need to understand every file. If a change touches more
  than ~3 files, stop and check with me.
- **Teach as you go.** When you make a non-obvious technical choice, add one
  sentence on *why* and what the alternative was. I need to be able to defend
  every decision in an interview.
- **Flag your own uncertainty.** If you are guessing at an API signature or a
  corpus format, say so and verify rather than assuming.
- **Push back on me.** If I ask for something that is a bad idea, say so.

## 8. Known Prior Art (do not reinvent; do differentiate)

Silero VAD, Pipecat Smart Turn v2, TEN Turn Detection, Turnsense,
Speechmatics EOT, OpenAI `semantic_vad`.

Most are English-centric and evaluated on clean audio. **Our differentiation
is evaluation under realistic conditions** — telephony band, disfluencies,
and (if data is available) code-switched speech — plus publishing the full
tradeoff curve rather than a single operating point.

## 9. Ethics / Legal

- Inbound calls only. No outbound dialing to third parties.
- Any human test subject is told they are talking to an AI and that audio is
  recorded, before the call starts.
- Respect corpus licenses. Record the license of every dataset in
  `data/README.md` before downloading it.
## 10. Eval Path vs. Live Path (architectural decision)

**The eval path and the live path are separate. They do not share a transcript
source.**

**Eval path — the eval harness never calls an STT.** It replays force-aligned
gold transcripts from the corpus as a *simulated incremental stream*: words are
released at their aligned timestamps to imitate what a streaming recognizer
would have emitted. Metrics are therefore deterministic, free, and reproducible
— the same eval run twice produces byte-identical numbers, and a regression is
always a change in our model, never STT jitter, network latency, or a vendor
silently shipping a new model version.

**Live path — the live pipeline uses a real streaming STT.** It exists only for
demonstration and qualitative testing. It is never the source of a reported
number.

**Consequence:** any import of `src/stt/` from `eval/` is a bug, in the same way
that reading `data/eval/` during training is a bug (rule 1). Assert it and fail
loudly.

**Known limitation — the optimism gap.** Gold transcripts have no word errors,
no revised partials, and perfect timings. Real ASR has all three, and the gap
flatters us by an amount we have not measured. **Phase 6 adds a "real ASR
output" eval condition** that runs the same eval audio through the live STT and
re-reports every metric, to quantify that gap. Until Phase 6 runs, every number
we publish carries the caveat that it is measured on gold transcripts.
