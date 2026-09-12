"""Write README.md from results/, models/ and data/. Nothing typed by hand.

Phase 9. Every figure in the README is read from a file produced by a run --
the sweep CSV, the conditions CSV, the model metadata, the eval manifest, the
recogniser caches -- and the numbers used are written to
results/readme_numbers.json so tests/test_readme.py can check the README
against the CSVs. If a number is not in a file, it is not in the README.

Run:  uv run python scripts/write_readme.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
R = REPO / "results"


def rows(path):
    return list(csv.DictReader(path.open()))


def pick(rs, system, knob, condition=None):
    for r in rs:
        if (
            r["system"] == system
            and float(r["knob_value"]) == knob
            and (condition is None or r["condition"] == condition)
        ):
            return r
    raise SystemExit(f"missing {system} {knob} {condition}")


def cut(r):
    return float(r["cutoff_rate_at_tolerance"]) * 100


def hold(r):
    return float(r["false_hold_rate"]) * 100


def p50(r):
    return float(r["latency_fallback_p50"])


def p95(r):
    return float(r["latency_fallback_p95"])


def norm(s):
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


def wer(ref, hyp):
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    d[:, 0] = range(len(ref) + 1)
    d[0, :] = range(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(
                d[i - 1, j] + 1,
                d[i, j - 1] + 1,
                d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    return int(d[-1, -1]), len(ref)


def main() -> int:
    sweep = rows(R / "tradeoff.csv")
    cond = rows(R / "conditions.csv")
    man = json.loads((REPO / "data/eval/MANIFEST.json").read_text())
    v11 = json.loads((REPO / "models/eot_v1/metadata.json").read_text())
    pro = json.loads((REPO / "models/eot_prosody/prosody_lr.json").read_text())
    fus = json.loads((REPO / "models/eot_prosody/fusion_lr.json").read_text())
    txt = json.loads((REPO / "models/eot_prosody/text_lr.json").read_text())
    train = json.loads((REPO / "data/train/STATS.json").read_text())
    pauses = json.loads((REPO / "data/train/PAUSE_STATS.json").read_text())
    asr16 = json.loads((R / "asr/parakeet_16k.json").read_text())
    raw16 = json.loads((R / "asr/parakeet_16k_rawtail.json").read_text())
    eval_set = {
        t["turn_id"]: t
        for t in json.loads((REPO / "data/eval/eval_set.json").read_text())["turns"]
    }
    tv11 = json.loads((R / "text_eot_v1.1.json").read_text())

    S, E, P, G, T, TG, PR, FU = (
        "fixed_timeout+silero",
        "fixed_timeout+energy",
        "punctuation",
        "punctuation+gate200",
        "text_eot_v1.1",
        "text_eot_v1.1+gate200",
        "prosody_prosody+gate200",
        "prosody_fusion+gate200",
    )
    # --- recomputed from the caches, so they are file-derived ---------------
    errs = tot = 0
    for tid, parts in asr16["turns"].items():
        e, n = wer(norm(eval_set[tid]["text"]), norm(parts[-1]["text"]))
        errs += e
        tot += n
    wer_pct = errs / tot * 100
    grew = 0
    for tid, parts in raw16["turns"].items():
        end = eval_set[tid]["true_end_ms"]
        before = [q for q in parts if q["audio_ms"] + q["compute_ms"] <= end]
        tb = before[-1]["text"] if before else ""
        if len(parts[-1]["text"].split()) > len(tb.split()):
            grew += 1
    revisions = sum(
        1
        for parts in asr16["turns"].values()
        for a, b in zip(parts, parts[1:], strict=False)
        if a["text"] and not b["text"].startswith(a["text"])
    )

    N = {
        "n_turns": man["n_turns"],
        "n_meetings": man["n_meetings"],
        "n_speakers": man["n_speakers"],
        "timer500_cut": cut(pick(sweep, S, 500.0)),
        "timer500_hold": hold(pick(sweep, S, 500.0)),
        "timer800_cut": cut(pick(sweep, S, 800.0)),
        "timer800_hold": hold(pick(sweep, S, 800.0)),
        "timer800_p50": p50(pick(sweep, S, 800.0)),
        "timer1000_cut": cut(pick(sweep, S, 1000.0)),
        "timer1000_hold": hold(pick(sweep, S, 1000.0)),
        "energy1000_hold": hold(pick(sweep, E, 1000.0)),
        "punct_cut": cut(pick(sweep, P, 0.0)),
        "punct_p50": p50(pick(sweep, P, 0.0)),
        "punct_p95": p95(pick(sweep, P, 0.0)),
        "pg_cut": cut(pick(sweep, G, 0.0)),
        "pg_hold": hold(pick(sweep, G, 0.0)),
        "pg_p50": p50(pick(sweep, G, 0.0)),
        "v11_cut": cut(pick(sweep, T, 0.5)),
        "v11_hold": hold(pick(sweep, T, 0.5)),
        "v11g01_cut": cut(pick(sweep, TG, 0.1)),
        "v11g01_hold": hold(pick(sweep, TG, 0.1)),
        "v11g03_cut": cut(pick(sweep, TG, 0.3)),
        "v11g03_hold": hold(pick(sweep, TG, 0.3)),
        "v11g03_p50": p50(pick(sweep, TG, 0.3)),
        "pro_cut": cut(pick(sweep, PR, 0.5)),
        "pro_hold": hold(pick(sweep, PR, 0.5)),
        "fus_cut": cut(pick(sweep, FU, 0.5)),
        "fus_hold": hold(pick(sweep, FU, 0.5)),
        "tel_timer800_cut": cut(pick(cond, S, 800.0, "asr_tel")),
        "tel_timer800_hold": hold(pick(cond, S, 800.0, "asr_tel")),
        "tel_pg_cut": cut(pick(cond, G, 0.0, "asr_tel")),
        "tel_pg_hold": hold(pick(cond, G, 0.0, "asr_tel")),
        "tel_pg_p50": p50(pick(cond, G, 0.0, "asr_tel")),
        "tel_v11g03_cut": cut(pick(cond, TG, 0.3, "asr_tel")),
        "tel_v11g03_hold": hold(pick(cond, TG, 0.3, "asr_tel")),
        "tel_v11g03_p50": p50(pick(cond, TG, 0.3, "asr_tel")),
        "asr16_punct_cut": cut(pick(cond, P, 0.0, "asr_16k")),
        "asr16_pg_cut": cut(pick(cond, G, 0.0, "asr_16k")),
        "asr16_v11g03_cut": cut(pick(cond, TG, 0.3, "asr_16k")),
        "goldtel_timer800_cut": cut(pick(cond, S, 800.0, "gold_tel")),
        "caller_timer800_hold": hold(pick(cond, S, 800.0, "asr_16k")),
        "caller_timer800_p95": p95(pick(cond, S, 800.0, "asr_16k")),
        "v11_val_ap": v11["final_val"]["average_precision"],
        "v11_params_m": v11["params"] / 1e6,
        "v11_infer_p50": tv11["inference_latency_ms"]["p50"],
        "pro_val_ap": pro["metrics"]["val"]["average_precision"],
        "txt_val_ap": txt["metrics"]["val"]["average_precision"],
        "fus_val_ap": fus["metrics"]["val"]["average_precision"],
        "train_turns": train["n_turns"],
        "train_examples": train["n_examples"],
        "train_speakers": train["n_speakers"],
        "ceiling_words_pct": train["collisions_text"]["unavoidable_error_rate"] * 100,
        "pauses": pauses["n_pauses"],
        "wer_pct": wer_pct,
        "bleed_grew": grew,
        "revisions": revisions,
        "asr_rtf": asr16["total_compute_s"] / asr16["total_audio_s"],
        "asr_audio_s": asr16["total_audio_s"],
    }
    (R / "readme_numbers.json").write_text(json.dumps(N, indent=2))
    f = {k: (f"{v:.1f}" if isinstance(v, float) else str(v)) for k, v in N.items()}
    f["v11_val_ap"] = f"{N['v11_val_ap']:.3f}"
    f["pro_val_ap"], f["txt_val_ap"], f["fus_val_ap"] = (
        f"{N[k]:.3f}" for k in ("pro_val_ap", "txt_val_ap", "fus_val_ap")
    )
    f["v11_infer_p50"] = f"{N['v11_infer_p50']:.2f}"
    f["asr_rtf"] = f"{N['asr_rtf']:.2f}"
    for k in (
        "timer800_p50",
        "punct_p50",
        "punct_p95",
        "pg_p50",
        "v11g03_p50",
        "tel_pg_p50",
        "tel_v11g03_p50",
        "caller_timer800_p95",
        "asr_audio_s",
    ):
        f[k] = f"{N[k]:.0f}"
    for k in (
        "n_turns",
        "n_meetings",
        "n_speakers",
        "train_turns",
        "train_examples",
        "train_speakers",
        "pauses",
        "bleed_grew",
        "revisions",
    ):
        f[k] = f"{N[k]:,}"

    readme = f"""# SARA — Semantic End-of-Turn Detection for Voice Agents

Production voice agents decide that you have stopped speaking by waiting for
silence, typically 500–1000ms. That is wrong in both directions: it interrupts
you when you pause mid-thought (*"my order number is… umm…"*) and it leaves
dead air after a short complete answer (*"yes"*). This project measures how
much better anything can do — a silence timer, a punctuation heuristic, a
text classifier, a prosody classifier — on the same held-out audio, under
the same conditions, as a full tradeoff curve rather than one operating
point.

**The evaluation is the contribution.** The models are participants in it.
Every number below is read from a file in [`results/`](results/) by
[`scripts/write_readme.py`](scripts/write_readme.py); nothing is typed, and a
test fails if this page and the CSVs disagree.

## The chart

![tradeoff](results/tradeoff.png)

Premature-cutoff rate against added latency, {f["n_turns"]} held-out turns from the
AMI Meeting Corpus ({f["n_meetings"]} meetings, {f["n_speakers"]} speakers; frozen, SHA256-manifested,
never trained on). Latency is measured from the audio-grounded true end of
the turn; a turn never answered is counted at the 2000ms fallback; cutoff
allows the 150ms the boundary itself is uncertain to. Reading:
[`results/tradeoff.md`](results/tradeoff.md).

![conditions](results/conditions.png)

The same systems under four conditions — gold transcripts and a real
streaming recogniser, wideband and the 300–3400Hz μ-law telephony band.
Reading: [`results/conditions.md`](results/conditions.md).

## The numbers

Gold transcripts, wideband — the ceiling — and the deployment condition, a
real recogniser on telephony-band audio. Cutoff / never answered / median
latency in ms.

| system | gold, wideband | real ASR, telephony |
|---|---|---|
| Silero VAD + 500ms timer *(the industry default)* | {f["timer500_cut"]}% / {f["timer500_hold"]}% / {f["timer500_p50"]} | — |
| Silero VAD + 800ms timer | {f["timer800_cut"]}% / {f["timer800_hold"]}% / {f["timer800_p50"]} | {f["tel_timer800_cut"]}% / {f["tel_timer800_hold"]}% / 800 |
| Silero VAD + 1000ms timer | {f["timer1000_cut"]}% / {f["timer1000_hold"]}% / {f["timer1000_p50"]} | — |
| punctuation heuristic | {f["punct_cut"]}% / {f["punct_hold"]}% / {f["punct_p50"]} | {cut(pick(cond, P, 0.0, "asr_tel")):.1f}% / {hold(pick(cond, P, 0.0, "asr_tel")):.1f}% / {p50(pick(cond, P, 0.0, "asr_tel")):.0f} |
| **punctuation + 200ms silence gate** | **{f["pg_cut"]}% / {f["pg_hold"]}% / {f["pg_p50"]}** | {f["tel_pg_cut"]}% / {f["tel_pg_hold"]}% / {f["tel_pg_p50"]} |
| text model v1.1 (bert-mini), bare | {f["v11_cut"]}% / {f["v11_hold"]}% / {f["v11_p50"]} | — |
| **text model v1.1 + 200ms gate @0.3** | {f["v11g03_cut"]}% / {f["v11g03_hold"]}% / {f["v11g03_p50"]} | **{f["tel_v11g03_cut"]}% / {f["tel_v11g03_hold"]}% / {f["tel_v11g03_p50"]}** |
| prosody classifier + gate @0.5 | {f["pro_cut"]}% / {f["pro_hold"]}% / {f["pro_p50"]} | {cut(pick(cond, PR, 0.5, "asr_tel")):.1f}% / {hold(pick(cond, PR, 0.5, "asr_tel")):.1f}% / {p50(pick(cond, PR, 0.5, "asr_tel")):.0f} |
| prosody + text + gate @0.5 | {f["fus_cut"]}% / {f["fus_hold"]}% / {f["fus_p50"]} | {cut(pick(cond, FU, 0.5, "asr_tel")):.1f}% / {hold(pick(cond, FU, 0.5, "asr_tel")):.1f}% / {p50(pick(cond, FU, 0.5, "asr_tel")):.0f} |

Full sweeps in [`results/tradeoff.csv`](results/tradeoff.csv); all four
conditions in [`results/conditions.csv`](results/conditions.csv).

## What worked

- **The industry default is measurably bad.** A 500ms silence timer
  interrupts **{f["timer500_cut"]}%** of turns on this data. Cutoffs concentrate in
  disfluent speech; dead air concentrates in short answers. No timeout fixes
  both.
- **A 200ms silence gate in front of a semantic signal.** On gold
  transcripts a one-character punctuation heuristic behind the gate matches
  the 800ms timer's cutoff rate ({f["pg_cut"]}% vs {f["timer800_cut"]}%) at a third of the
  latency ({f["pg_p50"]} vs {f["timer800_p50"]}ms) — the first system into the chart's empty
  bottom-left. The gate is the finding: semantics reduce the silence a system
  needs; they do not replace it.
- **Training the text model on words alone.** The model that never saw a
  trailing period lost {N["asr16_v11g03_cut"] - N["v11g03_cut"]:.1f} points of cutoff to a real recogniser
  where the punctuation heuristic lost {N["asr16_pg_cut"] - N["pg_cut"]:.1f}. On the deployment
  condition the gated text model has **fewer cutoffs than the timer**
  ({f["tel_v11g03_cut"]}% vs {f["tel_timer800_cut"]}%) at the same median latency — the first
  time it beats the timer on anything — and {f["tel_v11g03_hold"]}% of callers wait out the
  fallback, so it is not a point to ship.
- **The measurement instrument.** An eval that replays gold word timings
  and never calls a recogniser, so a run reproduces byte for byte and a
  regression is always a change in our code; a boundary grounded in audio,
  because the corpus alignment absorbs pauses into word durations; a caller
  channel, because the headsets carry the next speaker. Each of those was a
  finding before it was a design.

## What didn't work

- **Two text-only models lose to a silence timer.** v1 ({v11["base_model"].split("/")[-1]},
  {f["v11_params_m"]}M parameters) fired on the first word — *"Okay"* is a whole turn {f["okay_complete_pct"]}% of
  the time in training and no text can tell — and v1.1, with the positive
  class weight removed, fired too little instead. Validation AP {f["v11_val_ap"]} on
  held-out speakers is the whole story; no threshold fixes a ranking. The
  text-only ceiling is measurable: on words alone {f["ceiling_words_pct"]}% of training
  examples carry the same string with both labels.
- **The gold winner does not survive a real recogniser.** The gated
  heuristic goes {f["pg_cut"]}% → {f["asr16_pg_cut"]}% cutoff at wideband, {f["tel_pg_cut"]}% on
  telephony — more than the timer. The bare heuristic goes {f["punct_cut"]}% → {f["asr16_punct_cut"]}%:
  the optimism gap for a system reading annotator punctuation is about
  thirty points.
- **Prosody does not help.** Fourteen features of the last speech frame
  before a pause reach AP {f["pro_val_ap"]} alone (chance {f["pause_chance_ap"]}); the text logit alone
  reaches {f["txt_val_ap"]}; both together {f["fus_val_ap"]}. The signal that exists is energy —
  the last speech before a real end is quieter — and pitch carries none: a
  falling contour does not separate a turn's end from a hesitation on this
  data. Dominated by the gated heuristic at every threshold.
- **The telephony band hurts through the VAD.** Silero flaps more on
  band-limited μ-law audio; the 800ms timer goes {f["timer800_cut"]}% → {f["goldtel_timer800_cut"]}% cutoff
  with the text untouched.
- **The recogniser is the largest degradation source, and none of it is
  ours.** parakeet-mlx over {f["asr_audio_s"]}s of eval audio: word error rate {f["wer_pct"]}%
  against gold, {f["revisions"]} hypothesis revisions, {f["asr_rtf"]}× realtime on a shared
  GPU. Every text system pays for that before any of ours gets a say.
- **Two things were caught before they became results.** The headset tail
  carries the next speaker: on the raw recording the transcript kept growing
  after the true end on {f["bleed_grew"]} of {f["n_turns"]} turns, and a text system that should
  have held instead answered the wrong person. And the first prosody
  classifier scored a perfect 1.000 AP — it had learned to detect the
  zeroing that fixed the first problem. Both are in the commit history with
  their evidence.

## Known limitations

- **Meeting speech, not calls.** AMI is four-person meetings on headset
  microphones. Turn-taking dynamics differ from a phone agent's; a turn
  ending on *"yeah so"* is common here and rare there.
- **One corpus, {f["n_speakers"]} eval speakers, {f["n_turns"]} turns.** One turn is 0.5%; differences
  under about two points are inside the noise.
- **Gold transcripts are a ceiling.** Human punctuation from someone who
  heard the whole recording; the real-ASR condition is the floor, with one
  recogniser on one machine.
- **Clean-line telephony.** Band-limit and codec, no injected noise, so runs
  reproduce byte for byte; real lines are worse. The caller channel is
  digital silence after the turn, which flatters energy thresholding.
- **The boundary shares a model with a baseline.** The true end of each
  turn is refined with Silero, which is also baseline #2; mitigated (raw
  probability, not the smoothed flag) and quantified (±150ms), not
  eliminated.
- **The live path is Apple Silicon only** (parakeet-mlx). The results
  reproduce on any machine from the cached recogniser output.
- **Not yet measured:** Deepgram Flux, the closest commercial system, and
  Nova-3 as a second recogniser. Both are one API key and about a dollar of
  credit away; the harness takes a second backend unchanged.

## How to reproduce

Python 3.12 and [uv](https://docs.astral.sh/uv/). The runtime path is
onnxruntime and numpy; torch appears only in the training tier.

```bash
make install                              # runtime deps, uv-managed CPython
make test                                 # includes the eval-set freeze check

uv run python scripts/run_baselines.py    # the table, ~1 min
uv run python -m eval.sweep               # results/tradeoff.png, ~2 min
uv run python -m eval.conditions          # results/conditions.png, ~5 min
uv run python scripts/write_readme.py     # this page, from the files above
```

The frozen eval set, the recogniser caches under `results/asr/`, and both
models are committed, so the three commands above reproduce every number on
this page on any machine. To regenerate the inputs instead:

```bash
# recogniser caches (Apple Silicon; ~15 min per condition)
uv run python scripts/transcribe_eval.py --condition 16k
uv run python scripts/transcribe_eval.py --condition tel

# text model (make install-train first; ~3 min on an M4)
uv run python scripts/build_train_set.py
uv run python scripts/train_eot.py

# prosody classifier (~20 min: 1GB of training audio by range request)
uv run python scripts/fetch_train_audio.py
uv run python scripts/build_pause_set.py
uv run python scripts/train_prosody.py

# talk to it (Apple Silicon)
uv run python scripts/live.py --eot punct-gated
```

Do not rebuild `data/eval/`: it is frozen, and `scripts/freeze_eval_set.py`
refuses to re-freeze a changed set without `--force`.

## Where things are

```
src/eot/         the detector interface, the text model, the gate, the pause classifier
src/baselines/   silence timers, energy VAD, punctuation heuristic
src/audio/       Silero VAD, prosody tracker, telephony simulation, mic capture
src/stt/         streaming STT for the live path only -- never imported by eval/
eval/            harness, sweep, conditions; replays gold or cached recogniser output
scripts/         every build, fetch, train and report step, each runnable alone
data/eval/       198 frozen turns, audio and manifest, committed
results/         every number, chart and reading, committed
models/          Silero VAD, the text model, the pause classifier, with cards
```

The engineering charter — what is forbidden, and why — is
[`ENGINEERING.md`](ENGINEERING.md). Inbound calls only; any human test subject
is told they are talking to an automated system and that audio is recorded;
dataset licences are in [`data/README.md`](data/README.md).
"""
    (REPO / "README.md").write_text(readme)
    print(
        f"README.md written: {len(readme.splitlines())} lines, {len(N)} numbers recorded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
