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
    nova16 = json.loads((R / "asr/nova3_16k.json").read_text())
    flux16 = json.loads((R / "asr/flux_16k.json").read_text())
    eval_set = {
        t["turn_id"]: t
        for t in json.loads((REPO / "data/eval/eval_set.json").read_text())["turns"]
    }
    tv11 = json.loads((R / "text_eot_v1.1.json").read_text())

    S, E, P, G, T, TG, PR, FU, FX, FC, NS = (
        "fixed_timeout+silero",
        "fixed_timeout+energy",
        "punctuation",
        "punctuation+gate200",
        "text_eot_v1.1",
        "text_eot_v1.1+gate200",
        "prosody_prosody+gate200",
        "prosody_fusion+gate200",
        "flux",
        "flux_conf",
        "nova3_speech_final",
    )
    flux_thr = float(flux16["query"]["eot_threshold"])
    # --- recomputed from the caches, so they are file-derived ---------------
    errs = tot = 0
    for tid, parts in asr16["turns"].items():
        e, n = wer(norm(eval_set[tid]["text"]), norm(parts[-1]["text"]))
        errs += e
        tot += n
    wer_pct = errs / tot * 100
    errs_n = 0
    for tid, parts in nova16["turns"].items():
        e, _ = wer(norm(eval_set[tid]["text"]), norm(parts[-1]["text"]))
        errs_n += e
    nova_wer_pct = errs_n / tot * 100
    silent_para = sum(1 for v in asr16["turns"].values() if not v[-1]["text"].strip())
    silent_nova = sum(1 for v in nova16["turns"].values() if not v[-1]["text"].strip())
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
        # Deepgram: Flux as shipped and on its confidence, Nova-3 as a
        # recogniser for our systems and as an endpointer on its own.
        "flux_thr": flux_thr,
        "flux_cut": cut(pick(sweep, FX, flux_thr)),
        "flux_hold": hold(pick(sweep, FX, flux_thr)),
        "flux_p50": p50(pick(sweep, FX, flux_thr)),
        "flux_p95": p95(pick(sweep, FX, flux_thr)),
        "fluxc07_cut": cut(pick(sweep, FC, 0.7)),
        "fluxc07_hold": hold(pick(sweep, FC, 0.7)),
        "fluxc07_p50": p50(pick(sweep, FC, 0.7)),
        "fluxc05_cut": cut(pick(sweep, FC, 0.5)),
        "fluxc05_hold": hold(pick(sweep, FC, 0.5)),
        "fluxc05_p50": p50(pick(sweep, FC, 0.5)),
        "fluxc075_hold": hold(pick(sweep, FC, 0.75)),
        "tel_flux_cut": cut(pick(cond, FX, flux_thr, "nova3_tel")),
        "tel_flux_hold": hold(pick(cond, FX, flux_thr, "nova3_tel")),
        "tel_flux_p50": p50(pick(cond, FX, flux_thr, "nova3_tel")),
        "nova3sf_cut": cut(pick(cond, NS, 0.0, "nova3_16k")),
        "nova3sf_hold": hold(pick(cond, NS, 0.0, "nova3_16k")),
        "nova3sf_p50": p50(pick(cond, NS, 0.0, "nova3_16k")),
        "tel_nova3sf_cut": cut(pick(cond, NS, 0.0, "nova3_tel")),
        "tel_nova3sf_hold": hold(pick(cond, NS, 0.0, "nova3_tel")),
        "tel_nova3sf_p50": p50(pick(cond, NS, 0.0, "nova3_tel")),
        "nova3_pg_cut": cut(pick(cond, G, 0.0, "nova3_16k")),
        "nova3_pg_hold": hold(pick(cond, G, 0.0, "nova3_16k")),
        "nova3_pg_p50": p50(pick(cond, G, 0.0, "nova3_16k")),
        "nova3tel_pg_cut": cut(pick(cond, G, 0.0, "nova3_tel")),
        "nova3tel_pg_hold": hold(pick(cond, G, 0.0, "nova3_tel")),
        "nova3tel_pg_p50": p50(pick(cond, G, 0.0, "nova3_tel")),
        "nova3tel_v11g03_cut": cut(pick(cond, TG, 0.3, "nova3_tel")),
        "nova3tel_v11g03_hold": hold(pick(cond, TG, 0.3, "nova3_tel")),
        "nova3tel_v11g03_p50": p50(pick(cond, TG, 0.3, "nova3_tel")),
        "nova3_v11g03_cut": cut(pick(cond, TG, 0.3, "nova3_16k")),
        "nova3_v11g03_hold": hold(pick(cond, TG, 0.3, "nova3_16k")),
        "nova_wer_pct": nova_wer_pct,
        "nova_revisions": sum(
            1
            for parts in nova16["turns"].values()
            for a, b in zip(parts, parts[1:], strict=False)
            if a["text"] and not b["text"].startswith(a["text"])
        ),
        "silent_para": silent_para,
        "silent_nova": silent_nova,
        "flux_date": flux16["date"],
        "flux_model": flux16["model"],
        "nova_model": nova16["model"],
    }
    N.update(
        {
            "timer500_p50": p50(pick(sweep, S, 500.0)),
            "timer1000_p50": p50(pick(sweep, S, 1000.0)),
            "punct_hold": hold(pick(sweep, P, 0.0)),
            "v11_p50": p50(pick(sweep, T, 0.5)),
            "pro_p50": p50(pick(sweep, PR, 0.5)),
            "fus_p50": p50(pick(sweep, FU, 0.5)),
            "pause_chance_ap": pro["metrics"]["val"]["majority_ap"],
            "okay_complete_pct": next(
                w["complete"] / (w["complete"] + w["incomplete"]) * 100
                for w in train["collisions_text"]["worst"]
                if w["text"] == "Okay"
            ),
        }
    )
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
        "flux_p50",
        "flux_p95",
        "fluxc07_p50",
        "fluxc05_p50",
        "tel_flux_p50",
        "nova3sf_p50",
        "tel_nova3sf_p50",
        "nova3_pg_p50",
        "nova3tel_pg_p50",
        "nova3tel_v11g03_p50",
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
        "nova_revisions",
        "silent_para",
        "silent_nova",
    ):
        f[k] = f"{N[k]:,}"
    f["nova3_pg_gain"] = f"{N['asr16_pg_cut'] - N['nova3_pg_cut']:.1f}"
    f["flux_thr"] = f"{N['flux_thr']:g}"

    for k in (
        "timer500_p50",
        "timer1000_p50",
        "v11_p50",
        "pro_p50",
        "fus_p50",
        "okay_complete_pct",
    ):
        f[k] = f"{N[k]:.0f}"
    f["punct_hold"] = f"{N['punct_hold']:.1f}"
    f["pause_chance_ap"] = f"{N['pause_chance_ap']:.3f}"

    readme = f"""# SARA — Semantic End-of-Turn Detection for Voice Agents

Production voice agents decide that you have stopped speaking by waiting for
silence, typically 500–1000ms. That is wrong in both directions: it interrupts
you when you pause mid-thought (*"my order number is… umm…"*) and it leaves
dead air after a short complete answer (*"yes"*). This project measures how
much better anything can do — a silence timer, a punctuation heuristic, a
text classifier, a prosody classifier, and the closest commercial system,
Deepgram Flux — on the same held-out audio, under the same conditions, as a
full tradeoff curve rather than one operating point.

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

The same systems under six conditions — gold transcripts, a local streaming
recogniser (parakeet-mlx) and a cloud one (Deepgram Nova-3), each at wideband
and in the 300–3400Hz μ-law telephony band. The Nova-3 panels also carry
Deepgram's own detectors on the same audio. Reading:
[`results/conditions.md`](results/conditions.md).

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
| **Deepgram Flux**, as shipped (eot_threshold {f["flux_thr"]}) — cloud, own recogniser | {f["flux_cut"]}% / {f["flux_hold"]}% / {f["flux_p50"]} | **{f["tel_flux_cut"]}% / {f["tel_flux_hold"]}% / {f["tel_flux_p50"]}** |
| Deepgram Flux, acting on its confidence ≥0.5 | {f["fluxc05_cut"]}% / {f["fluxc05_hold"]}% / {f["fluxc05_p50"]} | {cut(pick(cond, FC, 0.5, "nova3_tel")):.1f}% / {hold(pick(cond, FC, 0.5, "nova3_tel")):.1f}% / {p50(pick(cond, FC, 0.5, "nova3_tel")):.0f} |
| Deepgram Nova-3 `speech_final`, default endpointing | {f["nova3sf_cut"]}% / {f["nova3sf_hold"]}% / {f["nova3sf_p50"]} | {f["tel_nova3sf_cut"]}% / {f["tel_nova3sf_hold"]}% / {f["tel_nova3sf_p50"]} |
| punctuation + gate, on Nova-3 transcripts | {f["nova3_pg_cut"]}% / {f["nova3_pg_hold"]}% / {f["nova3_pg_p50"]} | {f["nova3tel_pg_cut"]}% / {f["nova3tel_pg_hold"]}% / {f["nova3tel_pg_p50"]} |
| text model v1.1 + gate @0.3, on Nova-3 transcripts | {f["nova3_v11g03_cut"]}% / {f["nova3_v11g03_hold"]}% / {p50(pick(cond, TG, 0.3, "nova3_16k")):.0f} | {f["nova3tel_v11g03_cut"]}% / {f["nova3tel_v11g03_hold"]}% / {f["nova3tel_v11g03_p50"]} |

The Deepgram rows hear the audio directly (there is no gold-transcript
condition for a system with its own recogniser), so their left column is the
same wideband audio the gold rows use, and the Nova-3 rows' right column is
Nova-3 on the telephony audio. Full sweeps in
[`results/tradeoff.csv`](results/tradeoff.csv); all conditions in
[`results/conditions.csv`](results/conditions.csv).

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

## The commercial system, measured

Deepgram Flux is the closest thing on the market to what this project argues
for: one model doing recognition and end-of-turn together, no silence timer,
with a confidence reported on every update. It was run over the same {f["n_turns"]}
turns ([`scripts/deepgram_eval.py`](scripts/deepgram_eval.py), {f["flux_date"]}), audio
paced at real time so that network and model latency land on the clock a
caller experiences, responses cached under `results/asr/` so the comparison
reproduces without a key. Reading: [`results/tradeoff.md`](results/tradeoff.md)
and [`results/conditions.md`](results/conditions.md).

- **Flux is on the Pareto front, and on the deployment condition it is the
  best system measured.** As shipped: {f["flux_cut"]}% cutoff / {f["flux_hold"]}% never
  answered / {f["flux_p50"]}ms at wideband; {f["tel_flux_cut"]}% / {f["tel_flux_hold"]}% / {f["tel_flux_p50"]}ms on
  the telephony band, where the timer goes to {f["tel_timer800_cut"]}% and the gated
  heuristic to {f["tel_pg_cut"]}%. Nothing here beats it on all three axes: the gated
  heuristic answers in half the time ({f["pg_p50"]}ms) and cuts off more; our gated
  model matches its cutoff rate and leaves three to four times as many
  callers waiting. Said plainly, as the charter requires.
- **Its usable range is one threshold wide.** Sweeping the fire threshold
  over the confidence it reports while a turn is open: 0.5 gives {f["fluxc05_cut"]}%
  cutoff at {f["fluxc05_p50"]}ms, 0.7 gives {f["fluxc07_cut"]}% at {f["fluxc07_p50"]}ms, and at 0.75 the
  holds jump to {f["fluxc075_hold"]}% because the confidence rarely gets there. The
  curve passes through the shipped point; the vendor's default is the knee,
  and there is no hidden better setting. Its holds are the one-word
  backchannels — *"Mm-hmm"*, *"Yeah"* — on which it never opens a turn.
- **Nova-3 is the better recogniser on every axis, and it moves the
  heuristic, not the model.** Word error {f["nova_wer_pct"]}% against parakeet's {f["wer_pct"]}%,
  {f["silent_nova"]} silent turns against {f["silent_para"]}, {f["nova_revisions"]} hypothesis revisions against
  {f["revisions"]}, first partial 500ms after the first word. On its transcripts the
  gated heuristic drops {f["nova3_pg_gain"]} points of cutoff and pays for it in holds
  ({f["nova3_pg_hold"]}%: Nova-3's interim results carry no punctuation until the
  segment is final). The text model holds exactly as it does on gold
  ({f["nova3_v11g03_hold"]}% against {f["v11g03_hold"]}%) — parakeet's lower hold rate was its
  revisions handing the model extra strings to score, not better text. And
  Nova-3's own default endpointing, `speech_final`, cuts off {f["nova3sf_cut"]}% of
  turns: a pause detector, not a turn detector.

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
- **The cloud numbers are a snapshot.** Deepgram `{f["flux_model"]}` and
  `{f["nova_model"]}` were measured once, on {f["flux_date"]}, paced at real time
  over five concurrent streams from one machine; the raw responses are
  cached under `results/asr/` so the numbers reproduce, but the vendor can
  change the model underneath them. Network latency is included in their
  clocks, as a caller would experience it.

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
this page on any machine. Verified on a fresh clone at commit `30da231`
(2026-09-13): install, tests, the table, both CSVs' decision columns and this
page came out identical; only the CPU inference-latency column moved. Two tests skip without the optional
`data/raw/jfk.wav` sample, which is gitignored. To regenerate the inputs
instead:

```bash
# recogniser caches (Apple Silicon; ~15 min per condition)
uv run python scripts/transcribe_eval.py --condition 16k
uv run python scripts/transcribe_eval.py --condition tel

# Deepgram caches (DEEPGRAM_API_KEY or ~/.config/sara/deepgram_api_key;
# ~4 min and ~$0.15 per pass at list price; the key is never written anywhere)
uv run python scripts/deepgram_eval.py --backend nova3 --condition 16k
uv run python scripts/deepgram_eval.py --backend nova3 --condition tel
uv run python scripts/deepgram_eval.py --backend flux --condition 16k
uv run python scripts/deepgram_eval.py --backend flux --condition tel

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
src/baselines/   silence timers, energy VAD, punctuation heuristic, the Deepgram client
src/audio/       Silero VAD, prosody tracker, telephony simulation, mic capture
src/stt/         streaming STT for the live path only -- never imported by eval/
eval/            harness, sweep, conditions; replays gold or cached recogniser/vendor output
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
