"""Phase 6: every system under four conditions, and the delta per system.

  gold_16k  gold transcripts, wideband audio           -- the chart so far
  gold_tel  gold transcripts, telephony-band audio     -- silence changes, text does not
  asr_16k   parakeet-mlx transcripts, wideband audio   -- the optimism gap
  asr_tel   parakeet-mlx transcripts, telephony audio  -- the deployment target

Plus asr_16k_content (CSV only): the recogniser's partials with compute time
set to zero, isolating what its ERRORS cost from what its SPEED costs.

The boundary (true_end_ms) is frozen and shared across conditions: the true
end of a turn is a property of the speech, not of the channel it came down.

In the asr_* conditions every audio consumer -- the recogniser and the VAD --
hears the CALLER CHANNEL: the frozen audio zeroed from true_end + 150ms, because
AMI headsets carry the next speaker at low level and a recogniser transcribes
it (see eval.harness.caller_channel). The gold_* conditions keep the raw tail:
a VAD at threshold sees it as silence already, verified below by the timer's
numbers on both.

ASR timelines come from results/asr/parakeet_<cond>.json, written once by
scripts/transcribe_eval.py and cached with model, version, date and machine.
eval/ never runs a recogniser (section 10); it replays the cache.

Run from the repo root:  uv run python -m eval.conditions
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from eval.harness import (
    RESULTS,
    FrameCache,
    asr_timeline,
    caller_channel,
    evaluate,
    precompute,
)
from eval.sweep import (
    FIELDS,
    INK,
    INK2,
    SURFACE,
    THRESHOLDS,
    TIMEOUTS_MS,
    draw_panel,
    legend_handles,
    row,
)
from src.audio.telephony import degrade
from src.audio.vad import SileroVAD
from src.baselines.energy import EnergyVAD
from src.baselines.punctuation import PunctuationHeuristic
from src.baselines.silence import FixedSilenceTimeout
from src.eot.gated import DEFAULT_GATE_MS, SilenceGated
from src.eot.model import TextEOT

ASR_DIR = RESULTS / "asr"
CSV_PATH = RESULTS / "conditions.csv"
PNG_PATH = RESULTS / "conditions.png"
CONDITIONS = ("gold_16k", "gold_tel", "asr_16k", "asr_tel", "asr_16k_content")
CHARTED = ("gold_16k", "gold_tel", "asr_16k", "asr_tel")
CFIELDS = ("condition", *FIELDS)


def load_asr(condition: str, suffix: str = "") -> tuple[dict, dict]:
    path = ASR_DIR / f"parakeet_{condition}{suffix}.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Produce it first (section 10 exception path):\n"
            f"  uv run python scripts/transcribe_eval.py --condition {condition}\n"
            f"Not substituting anything (rule 3)."
        )
    spec = json.loads(path.read_text())
    meta = {k: v for k, v in spec.items() if k != "turns"}
    return meta, spec["turns"]


def systems_for(
    cache: FrameCache, energy: FrameCache, transcripts, label
) -> list[dict]:
    """Every system at every operating point, for one condition."""
    kw = {"transcripts": transcripts, "transcript_label": label}
    out = []
    for ms in TIMEOUTS_MS:
        s = evaluate(FixedSilenceTimeout(ms), cache=cache, name="c", **kw)
        out.append(row("fixed_timeout+silero", "timeout_ms", ms, s))
    for ms in (500.0, 800.0, 1000.0):
        s = evaluate(FixedSilenceTimeout(ms), cache=energy, name="c", **kw)
        out.append(row("fixed_timeout+energy", "timeout_ms", ms, s))
    s = evaluate(PunctuationHeuristic(), cache=cache, name="c", **kw)
    out.append(row("punctuation", "none", 0.0, s))
    pg = SilenceGated(PunctuationHeuristic(), DEFAULT_GATE_MS)
    s = evaluate(pg, cache=cache, name="c", **kw)
    out.append(row(pg.name, "none", 0.0, s))
    model, gated = TextEOT(), SilenceGated(TextEOT(), DEFAULT_GATE_MS)
    for det in (model, gated):
        for thr in THRESHOLDS:
            s = evaluate(det, cache=cache, name="c", threshold=thr, **kw)
            out.append(row(det.name, "threshold", thr, s))
    return out


def run_all(tail: str = "caller") -> tuple[list[dict], dict]:
    """`tail="caller"` is the condition that means something: after the turn
    ends, every audio consumer hears what a caller's channel carries -- nothing.
    `tail="raw"` reproduces the first pass on the headset as recorded, kept
    because it is the evidence for why the caller channel exists."""

    def tel(a, _t):
        return degrade(a)

    def zero_tel(a, t):
        return degrade(caller_channel(a, t))

    suffix = "" if tail == "caller" else "_rawtail"
    print("silence tracks: wideband and telephony, raw tail ...", flush=True)
    sil16, en16 = precompute(SileroVAD()), precompute(EnergyVAD())
    sil_tel = precompute(SileroVAD(), transform=tel, label="silero_tel")
    en_tel = precompute(EnergyVAD(), transform=tel, label="energy_tel")
    if tail == "caller":
        print("silence tracks: caller channel, wideband and telephony ...", flush=True)
        sil_a = precompute(SileroVAD(), transform=caller_channel, label="silero_caller")
        en_a = precompute(EnergyVAD(), transform=caller_channel, label="energy_caller")
        sil_ta = precompute(SileroVAD(), transform=zero_tel, label="silero_tel_caller")
        en_ta = precompute(EnergyVAD(), transform=zero_tel, label="energy_tel_caller")
    else:
        sil_a, en_a, sil_ta, en_ta = sil16, en16, sil_tel, en_tel
    meta16, asr16 = load_asr("16k", suffix)
    meta_tel, asr_tel = load_asr("tel", suffix)
    missing = [t for t in sil16.frames if t not in asr16 or t not in asr_tel]
    if missing:
        raise SystemExit(f"ASR cache lacks {len(missing)} turns, e.g. {missing[:3]}")

    tl16 = {k: asr_timeline(v) for k, v in asr16.items()}
    tl_tel = {k: asr_timeline(v) for k, v in asr_tel.items()}
    tl16_content = {k: asr_timeline(v, include_compute=False) for k, v in asr16.items()}
    plan = {
        "gold_16k": (sil16, en16, None, "gold"),
        "gold_tel": (sil_tel, en_tel, None, "gold"),
        "asr_16k": (sil_a, en_a, tl16, f"parakeet_16k{suffix}"),
        "asr_tel": (sil_ta, en_ta, tl_tel, f"parakeet_tel{suffix}"),
        "asr_16k_content": (sil_a, en_a, tl16_content, f"parakeet_16k{suffix}_content"),
    }
    rows = []
    for cond, (cache, energy, transcripts, label) in plan.items():
        print(f"condition {cond} ...", flush=True)
        for r in systems_for(cache, energy, transcripts, label):
            rows.append({"condition": cond, **r})
    return rows, {"asr_16k": meta16, "asr_tel": meta_tel}


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CFIELDS)
        w.writeheader()
        w.writerows(rows)


def plot(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), facecolor=SURFACE)
    titles = {
        "gold_16k": "gold transcripts, wideband",
        "gold_tel": "gold transcripts, telephony band",
        "asr_16k": "parakeet-mlx transcripts, wideband",
        "asr_tel": "parakeet-mlx transcripts, telephony band",
    }
    for ax, cond in zip(axes.ravel(), CHARTED, strict=True):
        draw_panel(
            ax,
            [r for r in rows if r["condition"] == cond],
            "latency_fallback_p50",
            titles[cond],
            tags_on=False,
        )
    fig.legend(
        handles=legend_handles([r for r in rows if r["condition"] == "gold_16k"]),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        labelcolor=INK,
        bbox_to_anchor=(0.5, -0.01),
    )
    n = rows[0]["n_turns"]
    fig.suptitle(
        f"The same systems under four conditions  ({n} turns; cutoff at the "
        f"150ms tolerance; p50 latency, never-answered turns at the 2000ms fallback)",
        color=INK,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.945,
        "top: gold transcripts. bottom: a real streaming recogniser, partials "
        "available when the caller would have them. right column: 300-3400Hz, "
        "G.711 mu-law.",
        color=INK2,
        fontsize=9,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(PNG_PATH, dpi=160, facecolor=SURFACE)
    plt.close(fig)


HEADLINE = (
    ("fixed_timeout+silero", "timeout_ms", 800.0, "Silero timer 800ms"),
    ("punctuation", "none", 0.0, "punctuation"),
    ("punctuation+gate200", "none", 0.0, "punctuation + 200ms gate"),
    ("text_eot_v1.1", "threshold", 0.5, "text EOT v1.1 @0.5"),
    ("text_eot_v1.1+gate200", "threshold", 0.3, "text EOT v1.1 + gate @0.3"),
    ("text_eot_v1.1+gate200", "threshold", 0.5, "text EOT v1.1 + gate @0.5"),
)


def table(rows: list[dict]) -> str:
    def pick(cond, system, knob):
        for r in rows:
            if (
                r["condition"] == cond
                and r["system"] == system
                and float(r["knob_value"]) == knob
            ):
                return r
        return None

    lines = []
    for system, _knob, val, label in HEADLINE:
        lines.append(f"\n{label}")
        lines.append(
            f"  {'condition':>16} {'cut>150':>8} {'hold':>7} {'p50':>6} {'p95':>6}"
            f"   delta vs gold_16k"
        )
        base = pick("gold_16k", system, val)
        for cond in CONDITIONS:
            r = pick(cond, system, val)
            if r is None:
                continue
            d = ""
            if base is not None and cond != "gold_16k":
                dc = (
                    r["cutoff_rate_at_tolerance"] - base["cutoff_rate_at_tolerance"]
                ) * 100
                dh = (r["false_hold_rate"] - base["false_hold_rate"]) * 100
                dp = r["latency_fallback_p50"] - base["latency_fallback_p50"]
                d = f"cut {dc:+.1f}  hold {dh:+.1f}  p50 {dp:+.0f}"
            lines.append(
                f"  {cond:>16} {r['cutoff_rate_at_tolerance'] * 100:7.1f}% "
                f"{r['false_hold_rate'] * 100:6.1f}% {r['latency_fallback_p50']:6.0f} "
                f"{r['latency_fallback_p95']:6.0f}   {d}"
            )
    return "\n".join(lines)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", choices=("caller", "raw"), default="caller")
    args = ap.parse_args()

    rows, meta = run_all(args.tail)
    csv_path = CSV_PATH if args.tail == "caller" else RESULTS / "conditions_rawtail.csv"
    write_csv(rows, csv_path)
    if args.tail == "caller":
        plot(rows)
    print(table(rows))

    def timer800(cond):
        return next(
            r
            for r in rows
            if r["condition"] == cond
            and r["system"] == "fixed_timeout+silero"
            and r["knob_value"] == 800.0
        )

    g, a = timer800("gold_16k"), timer800("asr_16k")
    print(
        "\nsanity -- the 800ms timer reads no text, so gold_16k vs asr_16k isolates "
        "the tail treatment on the VAD alone:"
    )
    print(
        f"  cutoff {g['cutoff_rate_at_tolerance'] * 100:.1f}% -> "
        f"{a['cutoff_rate_at_tolerance'] * 100:.1f}%, hold "
        f"{g['false_hold_rate'] * 100:.1f}% -> {a['false_hold_rate'] * 100:.1f}%"
    )
    for cond, m in meta.items():
        rtf = m["total_compute_s"] / m["total_audio_s"]
        print(
            f"\n{cond}: {m['model']} parakeet-mlx {m['parakeet_mlx']} on "
            f"{m['machine']}, {m['date']}, tail={m.get('tail', 'raw')}; "
            f"{m['total_audio_s']:.0f}s audio in {m['total_compute_s']:.0f}s "
            f"({rtf:.2f}x realtime)"
        )
    root = RESULTS.parent
    print(
        f"\nwrote {csv_path.relative_to(root)}"
        + (f" and {PNG_PATH.relative_to(root)}" if args.tail == "caller" else "")
    )
    for p in RESULTS.glob("sweep_*.json"):
        p.unlink()
    (RESULTS / "c.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
