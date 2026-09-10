"""Sweep every system across its operating range and draw the tradeoff curve.

This curve is the project's success criterion (section 3). A single operating
point is not a result: a timer at 500ms and the same timer at 1000ms are two
points on one line, and the question is never "which point" but "which line".

Timers sweep on their timeout, by constructing one instance per value. Systems
that emit a probability sweep on the fire threshold against one cached run. The
punctuation heuristic has no knob and is a single point.

x is added latency WITH the fallback counted: a false hold means the caller
waits out the horizon, not that nothing happened. y is cutoff rate at the
boundary tolerance. Strict cutoff, hold rate, and every percentile stay in the
CSV so no reading is hidden.

Never imports src/stt/ (section 10).

Run from the repo root as a module, so `eval` resolves as a package:
    uv run python -m eval.sweep
"""

from __future__ import annotations

import csv

from eval.harness import RESULTS, FrameCache, evaluate, precompute
from src.audio.vad import SileroVAD
from src.baselines.energy import EnergyVAD
from src.baselines.punctuation import PunctuationHeuristic
from src.baselines.silence import FixedSilenceTimeout
from src.eot.gated import DEFAULT_GATE_MS, SilenceGated
from src.eot.model import TextEOT

TIMEOUTS_MS = tuple(range(100, 2001, 100))
THRESHOLDS = tuple(round(x * 0.05, 2) for x in range(1, 20))  # 0.05 .. 0.95
GATES_MS = (100.0, 200.0, 300.0, 500.0)
"""Gate values reported for sensitivity. Only DEFAULT_GATE_MS goes on the
chart; the gate is chosen on principle, not by picking the best of these."""
CSV_PATH = RESULTS / "tradeoff.csv"
PNG_PATH = RESULTS / "tradeoff.png"
READING_PATH = RESULTS / "tradeoff.md"

FIELDS = (
    "system",
    "source",
    "knob",
    "knob_value",
    "n_turns",
    "cutoff_rate",
    "cutoff_rate_at_tolerance",
    "false_hold_rate",
    "latency_p50",
    "latency_p95",
    "latency_p99",
    "latency_fallback_p50",
    "latency_fallback_p95",
    "latency_fallback_p99",
)


def row(system: str, knob: str, value: float, s: dict) -> dict:
    a, f = s["added_latency_ms"], s["added_latency_with_fallback_ms"]
    return {
        "system": system,
        "source": s["silence_source"],
        "knob": knob,
        "knob_value": value,
        "n_turns": s["n_turns"],
        "cutoff_rate": s["cutoff_rate"],
        "cutoff_rate_at_tolerance": s["cutoff_rate_at_tolerance"],
        "false_hold_rate": s["false_hold_rate"],
        "latency_p50": a["p50"],
        "latency_p95": a["p95"],
        "latency_p99": a["p99"],
        "latency_fallback_p50": f["p50"],
        "latency_fallback_p95": f["p95"],
        "latency_fallback_p99": f["p99"],
    }


def sweep_timers(cache: FrameCache) -> list[dict]:
    system = f"fixed_timeout+{cache.source_name}"
    out = []
    for ms in TIMEOUTS_MS:
        s = evaluate(FixedSilenceTimeout(ms), cache=cache, name=f"sweep_{system}_{ms}")
        out.append(row(system, "timeout_ms", ms, s))
    return out


def sweep_thresholds(det, cache: FrameCache) -> list[dict]:
    """A detector that emits a probability sweeps on the fire threshold.

    One evaluate() per threshold. The VAD frames are cached and the model
    caches by text, so each pass costs only the few real inferences per turn.
    """
    out = []
    for thr in THRESHOLDS:
        s = evaluate(det, cache=cache, name=f"sweep_{det.name}_{thr}", threshold=thr)
        out.append(row(det.name, "threshold", thr, s))
    return out


def gate_sensitivity(cache: FrameCache) -> list[dict]:
    """How much the gate value matters, at the default threshold. CSV only."""
    out = []
    for g in GATES_MS:
        det = SilenceGated(TextEOT(), g)
        s = evaluate(det, cache=cache, name=f"sweep_gate_sens_{int(g)}")
        out.append(row("text_eot+gate_sensitivity", "gate_ms", g, s))
        pg = SilenceGated(PunctuationHeuristic(), g)
        s = evaluate(pg, cache=cache, name=f"sweep_punct_gate_{int(g)}")
        out.append(row("punctuation+gate_sensitivity", "gate_ms", g, s))
    return out


def run_sweep() -> list[dict]:
    rows: list[dict] = []
    silero = precompute(SileroVAD())
    print("sweeping fixed_timeout+silero ...", flush=True)
    rows += sweep_timers(silero)
    print("sweeping fixed_timeout+energy ...", flush=True)
    rows += sweep_timers(precompute(EnergyVAD()))
    print("punctuation (single point) ...", flush=True)
    s = evaluate(PunctuationHeuristic(), cache=silero, name="sweep_punctuation")
    rows.append(row("punctuation", "none", 0.0, s))
    model = TextEOT()
    print(f"sweeping {model.name} on threshold ...", flush=True)
    rows += sweep_thresholds(model, silero)
    gated = SilenceGated(TextEOT(), DEFAULT_GATE_MS)
    print(f"sweeping {gated.name} on threshold ...", flush=True)
    rows += sweep_thresholds(gated, silero)
    # The fair fight for the gated model: the same gate on the heuristic.
    pg = SilenceGated(PunctuationHeuristic(), DEFAULT_GATE_MS)
    print(f"{pg.name} (single point) ...", flush=True)
    s = evaluate(pg, cache=silero, name=f"sweep_{pg.name}")
    rows.append(row(pg.name, "none", 0.0, s))
    print("gate sensitivity ...", flush=True)
    rows += gate_sensitivity(silero)
    return rows


def write_csv(rows: list[dict]) -> None:
    RESULTS.mkdir(exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def plot(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Validated categorical palette, fixed slot order. Aqua is below 3:1 on the
    # light surface, so every series is direct-labelled and marker shape carries
    # identity alongside colour.
    def style_for(system: str):
        # Validated 6-slot categorical palette, fixed order; colour follows the
        # entity. Aqua, yellow and magenta sit under 3:1 on the light surface,
        # so every series is direct-labelled and marker shape carries identity.
        if system == "fixed_timeout+silero":
            return "#2a78d6", "o", "fixed timeout, Silero VAD"
        if system == "fixed_timeout+energy":
            return "#eb6834", "s", "fixed timeout, energy VAD"
        if system == "punctuation":
            return "#1baf7a", "D", "punctuation heuristic"
        if system.startswith("text_eot") and "+gate" in system:
            g = system.split("+gate")[1]
            return "#e87ba4", "v", f"text EOT model + {g}ms gate"
        if system.startswith("text_eot"):
            return "#eda100", "^", f"text EOT model {system.split('text_eot_')[1]}"
        if system.startswith("punctuation+gate"):
            g = system.split("+gate")[1]
            return "#008300", "P", f"punctuation + {g}ms gate"
        return None

    charted = [s for s in dict.fromkeys(r["system"] for r in rows) if style_for(s)]
    style = {s: style_for(s) for s in charted}
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=surface)
    panels = (("latency_fallback_p50", "p50"), ("latency_fallback_p95", "p95"))

    for ax, (xkey, pct) in zip(axes, panels, strict=True):
        ax.set_facecolor(surface)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(grid)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=ink2, labelsize=9)

        for system, (colour, marker, label) in style.items():
            pts = [r for r in rows if r["system"] == system]
            pts.sort(key=lambda r: r["knob_value"])
            xs = [r[xkey] for r in pts]
            ys = [r["cutoff_rate_at_tolerance"] * 100 for r in pts]
            if len(pts) > 1:
                ax.plot(xs, ys, color=colour, linewidth=2, zorder=2)
            # Hollow marker: a point where more than one turn in ten never got
            # an answer at all. The fallback latency already prices that in on
            # x; the marker makes it visible on its own.
            for r, x, y in zip(pts, xs, ys, strict=True):
                hollow = r["false_hold_rate"] > 0.10
                ax.plot(
                    x,
                    y,
                    marker=marker,
                    markersize=8,
                    color=colour,
                    markerfacecolor=surface if hollow else colour,
                    markeredgewidth=2,
                    linestyle="none",
                    zorder=3,
                )
            # Direct label at the first point of the line. Series start at
            # different heights, so labels separate there; mid-curve the two
            # timers overlap, and at the right edge they pile up on the wall.
            if len(pts) > 1:
                first = pts[0]
                ax.annotate(
                    label,
                    (first[xkey], first["cutoff_rate_at_tolerance"] * 100),
                    xytext=(10, 2),
                    textcoords="offset points",
                    fontsize=9,
                    color=ink,
                    ha="left",
                    va="center",
                )
            else:
                r = pts[0]
                ax.annotate(
                    label,
                    (r[xkey], r["cutoff_rate_at_tolerance"] * 100),
                    xytext=(10, -4),
                    textcoords="offset points",
                    fontsize=9,
                    color=ink,
                    ha="left",
                )
            # Knob values on a few points only, offset differently per series
            # so the two timer curves' tags never land on each other.
            tags = {
                "fixed_timeout+silero": ((300, 500, 1000), (6, 5)),
                "fixed_timeout+energy": ((500, 1000), (6, -12)),
            }
            if system.startswith("text_eot"):
                tags[system] = (
                    (0.3, 0.5, 0.7, 0.9),
                    (-30, -12) if "+gate" not in system else (6, 5),
                )
            if system in tags:
                values, offset = tags[system]
                for r in pts:
                    if r["knob_value"] in values and r[xkey] < 1900:
                        is_thr = system.startswith("text_eot")
                        ax.annotate(
                            f"p≥{r['knob_value']}"
                            if is_thr
                            else f"{int(r['knob_value'])}ms",
                            (r[xkey], r["cutoff_rate_at_tolerance"] * 100),
                            xytext=offset,
                            textcoords="offset points",
                            fontsize=7.5,
                            color=ink2,
                        )

        ax.axvline(2000, color=grid, linewidth=1.2, zorder=1)
        ax.annotate(
            "2000ms fallback:\nnever answered",
            (2000, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 40),
            xytext=(-6, 0),
            textcoords="offset points",
            fontsize=7.5,
            color=ink2,
            ha="right",
            va="top",
        )
        ax.set_xlabel(f"added latency at {pct} (ms)", color=ink2, fontsize=9)
        ax.set_ylabel("premature cutoff rate (%)", color=ink2, fontsize=9)
        ax.set_title(f"cutoff vs latency at {pct}", color=ink, fontsize=11, loc="left")
        ax.set_ylim(-1, 44)
        ax.set_xlim(0, 2150)

    handles = [
        plt.Line2D([], [], color=c, marker=m, markersize=8, linewidth=2, label=lab)
        for c, m, lab in style.values()
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            color=ink2,
            marker="o",
            markersize=8,
            linewidth=0,
            markerfacecolor=surface,
            markeredgewidth=2,
            label="hollow: >10% of turns never answered",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        labelcolor=ink,
        bbox_to_anchor=(0.5, -0.02),
    )
    n = rows[0]["n_turns"]
    fig.suptitle(
        f"End-of-turn detection: premature cutoffs against added latency  "
        f"({n} turns, AMI development split, gold transcripts)",
        color=ink,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.905,
        "latency is measured from the audio-grounded true end of the turn; a "
        "turn never answered is counted at the 2000ms fallback timeout",
        color=ink2,
        fontsize=9,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(PNG_PATH, dpi=160, facecolor=surface)
    plt.close(fig)


def main() -> int:
    rows = run_sweep()
    write_csv(rows)
    plot(rows)
    print(f"\nwrote {CSV_PATH.relative_to(RESULTS.parent)}")
    print(f"wrote {PNG_PATH.relative_to(RESULTS.parent)}")

    # Remove the per-run JSONs the sweep produced; the CSV is the record.
    for p in RESULTS.glob("sweep_*.json"):
        p.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
