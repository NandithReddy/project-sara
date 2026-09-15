"""The two charts a reader meets first, drawn from the CSVs the eval wrote.

  results/headline.png       one panel, the deployment condition (telephony
                             band, a real recogniser), five systems
  results/summary_bars.png   the same five systems as bars: interruptions,
                             never answered, median wait

Both read results/tradeoff.csv and results/conditions.csv and never run an
evaluation, so `make readme` can redraw them in a second. The points they
plot are written to results/headline.json, and a test checks that file
against the CSVs, so the pictures cannot drift from the numbers either.

Our text systems are taken from the `asr_tel` condition (parakeet-mlx
transcripts, telephony band); Deepgram's from `nova3_tel`, the same audio
through its own recogniser. Run from the repo root:  uv run python -m eval.headline
"""

from __future__ import annotations

import csv
import json

from eval.harness import RESULTS
from eval.sweep import GRID, INK, INK2, SURFACE, style_for

HEADLINE_PNG = RESULTS / "headline.png"
BARS_PNG = RESULTS / "summary_bars.png"
POINTS_JSON = RESULTS / "headline.json"
OURS, DEEPGRAM = "asr_tel", "nova3_tel"

# system, knob value, condition, label on the chart. Order is the bar order.
BARS = (
    ("fixed_timeout+silero", 500.0, OURS, "500ms silence timer\n(industry default)"),
    ("fixed_timeout+silero", 800.0, OURS, "800ms silence timer"),
    ("punctuation+gate200", 0.0, OURS, "punctuation\n+ 200ms gate"),
    ("text_eot_v1.1+gate200", 0.3, OURS, "text model + gate\n(ours, p≥0.3)"),
    ("flux", 0.7, DEEPGRAM, "Deepgram Flux\n(as shipped)"),
)
CURVES = (  # system, condition, values to tag
    ("fixed_timeout+silero", OURS, (500.0, 800.0, 1000.0)),
    ("text_eot_v1.1+gate200", OURS, (0.3,)),
    ("flux_conf", DEEPGRAM, (0.7,)),
)
POINTS = (  # single operating points on the headline chart
    ("punctuation+gate200", 0.0, OURS),
    ("flux", 0.7, DEEPGRAM),
    ("nova3_speech_final", 0.0, DEEPGRAM),
)
GOOD_X, GOOD_Y = 500.0, 10.0  # the region nothing reaches: under 10% and under 500ms
Y_MAX, X_MAX = 45.0, 2150.0


def load() -> list[dict]:
    rows = []
    with (RESULTS / "conditions.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    **r,
                    "knob_value": float(r["knob_value"]),
                    "cut": float(r["cutoff_rate_at_tolerance"]) * 100,
                    "hold": float(r["false_hold_rate"]) * 100,
                    "p50": float(r["latency_fallback_p50"]),
                    "n_turns": int(r["n_turns"]),
                }
            )
    return rows


def pick(rows, system, knob, condition):
    for r in rows:
        if (
            r["system"] == system
            and r["condition"] == condition
            and abs(r["knob_value"] - knob) < 1e-9
        ):
            return r
    raise KeyError(f"{system} {knob} {condition} not in conditions.csv")


def _axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_axisbelow(True)


def draw_headline(rows, plt) -> dict:
    fig, ax = plt.subplots(figsize=(11, 6.4), facecolor=SURFACE)
    _axes(ax)
    ax.grid(True, color=GRID, linewidth=0.8)
    # The region every system is trying to reach.
    ax.add_patch(
        plt.Rectangle(
            (0, -1), GOOD_X, GOOD_Y + 1, facecolor="#1baf7a", alpha=0.07, zorder=0
        )
    )
    ax.annotate(
        "the target: interrupts fewer than 1 in 10,\nanswers within half a second",
        (GOOD_X / 2, GOOD_Y / 2),
        ha="center",
        va="center",
        fontsize=8.5,
        color="#0d6b48",
        zorder=1,
    )
    plotted = {}
    for system, cond, tag_values in CURVES:
        colour, marker, _ = style_for(system)
        pts = sorted(
            (r for r in rows if r["system"] == system and r["condition"] == cond),
            key=lambda r: r["knob_value"],
        )
        pts = [r for r in pts if r["cut"] <= Y_MAX]
        ax.plot(
            [r["p50"] for r in pts],
            [r["cut"] for r in pts],
            color=colour,
            linewidth=2,
            zorder=2,
        )
        for r in pts:
            hollow = r["hold"] > 10.0
            ax.plot(
                r["p50"],
                r["cut"],
                marker=marker,
                markersize=8,
                color=colour,
                markerfacecolor=SURFACE if hollow else colour,
                markeredgewidth=2,
                linestyle="none",
                zorder=3,
            )
            if r["knob_value"] in tag_values:
                is_thr = system != "fixed_timeout+silero"
                tag = (
                    f"p≥{r['knob_value']:g}" if is_thr else f"{int(r['knob_value'])}ms"
                )
                ax.annotate(
                    tag,
                    (r["p50"], r["cut"]),
                    xytext=(7, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color=INK2,
                )
                plotted[f"{system}@{r['knob_value']:g}"] = {
                    "condition": cond,
                    "cut": r["cut"],
                    "hold": r["hold"],
                    "p50": r["p50"],
                }
    for system, knob, cond in POINTS:
        colour, marker, _ = style_for(system)
        r = pick(rows, system, knob, cond)
        hollow = r["hold"] > 10.0
        ax.plot(
            r["p50"],
            r["cut"],
            marker=marker,
            markersize=11,
            color=colour,
            markerfacecolor=SURFACE if hollow else colour,
            markeredgewidth=2.5,
            linestyle="none",
            zorder=4,
        )
        plotted[f"{system}@{knob:g}"] = {
            "condition": cond,
            "cut": r["cut"],
            "hold": r["hold"],
            "p50": r["p50"],
        }

    # Boxed callouts with a leader to the point they name: (text, box, target).
    labels = {
        "fixed_timeout+silero": (
            "silence timer, Silero VAD\n(what most voice agents ship)",
            (1230, 17.5),
            (800, pick(rows, "fixed_timeout+silero", 800.0, OURS)["cut"]),
        ),
        "text_eot_v1.1+gate200": (
            "our text model + 200ms gate\n(hollow: a third of callers never answered)",
            (1200, 12.0),
            tuple(
                pick(rows, "text_eot_v1.1+gate200", 0.3, OURS)[k]
                for k in ("p50", "cut")
            ),
        ),
        "flux_conf": (
            "Deepgram Flux, acting on\nits confidence",
            (560, 30.0),
            tuple(pick(rows, "flux_conf", 0.5, DEEPGRAM)[k] for k in ("p50", "cut")),
        ),
        "punctuation+gate200": (
            "punctuation + 200ms gate",
            (640, 24.5),
            tuple(
                pick(rows, "punctuation+gate200", 0.0, OURS)[k] for k in ("p50", "cut")
            ),
        ),
        "flux": (
            "Deepgram Flux, as shipped",
            (600, 5.0),
            tuple(pick(rows, "flux", 0.7, DEEPGRAM)[k] for k in ("p50", "cut")),
        ),
        "nova3_speech_final": (
            "Deepgram Nova-3 endpointing\n(default)",
            (430, 44.2),
            tuple(
                pick(rows, "nova3_speech_final", 0.0, DEEPGRAM)[k]
                for k in ("p50", "cut")
            ),
        ),
    }
    for system, (text, box, target) in labels.items():
        colour = style_for(system)[0]
        ax.annotate(
            text,
            xy=target,
            xytext=box,
            textcoords="data",
            fontsize=9.5,
            color=INK,
            ha="left",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": SURFACE,
                "edgecolor": colour,
                "linewidth": 1.2,
            },
            arrowprops={"arrowstyle": "-", "color": colour, "linewidth": 1.0},
            zorder=5,
        )
    ax.axvline(2000, color=GRID, linewidth=1.2, zorder=1)
    ax.annotate(
        "2000ms = never answered;\nthe agent's fallback timeout fires",
        (2000, Y_MAX * 0.97),
        xytext=(-6, 0),
        textcoords="offset points",
        fontsize=8,
        color=INK2,
        ha="right",
        va="top",
    )
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-1, Y_MAX)
    ax.set_xlabel(
        "how long the caller waits after finishing (median, ms)",
        color=INK2,
        fontsize=10,
    )
    ax.set_ylabel(
        "how often the caller is interrupted (% of turns)", color=INK2, fontsize=10
    )
    n = rows[0]["n_turns"]
    ax.set_title(
        f"Phone-quality audio, a real speech recogniser: {n} turns of meeting speech",
        color=INK,
        fontsize=12,
        loc="left",
        pad=16,
    )
    fig.text(
        0.01,
        0.005,
        "Lower-left is better. Each curve is one system at different settings; "
        "hollow markers never answer more than 1 in 10 callers.\nOur systems read "
        "parakeet-mlx transcripts; Deepgram's use their own recogniser on the same "
        "audio.",
        color=INK2,
        fontsize=8,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(HEADLINE_PNG, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return plotted


def draw_bars(rows, plt) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), facecolor=SURFACE)
    metrics = (
        ("cut", "interrupts the caller\n(% of turns)", "%"),
        ("hold", "never answers within 2s\n(% of turns)", "%"),
        ("p50", "typical wait after the caller finishes\n(median, ms)", "ms"),
    )
    picked = [(label, pick(rows, s, k, c), style_for(s)[0]) for s, k, c, label in BARS]
    out = {}
    for ax, (key, title, unit) in zip(axes, metrics, strict=True):
        _axes(ax)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ys = range(len(picked))[::-1]
        for y, (_label, r, colour) in zip(ys, picked, strict=True):
            ax.barh(y, r[key], color=colour, height=0.62, zorder=2)
            ax.annotate(
                f"{r[key]:.1f}%" if unit == "%" else f"{r[key]:.0f} ms",
                (r[key], y),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=INK,
            )
        ax.set_yticks(list(ys))
        ax.set_yticklabels([label for label, _, _ in picked], fontsize=9, color=INK)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")
        top = max(r[key] for _, r, _ in picked)
        ax.set_xlim(0, top * 1.28)
        ax.tick_params(axis="x", labelsize=8)
    for label, r, _ in picked:
        out[label.replace("\n", " ")] = {
            "cut": r["cut"],
            "hold": r["hold"],
            "p50": r["p50"],
        }
    n = rows[0]["n_turns"]
    fig.suptitle(
        f"Five systems on phone-quality audio with a real recogniser ({n} turns). "
        f"Lower is better on every panel.",
        color=INK,
        fontsize=11.5,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(BARS_PNG, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load()
    points = {"headline": draw_headline(rows, plt), "bars": draw_bars(rows, plt)}
    POINTS_JSON.write_text(json.dumps(points, indent=1))
    root = RESULTS.parent
    for p in (HEADLINE_PNG, BARS_PNG, POINTS_JSON):
        print(f"wrote {p.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
