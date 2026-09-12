# The tradeoff curve — reading

![tradeoff](tradeoff.png)

Every number below is from `tradeoff.csv`, produced by `uv run python -m
eval.sweep` over the 198 frozen turns in `data/eval/`. Latency is measured from
the audio-grounded true end of the turn; a turn never answered is counted at
the 2000ms fallback. Cutoff rate is at the 150ms boundary tolerance.

**Two iterations of the text-only model are measured. Neither beats the
timer. The silence gate is the finding.** Read this section first; the
baseline reading below it is unchanged.

## text EOT model, v1 and v1.1 — measured; neither beats the timer

Latency here is the chart's convention: a turn never answered is counted at
the 2000ms fallback. v1 rows are from the sweep at commit `809a9d5`; every
other row is from `tradeoff.csv` in this commit.

| system | cutoff (tol) | hold | p50 | p95 |
|---|---|---|---|---|
| Silero timer, 800ms | 10.6% | 4.5% | 800 | 1859 |
| punctuation heuristic | 24.7% | 6.6% | 96 | 2000 |
| **punctuation + 200ms gate** | 11.1% | 9.1% | 256 | 2000 |
| text EOT v1 @0.5 *(commit 809a9d5)* | 53.5% | 10.1% | 160 | 2000 |
| text EOT v1 + gate @0.5 *(commit 809a9d5)* | 15.7% | 24.2% | 256 | 2000 |
| text EOT v1.1 @0.5 | 15.7% | 58.1% | 2000 | 2000 |
| text EOT v1.1 + gate @0.1 | 19.7% | 17.2% | 256 | 2000 |
| text EOT v1.1 + gate @0.3 | 7.6% | 44.9% | 352 | 2000 |
| text EOT v1.1 + gate @0.5 | 4.5% | 67.2% | 2000 | 2000 |

**What v1.1 changed.** One thing: the loss weight on the "complete" class,
6.1× in v1, set to 1. Validation (10 held-out speakers): AP 0.505 → 0.551,
log-loss 0.410 → 0.269. Regularisation (freezing the lower two layers,
dropout 0.2) was tried in the same batch and made things worse on its own and
slightly worse on top of the weight fix, so it is not in v1.1. The earlier
claim that v1 was memorising the corpus was mostly wrong: the rising
validation loss was the weighted objective's calibration.

**What that bought on the eval set.** The model went from firing too much to
firing too little — bare cutoff 53.5% → 15.7%, hold 10.1% → 58.1%. Gated, it
reaches 4.5% cutoff at threshold 0.5 while refusing to answer 67% of turns;
sweeping the threshold down walks along that trade (19.7% / 17.2% at 0.1)
without ever crossing the timer's curve or reaching the gated heuristic's
point. Turn-level separability — the full turn outscoring every one of its
own prefixes — is 57.1% (disfluent 31/66 ordinary 24/66 short 58/66), against 55.1% for v1.

**Why it still loses, in one number: AP 0.55.** The ranking is not good
enough for a streaming decision, and no threshold fixes a ranking. Three
reasons it is weak, none of them fixable by another training flag:

- 5,747 training turns from one meeting scenario is a small corpus to learn
  syntactic completion from, and the eval speakers are, correctly, unseen.
- The Phase 3 ceiling: on the short-answer stratum, "Yeah" / "Okay" carry
  both labels and no function of the text separates them. Gating recovers
  some of this; it cannot recover the ranking.
- Meeting speech trails off — turns ending on "so", "and", "um" are labelled
  complete because the speaker stopped — which teaches exactly the wrong
  thing about telephony turns.

**The gate is the contribution; the heuristic's point was not real.** On
gold, the same 200ms gate on a one-character heuristic sits at 11.1% cutoff,
9.1% hold, p50 256ms — the first point in the chart's empty bottom-left. Phase
6 measured it against a real recogniser on a caller's channel: **19.7% cutoff**
on the deployment condition, more than the timer. The model behind the same
gate lost 2.5 points where the heuristic lost 8.6. Read
[`conditions.md`](conditions.md) before quoting anything from this file.

**What would move the model, as proposals.** More training turns from a
second corpus; a larger encoder with a frozen body (latency allows 10× —
v1.1 is 0.45ms per inference); prosody (Phase 7, the ceiling argument);
punctuation as an input, with the same caveat as the heuristic. None of these
were tried. The model has not been touched since the number was seen.

## What the chart says

**1. There is an empty region, and it is the whole point.**
Nothing on this chart achieves both under 10% premature cutoffs *and* under
500ms median latency. The best timer needs **900ms** to get under 10% cutoff
(p50 944ms). The punctuation heuristic answers at **96ms** but cuts off
**24.7%** of turns. Bottom-left is unoccupied. A semantic model earns its place
by putting a point there; if it cannot, the project reports that plainly.

**2. At equal cutoff rate, the semantic signal is 3.3× faster.**
The Silero timer at 300ms and the punctuation heuristic both cut off 24.7% of
turns. The timer does it at p50 **320ms**; punctuation at **96ms**. Same
damage, a third of the wait — because a signal that reads *what was said* does
not have to wait out silence to know the sentence ended. This is the mechanism
the project is betting on, demonstrated by a heuristic that looks at one
character.

**3. But the heuristic's tail is a wall.**
Punctuation's p95 is **2000ms** — the fallback. 6.6% of turns never end in
terminal punctuation, so on those the caller waits the full timeout. A median
of 96ms hides a p95 of two seconds. The charter's rule about never reporting a
single percentile exists for exactly this.

**4. The naive silence source is not viable.**
Energy thresholding looks competitive at p50 and collapses at p95: its p95 hits
the 2000ms wall from a 300ms timeout onward, because it holds the floor on
12.1% of turns at 500ms, 31.3% at 800ms, 46.5% at 1000ms — re-triggering on
room tone after the speaker stops. Baseline #2 (Silero) exists because
baseline #1 does not work.

**5. The industry default is measurably bad on this data.**
A 500ms Silero timer interrupts **17.2%** of turns. At 1000ms that falls to
4.0% — and 7.6% of turns now wait the full fallback, with p95 at the wall.
There is no timeout on this set that is both safe and fast.

## Where a timer beats the heuristic, and where it does not

| you need | timer | punctuation | winner |
|---|---|---|---|
| cutoffs under 20% | 500ms, p50 512ms | cannot (stuck at 24.7%) | timer |
| cutoffs under 10% | 900ms, p50 944ms | cannot | timer |
| median response under 150ms | 100ms, 40.9% cutoff | 96ms, 24.7% cutoff | punctuation |
| a bounded p95 | any timeout ≤ 500ms | never (p95 = fallback) | timer |

The heuristic wins only if ~25% cutoffs are acceptable. They almost never are.
So the practical conclusion is not "use punctuation" — it is that **the
latency headroom is real and large, and a model that keeps the semantic speed
while fixing the semantic cutoffs has ~600ms to gain over the best timer.**

## Caveats that travel with these numbers

- **Gold transcripts.** Punctuation here is placed by human annotators who
  heard the whole recording. Phase 6 measured the gap: the bare heuristic goes
  24.7% → 54.0% cutoff on a real recogniser. Every number in this file is
  the gold ceiling; [`conditions.md`](conditions.md) has the floor.
- **Meeting speech, 16kHz headset.** Phase 6 adds the telephony band: the
  timer alone goes 10.6% → 17.2% cutoff, all of it through the VAD. And the
  headset's raw tail carries the next speaker: the timer's 4.5% false holds
  here are that, and read 0.0% on a caller's channel.
- **The boundary is Silero-refined, and Silero is baseline #2.** Mitigated
  (raw probability, not the smoothed flag) and quantified (boundary delta
  p10 −144ms, p90 +145ms), but the ground truth shares a model with one system
  under test.
- **19 speakers, 198 turns.** A cutoff-rate difference of one turn is 0.5%.
  Differences under ~2% between systems are inside the noise.
