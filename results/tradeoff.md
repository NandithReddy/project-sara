# The tradeoff curve — reading

![tradeoff](tradeoff.png)

Every number below is from `tradeoff.csv`, produced by `uv run python -m
eval.sweep` over the 198 frozen turns in `data/eval/`. Latency is measured from
the audio-grounded true end of the turn; a turn never answered is counted at
the 2000ms fallback. Cutoff rate is at the 150ms boundary tolerance.

**Two iterations of the text-only model and a prosody classifier are
measured. None beats the timer; prosody does not help; the silence gate is
the finding.** The Phase 4 and Phase 7 sections come first; the baseline
reading below them is unchanged.

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

## Phase 7 — prosody, measured; it does not help

Every number is from `tradeoff.csv` (gold transcripts, wideband) and
`conditions.csv`; the models and their validation numbers are in
`models/README.md`. The systems: a logistic regression over 14 prosodic
features of the last speech frame before a pause, fired behind the same 200ms
gate as everything else; the same with the text model's logit added.

| system | cutoff (tol) | hold | p50 | p95 |
|---|---|---|---|---|
| Silero timer, 200ms | 29.8% | 0.0% | 224 | 256 |
| Silero timer, 800ms | 10.6% | 4.5% | 800 | 1859 |
| punctuation + 200ms gate | 11.1% | 9.1% | 256 | 2000 |
| text EOT v1.1 + gate @0.1 | 19.7% | 17.2% | 256 | 2000 |
| text EOT v1.1 + gate @0.3 | 7.6% | 44.9% | 352 | 2000 |
| **prosody only + gate @0.3** | 26.3% | 2.5% | 256 | 429 |
| **prosody only + gate @0.5** | 24.7% | 9.6% | 256 | 2000 |
| **prosody only + gate @0.7** | 6.6% | 62.6% | 2000 | 2000 |
| **prosody + text + gate @0.5** | 23.7% | 12.1% | 256 | 2000 |
| **prosody + text + gate @0.7** | 12.6% | 30.8% | 256 | 2000 |

**1. Prosody alone is dominated by the one-character heuristic at every
threshold.** At matched hold rates (~10%) it cuts off 24.7% of turns against
the gated heuristic's 11.1%. Below threshold 0.5 its curve is flat at
26.3% / 2.5% — that is the 200ms timer itself, a classifier saying yes to
every pause — and above it the holds explode before the cutoffs come down.

**2. Fusion adds nothing over the text.** On held-out speakers the text logit
alone scores AP 0.860 at the pause; adding prosody gives 0.857. On the eval,
fusion @0.7 sits at 12.6% / 30.8% against the gated text model's
19.7% / 17.2% @0.1 and 7.6% / 44.9% @0.3 — the same trade, no new frontier.
The text logit's weight in the fusion model is +1.38; nothing prosodic comes close.

**3. The signal that exists is energy, not pitch.** Prosody alone reaches AP
0.716 (chance 0.585) on held-out speakers, and its weights say where:
`energy_db` -0.25, `voiced_fraction` -0.24, `energy_slope` +0.19
— the last speech before a real end is quieter and trailing off. The
final-fall and pitch-slope weights are +0.01 and +0.01: at this
tracker's resolution on AMI headsets, a falling contour does not separate a
turn's end from a mid-turn pause. Many mid-turn pauses follow a complete,
falling sentence — the hard negative Phase 3 named.

**4. Under a real recogniser and the telephony band.**

| system | gold, wideband | gold, telephony | real ASR, wideband | real ASR, telephony |
|---|---|---|---|---|
| Silero timer 800ms | 10.6% / 4.5% | 17.2% / 2.5% | 10.6% / 0.0% | 17.2% / 0.0% |
| punctuation + gate | 11.1% / 9.1% | 14.1% / 8.6% | 19.7% / 6.1% | 20.2% / 8.6% |
| text EOT v1.1 + gate @0.3 | 7.6% / 44.9% | 14.6% / 41.9% | 10.6% / 30.8% | 10.1% / 34.3% |
| prosody only + gate @0.5 | 24.7% / 9.6% | 25.8% / 19.7% | 24.7% / 10.6% | 25.8% / 20.2% |
| prosody + text + gate @0.5 | 23.7% / 12.1% | 24.2% / 21.2% | 21.2% / 9.1% | 21.7% / 19.7% |

Prosody-only is invariant to the recogniser by construction (24.7% → 24.7%)
and loses about ten points of hold rate at the telephony band (9.6% → 19.7%):
the features were fit on wideband audio and are not band-invariant. Neither
prosody system beats the gated heuristic or the gated text model on any
condition.

**5. Why, honestly, and what would move it — as proposals.** The features
are weak (AP 0.72 on validation is the ceiling of this feature set, not of
the model: a linear fit on standardised inputs is not what is limiting it).
The F0 estimator is a 64ms autocorrelation on headset audio with crosstalk;
a pitch tracker built for speech (pYIN, CREPE) is a dependency this project
has not taken. Window statistics over 1.5s lose the syllable-level timing
that final lengthening lives at. 10.7% of the mid-turn labels are a laugh
or a breath, not a word. And the systems that make prosody work in
production (Smart Turn v2) use a pretrained audio encoder, not fourteen
hand-built numbers. None of these were tried; the model was not touched
after the number was seen.

**One thing was caught before it became a result.** The first classifier
scored a perfect 1.000 AP on held-out speakers. It had learned to detect the
caller-channel zeroing — features sampled 200ms into an end pause read −120dB
of digital silence. Features come from the last speech frame now, the leaked
models are not in the repo, and the lesson is in `models/README.md`.

**Decision, per the plan:** prosody does not help; the text-only model stays
the model, and this is the negative result reported.

## Baseline #4 — Deepgram Flux, measured

Two series come from one cached pass over the same 198 turns
(`results/asr/flux_16k.json`: `flux-general-en`, 2026-09-13, caller-channel
audio paced at real time over five streams, so network and model latency sit
on the clock a caller would experience; ~$0.15 at list price). **As shipped**
is Flux's own `EndOfTurn` at its default `eot_threshold` of 0.7. **Acting on
its confidence** sweeps the fire threshold over the `end_of_turn_confidence`
Flux reports every ~240ms while it has a turn open — what a developer gets by
reading the number themselves. p95 is the 2000ms fallback for every row: Flux
never answers 9.6–11.1% of these turns at any threshold.

| Flux | cutoff (>150ms) | never answered | p50 latency |
|---|---|---|---|
| as shipped, `eot_threshold` 0.7 | **7.6%** | 11.1% | 576ms |
| confidence ≥ 0.5 | 20.7% | 9.6% | 320ms |
| confidence ≥ 0.6 | 14.6% | 9.6% | 416ms |
| confidence ≥ 0.65 | 10.6% | 10.6% | 480ms |
| confidence ≥ 0.7 | 8.6% | 11.1% | 544ms |
| confidence ≥ 0.75 | 4.5% | 61.1% | 2000ms |
| confidence ≥ 0.8 | 2.0% | 83.8% | 2000ms |

**It is on the Pareto front, and nothing here dominates it.** 7.6% cutoff at
576ms with 11.1% holds sits between the 800ms Silero timer (10.6%, 4.5%,
800ms) and the gated heuristic (11.1%, 9.1%, 256ms). The gated text model
matches its cutoff rate at 0.3 (7.6%) and leaves four times as many callers
waiting (44.9%). Per the charter, said plainly: at wideband the commercial
system is at least as good as anything built here, and on the telephony
band it is better than all of it ([`conditions.md`](conditions.md)).

**The curve passes through the shipped point, and the default is where it
turns.** On the median turn Flux's `EndOfTurn` *is* the first message at or
above 0.7 (p90: 165ms later), so acting on the confidence at 0.7 reproduces
the product to within a point. Below 0.65 cutoffs climb steeply — 0.5 costs
20.7% — and above 0.7 the confidence mostly never arrives: 0.75 already holds
on 61% of turns. There is no hidden better setting; the vendor's default is
the knee.

**Its holds are structural.** On 17 of the 198 turns Flux never opens a turn
at all — the one-word backchannels, *"Mm-hmm"*, *"Yeah"*, *"Okay"*, the same
turns Nova-3 returns no text for. Its idle-state confidence drifts past 0.7
on several of them, but there is no turn for it to end and its own
`EndOfTurn` cannot fire; the sweep counts only in-turn messages for that
reason (the raw feed would have answered a few of them, late, by accident).
No threshold recovers those callers.

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

**6. The commercial fused model does not enter the empty region either.**
Flux's 7.6% at 576ms is the best point on the chart that also answers nearly
everyone, and it is still outside the under-10%, under-500ms corner. A
product built the way this project argues for — acoustics and semantics in
one pass, no silence timer — lands 76ms outside the region, which says the
headroom in point 1 is real and that the last stretch of it is hard.

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
- **Flux is a snapshot of a cloud service.** One pass on 2026-09-13, five
  concurrent streams from one machine, network latency included in its clock.
  The vendor can change the model underneath the number; the cached responses
  cannot change, and the comparison reproduces from them without a key.
- **The boundary is Silero-refined, and Silero is baseline #2.** Mitigated
  (raw probability, not the smoothed flag) and quantified (boundary delta
  p10 −144ms, p90 +145ms), but the ground truth shares a model with one system
  under test.
- **19 speakers, 198 turns.** A cutoff-rate difference of one turn is 0.5%.
  Differences under ~2% between systems are inside the noise.
