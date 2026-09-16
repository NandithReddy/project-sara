# A plain-language guide to this project

This page explains the whole project in everyday words. No maths, no jargon
without an explanation, nothing you need to know in advance. If you can
follow a recipe, you can follow this. The [README](README.md) has the full
results; this page tells you what they mean and how to make them yourself.

---

## 1. The problem, in one story

Imagine phoning a company and a computer answers. You say:

> "My order number is… umm… four four seven… one."

A computer that is **too quick** hears the pause after "is… umm…" and jumps
in: *"Sorry, I didn't catch that."* It talked over you. Annoying.

A computer that is **too slow** waits a long time after you say "one." to be
sure you have finished. You sit there in silence wondering if it heard you.
Also annoying.

Nearly every phone robot today makes this decision the same simple way: **it
waits for a fixed amount of silence** — usually between half a second and one
second — and then assumes you are done. That is too quick for people who
pause to think, and too slow for people who answer "yes."

This project asks one question: **can anything decide better than a silence
timer, and by how much?** Not "does it feel better" — measured, on the same
recordings, with the same rules for everyone.

---

## 2. The words we use

| word | what it means here |
|---|---|
| **turn** | one go at talking — everything one person says until it's someone else's go |
| **end of turn** | the exact moment the person stopped talking |
| **detector** | any program that listens and says "they've finished now" |
| **interrupt** (cutoff) | the detector said "finished" *before* the person really was — it talked over them |
| **never answers** (hold) | two seconds after the person stopped, the detector still hadn't decided, so a backup timer had to step in — dead air |
| **wait** (latency) | how long after the person stopped the detector said "finished" |
| **silence timer** | the simple detector everyone uses: wait N milliseconds of quiet, then say "finished" |
| **transcript** | the words, written down. A *perfect* transcript is typed by a person; a *real* one comes from speech-recognition software, mistakes and all |
| **speech recogniser** | software that turns sound into words while you are still talking |
| **gate** | a rule that says "even if you think they've finished, wait 200 milliseconds of quiet first, just in case" |
| **phone-quality audio** | sound squeezed the way a phone line squeezes it: muffled, narrow, slightly crackly |

A millisecond (ms) is a thousandth of a second. 500 ms is half a second.

---

## 3. How the contest works

Think of it as a school sports day for detectors. Every contestant runs the
same race on the same track, and the same three judges score them.

**The track.** 198 short recordings of real people talking in meetings (from
a public research collection called the AMI Meeting Corpus). For every
recording we know, to within a fraction of a second, the exact moment the
person stopped — that is the finish line. The recordings are locked: a
checksum file guarantees nobody quietly swaps one, and the code refuses to
train a model on them. They are only ever used to test.

**The race.** Each recording is played to each detector *as if it were
happening live* — the sound arrives bit by bit, the words arrive as a
recogniser would produce them, and the detector has to shout "finished!" at
some point. We write down when it shouted.

**The three judges.**

1. **Did it interrupt?** Shouted more than 150 ms before the real finish → yes.
2. **Did it never answer?** Still silent 2 seconds after the finish → yes.
3. **How long did the person wait?** Time from the real finish to the shout.

Every detector gets a score from all three judges, on every recording, and
we report the average. You cannot win by being good at one judge only:
shouting early pleases judge 3 and angers judge 1; shouting late does the
opposite. That is why the results are drawn as *curves*, not single numbers —
each detector is tried at every setting it has, from hasty to cautious.

**The track conditions.** The race is run twice:

- **Clean**: crisp recordings and perfect, human-typed words. The best case.
- **Phone**: the same recordings squeezed to phone quality, with the words
  coming from real speech-recognition software. The real-world case.

Detectors that look brilliant on the clean track often fall apart on the
phone track. That gap is one of the main findings.

---

## 4. The contestants

| contestant | how it decides | where it comes from |
|---|---|---|
| **Silence timer** | waits for N ms of quiet (a small "is anyone talking?" model called Silero decides what quiet is) | what almost every product ships; the one to beat |
| **Punctuation check** | if the latest words end in `.` `?` or `!`, they've finished | a one-line rule, as a sanity check |
| **Punctuation + gate** | the same, but also waits 200 ms of quiet | the cheapest idea that actually helps |
| **Text model** | a small AI (11 million numbers, runs on a laptop CPU) that reads the words so far and estimates "how likely is it that this sentence is complete?" | trained in this project on 5,700 other meeting turns, never on the test recordings |
| **Prosody model** | listens to how the voice sounds at the last moment of speech — is it getting quieter, dropping in pitch? | trained here; a negative result, kept because negative results count |
| **Deepgram Flux** | a commercial product from a speech company that does recognition and "are they finished?" in one go, in the cloud | the closest competitor; included so the comparison is fair |

---

## 5. What was found

Short version, phone-quality track unless stated:

1. **The everyday silence timer interrupts a lot.** At the common half-second
   setting it talks over people on about 17% of turns on clean audio and 24%
   on phone audio — roughly one turn in four. Making it wait longer fixes
   that, but then everyone gets dead air instead. No setting fixes both.

2. **Reading the words helps — until real speech recognition gets in the
   way.** With perfect transcripts, "punctuation + gate" interrupts about as
   often as an 800 ms timer but answers three times faster (256 ms instead of
   800 ms). With real recognition software on phone audio, its mistakes
   (about one word in five is wrong) push the interruptions up to 20%. The
   text model is the only home-made detector that interrupts less than the
   timer on the phone track (10% vs 17%), but it leaves a third of callers
   with no answer, so it isn't something you could switch on for real.

3. **The commercial product is the best on the phone track.** Deepgram Flux
   interrupts under 10% of turns, never answers about 12%, and takes about
   half a second. Nothing built here beats it on all three judges at once.
   The project says so plainly — the point was to measure, not to win.

The [README](README.md) has every number, and `results/` has the charts and
the longer explanations with the surprises along the way (a headset that
picked up the *next* speaker, a model that scored a suspiciously perfect
mark, a recogniser whose mistakes accidentally helped).

---

## 6. Run it yourself

You need a computer with a terminal (Mac, Linux, or Windows with WSL),
about 2 GB of free disk, and 15 minutes. Nothing is sent anywhere; nothing
needs an account or a key.

**Step 1 — get `uv`.** It's a tool that fetches the right Python and the
right libraries for you, so nothing on your computer gets changed.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Step 2 — get the project.**

```bash
git clone https://github.com/NandithReddy/project-sara.git
cd project-sara
```

**Step 3 — install and check.** `make install` downloads Python 3.12 and the
libraries into a private folder inside the project. `make test` runs 160-odd
checks, including one that confirms the 198 test recordings are untouched.

```bash
make install
make test
```

You should see something like `165 passed, 2 skipped`. (The two skips are
for an optional sample file; they are fine.)

**Step 4 — run the contest.** Three commands, one per chart. Each prints its
results as it goes.

```bash
uv run python scripts/run_baselines.py    # the summary table         (~1 minute)
uv run python -m eval.sweep               # results/tradeoff.png      (~2 minutes)
uv run python -m eval.conditions          # results/conditions.png    (~5 minutes)
```

**Step 5 — rebuild the README from what you just produced.**

```bash
make readme
```

This redraws the two headline charts and rewrites `README.md` from the
files in `results/`. If your numbers match the ones on GitHub — and they
should, to the decimal — you have just reproduced the whole project.

### What you don't need to run

Everything below is already done and its output is committed; the commands
are here so you can see there is no magic.

- **Speech recognition of the test recordings** (`scripts/transcribe_eval.py`)
  needs an Apple Silicon Mac and about 15 minutes per condition. Its output
  is cached under `results/asr/`.
- **The Deepgram passes** (`scripts/deepgram_eval.py`) need a Deepgram
  account and about 60 cents of credit. Their raw responses are cached under
  `results/asr/` too, with the model name and date.
- **Training the text model** (`scripts/build_train_set.py`,
  `scripts/train_eot.py`) needs `make install-train` (adds PyTorch) and a
  few minutes; the trained model is in `models/`.
- **Training the prosody model** needs about 1 GB of extra audio, fetched on
  demand by `scripts/fetch_train_audio.py`.

---

## 7. Try it live (Mac only)

On an Apple Silicon Mac you can talk to it. It listens on the microphone,
recognises your words as you speak, decides when you have finished, and
replies out loud.

```bash
uv run python scripts/live.py --eot punct-gated
```

Say something, pause, and it answers. `--eot timeout` makes it behave like
an ordinary silence timer so you can feel the difference; `--eot text` uses
the text model. The live path is a demonstration, not the measurement — the
numbers on the README come only from the recorded contest above, where every
run is repeatable.

---

## 8. A map of the folders

```
README.md        the results, generated from the files below — never edited by hand
GUIDE.md         this page
ENGINEERING.md   the rules the project works by, and why
src/             the detectors, the audio processing, the live demo
eval/            the contest: the harness, the sweeps, the charts
scripts/         one script per job: fetch data, train, transcribe, report
tests/           the checks (`make test`)
data/eval/       the 198 locked test recordings and their checksums
data/train/      the lists that describe the training data (the audio itself is fetched on demand)
models/          the small "is anyone talking?" model, the text model, the prosody model
results/         every number, chart and write-up; the cached recogniser and Deepgram output
```

---

## 9. The rules the project keeps

These are written down in [ENGINEERING.md](ENGINEERING.md) and enforced by
tests where a test can enforce them. In plain words:

- **Never touch the test recordings.** Not for training, not for tuning.
  A checksum file catches changes; a guard stops training code from reading them.
- **Never make a number up.** Every figure on the README comes from a file
  produced by a command you can run. If something hasn't been measured, the
  page says so instead of guessing.
- **Never fake a component.** If a model file or a cache is missing, the code
  stops and tells you, rather than quietly substituting something.
- **Never loosen a failing test to make it pass.** Fix the code or explain
  why the test was wrong.
- **Report the competitor honestly.** If the commercial product wins, say so.
  It did, on the phone track, and the README says so.

---

## 10. Questions people ask

**Why meetings and not phone calls?** Because a good, free, licensed
collection of phone calls with word-by-word timing doesn't exist, and this
one does. Meeting talk is harder in some ways (people trail off, overlap,
say "yeah so" and stop) and easier in others (no line noise). The
telephony condition squeezes the audio to phone quality to close part of
that gap; it is stated as a limitation, not hidden.

**Why 198 recordings?** Enough that one recording is half a percent, so a
difference of two points or more is meaningful, and small enough that every
contest runs in minutes on a laptop. More would be better; the harness
doesn't care how many there are.

**Can I add my own detector?** Yes, and it is the intended use. A detector
is a small class with three parts: a `name`, a `reset()` that forgets the
current turn, and an `update(...)` that receives the latest words and how
much silence there has been and returns a number from 0 to 1 — how sure it
is that the turn is over. Look at `src/baselines/punctuation.py` for the
shortest example, `src/eot/base.py` for the interface, and add yours to
`scripts/run_baselines.py` and `eval/sweep.py` to get it on the chart.

**Does it need the internet?** No. Everything runs on your machine. The only
step that ever talked to a server was the one-time Deepgram measurement,
and its answers are saved in the repo.

**Why does the README say "generated"?** So the numbers can't drift from the
files. If you edit the README by hand and change a figure, a test fails.
Change the code or the results instead, then run `make readme`.

**Where do I read more?** `results/tradeoff.md` and `results/conditions.md`
are the long-form write-ups, with what went wrong along the way.
`ENGINEERING.md` is the charter. `models/README.md` describes the models.
`data/README.md` records the data licences.
