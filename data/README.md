# Datasets

ENGINEERING.md section 9: record the license of every dataset here **before**
downloading it.

| Dataset | Version | License | Used for | Downloaded |
|---|---|---|---|---|
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | manual annotations v1.6.2 | [CC BY 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) | eval + train (disjoint meetings) | annotations 2026-09-10; eval audio 2026-09-10; training audio 2026-09-12 |

`data/raw/` is gitignored. `data/eval/` is frozen once created and is never
trained or tuned on (ENGINEERING.md rule 1).

## AMI Meeting Corpus

**License:** CC BY 4.0 — commercial use and redistribution permitted, attribution
required. Verified against the corpus licence page, not assumed.

**Attribution.** Carletta, J. et al. *The AMI Meeting Corpus: A
Pre-Announcement.* Machine Learning for Multimodal Interaction (2006).

**Why this corpus.** Chosen over three alternatives on one requirement: section
10 replays *force-aligned gold transcripts*, and "gold" means human
transcription with word-level timings. AMI has manual orthographic transcription
with word timings produced by forced alignment.

- **CANDOR** (850h, dyadic, free on request) was rejected despite being the best
  structural fit: its word timings come from Speechmatics ASR, not human
  transcription. Building a gold eval on ASR output imports word errors into the
  ground truth and silently invalidates every number downstream.
- **Switchboard** (LDC97S62) is the better corpus for this problem — native 8kHz
  two-party telephone speech, and the Mississippi State human transcripts carry
  no licence restrictions. Only the audio is paywalled, at a four-figure fee for
  non-members. Not viable without an institutional LDC membership.
- **TurnBench** (Sesame) is purpose-built for turn-taking, with `mid-turn pause`
  as a labelled class, but is studio-recorded voice actors. Acted speech
  contradicts the "realistic conditions" claim that is our stated contribution
  (section 8).

**Known limitations of this choice, stated up front:**

1. **Four-person meetings, not dyadic.** Turn ends are extracted where one
   speaker's contiguous contribution is followed by a different speaker, with
   overlapped regions and backchannels discarded. Each speaker has an individual
   headset mic, so speaker attribution is unambiguous. ~200 turns from ~100
   hours means we can afford to be extremely selective.
2. **16kHz headset audio, not telephony.** This is the correct development band
   per section 5; the 8kHz telephony condition is produced by downsampling and
   band-limiting in Phase 6, not sourced natively.
3. **Meeting speech, not customer-service speech.** The deployment target is a
   phone agent. Meeting turn-taking has different dynamics, and any result
   carries that caveat.

**Downloaded so far:** annotations only, not audio.
`ami_public_manual_1.6.2.zip`, 22,887,865 bytes,
SHA256 `b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d`.
Audio is fetched later and only for the meetings the eval set actually samples,
to keep the download bounded.

**Format**, verified against the extracted files rather than the documentation:

```xml
<w nite:id="ES2002a.B.words0" starttime="50.42" endtime="50.99">Okay</w>
<w nite:id="ES2002a.B.words1" starttime="50.99" endtime="50.99" punc="true">.</w>
<vocalsound nite:id="ES2002a.B.words4" starttime="55.415" endtime="55.415" type="other"/>
```

One file per meeting per speaker, `<meeting>.<speaker>.words.xml`, 687 files.
Element counts across all of them:

| element | count | handling |
|---|---|---|
| `<w>` | 1,147,783 | the transcript; `punc="true"` marks zero-duration punctuation |
| `<disfmarker>` | 27,395 | inline disfluency marker -- used to select section 3's disfluent cases |
| `<vocalsound>` | 27,073 | laughter, coughs; excluded from the word stream |
| `<gap>` | 5,125 | untranscribable; turns containing one are discarded |
| `<transformerror>` | 30 | excluded |

**Caveat for the punctuation baseline.** AMI punctuation is inserted by human
annotators, so baseline #3 reads cleaner punctuation than any real STT emits.
That baseline is therefore flattered on gold transcripts, and the gap is
quantified by the Phase 6 real-ASR condition, not assumed away.

**Training audio (Phase 7), fetched 2026-09-12.** 5,747 turn segments from the
`training` split, by HTTP range request through the same function as the eval
audio (`scripts/fetch_eval_audio.fetch_turns`, called by
`scripts/fetch_train_audio.py`): 718MB fetched, 294MB as FLAC under
`data/train/audio/`, **gitignored** and reproducible from `data/train/turns.jsonl`.
Segment = 500ms pre-roll + turn + 1500ms; boundary refined with Silero as for
eval (5,525 from audio, 222 quiet single-word turns kept on the annotation;
annotated minus audio end p50 −84ms, p90 +107ms). `data/train/audio_set.json`
carries the refined boundaries.

Four headset files in `ES2010d` are stereo. The fetch checks whether the two
channels are the same signal before using one: all four are dual-mono at L/R
correlation 1.000, so the left channel is used and no turn is skipped. A
stereo file whose channels differed would have had its turns skipped, out
loud, rather than downmixed on a guess.

**Pause examples (Phase 7), built 2026-09-12.** `data/train/pauses.jsonl`:
8,873 pauses from 5,747 training turns — one per point where silence reached
200ms inside a turn, on the caller channel. Label 1 (5,521, 62.2%) if the
speaker never spoke again; 0 (3,352, 37.8%) if they did. Features are the 14
prosodic numbers at the **last speech frame before the pause** (see
`models/README.md` for why not at the pause), the transcript so far, and the
frozen text model's P(complete) on it. 359 of the mid-turn pauses (10.7%) are
followed by VAD speech but no word — label noise, counted and kept for
consistency with the VAD-based eval boundary. Review: `data/train/PAUSE_REVIEW.md`.

**Sent to a third party (Phase 10).** For the Deepgram baselines, the 198 eval
segments (20 minutes, caller channel, once at wideband and once through the
telephony simulation) were streamed to Deepgram's Nova-3 and Flux APIs on
2026-09-13 by `scripts/deepgram_eval.py`. AMI's CC BY 4.0 licence permits
this; no other data leaves the machine, and the responses — not the audio —
are cached under `results/asr/` with the model, parameters and date, so the
measurement reproduces without sending anything again.
