# Vendored models

Committed rather than downloaded at runtime so a clone reproduces exactly and
runs offline. ENGINEERING.md rule 3: nothing here is fetched silently.

| File | Source | Version | License | SHA256 |
|---|---|---|---|---|
| `silero_vad.onnx` | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | v5 ONNX export, retrieved 2026-09-09 | MIT | `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` |
| `eot_v1/model.onnx` | fine-tuned from [`google/bert_uncased_L-4_H-256_A-4`](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4) on AMI `training` split | v1.1, trained 2026-09-10, epoch 2 of 4, INT8 | Apache 2.0 (base) · CC BY 4.0 (data) | `f9f14f8f993f2e6d3108403b6bc2d22f0c8275e0fc3c441bd80f1b6ee8a8054d` |

| `eot_prosody/{prosody,text,fusion}_lr.json` | logistic regression on the pause set, `scripts/train_prosody.py` | v1, trained 2026-09-12 | CC BY 4.0 (data) | prosody `6eb08e55f6cf0b78…` · fusion `08a13e05dc3a31a8…` |

## silero_vad.onnx

Signature, verified against the file rather than assumed:

| | name | shape | dtype |
|---|---|---|---|
| in | `input` | `[batch, samples]` | float32 |
| in | `state` | `[2, batch, 128]` | float32 |
| in | `sr` | scalar | int64 |
| out | `output` | `[batch, 1]` | float32 — P(speech) |
| out | `stateN` | `[2, batch, 128]` | float32 — feed back as `state` |

**`samples` must be 576 at 16kHz, not 512.** The model consumes 512-sample
(32ms) frames, but each call must be prefixed with the previous frame's last 64
samples. Omitting the prefix does not raise — it silently returns
P(speech) ≈ 0.001 for every frame of clear speech. Measured on an 11s clip:
0% speech detected without the prefix, 68% with it.

## eot_v1/model.onnx

**The text-only end-of-turn classifier, v1.1.** 11.2M parameters,
11.4MB after dynamic INT8 quantisation (fp32 export kept beside it, gitignored;
validation AP fp32 0.5512, INT8 0.5517). Inputs `input_ids`,
`attention_mask` (int64, left-truncated to 64 tokens); output `logits`
`[batch, 2]`, index 1 = complete. Tokenizer in `tokenizer.json`, for the
`tokenizers` library. Training record and validation metrics in `metadata.json`.

**Measured status: does not beat the fixed-timeout baseline** — bare or behind
the 200ms silence gate. Validation on 10 held-out speakers: average
precision 0.551, log-loss 0.269. See `results/tradeoff.md`. Kept and
committed so the negative result reproduces from the artefact in git. v1
(fp32, positive class weight 6.1×, AP 0.505) is at commit `a7df08c`.

Trained with `pos_weight=1.0`, no frozen layers, no extra dropout, and
`SARA_TRAINING=1`; the trainer refuses to run unless the eval-set guard fires.
CPU inference, one example, measured at export: p50 0.45ms, p99 1.20ms
(budget 20ms).

## eot_prosody/ — the pause classifier (Phase 7)

Three logistic regressions over the same pause examples, so "did prosody add
anything over the transcript" is a comparison between rows of one table:

| model | features | val AP | val log-loss |
|---|---|---|---|
| `prosody_lr.json` | 14 prosodic features at the last speech frame | 0.716 | 0.645 |
| `text_lr.json` | logit P(complete) of the frozen text model v1.1 | 0.860 | 0.561 |
| `fusion_lr.json` | both | 0.857 | 0.534 |

Chance AP (positive rate on validation): 0.585. Validation is
10 held-out speakers; 8,873 pauses from 5,747 training turns
(5,521 end, 3,352 mid-turn), gate 200ms, L2 1.0. Each JSON
carries weights, bias, feature mean and std, and the feature names, which
`src/eot/prosody_eot.py` checks against the tracker before loading. Inference
is one dot product; the fusion model also runs the text model once per pause.

**Which cues carry the signal** (standardised coefficients, `prosody_lr.json`;
positive means "more like an end"):

| feature | weight |
|---|---|
| `energy_db` | -0.251 |
| `voiced_fraction` | -0.239 |
| `energy_slope_db_per_s` | +0.190 |
| `energy_drop_from_peak_db` | -0.147 |
| `f0_range_st` | +0.136 |
| `energy_std_db` | -0.114 |
| `ms_since_voiced` | -0.085 |
| `last_run_over_mean_run` | -0.074 |
| `last_run_ms` | -0.066 |
| `n_runs` | -0.033 |
| `f0_last_semitones_vs_median` | -0.031 |
| `last_run_energy_vs_window_db` | -0.022 |
| `f0_slope_st_per_s` | +0.014 |
| `last_run_f0_fall_st` | +0.010 |

Energy carries it — the last speech frame before a real end is quieter and
sits further below the window's peak. **Pitch carries almost nothing** at this
tracker's resolution on AMI headsets: the final-fall and slope weights are
near zero. In the fusion model the text logit dominates (`logit_p_text`
+1.38); the energy terms keep their sign.

**A leak, found and fixed before any eval.** The first build sampled the
features 200ms into the pause. For an end pause that is inside the region the
caller channel zeroes (true_end + 150ms), so `energy_db` read −120dB of digital
silence and the classifier scored a perfect 1.000 AP on held-out speakers — a
digital-silence detector. Features now come from the last speech frame before
the pause, which the zeroed tail never reaches. The leaked models are not in
the repo. Treat any AP near 1.0 on this data as a leak until proven otherwise.

**Label noise, counted:** 359 of 3,352 mid-turn pauses (10.7%) are followed
by VAD speech but no further word — a laugh, a breath, or bleed. Kept, because
the eval boundary is VAD-based too and the labels should mean the same thing.
