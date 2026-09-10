# Vendored models

Committed rather than downloaded at runtime so a clone reproduces exactly and
runs offline. ENGINEERING.md rule 3: nothing here is fetched silently.

| File | Source | Version | License | SHA256 |
|---|---|---|---|---|
| `silero_vad.onnx` | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | v5 ONNX export, retrieved 2026-09-09 | MIT | `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` |
| `eot_v1/model.onnx` | fine-tuned from [`google/bert_uncased_L-4_H-256_A-4`](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4) on AMI `training` split | v1, trained 2026-09-10, epoch 1 of 3 | Apache 2.0 (base) · CC BY 4.0 (data) | `be7dc53229d18cab230b3fff068f8a99aa535f62d73bc4152cfaf8e2db878814` |

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

**The text-only end-of-turn classifier, v1.** 11.2M parameters,
44.8MB fp32. Inputs `input_ids`, `attention_mask` (int64, left-truncated
to 64 tokens); output `logits` `[batch, 2]`, index 1 = complete.
Tokenizer in `tokenizer.json`, for the `tokenizers` library. Training record and
validation metrics in `metadata.json`.

**Measured status: does not beat the fixed-timeout baseline.** Validation on
10 held-out speakers: average precision 0.505,
log-loss 0.409. On the frozen eval set it is dominated by the Silero
timer at every threshold -- see `results/tradeoff.md` for the numbers and the
diagnosis. Kept and committed so the negative result is reproducible.

CPU inference, one example: p50 0.41ms, p99 1.11ms
(budget 20ms). Trained with `SARA_TRAINING=1`; the trainer
refuses to run unless the eval-set guard fires.
