# Vendored models

Committed rather than downloaded at runtime so a clone reproduces exactly and
runs offline. ENGINEERING.md rule 3: nothing here is fetched silently.

| File | Source | Version | License | SHA256 |
|---|---|---|---|---|
| `silero_vad.onnx` | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | v5 ONNX export, retrieved 2026-09-09 | MIT | `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` |

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
