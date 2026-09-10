"""THE DELIVERABLE: the text-only end-of-turn classifier, at inference.

Runs the exported ONNX model on CPU through onnxruntime and the Rust
`tokenizers` library -- no torch, no transformers on this path (section 5).

Text-only, by design of Phase 4: `silence_ms` is ignored. The model reads the
transcript so far and emits P(turn_complete). It therefore cannot resolve the
cases the ceiling analysis flagged -- "Yeah" is a finished turn 63% of the time
on the training data and no function of the text can tell which -- and it
cannot fire before the recogniser has emitted any text at all. Both are known
and measured, not surprises.

Punctuation is stripped before encoding, because the model was trained on
words alone. The harness replays gold text WITH annotator punctuation so that
the punctuation baseline can read it; taking that cue here would hand this
model the same optimism gap.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from src.eot.base import Update

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "eot_v1"

_PUNCT = re.compile(r"[.?!,]+")
_SPACES = re.compile(r"\s+")


def strip_punctuation(text: str) -> str:
    """Remove the marks AMI attaches as punc_after; leave in-word marks alone.

    "it's", "kick-off" and "L_C_D_" survive; "hello there." becomes
    "hello there", matching how the training examples were rendered.
    """
    return _SPACES.sub(" ", _PUNCT.sub("", text)).strip()


class TextEOT:
    """P(turn_complete) from the transcript so far."""

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR) -> None:
        model_dir = Path(model_dir)
        onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"{onnx_path} not found. Train and export it first:\n"
                f"  make install-train && uv run python scripts/train_eot.py\n"
                f"Not substituting anything (rule 3)."
            )
        meta = json.loads((model_dir / "metadata.json").read_text())
        self.name = f"text_eot_v{meta.get('version', '1')}"
        self.max_len = int(meta["max_len"])
        self.meta = meta

        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        # Keep the END of long prefixes: that is where "did it finish?" lives.
        self._tok.enable_truncation(self.max_len, direction=meta["truncation_side"])
        self._tok.no_padding()
        self._session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self._latencies_ms: list[float] = []
        self.reset()

    def reset(self) -> None:
        # Cache by text: the harness and the live loop call update() ~31 times a
        # second but the transcript only changes a few times a second. Without
        # this the model would re-encode an identical string 30 times over.
        self._last_text: str | None = None
        self._last_p = 0.0

    def update(self, u: Update) -> float:
        text = strip_punctuation(u.text)
        if not text:
            return 0.0  # nothing said yet: a turn that never started cannot end
        if text == self._last_text:
            return self._last_p
        self._last_text, self._last_p = text, self._infer(text)
        return self._last_p

    def _infer(self, text: str) -> float:
        enc = self._tok.encode(text)
        ids = np.asarray([enc.ids], dtype=np.int64)
        mask = np.asarray([enc.attention_mask], dtype=np.int64)
        t0 = time.perf_counter()
        logits = self._session.run(None, {"input_ids": ids, "attention_mask": mask})[0][
            0
        ]
        self._latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        z = logits - logits.max()
        p = np.exp(z) / np.exp(z).sum()
        return float(p[1])

    def inference_latency_ms(self) -> dict:
        """Model-only cost per real inference, excluding cache hits.

        The harness measures every update() call, most of which hit the cache
        and cost microseconds; this is the honest number against the 20ms
        budget in section 5.
        """
        if not self._latencies_ms:
            return {"n": 0}
        a = np.asarray(self._latencies_ms)
        return {
            "n": int(len(a)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "budget_ms": 20.0,
        }
