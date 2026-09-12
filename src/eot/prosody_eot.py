"""The pause classifier: at a pause, decide end-of-turn or hesitation.

Phase 7's model. It runs only once silence reaches the gate, and reads the
prosody of the speech that just ended -- final fall, lengthening, energy
trailing off -- and optionally the text model's opinion of the transcript so
far. A logistic regression over 14 (or 15) standardised features, stored as
JSON: weights, bias, mean, std, feature names. Inference is a dot product.

Text-only detectors ignore Update.prosody; this one ignores nothing it is
given. With `uses_text` it wraps the frozen text model and adds logit(P_text)
as a feature, so "did prosody add information over the transcript" is a
comparison between two rows of the same table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.audio.prosody import FEATURE_NAMES
from src.eot.base import Update
from src.eot.gated import DEFAULT_GATE_MS

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "eot_prosody"
TEXT_FEATURE = "logit_p_text"


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


class ProsodyEOT:
    """P(turn_complete) at a pause, from prosody (and optionally the text)."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        kind: str = "fusion",
        gate_ms: float = DEFAULT_GATE_MS,
    ) -> None:
        path = Path(model_path) if model_path else DEFAULT_MODEL_DIR / f"{kind}_lr.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Train it first:\n"
                f"  uv run python scripts/build_pause_set.py\n"
                f"  uv run python scripts/train_prosody.py\n"
                f"Not substituting anything (rule 3)."
            )
        m = json.loads(path.read_text())
        self.meta = m
        self.kind = m["kind"]
        self.features = tuple(m["features"])
        self.uses_text = bool(m.get("uses_text", False))
        expected = FEATURE_NAMES + ((TEXT_FEATURE,) if self.uses_text else ())
        if self.features != expected:
            raise ValueError(
                f"{path}: feature order {self.features} does not match the "
                f"tracker's {expected}; the model and the tracker disagree."
            )
        self._w = np.asarray(m["w"], dtype=np.float64)
        self._b = float(m["b"])
        self._mean = np.asarray(m["mean"], dtype=np.float64)
        self._std = np.asarray(m["std"], dtype=np.float64)
        self.gate_ms = float(gate_ms)
        self._text = None
        if self.uses_text:
            from src.eot.model import TextEOT

            self._text = TextEOT()
        self.name = f"prosody_{self.kind}+gate{int(self.gate_ms)}"

    def reset(self) -> None:
        if self._text is not None:
            self._text.reset()

    def update(self, u: Update) -> float:
        if u.silence_ms < self.gate_ms:
            # Keep the text model's cache warm so a fire is not delayed by an
            # inference at the moment the gate opens.
            if self._text is not None and u.text:
                self._text.update(u)
            return 0.0
        if not u.prosody:
            raise ValueError(
                f"{self.name} needs Update.prosody; this source supplies none. "
                f"Not guessing (rule 3)."
            )
        x = list(u.prosody)
        if self._text is not None:
            x.append(logit(self._text.update(u)))
        z = (np.asarray(x, dtype=np.float64) - self._mean) / self._std
        return float(1.0 / (1.0 + np.exp(-(float(self._w @ z) + self._b))))

    def inference_latency_ms(self) -> dict:
        fn = getattr(self._text, "inference_latency_ms", None)
        return fn() if fn else {"n": 0}
