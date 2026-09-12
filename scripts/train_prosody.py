"""Fit the pause classifier: logistic regression, three feature sets.

  prosody   the 14 tracker features
  text      logit(P_text) alone -- the frozen text model, as a one-weight
            reference so the comparison is inside one table
  fusion    both

Held-out SPEAKERS for validation, as every trainer here. IRLS with a small L2
penalty on standardised features: 15 dimensions, closed-form-ish, no torch.
Writes models/eot_prosody/<kind>_lr.json with weights, bias, mean, std and the
feature names, which src/eot/prosody_eot.py checks against the tracker.

The coefficients are the point: which cues carry the signal, with signs.

Run:
    uv run python scripts/train_prosody.py
"""

from __future__ import annotations

import os

os.environ["SARA_TRAINING"] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.dataset import EvalSetViolation, load_eval_set  # noqa: E402
from src.audio.prosody import FEATURE_NAMES  # noqa: E402
from src.eot.prosody_eot import TEXT_FEATURE, logit  # noqa: E402

TRAIN_DIR = REPO / "data" / "train"
OUT_DIR = REPO / "models" / "eot_prosody"
VAL_SPEAKER_FRACTION = 0.10


def prove_rule_1_is_live() -> None:
    try:
        load_eval_set()
    except EvalSetViolation:
        return
    raise SystemExit("FATAL: eval-set guard did not fire with SARA_TRAINING set.")


def average_precision(scores, labels):
    order = np.argsort(-scores, kind="stable")
    hits = labels[order]
    prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((prec * hits).sum() / max(int(hits.sum()), 1))


def log_loss(p, y, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_lr(X, y, l2=1.0, iters=50):
    """IRLS (Newton) for L2-regularised logistic regression on standardised X."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    reg = np.eye(d + 1) * l2
    reg[-1, -1] = 0.0  # never shrink the bias
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Xb @ w)))
        g = Xb.T @ (p - y) + reg @ w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + reg
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w[:-1], float(w[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pauses", type=Path, default=TRAIN_DIR / "pauses.jsonl")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args()
    prove_rule_1_is_live()
    print("rule 1 guard: live")

    ex = [json.loads(line) for line in args.pauses.open()]
    speakers = sorted({e["global_name"] for e in ex})
    rng = random.Random(args.seed)
    rng.shuffle(speakers)
    n_val = max(1, int(len(speakers) * VAL_SPEAKER_FRACTION))
    val_spk = set(speakers[:n_val])
    tr = [e for e in ex if e["global_name"] not in val_spk]
    va = [e for e in ex if e["global_name"] in val_spk]
    if not tr or not va:
        raise SystemExit(
            f"split left train={len(tr)} val={len(va)} examples over "
            f"{len(speakers)} speakers; need more turns (rule 3: not proceeding)."
        )
    print(
        f"train {len(tr):,} pauses / val {len(va):,} ({n_val} held-out speakers)  "
        f"positives: train {np.mean([e['label'] for e in tr]) * 100:.1f}%"
    )

    def matrix(rows, kind):
        cols = []
        if kind in ("prosody", "fusion"):
            cols.append(np.array([e["prosody"] for e in rows], dtype=np.float64))
        if kind in ("text", "fusion"):
            cols.append(
                np.array([[logit(e["p_text"])] for e in rows], dtype=np.float64)
            )
        return np.hstack(cols), np.array([e["label"] for e in rows], dtype=np.float64)

    names = {
        "prosody": list(FEATURE_NAMES),
        "text": [TEXT_FEATURE],
        "fusion": list(FEATURE_NAMES) + [TEXT_FEATURE],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for kind in ("prosody", "text", "fusion"):
        X, y = matrix(tr, kind)
        mean, std = np.atleast_1d(X.mean(0)), np.atleast_1d(X.std(0))
        std = np.where(std == 0, 1.0, std)
        w, b = fit_lr((X - mean) / std, y, l2=args.l2)
        metrics = {}
        for split, rows in (("train", tr), ("val", va)):
            if not rows:
                continue
            Xs, ys = matrix(rows, kind)
            p = 1 / (1 + np.exp(-(((Xs - mean) / std) @ w + b)))
            metrics[split] = {
                "n": int(len(ys)),
                "average_precision": average_precision(p, ys),
                "log_loss": log_loss(p, ys),
                "majority_ap": float(ys.mean()),
            }
        model = {
            "kind": kind,
            "features": names[kind],
            "w": w.tolist(),
            "b": b,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "uses_text": kind in ("text", "fusion"),
            "l2": args.l2,
            "seed": args.seed,
            "val_speakers": n_val,
            "metrics": metrics,
            "pauses_file": str(args.pauses.relative_to(REPO))
            if args.pauses.is_relative_to(REPO)
            else str(args.pauses),
        }
        (args.out / f"{kind}_lr.json").write_text(json.dumps(model, indent=2))
        v = metrics.get("val", {})
        ap_, ch, ll = (
            v.get("average_precision", float("nan")),
            v.get("majority_ap", float("nan")),
            v.get("log_loss", float("nan")),
        )
        print(f"{kind:8} val AP={ap_:.3f} (chance {ch:.3f})  logloss={ll:.3f}")
        summary[kind] = v
        if kind != "text":
            order = np.argsort(-np.abs(w))
            print("   " + "  ".join(f"{names[kind][i]}={w[i]:+.2f}" for i in order[:6]))
    print(f"wrote {args.out.relative_to(REPO)}/{{prosody,text,fusion}}_lr.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
