"""Train the text-only EOT classifier (Phase 4, v1) and export it to ONNX.

Runs with SARA_TRAINING=1 from the first line, so the guard in eval/dataset.py
is live for the whole process, and proves it by trying to read the eval set
and requiring that to fail (rule 1, executable).

Data: data/train/examples.jsonl rendered WITHOUT punctuation. Validation is a
held-out set of SPEAKERS from the training split, never the eval set.

Two choices worth defending:

- Left truncation. A prefix can run to 196 words, but the decision "did this
  end?" lives at the END of the text. Standard truncation keeps the start and
  throws the end away, which is exactly backwards here. The tokenizer is set to
  truncate on the left so the last MAX_LEN tokens survive.

- Early stopping on validation log-loss, keeping the best epoch's weights.
  Every run so far had its best generalisation at epoch 1 and got worse from
  there while train loss kept falling: the AMI scenario meetings all discuss
  the same fictional product, so there is content to memorise. Validation is
  by held-out speakers precisely so this shows up.

- Average precision alongside F1. The harness sweeps the fire threshold, so
  the number that matters is whether P(complete) RANKS true ends above
  mid-turn prefixes, not how the model does at 0.5.

- Hard negatives up-weighted. The prefix one word short of the end (k = n-1)
  is the only negative that looks like a positive; every other negative is
  obviously mid-sentence. It gets extra weight so the model is graded on the
  distinction that matters rather than the easy ones.

Run:
    make install-train
    uv run python scripts/train_eot.py
    uv run python scripts/train_eot.py --model google/bert_uncased_L-4_H-256_A-4
"""

from __future__ import annotations

import os

os.environ["SARA_TRAINING"] = "1"  # before ANY import that could reach data/eval

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from ami import render  # noqa: E402

from eval.dataset import EvalSetViolation, load_eval_set  # noqa: E402

TRAIN_DIR = REPO / "data" / "train"
OUT_DIR = REPO / "models" / "eot_v1"
DEFAULT_MODEL = "google/bert_uncased_L-2_H-128_A-2"  # BERT-tiny, 4.4M params
MAX_LEN = 64
HARD_NEGATIVE_WEIGHT = 2.0
VAL_SPEAKER_FRACTION = 0.10


def prove_rule_1_is_live() -> None:
    try:
        load_eval_set()
    except EvalSetViolation:
        return
    raise SystemExit(
        "FATAL: the eval-set guard did not fire with SARA_TRAINING set. "
        "Refusing to train until rule 1 is enforceable."
    )


def load_examples() -> tuple[list[dict], list[dict]]:
    turns = {}
    with (TRAIN_DIR / "turns.jsonl").open() as f:
        for line in f:
            t = json.loads(line)
            turns[t["turn_id"]] = t
    rows = []
    with (TRAIN_DIR / "examples.jsonl").open() as f:
        for line in f:
            e = json.loads(line)
            words = [tuple(w) for w in turns[e["turn_id"]]["words"]]
            e["text"] = render(words, e["k"], punctuated=False)
            rows.append(e)
    return rows, list(turns.values())


def split_by_speaker(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    speakers = sorted({r["global_name"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(speakers)
    n_val = max(1, int(len(speakers) * VAL_SPEAKER_FRACTION))
    val_speakers = set(speakers[:n_val])
    train = [r for r in rows if r["global_name"] not in val_speakers]
    val = [r for r in rows if r["global_name"] in val_speakers]
    assert not ({r["global_name"] for r in train} & val_speakers)
    return train, val


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold-free ranking quality of P(complete). 1.0 = every true end
    outranks every mid-turn prefix."""
    order = np.argsort(-scores, kind="stable")
    hits = labels[order]
    precision_at_k = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision_at_k * hits).sum() / max(int(hits.sum()), 1))


def metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    pred = logits.argmax(-1)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    z = logits - logits.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    p_complete = np.exp(logp[:, 1])
    return {
        "average_precision": average_precision(p_complete, labels),
        "n": int(len(labels)),
        "accuracy": float((pred == labels).mean()),
        "majority_accuracy": float(max(labels.mean(), 1 - labels.mean())),
        "precision_complete": p,
        "recall_complete": r,
        "f1_complete": 2 * p * r / (p + r) if p + r else 0.0,
        "log_loss": float(-logp[np.arange(len(labels)), labels].mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap examples")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    prove_rule_1_is_live()
    print("rule 1 guard: live (eval set unreadable while training)")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    rows, _turns = load_examples()
    if args.limit:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.limit]
    train, val = split_by_speaker(rows, args.seed)
    n_pos = sum(r["label"] for r in train)
    print(
        f"train: {len(train):,} examples ({n_pos:,} complete)  "
        f"val: {len(val):,} examples, {len({r['global_name'] for r in val})} "
        f"held-out speakers  device: {device}"
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2
    ).to(device)

    # Class weights balance the 6:1 skew; the k = n-1 prefix gets extra weight.
    w_pos = (len(train) - n_pos) / n_pos
    class_w = torch.tensor([1.0, w_pos], dtype=torch.float32, device=device)

    def batches(data, shuffle):
        idx = list(range(len(data)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), args.batch):
            chunk = [data[j] for j in idx[i : i + args.batch]]
            enc = tok(
                [r["text"] for r in chunk],
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            labels = torch.tensor([r["label"] for r in chunk])
            hard = torch.tensor(
                [HARD_NEGATIVE_WEIGHT if r["k"] == r["n"] - 1 else 1.0 for r in chunk]
            )
            yield enc["input_ids"], enc["attention_mask"], labels, hard

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_total = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, (s + 1) / (0.06 * steps_total)) * max(0.0, 1 - s / steps_total)
        ),
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_w, reduction="none")

    def evaluate(data):
        model.eval()
        outs, labs = [], []
        with torch.no_grad():
            for ids, mask, labels, _ in batches(data, shuffle=False):
                logits = model(
                    input_ids=ids.to(device), attention_mask=mask.to(device)
                ).logits
                outs.append(logits.float().cpu().numpy())
                labs.append(labels.numpy())
        model.train()
        return metrics(np.concatenate(outs), np.concatenate(labs))

    history = []
    best_loss, best_state, best_epoch = float("inf"), None, 0
    t_start = time.perf_counter()
    model.train()
    for epoch in range(1, args.epochs + 1):
        running, n_steps = 0.0, 0
        for ids, mask, labels, hard in batches(train, shuffle=True):
            logits = model(
                input_ids=ids.to(device), attention_mask=mask.to(device)
            ).logits
            loss = (loss_fn(logits, labels.to(device)) * hard.to(device)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            n_steps += 1
        m = evaluate(val)
        m["epoch"], m["train_loss"] = epoch, running / n_steps
        history.append(m)
        improved = m["log_loss"] < best_loss
        if improved:
            best_loss, best_epoch = m["log_loss"], epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        print(
            f"epoch {epoch}: train_loss={m['train_loss']:.4f}  "
            f"val AP={m['average_precision']:.3f} logloss={m['log_loss']:.4f}  "
            f"P={m['precision_complete']:.3f} R={m['recall_complete']:.3f} "
            f"F1={m['f1_complete']:.3f}{'  <- best' if improved else ''}"
        )
    train_s = time.perf_counter() - t_start
    model.load_state_dict(best_state)
    print(f"keeping epoch {best_epoch} (val logloss {best_loss:.4f})")

    # ---- export -------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    sample = tok(["my order number is", "yes"], padding=True, return_tensors="pt")
    ids, mask = sample["input_ids"], sample["attention_mask"]
    onnx_path = args.out / "model.onnx"
    torch.onnx.export(
        model,
        (ids, mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    tok.backend_tokenizer.save(str(args.out / "tokenizer.json"))

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=mask).logits.numpy()
    out = sess.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]
    max_diff = float(np.abs(out - ref).max())
    if max_diff > 1e-3:
        raise SystemExit(f"FATAL: ONNX disagrees with torch by {max_diff:.2e}")

    # Single-example CPU latency: the live setting, one update at a time.
    lat = []
    for r in val[:400]:
        enc = tok(r["text"], truncation=True, max_length=MAX_LEN, return_tensors="np")
        feed = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        t0 = time.perf_counter()
        sess.run(None, feed)
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat_ms = {
        "p50": float(np.percentile(lat, 50)),
        "p95": float(np.percentile(lat, 95)),
        "p99": float(np.percentile(lat, 99)),
        "budget_ms": 20.0,
    }

    meta = {
        "base_model": args.model,
        "params": int(sum(p.numel() for p in model.parameters())),
        "max_len": MAX_LEN,
        "truncation_side": "left",
        "punctuated_input": False,
        "hard_negative_weight": HARD_NEGATIVE_WEIGHT,
        "class_weight_complete": float(w_pos),
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "train_examples": len(train),
        "val_examples": len(val),
        "val_speakers": len({r["global_name"] for r in val}),
        "train_seconds": round(train_s, 1),
        "history": history,
        "best_epoch": best_epoch,
        "final_val": history[best_epoch - 1],
        "onnx_vs_torch_max_abs_diff": max_diff,
        "onnx_bytes": onnx_path.stat().st_size,
        "cpu_latency_ms_single_example": lat_ms,
    }
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    shown = onnx_path.relative_to(REPO) if onnx_path.is_relative_to(REPO) else onnx_path
    print(
        f"\nexported {shown} ({meta['onnx_bytes'] / 1e6:.1f}MB), "
        f"onnx==torch to {max_diff:.1e}"
    )
    print(
        f"CPU latency, one example: p50={lat_ms['p50']:.2f}ms "
        f"p95={lat_ms['p95']:.2f}ms p99={lat_ms['p99']:.2f}ms  (budget 20ms)"
    )
    print(f"trained in {train_s:.0f}s on {device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
