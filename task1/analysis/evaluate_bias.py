"""Prediction-level metrics for Task 1."""
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def prediction_metrics(logits: np.ndarray, y: np.ndarray, n_classes: int = 10) -> Dict[str, float]:
    p = softmax(logits)
    pred = p.argmax(1)
    return {
        "top1": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(n_classes)), zero_division=0)),
        "mean_max_conf": float(p.max(1).mean()),
    }


def consistency(pred_clean: np.ndarray, pred_transformed: np.ndarray) -> float:
    """Fraction of images whose predicted class is unchanged by the intervention."""
    return float((pred_clean == pred_transformed).mean())


def shape_texture_counts(pred: Sequence[int], shape_label: Sequence[int], texture_label: Sequence[int]) -> Dict[str, float]:
    pred, s, t = map(np.asarray, (pred, shape_label, texture_label))
    n_shape = int((pred == s).sum())
    n_tex = int((pred == t).sum())
    n_other = int(len(pred) - n_shape - n_tex)
    denom = n_shape + n_tex
    return {
        "n_total": int(len(pred)),
        "n_shape": n_shape,
        "n_texture": n_tex,
        "n_other": n_other,
        "shape_bias_pct": 100.0 * n_shape / denom if denom else float("nan"),
        "coverage_pct": 100.0 * denom / len(pred) if len(pred) else float("nan"),
    }
