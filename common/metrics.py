"""Classification metrics shared by all tasks."""
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int = None) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(n_classes)) if n_classes is not None else None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
    }


def per_class_accuracy(y_true, y_pred, n_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = np.zeros(n_classes)
    for c in range(n_classes):
        m = y_true == c
        acc[c] = (y_pred[m] == c).mean() if m.any() else np.nan
    return acc


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def top_confusions(cm: np.ndarray, class_names: Sequence[str], k: int = 5):
    """Return the k largest off-diagonal entries as (true, pred, count, frac_of_true)."""
    out = []
    n = cm.shape[0]
    for i in range(n):
        row = cm[i].sum()
        for j in range(n):
            if i != j and cm[i, j] > 0:
                out.append((class_names[i], class_names[j], int(cm[i, j]), float(cm[i, j] / max(row, 1))))
    out.sort(key=lambda t: -t[2])
    return out[:k]
