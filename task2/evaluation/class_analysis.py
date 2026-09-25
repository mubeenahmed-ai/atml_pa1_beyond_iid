"""Per-class target accuracy changes and confusions relative to a baseline."""
from typing import Dict, List

import numpy as np

from common.metrics import confusion, per_class_accuracy, top_confusions


def class_report(y: np.ndarray, pred: np.ndarray, class_names: List[str], baseline_pred: np.ndarray = None, k: int = 5) -> Dict:
    n = len(class_names)
    acc = per_class_accuracy(y, pred, n)
    cm = confusion(y, pred, n)
    out = {"per_class_acc": {c: float(acc[i]) for i, c in enumerate(class_names)},
           "confusion_matrix": cm.tolist(),
           "top_confusions": [{"true": t, "pred": p, "count": c, "frac_of_true": round(f, 3)} for t, p, c, f in top_confusions(cm, class_names, k)]}
    if baseline_pred is not None:
        b = per_class_accuracy(y, baseline_pred, n)
        delta = acc - b
        out["per_class_delta_vs_baseline"] = {c: float(delta[i]) for i, c in enumerate(class_names)}
        out["most_improved"] = class_names[int(np.nanargmax(delta))]
        out["most_degraded"] = class_names[int(np.nanargmin(delta))]
    return out
