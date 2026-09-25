"""Incorrectly accepted unknowns under a given threshold (for the vanilla MLS analysis)."""
from typing import List

import numpy as np
import pandas as pd


def accepted_unknowns(u: np.ndarray, tau: float, pred_known: np.ndarray, fine_labels: np.ndarray,
                      fine_names: List[str], known_names: List[str], group: str, k: int = 12) -> pd.DataFrame:
    acc = np.where(u <= tau)[0]
    order = acc[np.argsort(u[acc])]  # most confidently accepted first
    rows = [{"group": group, "index_in_group": int(i), "unknown_class": fine_names[int(fine_labels[i])],
             "predicted_known": known_names[int(pred_known[i])], "score": float(u[i]), "threshold": float(tau)} for i in order[:k]]
    return pd.DataFrame(rows)


def acceptance_by_class(u: np.ndarray, tau: float, pred_known: np.ndarray, fine_labels: np.ndarray, fine_names, known_names):
    rows = []
    for c in np.unique(fine_labels):
        m = fine_labels == c
        acc = (u[m] <= tau)
        absorbed = pd.Series([known_names[p] for p in pred_known[m][acc]]).value_counts()
        rows.append({"unknown_class": fine_names[int(c)], "n": int(m.sum()), "accepted_frac": float(acc.mean()),
                     "top_absorbing_known": absorbed.index[0] if len(absorbed) else "-",
                     "top_absorbing_frac_of_accepted": float(absorbed.iloc[0] / acc.sum()) if acc.sum() else 0.0})
    return pd.DataFrame(rows).sort_values("accepted_frac", ascending=False)
