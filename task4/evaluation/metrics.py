import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(u_known: np.ndarray, u_unknown: np.ndarray) -> float:
    """AUROC treating unknown as the positive class and u as its score."""
    y = np.concatenate([np.zeros(len(u_known)), np.ones(len(u_unknown))])
    s = np.concatenate([u_known, u_unknown])
    return float(roc_auc_score(y, s))


def roc(u_known, u_unknown):
    y = np.concatenate([np.zeros(len(u_known)), np.ones(len(u_unknown))])
    s = np.concatenate([u_known, u_unknown])
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr
