"""Domain separability score.

Freeze the backbone, collect equal numbers of source-validation and target features,
70/30 split with seed 6304, balanced logistic regression (C=1). Held-out accuracy is
the separability score (50% = chance for two domains, 33.3% for three).
"""
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def separability(feature_groups: List[np.ndarray], seed: int = 6304, C: float = 1.0, test_size: float = 0.3) -> Dict:
    """feature_groups[i] = features of domain group i. Groups are subsampled to equal size."""
    rng = np.random.RandomState(seed)
    n = min(len(g) for g in feature_groups)
    X, y = [], []
    for i, g in enumerate(feature_groups):
        sel = rng.choice(len(g), size=n, replace=False)
        X.append(g[sel]); y.append(np.full(n, i))
    X, y = np.concatenate(X), np.concatenate(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    clf.fit(Xtr, ytr)
    return {"score": float(clf.score(Xte, yte)), "train_acc": float(clf.score(Xtr, ytr)), "n_per_group": int(n),
            "n_groups": len(feature_groups), "chance": 1.0 / len(feature_groups)}
