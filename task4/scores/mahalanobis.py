"""Mahalanobis unknownness with class means and ONE shared *diagonal* covariance
estimated from unaugmented CIFAR-10 training features (+1e-6 on the diagonal)."""
import numpy as np


def fit_mahalanobis(train_feats: np.ndarray, train_labels: np.ndarray, n_classes: int = 10, eps: float = 1e-6):
    mus = np.stack([train_feats[train_labels == c].mean(0) for c in range(n_classes)])
    centered = train_feats - mus[train_labels]
    var = centered.var(0) + eps  # shared diagonal covariance
    return {"mu": mus, "var": var}


def mahalanobis_unknownness(feats: np.ndarray, params) -> np.ndarray:
    """u_Mah = min_c (f - mu_c)^T Sigma^{-1} (f - mu_c)."""
    inv = 1.0 / params["var"]
    d = ((feats[:, None, :] - params["mu"][None]) ** 2 * inv).sum(-1)  # (N, C)
    return d.min(1)
