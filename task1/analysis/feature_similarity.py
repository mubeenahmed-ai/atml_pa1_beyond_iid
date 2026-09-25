"""Representation-stability measures (cosine similarity between clean and transformed features)."""
import numpy as np


def cosine_stability(f_clean: np.ndarray, f_trans: np.ndarray) -> float:
    a = f_clean / (np.linalg.norm(f_clean, axis=1, keepdims=True) + 1e-12)
    b = f_trans / (np.linalg.norm(f_trans, axis=1, keepdims=True) + 1e-12)
    return float((a * b).sum(1).mean())


def per_example_cosine(f_clean: np.ndarray, f_trans: np.ndarray) -> np.ndarray:
    a = f_clean / (np.linalg.norm(f_clean, axis=1, keepdims=True) + 1e-12)
    b = f_trans / (np.linalg.norm(f_trans, axis=1, keepdims=True) + 1e-12)
    return (a * b).sum(1)
