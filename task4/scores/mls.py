import numpy as np


def mls_unknownness(logits: np.ndarray) -> np.ndarray:
    """u_MLS = - max_k z_k."""
    return -logits.max(1)
