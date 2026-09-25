import numpy as np
from scipy.special import logsumexp


def energy_unknownness(logits: np.ndarray) -> np.ndarray:
    """u_Energy = - log sum_k exp(z_k)  (temperature 1)."""
    return -logsumexp(logits, axis=1)
