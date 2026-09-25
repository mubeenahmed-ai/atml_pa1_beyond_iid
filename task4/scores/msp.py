import numpy as np
from scipy.special import softmax


def msp_unknownness(logits: np.ndarray) -> np.ndarray:
    """u_MSP = 1 - max_k p_k."""
    return 1.0 - softmax(logits, axis=1).max(1)
