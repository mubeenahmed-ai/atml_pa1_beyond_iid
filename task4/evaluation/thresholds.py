"""Validation-calibrated rejection: tau = 95th percentile of u on CIFAR-10 validation;
accept x iff u(x) <= tau (aims to accept 95% of known examples)."""
import numpy as np


def calibrate_threshold(u_val_known: np.ndarray, tpr: float = 0.95) -> float:
    return float(np.percentile(u_val_known, 100 * tpr))


def operating_point(tau: float, u_test_known, u_near, u_far):
    acc_known = float((u_test_known <= tau).mean())
    return {"tau": tau, "known_test_acceptance": acc_known,
            "near_rejection": float((u_near > tau).mean()), "far_rejection": float((u_far > tau).mean()),
            "all_rejection": float((np.concatenate([u_near, u_far]) > tau).mean()),
            "fpr95_near": float((u_near <= tau).mean()), "fpr95_far": float((u_far <= tau).mean()),
            "fpr95_all": float((np.concatenate([u_near, u_far]) <= tau).mean())}
