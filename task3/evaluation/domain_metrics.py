"""Mean / worst source-domain metrics."""
from typing import Dict, List

import numpy as np


def mean_worst(per_domain: Dict[str, Dict[str, float]], domains: List[str]) -> Dict[str, float]:
    out = {}
    for k in ("accuracy", "macro_f1"):
        vals = [per_domain[d][k] for d in domains]
        out[f"mean_{k}"] = float(np.mean(vals))
        out[f"worst_{k}"] = float(np.min(vals))
        out[f"worst_domain_{k}"] = domains[int(np.argmin(vals))]
    return out
