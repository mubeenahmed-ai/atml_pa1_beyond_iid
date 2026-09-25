"""Common local sharpness proxy.

Fixed validation batch (32 images per source domain, seed 6304), model in eval mode:
    delta = L(theta + eps) - L(theta),  eps = rho * grad L / ||grad L||_2,  rho = 0.05
"""
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from shared.pacs import PACSSubset, SOURCE_DOMAINS, eval_transform


def fixed_val_batch(table, splits, per_domain: int = 32, seed: int = 6304):
    rng = np.random.RandomState(seed)
    idx = []
    for d in SOURCE_DOMAINS:
        pool = np.asarray(splits["domains"][d]["val"])
        idx.extend(rng.choice(pool, size=per_domain, replace=False).tolist())
    ds = PACSSubset(table, idx, eval_transform())
    xs, ys = zip(*[(ds[i][0], ds[i][1]) for i in range(len(ds))])
    return torch.stack(xs), torch.tensor(ys), idx


def sharpness_proxy(model, x, y, rho: float = 0.05) -> Dict[str, float]:
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.grad = None
    _, logits = model(x)
    loss0 = F.cross_entropy(logits, y)
    loss0.backward()
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    gnorm = torch.sqrt(sum((g**2).sum() for g in grads)) + 1e-12
    with torch.no_grad():
        eps = [rho * g / gnorm for g in grads]
        for p, e in zip(params, eps):
            p.add_(e)
        _, logits1 = model(x)
        loss1 = F.cross_entropy(logits1, y)
        for p, e in zip(params, eps):
            p.sub_(e)
    for p in params:
        p.grad = None
    return {"loss_val": float(loss0.item()), "loss_val_perturbed": float(loss1.item()),
            "delta_sharp": float((loss1 - loss0).item()), "grad_norm": float(gnorm.item()), "rho": rho}
