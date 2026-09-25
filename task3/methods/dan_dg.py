"""DAN-DG: ERM + (lambda/3) * sum over the three unordered source pairs of MMD^2.

Never touches Sketch. Same multi-RBF MMD as Task 2 (shared/mmd.py); the median
bandwidth is computed per domain pair in the current batch.
"""
import itertools

import torch
import torch.nn.functional as F

from shared.mmd import mmd2_multi_rbf
from task2.methods.base import Method


class DANDG(Method):
    needs_target = False

    def loss(self, xs, ys, xt, progress):
        m = self.cfg["method"]
        lam = float(m.get("lambda_dg", 1.0))
        n_dom = len(self.cfg["data"]["sources"])
        f, logits = self.model(xs)
        ce = F.cross_entropy(logits, ys)
        fd = f.chunk(n_dom, dim=0)  # batches are concatenated per source domain, 8 each
        pairs = list(itertools.combinations(range(n_dom), 2))
        mmd = sum(mmd2_multi_rbf(fd[i], fd[j], m.get("kernel_factors", [0.5, 1.0, 2.0])) for i, j in pairs) / len(pairs)
        return ce + lam * mmd, {"loss_cls": float(ce.item()), "mmd2": float(mmd.item()), "loss_align": float(lam * mmd.item())}
