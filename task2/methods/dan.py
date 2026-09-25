"""DAN: source CE + lambda * MMD^2(source features, target features)."""
import torch
import torch.nn.functional as F

from shared.mmd import mmd2_multi_rbf
from task2.methods.base import Method


class DAN(Method):
    needs_target = True

    def loss(self, xs, ys, xt, progress):
        lam = float(self.cfg["method"].get("lambda_mmd", 1.0))
        x = torch.cat([xs, xt], 0)
        f, logits = self.model(x)
        ns = xs.shape[0]
        ce = F.cross_entropy(logits[:ns], ys)
        mmd = mmd2_multi_rbf(f[:ns], f[ns:], self.cfg["method"].get("kernel_factors", [0.5, 1.0, 2.0]))
        return ce + lam * mmd, {"loss_cls": float(ce.item()), "mmd2": float(mmd.item()), "loss_align": float(lam * mmd.item())}
