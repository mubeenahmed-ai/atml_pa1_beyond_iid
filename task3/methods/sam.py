"""Sharpness-Aware Minimisation (Foret et al., 2021), non-adaptive.

    e = rho * g / ||g||_2        (ascent step from the ERM gradient)
    theta <- theta - lr * grad L(theta + e)   (AdamW update at the original point)

Two forward/backward passes per batch; BatchNorm running statistics stay frozen
in both passes (shared policy).
"""
import torch
import torch.nn.functional as F

from task2.methods.base import Method


class SAM(Method):
    needs_target = False

    def _params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def train_step(self, xs, ys, xt, progress, optimizer):
        rho = float(self.cfg["method"].get("rho", 0.05))
        params = self._params()
        # pass 1: gradient at theta
        _, logits = self.model(xs)
        loss1 = F.cross_entropy(logits, ys)
        optimizer.zero_grad(set_to_none=True)
        loss1.backward()
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        gnorm = torch.sqrt(sum((g**2).sum() for g in grads)) + 1e-12
        scale = rho / gnorm
        with torch.no_grad():
            e_ws = [g * scale for g in grads]
            for p, e in zip(params, e_ws):
                p.add_(e)
        # pass 2: gradient at theta + e, applied to theta
        optimizer.zero_grad(set_to_none=True)
        _, logits2 = self.model(xs)
        loss2 = F.cross_entropy(logits2, ys)
        loss2.backward()
        with torch.no_grad():
            for p, e in zip(params, e_ws):
                p.sub_(e)
        optimizer.step()
        return {"loss_cls": float(loss1.item()), "loss_perturbed": float(loss2.item()),
                "sharpness_gap": float((loss2 - loss1).item()), "grad_norm": float(gnorm.item()), "loss_total": float(loss2.item())}
