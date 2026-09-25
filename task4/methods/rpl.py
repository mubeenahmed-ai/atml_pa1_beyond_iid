"""Reciprocal Point Learning (Chen et al., ECCV 2020) - optional extension.

* Reciprocal points: one learnable point P_k in R^512 per known class (nn.Parameter,
  K x d), initialised N(0, 0.1^2); they represent the *extra-class* space of class k.
* Distance: d(f, P_k) = ||f - P_k||^2 / d (squared Euclidean, normalised by dimension).
  A feature belongs to class k when it is FAR from P_k, so the class logits are the
  distances themselves: logits_k = d(f, P_k) / T, with T = 1, trained by cross-entropy.
* Open-space regularisation: the distance between class-k features and P_k is pulled
  toward a learnable radius R:  L_o = MSE(d(f_i, P_{y_i}), R) / 2, weighted by lambda_o
  (0.1). This bounds the known-class region and thereby the open space.
* Test score: known-class evidence s(x) = max_k d(f, P_k); unknownness u = -s(x).
Re-implemented from the description in the paper and the public ARPL code base
(github.com/iCGY96/ARPL, `RPLoss`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from task4.methods.base import OSRMethod


class ReciprocalPoints(nn.Module):
    def __init__(self, n_classes: int, feat_dim: int, seed: int = 6304):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.points = nn.Parameter(0.1 * torch.randn(n_classes, feat_dim, generator=g))
        self.radius = nn.Parameter(torch.zeros(1))

    def distances(self, f):  # (B,K): squared euclidean / d
        d2 = (f * f).sum(1, keepdim=True) - 2.0 * f @ self.points.t() + (self.points * self.points).sum(1)
        return d2.clamp_min(0.0) / f.shape[1]


class RPL(OSRMethod):
    def __init__(self, model, cfg, device):
        super().__init__(model, cfg, device)
        m = cfg["method"]
        self.rp = ReciprocalPoints(int(cfg["data"]["n_classes"]), model.feat_dim, int(cfg["seed"])).to(device)
        self.temp = float(m.get("temperature", 1.0))
        self.weight_o = float(m.get("lambda_open", 0.1))

    def parameters(self):
        # the model's fc is unused by RPL; exclude it so the comparison stays on the feature space
        return [p for n, p in self.model.named_parameters() if not n.startswith("net.fc")] + list(self.rp.parameters())

    def logits(self, f):
        return self.rp.distances(f) / self.temp

    def train_step(self, x, y, optimizer, scaler=None):
        with torch.autocast("cuda", enabled=scaler is not None):
            f = self.model.features(x)
        f = f.float()
        dist = self.rp.distances(f)
        loss_cls = F.cross_entropy(dist / self.temp, y)
        own = dist[torch.arange(len(y)), y]
        loss_open = F.mse_loss(own, self.rp.radius.expand_as(own)) / 2.0
        loss = loss_cls + self.weight_o * loss_open
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        return {"loss": float(loss.item()), "loss_cls": float(loss_cls.item()), "loss_open": float(loss_open.item()),
                "radius": float(self.rp.radius.item()), "acc": float((dist.argmax(1) == y).float().mean().item())}

    def extra_state(self):
        return {"reciprocal_points": self.rp.state_dict()}

    def load_extra_state(self, sd):
        self.rp.load_state_dict(sd["reciprocal_points"])
