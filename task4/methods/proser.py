"""PROSER (Zhou et al., CVPR 2021) - classifier and data placeholders.

Loss for a mini-batch split into two halves:
  * second half (classifier placeholders):
        L_cls   = CE([z_known, max dummy], y)                       # correct class still wins
        L_dummy = CE([z_known with y masked to -inf, max dummy], K) # dummy is 2nd best
        L1 = L_cls + beta * L_dummy                                 (beta = 1)
  * first half (data placeholders): manifold mixup after layer2 of two different-class
    examples, lambda ~ Beta(2,2); the mixed feature is trained toward the dummy class:
        L2 = CE([z_known(h~), max dummy(h~)], K)
  * total: L = L1 + gamma * L2                                      (gamma = 0.1)
The five dummy classifiers share the penultimate feature; the target index K (=10) refers
to the appended "max dummy" logit. Follows the structure of the reference implementation
(github.com/zhoudw-zdw/CVPR21-Proser), re-implemented here.
"""
import torch
import torch.nn.functional as F

from task4.methods.base import OSRMethod
from task4.methods.manifold_mixup import manifold_mixup


class PROSER(OSRMethod):
    def __init__(self, model, cfg, device):
        super().__init__(model, cfg, device)
        m = cfg["method"]
        self.beta = float(m.get("beta", 1.0))
        self.gamma = float(m.get("gamma", 0.1))
        self.mix_alpha = float(m.get("mixup_alpha", 2.0))
        self.K = int(cfg["data"]["n_classes"])
        self.gen = torch.Generator().manual_seed(int(cfg["seed"]))

    def _with_max_dummy(self, z_known, z_dummy):
        return torch.cat([z_known, z_dummy.max(dim=1, keepdim=True).values], dim=1)

    def train_step(self, x, y, optimizer, scaler=None):
        half = x.shape[0] // 2
        x_mix, y_mix = x[:half], y[:half]        # data placeholders
        x_cls, y_cls = x[half:], y[half:]        # classifier placeholders
        with torch.autocast("cuda", enabled=scaler is not None):
            # --- classifier placeholders on ordinary examples
            f = self.model.features(x_cls)
            z_known = self.model.net.fc(f)
            z_dummy = self.model.dummy_logits(f)
            full = self._with_max_dummy(z_known, z_dummy)
            loss_cls = F.cross_entropy(full, y_cls)
            masked = full.clone()
            masked[torch.arange(len(y_cls)), y_cls] = -1e4
            loss_dummy = F.cross_entropy(masked, torch.full_like(y_cls, self.K))
            # --- data placeholders via manifold mixup after layer2
            h = self.model.forward_pre(x_mix)
            h_mix, lam, _ = manifold_mixup(h, y_mix, self.mix_alpha, self.gen)
            f_mix = self.model.forward_post_feat(h_mix)
            full_mix = self._with_max_dummy(self.model.net.fc(f_mix), self.model.dummy_logits(f_mix))
            loss_mix = F.cross_entropy(full_mix, torch.full_like(y_mix, self.K))
            loss = loss_cls + self.beta * loss_dummy + self.gamma * loss_mix
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        return {"loss": float(loss.item()), "loss_cls": float(loss_cls.item()), "loss_dummy": float(loss_dummy.item()),
                "loss_mix": float(loss_mix.item()), "acc": float((z_known.argmax(1) == y_cls).float().mean().item()),
                "mix_to_dummy_rate": float((full_mix.argmax(1) == self.K).float().mean().item()), "lambda": lam}
