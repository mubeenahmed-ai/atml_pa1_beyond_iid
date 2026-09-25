"""CDAN: discriminator conditioned on the outer product of features and class probabilities.

g(x) = vec(f (x) ⊗ p(x)),  f in R^512, p = softmax(logits) in R^7  ->  R^3584.
No entropy conditioning; neither f nor p is detached (required implementation).
"""
import torch
import torch.nn.functional as F

from task2.methods.base import Method
from task2.models.domain_discriminator import DomainDiscriminator, grl_schedule


class CDAN(Method):
    needs_target = True

    def __init__(self, model, cfg, device):
        super().__init__(model, cfg, device)
        m = cfg["method"]
        n_classes = int(cfg["data"]["n_classes"])
        self.disc = DomainDiscriminator(model.backbone.feat_dim * n_classes, m.get("disc_hidden", 256), m.get("disc_dropout", 0.5)).to(device)
        self.max_alpha = float(m.get("max_alpha", 1.0))
        self.domain_weight = float(m.get("domain_weight", 1.0))

    def extra_modules(self):
        return [self.disc]

    def loss(self, xs, ys, xt, progress):
        x = torch.cat([xs, xt], 0)
        f, logits = self.model(x)
        ns = xs.shape[0]
        ce = F.cross_entropy(logits[:ns], ys)
        p = F.softmax(logits, dim=1)
        g = torch.bmm(f.unsqueeze(2), p.unsqueeze(1)).flatten(1)  # (B, 512*7)
        alpha = grl_schedule(progress, max_alpha=self.max_alpha)
        d_logits = self.disc(g, alpha)
        d_true = torch.cat([torch.zeros(ns, dtype=torch.long), torch.ones(xt.shape[0], dtype=torch.long)]).to(f.device)
        d_loss = F.cross_entropy(d_logits, d_true)
        d_acc = (d_logits.argmax(1) == d_true).float().mean()
        return ce + self.domain_weight * d_loss, {"loss_cls": float(ce.item()), "loss_domain": float(d_loss.item()),
                                                  "disc_acc": float(d_acc.item()), "grl_alpha": alpha}
