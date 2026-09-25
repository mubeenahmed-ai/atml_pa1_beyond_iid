"""DANN: source CE + domain CE through a gradient-reversal layer."""
import torch
import torch.nn.functional as F

from task2.methods.base import Method
from task2.models.domain_discriminator import DomainDiscriminator, grl_schedule


class DANN(Method):
    needs_target = True

    def __init__(self, model, cfg, device):
        super().__init__(model, cfg, device)
        m = cfg["method"]
        self.disc = DomainDiscriminator(model.backbone.feat_dim, m.get("disc_hidden", 256), m.get("disc_dropout", 0.5)).to(device)
        self.max_alpha = float(m.get("max_alpha", 1.0))
        self.domain_weight = float(m.get("domain_weight", 1.0))

    def extra_modules(self):
        return [self.disc]

    def loss(self, xs, ys, xt, progress):
        x = torch.cat([xs, xt], 0)
        f, logits = self.model(x)
        ns = xs.shape[0]
        ce = F.cross_entropy(logits[:ns], ys)
        alpha = grl_schedule(progress, max_alpha=self.max_alpha)
        d_logits = self.disc(f, alpha)
        d_true = torch.cat([torch.zeros(ns, dtype=torch.long), torch.ones(xt.shape[0], dtype=torch.long)]).to(f.device)
        d_loss = F.cross_entropy(d_logits, d_true)
        d_acc = (d_logits.argmax(1) == d_true).float().mean()
        return ce + self.domain_weight * d_loss, {"loss_cls": float(ce.item()), "loss_domain": float(d_loss.item()),
                                                  "disc_acc": float(d_acc.item()), "grl_alpha": alpha}
