"""Source-only ERM: cross-entropy on domain-balanced source batches."""
import torch.nn.functional as F

from task2.methods.base import Method


class SourceOnly(Method):
    needs_target = False

    def loss(self, xs, ys, xt, progress):
        _, logits = self.model(xs)
        ce = F.cross_entropy(logits, ys)
        return ce, {"loss_cls": float(ce.item())}
