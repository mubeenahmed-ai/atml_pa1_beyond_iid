"""Gradient-reversal layer and the DANN/CDAN domain discriminator."""
import torch
import torch.nn as nn


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grad_reverse(x, alpha: float):
    return GradReverse.apply(x, alpha)


def grl_schedule(progress: float, gamma: float = 10.0, max_alpha: float = 1.0) -> float:
    """alpha(p) = max_alpha * (2 / (1 + exp(-gamma p)) - 1), p in [0,1]."""
    import math

    return max_alpha * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)


class DomainDiscriminator(nn.Module):
    """in_dim -> 256 -> ReLU -> Dropout(0.5) -> 2 (source / target)."""

    def __init__(self, in_dim: int = 512, hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, g, alpha: float):
        return self.net(grad_reverse(g, alpha))
