"""Manifold mixup between examples of *different* known classes (PROSER data placeholders)."""
import torch


def different_class_partner(y: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Return a permutation index j(i) with y[j(i)] != y[i] wherever possible."""
    n = len(y)
    perm = torch.randperm(n, generator=gen).to(y.device)
    same = perm.new_tensor(y[perm] == y)
    for i in torch.where(same)[0].tolist():
        cand = torch.where(y != y[i])[0]
        if len(cand) > 0:
            perm[i] = cand[torch.randint(len(cand), (1,), generator=gen).item()]
    return perm


def manifold_mixup(h: torch.Tensor, y: torch.Tensor, alpha: float, gen: torch.Generator):
    """h: (B,C,H,W) intermediate features. Returns mixed features, lambda and partner index."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    j = different_class_partner(y, gen)
    return lam * h + (1.0 - lam) * h[j], lam, j
