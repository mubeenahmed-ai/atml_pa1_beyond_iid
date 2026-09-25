"""Multi-kernel MMD^2 shared by Task 2 (DAN) and Task 3 (DAN-DG).

Kernel: sum of three RBF kernels k(x,y) = exp(-||x-y||^2 / bw), with
bw = factor * median pairwise squared distance in the current *combined* batch,
factor in {0.5, 1, 2}.

Estimator: the UNBIASED U-statistic (Gretton et al., 2012; used by DAN):
    MMD^2 = 1/(n(n-1)) sum_{i!=j} k(s_i,s_j) + 1/(m(m-1)) sum_{i!=j} k(t_i,t_j) - 2/(nm) sum_{i,j} k(s_i,t_j).
The biased V-statistic (which keeps the diagonal k(x,x)=1 terms) is available as
`mmd2_multi_rbf_biased`. With the small per-domain batches of this protocol (8 images per
domain in DAN-DG, 24 in DAN) its diagonal terms can only be removed by making all features
identical, and AdamW finds that solution within a few dozen steps: the backbone's ReLU
features die, the classifier outputs a constant and CE sticks at ln 7 (see
task3/results/dan_dg_biased and task2/results/dan_lambda10_biased). The unbiased estimator
does not have this incentive.
"""
from typing import Sequence

import torch


def pairwise_sq_dists(z: torch.Tensor) -> torch.Tensor:
    sq = (z * z).sum(1, keepdim=True)
    d2 = sq + sq.t() - 2.0 * z @ z.t()
    return d2.clamp_min(0.0)


def _multi_rbf(z: torch.Tensor, factors: Sequence[float]) -> torch.Tensor:
    d2 = pairwise_sq_dists(z)
    # median over distinct pairs (exclude the zero diagonal); detached so it acts as a fixed bandwidth
    iu = torch.triu_indices(z.shape[0], z.shape[0], offset=1, device=z.device)
    med = d2[iu[0], iu[1]].detach().median().clamp_min(1e-8)
    return sum(torch.exp(-d2 / (f * med)) for f in factors)


def mmd2_multi_rbf(fs: torch.Tensor, ft: torch.Tensor, factors: Sequence[float] = (0.5, 1.0, 2.0)) -> torch.Tensor:
    """Unbiased MMD^2 estimate between feature batches fs (n,d) and ft (m,d)."""
    n, m = fs.shape[0], ft.shape[0]
    K = _multi_rbf(torch.cat([fs, ft], 0), factors)
    K_ss, K_tt, K_st = K[:n, :n], K[n:, n:], K[:n, n:]
    ss = (K_ss.sum() - K_ss.diagonal().sum()) / (n * (n - 1))
    tt = (K_tt.sum() - K_tt.diagonal().sum()) / (m * (m - 1))
    return ss + tt - 2.0 * K_st.mean()


def mmd2_multi_rbf_biased(fs: torch.Tensor, ft: torch.Tensor, factors: Sequence[float] = (0.5, 1.0, 2.0)) -> torch.Tensor:
    """Biased V-statistic (kept for reference / the collapsed runs)."""
    n = fs.shape[0]
    K = _multi_rbf(torch.cat([fs, ft], 0), factors)
    return K[:n, :n].mean() + K[n:, n:].mean() - 2.0 * K[:n, n:].mean()
