"""Reproducibility helpers shared by all tasks."""
import os
import random

import numpy as np
import torch

SEED = 6304


def seed_everything(seed: int = SEED, deterministic: bool = False) -> None:
    """Fix Python, NumPy and PyTorch RNGs.

    `deterministic=True` additionally forces cuDNN into deterministic mode,
    which is slower; we keep it off for training and rely on the fixed seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn so that augmentation RNG is reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = SEED) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
