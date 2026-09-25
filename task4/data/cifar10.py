"""CIFAR-10 known classes: stratified 90/10 train/val split (seed 6304) and transforms."""
from typing import Tuple

import numpy as np
import torchvision
from torch.utils.data import Dataset, Subset
from torchvision import transforms as T

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def train_transform(randaugment: bool = False) -> T.Compose:
    ops = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if randaugment:
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    ops += [T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)]
    return T.Compose(ops)


def eval_transform() -> T.Compose:
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])


class TransformSubset(Dataset):
    """Subset of a torchvision dataset with its own transform (raw PIL underneath)."""

    def __init__(self, base, indices, transform):
        self.base, self.indices, self.transform = base, np.asarray(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, y = self.base[int(self.indices[i])]
        return self.transform(img), y


def cifar10_train_raw(root: str):
    return torchvision.datasets.CIFAR10(root, train=True, download=False, transform=None)


def cifar10_test(root: str):
    return torchvision.datasets.CIFAR10(root, train=False, download=False, transform=eval_transform())
