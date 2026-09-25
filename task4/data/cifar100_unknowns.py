"""Fixed CIFAR-100 test classes used as unknowns (evaluation only)."""
import numpy as np
import torchvision
from torch.utils.data import Subset

from task4.data.cifar10 import eval_transform

NEAR = ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"]
FAR = ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"]


def unknown_subsets(root: str):
    ds = torchvision.datasets.CIFAR100(root, train=False, download=False, transform=eval_transform())
    name2id = {n: i for i, n in enumerate(ds.classes)}
    targets = np.asarray(ds.targets)
    out = {}
    for group, names in (("near", NEAR), ("far", FAR)):
        ids = [name2id[n] for n in names]
        idx = np.where(np.isin(targets, ids))[0]
        out[group] = {"subset": Subset(ds, idx), "indices": idx, "class_ids": ids, "class_names": names,
                      "fine_labels": targets[idx]}
    return out, ds.classes
