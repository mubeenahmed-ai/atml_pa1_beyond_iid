"""Stratified 90/10 split of the CIFAR-10 training partition (seed 6304)."""
import argparse
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.logging import save_json  # noqa: E402
from task4.data.cifar10 import cifar10_train_raw  # noqa: E402

SPLIT_PATH = "task4/results/cifar10_split_seed6304.json"


def make(root: str, seed: int = 6304, val_fraction: float = 0.1, out: str = SPLIT_PATH):
    ds = cifar10_train_raw(root)
    y = np.asarray(ds.targets)
    tr, va = train_test_split(np.arange(len(y)), test_size=val_fraction, random_state=seed, stratify=y)
    s = {"seed": seed, "train_idx": sorted(int(i) for i in tr), "val_idx": sorted(int(i) for i in va),
         "counts": {"train": len(tr), "val": len(va), "val_per_class": np.bincount(y[va]).tolist()}}
    save_json(s, out)
    print("saved", out, s["counts"])
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    make(ap.parse_args().root)
