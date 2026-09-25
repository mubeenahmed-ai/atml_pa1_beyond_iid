"""Create the fixed Task 1 splits (seed 6304) and save identifiers.

* stratified 80/20 train/validation split of the official STL-10 training partition
* class-balanced 500-image evaluation subset of the official test partition
"""
import argparse
import os
import sys

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.logging import save_json  # noqa: E402
from task1.data.stl10 import CLASS_NAMES, STL10Parquet  # noqa: E402


def make_splits(cfg: dict) -> dict:
    seed = int(cfg["seed"])
    train = STL10Parquet(cfg["data_root"], "train")
    test = STL10Parquet(cfg["data_root"], "test")

    idx = np.arange(len(train))
    tr_idx, va_idx = train_test_split(
        idx, test_size=cfg["split"]["val_fraction"], random_state=seed, stratify=train.labels
    )

    n_total = int(cfg["split"]["eval_subset_size"])
    n_classes = len(CLASS_NAMES)
    per_class = n_total // n_classes
    rng = np.random.RandomState(seed)
    eval_idx, imbalance = [], {}
    for c in range(n_classes):
        pool = np.where(test.labels == c)[0]
        k = min(per_class, len(pool))
        if k < per_class:
            imbalance[CLASS_NAMES[c]] = int(len(pool))
        eval_idx.extend(rng.choice(pool, size=k, replace=False).tolist())
    eval_idx = sorted(int(i) for i in eval_idx)

    return {
        "seed": seed,
        "dataset": "stl10",
        "class_names": CLASS_NAMES,
        "train_idx": sorted(int(i) for i in tr_idx),
        "val_idx": sorted(int(i) for i in va_idx),
        "eval_idx": eval_idx,
        "eval_labels": [int(test.labels[i]) for i in eval_idx],
        "eval_per_class_imbalance": imbalance,
        "counts": {
            "train": int(len(tr_idx)),
            "val": int(len(va_idx)),
            "eval": len(eval_idx),
            "eval_per_class": np.bincount(test.labels[eval_idx], minlength=n_classes).tolist(),
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task1/configs/task1.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    splits = make_splits(cfg)
    out = os.path.join(cfg["results_dir"], "splits_seed%d.json" % cfg["seed"])
    save_json(splits, out)
    print("saved", out, splits["counts"], "imbalance:", splits["eval_per_class_imbalance"])
