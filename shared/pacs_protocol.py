"""Shared PACS protocol for Task 2 (UDA) and Task 3 (DG).

* Sketch is the target; Photo / Art Painting / Cartoon are labelled sources.
* Within each source domain: stratified 80/20 train/val split with seed 6304.
* Each training step uses 8 images per source domain (24 source images) and, in
  Task 2 only, 24 unlabelled Sketch images.
* Checkpoints are selected by mean macro-F1 over the three source validation sets.
* BatchNorm running statistics are frozen at the ImageNet values; gamma/beta train.
"""
import itertools
import json
import os
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from common.seed import SEED, make_generator, seed_worker
from shared.pacs import PACSSubset, PACSTable, SOURCE_DOMAINS, TARGET_DOMAIN, eval_transform, train_transform

SPLIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits", f"pacs_sketch_seed{SEED}.json")


# --------------------------------------------------------------------------- splits
def make_splits(table: PACSTable, seed: int = SEED, val_fraction: float = 0.2) -> Dict:
    out = {"seed": seed, "target": TARGET_DOMAIN, "sources": SOURCE_DOMAINS, "val_fraction": val_fraction, "domains": {}}
    for d in SOURCE_DOMAINS:
        idx = table.indices_of(d)
        tr, va = train_test_split(idx, test_size=val_fraction, random_state=seed, stratify=table.labels[idx])
        out["domains"][d] = {"train": sorted(int(i) for i in tr), "val": sorted(int(i) for i in va)}
    out["domains"][TARGET_DOMAIN] = {"all": sorted(int(i) for i in table.indices_of(TARGET_DOMAIN))}
    out["counts"] = {d: {k: len(v) for k, v in s.items()} for d, s in out["domains"].items()}
    return out


def load_splits(path: str = SPLIT_PATH) -> Dict:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- loaders
def _loader(ds, batch_size, shuffle, seed, num_workers, drop_last):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                      drop_last=drop_last, worker_init_fn=seed_worker, generator=make_generator(seed),
                      persistent_workers=num_workers > 0)


def source_train_loaders(table, splits, per_domain_batch=8, seed=SEED, num_workers=6) -> Dict[str, DataLoader]:
    return {d: _loader(PACSSubset(table, splits["domains"][d]["train"], train_transform()), per_domain_batch, True,
                       seed + i, num_workers, drop_last=True) for i, d in enumerate(SOURCE_DOMAINS)}


def source_val_loaders(table, splits, batch_size=128, num_workers=6) -> Dict[str, DataLoader]:
    return {d: _loader(PACSSubset(table, splits["domains"][d]["val"], eval_transform()), batch_size, False, SEED,
                       num_workers, drop_last=False) for d in SOURCE_DOMAINS}


def target_unlabelled_loader(table, splits, batch_size=24, seed=SEED, num_workers=6) -> DataLoader:
    """Task 2 adaptation set: all Sketch images, labels replaced by -1."""
    return _loader(PACSSubset(table, splits["domains"][TARGET_DOMAIN]["all"], train_transform(), with_labels=False),
                   batch_size, True, seed + 100, num_workers, drop_last=True)


def target_eval_loader(table, splits, batch_size=128, num_workers=6) -> DataLoader:
    """Final evaluation only (labels visible). Never imported by training scripts."""
    return _loader(PACSSubset(table, splits["domains"][TARGET_DOMAIN]["all"], eval_transform()), batch_size, False,
                   SEED, num_workers, drop_last=False)


def infinite(loader: DataLoader) -> Iterator:
    while True:
        for b in loader:
            yield b


def steps_per_epoch(splits, per_domain_batch=8) -> int:
    """One 'source epoch' = enough domain-balanced steps to see as many source images as exist."""
    n = sum(len(splits["domains"][d]["train"]) for d in SOURCE_DOMAINS)
    return int(np.ceil(n / (per_domain_batch * len(SOURCE_DOMAINS))))


# --------------------------------------------------------------------------- BN policy
def set_bn_eval(model: nn.Module) -> None:
    """Call after model.train(): keep running mean/var frozen, gamma/beta trainable."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def train_mode_frozen_bn(model: nn.Module) -> None:
    model.train()
    set_bn_eval(model)


# --------------------------------------------------------------------------- evaluation helper
@torch.no_grad()
def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (features, logits, labels) for a loader. Model must expose forward -> (features, logits)."""
    model.eval()
    F, L, Y = [], [], []
    for x, y, _ in loader:
        f, z = model(x.to(device, non_blocking=True))
        F.append(f.float().cpu().numpy()); L.append(z.float().cpu().numpy()); Y.append(np.asarray(y))
    return np.concatenate(F), np.concatenate(L), np.concatenate(Y)
