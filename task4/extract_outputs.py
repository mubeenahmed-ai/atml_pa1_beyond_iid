"""Extract penultimate features and logits (fp32, no augmentation) for every fixed model on
CIFAR-10 train / val / test and the fixed CIFAR-100 near / far unknowns.

    python task4/extract_outputs.py --models vanilla gcsc proser rpl

Outputs task4/cache/<model>.npz. Every score in evaluate_osr.py reads these same arrays.
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.logging import load_json  # noqa: E402
from task4.data.cifar10 import TransformSubset, cifar10_test, cifar10_train_raw, eval_transform  # noqa: E402
from task4.data.cifar100_unknowns import unknown_subsets  # noqa: E402
from task4.methods.rpl import ReciprocalPoints  # noqa: E402
from task4.models.resnet_cifar import ResNet18CIFAR  # noqa: E402


@torch.no_grad()
def run(model, loader, device, rp=None):
    model.eval()
    F, Z, Y, D, R = [], [], [], [], []
    for x, y in loader:
        f, z = model(x.to(device))
        F.append(f.cpu().numpy()); Z.append(z.cpu().numpy()); Y.append(np.asarray(y))
        if model.dummy is not None:
            D.append(model.dummy_logits(f).cpu().numpy())
        if rp is not None:
            R.append(rp.distances(f).cpu().numpy())
    out = {"feats": np.concatenate(F), "logits": np.concatenate(Z), "labels": np.concatenate(Y)}
    if D:
        out["dummy_logits"] = np.concatenate(D)
    if R:
        out["rp_distances"] = np.concatenate(R)
    return out


def main(models, results_root, cache_dir, root, split_path, device):
    split = load_json(split_path)
    raw = cifar10_train_raw(root)
    loaders = {
        "train": DataLoader(TransformSubset(raw, split["train_idx"], eval_transform()), batch_size=512, num_workers=6),
        "val": DataLoader(TransformSubset(raw, split["val_idx"], eval_transform()), batch_size=512, num_workers=6),
        "test": DataLoader(cifar10_test(root), batch_size=512, num_workers=6),
    }
    unk, fine_names = unknown_subsets(root)
    for g in ("near", "far"):
        loaders[g] = DataLoader(unk[g]["subset"], batch_size=512, num_workers=6)
    os.makedirs(cache_dir, exist_ok=True)
    for m in models:
        ck = torch.load(os.path.join(results_root, m, "best.pt"), map_location=device, weights_only=False)
        n_dummy = int(ck["cfg"]["method"].get("n_dummy", 0))
        model = ResNet18CIFAR(10, n_dummy).to(device)
        model.load_state_dict(ck["model"])
        rp = None
        if "reciprocal_points" in ck.get("extra", {}):
            rp = ReciprocalPoints(10, model.feat_dim).to(device)
            rp.load_state_dict(ck["extra"]["reciprocal_points"])
        arrays = {"checkpoint_epoch": ck["epoch"], "checkpoint_val_acc": ck["val_acc"]}
        for name, ld in loaders.items():
            o = run(model, ld, device, rp)
            for k, v in o.items():
                arrays[f"{name}_{k}"] = v
            print(f"{m:8s} {name:5s} feats {o['feats'].shape} logits {o['logits'].shape}")
        np.savez_compressed(os.path.join(cache_dir, f"{m}.npz"), **arrays)
    np.save(os.path.join(cache_dir, "cifar100_fine_names.npy"), np.array(fine_names))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vanilla", "gcsc", "proser", "rpl"])
    ap.add_argument("--results_root", default="task4/results")
    ap.add_argument("--cache_dir", default="task4/cache")
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="task4/results/cifar10_split_seed6304.json")
    a = ap.parse_args()
    main(a.models, a.results_root, a.cache_dir, a.root, a.split, "cuda" if torch.cuda.is_available() else "cpu")
