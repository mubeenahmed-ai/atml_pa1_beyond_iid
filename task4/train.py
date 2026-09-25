"""Train one Task 4 model on CIFAR-10 only.   python task4/train.py --config task4/configs/vanilla.yaml

Checkpoint = highest CIFAR-10 validation accuracy. CIFAR-100 is never imported here.
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.logging import CurveLogger, load_json, save_json  # noqa: E402
from common.seed import make_generator, seed_everything, seed_worker  # noqa: E402
from task4.data.cifar10 import TransformSubset, cifar10_train_raw, eval_transform, train_transform  # noqa: E402
from task4.methods import REGISTRY  # noqa: E402
from task4.models.resnet_cifar import ResNet18CIFAR  # noqa: E402


def load_cfg(path):
    base = yaml.safe_load(open(os.path.join(os.path.dirname(path), "base.yaml")))
    over = yaml.safe_load(open(path))
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v
    return base


@torch.no_grad()
def val_accuracy(model, method, loader, device):
    model.eval()
    correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        f, z = model(x)
        if hasattr(method, "logits"):  # RPL classifies by distance to reciprocal points
            z = method.logits(f)
        correct += (z.argmax(1) == y).sum().item(); n += len(y)
    return correct / n


def main(cfg_path):
    cfg = load_cfg(cfg_path)
    seed_everything(int(cfg["seed"]))
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    tr = cfg["train"]
    out_dir = os.path.join(cfg["results_root"], cfg["name"])
    os.makedirs(out_dir, exist_ok=True)

    split = load_json(cfg["data"]["split"])
    raw = cifar10_train_raw(cfg["data"]["root"])
    train_ds = TransformSubset(raw, split["train_idx"], train_transform(bool(tr.get("randaugment", False))))
    val_ds = TransformSubset(raw, split["val_idx"], eval_transform())
    nw = int(cfg["data"]["num_workers"])
    train_ld = DataLoader(train_ds, batch_size=int(tr["batch_size"]), shuffle=True, num_workers=nw, pin_memory=True, drop_last=True,
                          worker_init_fn=seed_worker, generator=make_generator(int(cfg["seed"])), persistent_workers=True)
    val_ld = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=nw, pin_memory=True)

    n_dummy = int(cfg["method"].get("n_dummy", 0))
    model = ResNet18CIFAR(int(cfg["data"]["n_classes"])).to(device)
    if cfg.get("init_from"):
        sd = torch.load(cfg["init_from"], map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        print("initialised from", cfg["init_from"], "(val acc", round(sd.get("val_acc", -1), 4), ")")
    if n_dummy:
        model.add_dummy_classifiers(n_dummy, int(cfg["seed"]))
    method = REGISTRY[cfg["method"]["type"]](model, cfg, device)

    opt = torch.optim.SGD(method.parameters(), lr=float(tr["lr"]), momentum=float(tr["momentum"]), weight_decay=float(tr["weight_decay"]))
    epochs = int(tr["epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if (tr.get("amp", True) and device == "cuda") else None

    log = CurveLogger(os.path.join(out_dir, "curves_epoch.csv"))
    best, best_ep, t0 = -1.0, -1, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        agg = {}
        for x, y in train_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logs = method.train_step(x, y, opt, scaler)
            for k, v in logs.items():
                agg.setdefault(k, []).append(v)
        sched.step()
        va = val_accuracy(model, method, val_ld, device)
        row = {"epoch": ep, "lr": opt.param_groups[0]["lr"], **{f"train_{k}": float(np.mean(v)) for k, v in agg.items()}, "val_acc": va}
        if va > best:
            best, best_ep = va, ep
            torch.save({"model": model.state_dict(), "extra": method.extra_state(), "epoch": ep, "val_acc": va, "cfg": cfg},
                       os.path.join(out_dir, "best.pt"))
        row["best_val_acc"] = best
        log.log(**row)
        print(f"[{cfg['name']}] ep {ep:03d} " + " ".join(f"{k}={v:.4f}" for k, v in row.items() if k.startswith("train_"))
              + f" | val acc {va:.4f} (best {best:.4f} @ {best_ep}) [{time.time()-t0:.0f}s]", flush=True)
    save_json({"name": cfg["name"], "best_epoch": best_ep, "best_val_acc": best, "epochs": epochs,
               "wall_time_s": round(time.time() - t0, 1), "config": cfg}, os.path.join(out_dir, "train_summary.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
