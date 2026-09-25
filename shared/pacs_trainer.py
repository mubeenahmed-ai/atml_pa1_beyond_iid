"""Common training loop for every PACS method in Tasks 2 and 3.

Fixed across methods: initialisation, domain-balanced source sampling (8 / domain),
augmentation, AdamW(lr 1e-4, wd 1e-4), <=30 source epochs, early stopping after 5
epochs without improvement of mean source-validation macro-F1, frozen BN statistics,
seed 6304. Only the loss (Method) differs.
"""
import os
import time
from typing import Dict

import numpy as np
import torch

from common.logging import CurveLogger, save_json
from common.metrics import classification_metrics
from common.seed import seed_everything
from shared.pacs import SOURCE_DOMAINS
from shared.pacs_protocol import (infinite, predict, source_train_loaders, source_val_loaders, steps_per_epoch,
                                  target_unlabelled_loader, train_mode_frozen_bn)


def evaluate_sources(model, val_loaders, device, n_classes=7) -> Dict:
    out = {}
    for d, ld in val_loaders.items():
        _, logits, y = predict(model, ld, device)
        out[d] = classification_metrics(y, logits.argmax(1), n_classes)
    out["mean"] = {k: float(np.mean([out[d][k] for d in SOURCE_DOMAINS])) for k in ("accuracy", "macro_f1")}
    out["worst"] = {k: float(np.min([out[d][k] for d in SOURCE_DOMAINS])) for k in ("accuracy", "macro_f1")}
    return out


def train(model, method, table, splits, cfg: dict, out_dir: str, device) -> Dict:
    seed_everything(int(cfg["seed"]))
    tr = cfg["train"]
    os.makedirs(out_dir, exist_ok=True)
    per_dom = int(tr["per_domain_batch"])
    n_workers = int(tr.get("num_workers", 6))

    src_loaders = source_train_loaders(table, splits, per_dom, int(cfg["seed"]), n_workers)
    val_loaders = source_val_loaders(table, splits, num_workers=n_workers)
    src_iters = {d: infinite(ld) for d, ld in src_loaders.items()}
    tgt_iter = infinite(target_unlabelled_loader(table, splits, int(tr["target_batch"]), int(cfg["seed"]), n_workers)) if method.needs_target else None

    # backbone + head always use the shared lr; method-specific modules (domain discriminators) may use a
    # multiple of it (cfg.method.extra_lr_mult, default 1 = identical optimiser settings everywhere)
    groups = [{"params": list(model.parameters()), "lr": float(tr["lr"])}]
    extra = [p for m in method.extra_modules() for p in m.parameters()]
    if extra:
        groups.append({"params": extra, "lr": float(tr["lr"]) * float(cfg["method"].get("extra_lr_mult", 1.0))})
    opt = torch.optim.AdamW(groups, lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))
    method.grad_clip = tr.get("grad_clip", None)  # optional global grad-norm clipping (off by default)

    spe = steps_per_epoch(splits, per_dom)
    max_epochs = int(tr["max_epochs"])
    total_steps = spe * max_epochs
    step_log = CurveLogger(os.path.join(out_dir, "curves_step.csv"))
    epoch_log = CurveLogger(os.path.join(out_dir, "curves_epoch.csv"))
    best, best_epoch, bad, gstep = -1.0, -1, 0, 0
    ckpt_path = os.path.join(out_dir, "best.pt")
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        train_mode_frozen_bn(model)
        for m in method.extra_modules():
            m.train()
        acc: Dict[str, list] = {}
        for _ in range(spe):
            xs, ys = [], []
            for d in SOURCE_DOMAINS:
                x, y, _ = next(src_iters[d])
                xs.append(x); ys.append(y)
            xs = torch.cat(xs).to(device, non_blocking=True)
            ys = torch.cat(ys).to(device, non_blocking=True)
            xt = next(tgt_iter)[0].to(device, non_blocking=True) if tgt_iter is not None else None
            progress = gstep / max(total_steps - 1, 1)
            logs = method.train_step(xs, ys, xt, progress, opt)
            gstep += 1
            for k, v in logs.items():
                acc.setdefault(k, []).append(v)
            if gstep % int(tr.get("log_every", 20)) == 0:
                step_log.log(step=gstep, epoch=epoch, progress=round(progress, 4),
                             **{k: float(np.mean(v[-int(tr.get("log_every", 20)):])) for k, v in acc.items()})
        val = evaluate_sources(model, val_loaders, device, int(cfg["data"]["n_classes"]))
        score = val["mean"]["macro_f1"]
        row = {"epoch": epoch, "step": gstep, **{f"train_{k}": float(np.mean(v)) for k, v in acc.items()},
               **{f"val_{d}_acc": val[d]["accuracy"] for d in SOURCE_DOMAINS},
               **{f"val_{d}_f1": val[d]["macro_f1"] for d in SOURCE_DOMAINS},
               "val_mean_acc": val["mean"]["accuracy"], "val_mean_f1": score, "val_worst_f1": val["worst"]["macro_f1"]}
        improved = score > best
        if improved:
            best, best_epoch, bad = score, epoch, 0
            torch.save({"model": model.state_dict(), "method": method.state_dict(), "epoch": epoch, "val": val, "cfg": cfg}, ckpt_path)
        else:
            bad += 1
        row["best_so_far"] = best
        epoch_log.log(**row)
        print(f"[{cfg['name']}] ep {epoch:02d} " + " ".join(f"{k}={v:.3f}" for k, v in row.items() if k.startswith("train_"))
              + f" | val mean F1 {score:.4f} (best {best:.4f} @ {best_epoch}) {'*' if improved else ''} [{time.time()-t0:.0f}s]", flush=True)
        if bad >= int(tr["patience"]):
            print(f"[{cfg['name']}] early stop at epoch {epoch}")
            break
    summary = {"name": cfg["name"], "best_epoch": best_epoch, "best_val_mean_macro_f1": best, "epochs_run": epoch,
               "steps_per_epoch": spe, "total_steps_run": gstep, "wall_time_s": round(time.time() - t0, 1), "config": cfg}
    save_json(summary, os.path.join(out_dir, "train_summary.json"))
    return summary
